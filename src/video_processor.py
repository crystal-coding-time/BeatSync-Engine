import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger import setup_environment

# Initialize environment
setup_environment()

# NOW import other modules (after CUDA and Python environment is set)
import argparse
import shutil
from dataclasses import dataclass, field
from typing import TypeAlias, List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import warnings
import time

from gpu_cpu_utils import (
    PARALLEL_WORKERS,
    GPU_INFO as gpu_info,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    VIDEOTOOLBOX_AVAILABLE,
    hw_encoder_available,
    set_gpu_mode,
)
from paths import (
    get_processing_dir,
)

# Import FFmpeg processing module
from ffmpeg_processing import (
    get_video_duration,
    get_cached_video_duration,
    get_cached_video_fps,
    get_video_fps,
    get_video_resolution,
    convert_to_prores_proxy,
    extract_clip_segment_ffmpeg,
    extract_prores_segment_random,
    concatenate_videos_ffmpeg,
    seconds_to_frame_count,
    frame_count_to_seconds,
    is_image_source,
    contributes_render_fps,
    build_ken_burns_filter,
    count_video_frames,
    retime_source_window,
    build_crossfade_chunk,
    truncate_segment_to_frames,
)
from auto_mode.stage6_av_planner import build_planned_clip_sequence, summarize_clip_plan
from effects import build_effect_filters, _stable_rng
from text_overlay import parse_text_entries, plan_text_windows, render_text_png


def _strip_entry_tags(text: str) -> str:
    """Clean text for the classic fallback of a styled render: never burn
    '[style:…]'/'[widget:…]' tags into the video. Prefers styled_text's own
    parser (lazy import — pure-Python-safe even when cairosvg is missing);
    a bare regex strip covers the pathological import failure."""
    try:
        from styled_text import parse_entry_tags
        return parse_entry_tags(text)[0]
    except Exception:
        import re
        return re.sub(r'\s*\[[^\][]*\]\s*', ' ', text).strip()

# Import mode modules
from auto_mode import analyze_beats_auto

warnings.filterwarnings('ignore', message='.*bytes wanted but 0 bytes read.*')

BeatTimes : TypeAlias = np.ndarray
VideoList : TypeAlias = List[str]


def _fmt_seconds(seconds: float) -> str:
    try:
        value = float(seconds)
    except Exception:
        value = 0.0
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _effective_clip_workers(requested_workers: int, use_nvenc: bool) -> int:
    """Choose a stable FFmpeg worker count for clip extraction.

    Multiple simultaneous NVENC encodes can fight for one hardware encoder and
    decode/IO bandwidth. Capping the default is a performance-only change: the
    same clip plan, source times, frame counts, and encoder settings are used.
    """
    requested_workers = max(1, int(requested_workers or 1))
    if not use_nvenc:
        return requested_workers

    cpu = os.cpu_count() or 4
    default_nvenc_workers = min(requested_workers, 4, max(1, cpu // 2))
    return _env_int(
        "BEATSYNC_NVENC_CLIP_WORKERS",
        default_nvenc_workers,
        lo=1,
        hi=requested_workers,
    )


def _assert_output_frames(output_file: str, render_info: Dict) -> None:
    """Run-level zero-drift guard: the assembled video must hold exactly the
    frame-locked timeline's frames. Per-segment counts are checked at extract
    time; this catches anything the assembly stage could still lose."""
    expected = int(render_info.get("timeline_frames") or 0)
    if expected <= 0:
        return
    actual = count_video_frames(output_file)
    if actual is None:
        print("   ⚠️  Could not verify final frame count; skipping the assembly guard")
        return
    if actual != expected:
        raise RuntimeError(
            f"Assembled output has {actual} frames but the frame-locked timeline "
            f"has {expected} — refusing to deliver a drifted video."
        )
    print(f"   ✓ Frame guard: output holds exactly {actual} timeline frames")


def _summarize_clip_timings(timings: List[float], total_duration: float) -> None:
    if not timings:
        return
    arr = np.asarray(timings, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    print(
        f"   ⏱ Clip extraction total: {_fmt_seconds(total_duration)} | "
        f"avg {float(np.mean(arr)):.2f}s, p50 {float(np.percentile(arr, 50)):.2f}s, "
        f"p90 {float(np.percentile(arr, 90)):.2f}s, max {float(np.max(arr)):.2f}s"
    )

# Log status on import if running directly
if __name__ == '__main__':
    if GPU_AVAILABLE:
        print(f"⚡ GPU available: {gpu_info['name']}")
    else:
        print(f"💻 GPU Acceleration: NOT AVAILABLE")

    if NVENC_AVAILABLE:
        print(f"🎬 NVIDIA NVENC: AVAILABLE - Hardware video encoding enabled")
    elif VIDEOTOOLBOX_AVAILABLE:
        print(f"🎬 Apple VideoToolbox: AVAILABLE - Hardware video encoding enabled")
    else:
        print(f"⚠️  Hardware encoding: NOT AVAILABLE - Using CPU encoding only")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create an Auto Mode music video with rhythmic audio-video cuts'
    )
    parser.add_argument(
        'mp3_file',
        type=str,
        help='Path to the input audio file (MP3/WAV/FLAC)'
    )
    parser.add_argument(
        'video_directory',
        type=str,
        help='Directory containing MP4/MKV video files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_music_video.mkv',
        help='Output video file path (default: output_music_video.mkv)'
    )
    parser.add_argument(
        '-s', '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds for audio processing (default: 0.0)'
    )
    parser.add_argument(
        '-e', '--end-time',
        type=float,
        default=None,
        help='End time in seconds for audio processing (default: full duration)'
    )
    parser.add_argument(
        '--lossless',
        action='store_true',
        help='Enable Lossless/Precise mode with ProRes 422 Proxy (frame-accurate cuts, no re-encoding)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (audio analysis + NVENC encoding)'
    )
    parser.add_argument(
        '--gpu-encoder',
        type=str,
        choices=['h264_nvenc', 'hevc_nvenc', 'h264_videotoolbox', 'hevc_videotoolbox', 'none'],
        default='h264_nvenc' if NVENC_AVAILABLE else ('h264_videotoolbox' if VIDEOTOOLBOX_AVAILABLE else 'none'),
        help='Hardware encoder: h264/hevc_nvenc (NVIDIA), h264/hevc_videotoolbox (Apple), none (CPU)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output FPS (frames per second). If not specified, auto-detect from input video (default: auto)'
    )

    return parser.parse_args()


def get_max_resolution(video_files: VideoList) -> Tuple[int, int]:
    """Target resolution = the highest-resolution source (by pixel area).

    Its aspect ratio wins too; other sources adapt via the frame-fit mode.
    Dimensions are rounded down to even for yuv420p encoders.
    """
    best = (0, 0)
    for f in dict.fromkeys(video_files):
        width, height = get_video_resolution(f)
        if width * height > best[0] * best[1]:
            best = (width, height)
    if best == (0, 0):
        best = (1920, 1080)
    return (max(2, best[0] // 2 * 2), max(2, best[1] // 2 * 2))


# Output canvas presets: key -> (human label, fixed WxH or None for legacy
# "highest-resolution source wins" behavior). A fixed canvas keeps one portrait
# phone clip from flipping the whole render; sources adapt via the frame-fit
# mode, which is decided elsewhere (ffmpeg_processing).
OUTPUT_FORMATS: Dict[str, Tuple[str, Optional[Tuple[int, int]]]] = {
    '16:9_1080p': ('16:9 · 1080p (1920×1080)', (1920, 1080)),
    '16:9_4k': ('16:9 · 4K (3840×2160)', (3840, 2160)),
    '9:16_portrait': ('9:16 · Portrait (1080×1920)', (1080, 1920)),
    'match_source': ('Match best source (legacy)', None),
}
DEFAULT_OUTPUT_FORMAT = '16:9_1080p'


def resolve_target_resolution(output_format: str, video_files: VideoList) -> Tuple[int, int]:
    """Resolve the output canvas from an OUTPUT_FORMATS key.

    Fixed formats return their exact WxH regardless of sources; 'match_source'
    keeps the legacy get_max_resolution() behavior. Unknown keys fall back to
    the default so stale saved settings can't crash a render.
    """
    if output_format not in OUTPUT_FORMATS:
        print(f"⚠️ Unknown output format {output_format!r}; using default {DEFAULT_OUTPUT_FORMAT}")
        output_format = DEFAULT_OUTPUT_FORMAT
    label, fixed_size = OUTPUT_FORMATS[output_format]
    if fixed_size is not None:
        target_size = fixed_size
        # The label carries the WxH for the dropdown; the log already prints
        # the dimensions, so keep just the aspect/name part here.
        reason = label.split(' (')[0]
    else:
        target_size = get_max_resolution(video_files)
        reason = 'matched best source, legacy'
    print(f"🖼️ Output canvas: {target_size[0]}x{target_size[1]} ({reason})")
    return target_size


def get_video_files(directory : str) -> VideoList:
    video_extensions = ['.mp4', '.MP4', '.mkv', '.MKV', '.mov', '.MOV',
                        '.webm', '.WEBM', '.m4v', '.M4V', '.avi', '.AVI',
                        '.gif', '.GIF',
                        # Still images (rendered with Ken Burns motion)
                        '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                        '.webp', '.WEBP', '.bmp', '.BMP']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))

    if not video_files:
        raise ValueError(f'No MP4/MKV files found in {directory}')

    return [str(f) for f in video_files]


def build_frame_aligned_cut_timeline(beat_times: BeatTimes, audio_duration: float,
                                     fps: float):
    """
    Build the output cut timeline on absolute video frame numbers.

    This is the critical sync stage: each cut boundary is quantized once from its
    absolute beat time. Segment durations are then frame differences between
    absolute boundaries, so rounding error cannot accumulate from clip to clip.

    The first and last boundaries stay locked to the audio timeline.
    """
    if fps <= 0:
        raise ValueError(f"Invalid FPS for cut timeline: {fps}")
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration: {audio_duration}")

    beats = np.asarray(beat_times, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]
    if beats.size == 0:
        raise ValueError("No valid beat times were provided.")

    beats = np.sort(beats)

    internal_beats = beats[(beats > 0.0) & (beats < audio_duration)]

    raw_cut_times = np.concatenate(([0.0], internal_beats, [audio_duration]))

    # Quantize ABSOLUTE cut positions, not per-segment durations. This removes
    # cumulative drift caused by round((beat[i+1] - beat[i]) * fps) on each clip.
    end_frame = max(1, seconds_to_frame_count(audio_duration, fps))
    cut_frames = np.rint(raw_cut_times * fps).astype(int)
    # Editor's cut-lead: interior cuts land a frame or two BEFORE the beat so
    # the new shot is already onscreen when the transient hits. A uniform
    # shift of interior boundaries only; first/last stay locked, so the total
    # frame count is unchanged.
    cut_lead = _env_int('BEATSYNC_CUT_LEAD_FRAMES', 1, lo=0, hi=2)
    if cut_lead and cut_frames.size > 2:
        cut_frames[1:-1] = cut_frames[1:-1] - cut_lead
    cut_frames = np.clip(cut_frames, 0, end_frame)
    cut_frames[0] = 0
    cut_frames[-1] = end_frame
    cut_frames = np.unique(cut_frames)

    if cut_frames.size == 0 or cut_frames[0] != 0:
        cut_frames = np.insert(cut_frames, 0, 0)
    if cut_frames[-1] != end_frame:
        cut_frames = np.append(cut_frames, end_frame)

    if cut_frames.size < 2:
        cut_frames = np.array([0, end_frame], dtype=int)

    segment_frames = np.diff(cut_frames).astype(int)
    segment_durations = segment_frames / fps
    cut_times = cut_frames / fps
    dropped_boundaries = max(0, raw_cut_times.size - cut_frames.size)

    return cut_times, segment_frames, segment_durations, dropped_boundaries


@dataclass
class ClipJob:
    """Everything one clip-extraction worker needs for one output segment."""
    index: int
    video_file: str
    final_duration: float
    target_size: Tuple[int, int]
    use_nvenc: bool
    gpu_encoder: str
    temp_dir: str
    fps: float
    planned_clip: Dict | None = None
    render_opts: Dict = field(default_factory=dict)
    # Crossfade A side: render this many EXTRA tail frames so the boundary-chunk
    # re-encode can dissolve into the next segment. 0 for every other segment
    # (byte-identical to the pre-crossfade engine).
    xfade_extend_frames: int = 0


# One bad source must not kill a full render: each failed segment gets this
# many re-extraction attempts from seeded fallback sources before we abort.
_CLIP_RESCUE_ATTEMPTS = 3


def create_clip_parallel(job: ClipJob):
    """
    Wrapper function for parallel clip creation using FFmpeg.

    Auto Mode usually passes a planned source moment. If visual planning is not
    available, this worker samples forward source content as a fallback.
    """
    clip_started = time.perf_counter()
    i = job.index
    video_file = job.video_file
    final_duration = job.final_duration
    target_size = job.target_size
    planned_clip = job.planned_clip
    render_opts = job.render_opts
    # Crossfade A side: D extra tail frames to blend into the next segment.
    # Effects/text/Ken Burns stay planned on the original source_duration; only
    # the decoded window and -vframes grow (in extract_clip_segment_ffmpeg).
    extend_frames = max(0, int(job.xfade_extend_frames or 0))
    extend_secs = frame_count_to_seconds(extend_frames, job.fps) if extend_frames else 0.0

    try:
        # Sources shorter than the segment keep the full requested duration:
        # extraction loops them (-stream_loop) instead of emitting short clips.
        retime = None
        partner = None
        if planned_clip:
            video_file = planned_clip.get('video_file') or video_file
            video_duration = get_cached_video_duration(video_file)
            retime = planned_clip.get('retime')
            if retime and extend_frames:
                # Belt-and-braces (mirrors the partner strip below): crossfade
                # selection excludes retimed boundaries, so this is unreachable
                # today — but a retimed A-side's extend window would be added in
                # output-clock seconds while the retime consumes speed× source,
                # clone-freezing the tail mid-dissolve. Drop the extend, keep
                # the retime.
                print(f"   ⚠️  Crossfade extend dropped for clip {i + 1}: retimed segments never extend")
                extend_frames = 0
                extend_secs = 0.0
            if retime:
                # Retime windows keep the plan's value (gated >= 0.6s upstream,
                # so the planner's 0.05s floor never actually engages here).
                source_duration = max(0.05, float(planned_clip.get('source_duration', final_duration)))
            else:
                # The frame-locked timeline (job.final_duration) is the duration
                # authority: the plan's source_duration carries the planner's
                # 0.05s floor, which inflates a sub-floor segment (a 1-frame
                # lead-in at 30fps, or 2 frames at 60fps) past its planned frame
                # count — the extracted clip then holds one frame too many and
                # the assembly guard aborts the render. For every segment the
                # floor never touched, plan source_duration == final_duration by
                # construction, so this is byte-identical.
                source_duration = max(1.0 / max(job.fps, 1.0), float(final_duration))
            if retime:
                # A retimed segment consumes source_window seconds of source;
                # if the clamped window can't fit, strip the ramp instead of
                # looping it (deterministic: depends only on probed duration).
                # source_fps matters for interp retimes (extra decode slack) —
                # pass it so this check, the planner and extraction all size
                # the same window.
                window = retime_source_window(source_duration, retime, job.fps,
                                              source_fps=get_cached_video_fps(video_file))
                if video_duration >= window:
                    max_start = max(0.0, video_duration - window)
                    clip_start = max(0.0, min(float(planned_clip.get('start_time', 0.0)), max_start))
                else:
                    print(f"   ⚠️  Retime dropped for clip {i + 1}: source too short for the ramp window")
                    retime = None
            if not retime:
                # A crossfade A side decodes source_duration + D frames, so
                # clamp the start against the larger window (keeps the tail
                # inside the source when possible; loop/tpad guards cover the
                # rest). extend_secs is 0 for every non-crossfade segment, so
                # this is byte-identical off the crossfade path.
                clamp_duration = source_duration + extend_secs
                if video_duration >= clamp_duration:
                    max_start = max(0.0, video_duration - clamp_duration)
                    clip_start = max(0.0, min(float(planned_clip.get('start_time', 0.0)), max_start))
                else:
                    clip_start = 0.0

            # Duo segments: clamp the partner's start against its own probed
            # duration (same policy as the primary above). Any doubt about the
            # partner drops it and renders the primary solo — a bad probe must
            # not kill the segment.
            partner = planned_clip.get('partner')
            if partner and retime:
                # The planner strips partners from retimed clips; belt-and-braces.
                print(f"   ⚠️  Partner dropped for clip {i + 1}: retimed segments never pair")
                partner = None
            if partner:
                partner_file = partner.get('video_file')
                if not partner_file or not os.path.exists(partner_file):
                    print(f"   ⚠️  Partner dropped for clip {i + 1}: partner source missing ({partner_file})")
                    partner = None
                else:
                    partner_duration = get_cached_video_duration(partner_file)
                    if partner_duration <= 0:
                        print(f"   ⚠️  Partner dropped for clip {i + 1}: could not probe partner duration")
                        partner = None
                    else:
                        # Copy before clamping: the planned clip dict is shared
                        # state (plan summaries, determinism re-runs).
                        partner = dict(partner)
                        partner_source = max(0.05, float(partner.get('source_duration', source_duration)))
                        if partner_duration >= partner_source:
                            partner_max_start = max(0.0, partner_duration - partner_source)
                            partner['start_time'] = max(
                                0.0, min(float(partner.get('start_time', 0.0)), partner_max_start))
                        else:
                            partner['start_time'] = 0.0
        else:
            # Seeded start time from video if visual planning is unavailable:
            # same inputs + settings must always render the same video.
            video_duration = get_cached_video_duration(video_file)

            source_duration = final_duration

            if video_duration >= source_duration:
                max_start = video_duration - source_duration
                clip_start = _stable_rng('clip_fallback', i, video_file).uniform(0.0, max_start)
            else:
                clip_start = 0

        
        # Output file
        temp_clip_path = os.path.join(job.temp_dir, f"temp_clip_{i}_{uuid.uuid4().hex}.mp4")

        opts = render_opts or {}
        effect_filters = build_effect_filters(
            planned_clip,
            opts.get('effect_style', 'clean'),
            opts.get('effect_intensity', 0.0),
            opts.get('tempo'),
            i,
            target_size,
            fps=job.fps,
            mode=opts.get('effect_mode', 'curated'),
            palette=opts.get('effect_palette'),
            palette_seed=opts.get('effect_seed', 0),
            local_beats=(opts.get('segment_beats') or {}).get(i),
            transitions=opts.get('transitions', True),
            semantic_fx=opts.get('semantic_fx', False),
        )

        if is_image_source(video_file):
            # Stills: any start shows the same frame, and each segment gets
            # deterministic Ken Burns motion so the photo reads as footage —
            # unless the effect chain already zooms (avoid double zoompan).
            clip_start = 0.0
            if not any('zoompan' in f for f in effect_filters):
                frame_count = max(1, seconds_to_frame_count(source_duration, job.fps))
                ken_burns = build_ken_burns_filter(
                    _stable_rng('kenburns', i, video_file),
                    target_size, job.fps, frame_count,
                )
                effect_filters = [ken_burns] + effect_filters

        extract_kwargs = {
            'video_file': video_file,
            'start_time': clip_start,
            'duration': source_duration,
            'output_file': temp_clip_path,
            'fps': job.fps,
            'target_size': target_size,
            'use_nvenc': job.use_nvenc,
            'gpu_encoder': job.gpu_encoder,
            'fit_mode': opts.get('fit_mode', 'crop'),
            'extra_filters': effect_filters,
            'text_overlay': (opts.get('text_plan') or {}).get(i),
            'look_cube': opts.get('look_cube'),
            'retime': retime,
            'anchor': (planned_clip or {}).get('subject_anchor'),
            'partner': partner,
            'extend_frames': extend_frames,
        }
        if opts.get('effect_style', 'clean') != 'clean':
            # Beat-reactive echo margins: the blur/hybrid background pulses
            # on these segment-local beat offsets. Key added ONLY for styled
            # renders — Minimal ('clean') never passes it, so its chains stay
            # byte-identical to the pulse-free engine.
            extract_kwargs['local_beats'] = (opts.get('segment_beats') or {}).get(i)

        success = extract_clip_segment_ffmpeg(**extract_kwargs)
        
        elapsed = time.perf_counter() - clip_started
        if not success:
            return (i, None, target_size, None, "FFmpeg extraction failed", elapsed)
        
        return (i, temp_clip_path, target_size, temp_clip_path, None, elapsed)
        
    except Exception as e:
        elapsed = time.perf_counter() - clip_started
        return (i, None, target_size, None, str(e), elapsed)


# --- Opt-in crossfades on calm boundaries -----------------------------------
# Targets whose boundaries are calm enough to dissolve rather than hard-cut.
_XFADE_TARGETS = frozenset({'soft', 'flow'})
# Roughly one in three eligible calm boundaries actually crossfades.
_XFADE_PROBABILITY = 1.0 / 3.0
# The dissolve length before clamping: 0.4s worth of frames.
_XFADE_SECONDS = 0.4
# A segment must be at least this long to lend a boundary to a crossfade.
_XFADE_MIN_SEG_SECONDS = 1.0


def _select_crossfade_boundaries(plan: List[Dict], segment_frames, fps: float) -> Dict[int, Dict]:
    """Deterministically pick calm boundaries to crossfade.

    Returns {i: {'frames': D, 'transition': name}} for each chosen boundary
    between planned clips i and i+1. Rules:
      * BOTH targets in {'soft','flow'};
      * NEITHER clip carries a retime or partner, clip i has no
        transition_out and clip i+1 no transition_in (never double up with a
        split transition or a duo/retime);
      * neither segment shorter than _XFADE_MIN_SEG_SECONDS;
      * ~1 in 3 of the eligible boundaries fire, via a dedicated per-boundary
        rng stream (_stable_rng('xfade', i, fileA, fileB)) — existing plans
        and rng streams are untouched, so plans stay byte-identical;
      * never two adjacent crossfades: a segment is in at most one, so chosen
        boundary indices are always at least 2 apart.
    D = round(_XFADE_SECONDS * fps), clamped to min(lenA, lenB)//3 frames.
    """
    if not plan or fps <= 0:
        return {}
    chosen: Dict[int, Dict] = {}
    last_selected = -2
    for i in range(len(plan) - 1):
        a, b = plan[i], plan[i + 1]
        if str(a.get('target', 'flow')) not in _XFADE_TARGETS:
            continue
        if str(b.get('target', 'flow')) not in _XFADE_TARGETS:
            continue
        if a.get('retime') or b.get('retime'):
            continue
        if a.get('partner') or b.get('partner'):
            continue
        if a.get('transition_out') or b.get('transition_in'):
            continue
        len_a = int(segment_frames[i])
        len_b = int(segment_frames[i + 1])
        if len_a / fps < _XFADE_MIN_SEG_SECONDS or len_b / fps < _XFADE_MIN_SEG_SECONDS:
            continue
        # Adjacency guard first, so a skipped-by-adjacency boundary leaves its
        # own rng stream untouched (each boundary's stream is independent).
        if i <= last_selected + 1:
            continue
        rng = _stable_rng('xfade', i, a.get('video_file'), b.get('video_file'))
        if rng.random() >= _XFADE_PROBABILITY:
            continue
        d_frames = round(_XFADE_SECONDS * fps)
        d_frames = min(d_frames, min(len_a, len_b) // 3)
        if d_frames < 1:
            continue
        # Occasional wipes; fade is the common case. Same rng draw.
        transition_roll = rng.random()
        if transition_roll < 0.75:
            transition = 'fade'
        elif transition_roll < 0.83:
            transition = 'wipeleft'
        elif transition_roll < 0.91:
            transition = 'wiperight'
        else:
            transition = 'smoothup'
        chosen[i] = {'frames': int(d_frames), 'transition': transition}
        last_selected = i
    return chosen


def _assemble_crossfade_chunks(clip_files: List[str], xfade_boundaries: Dict[int, Dict],
                               segment_frames, fps: float, temp_dir: str,
                               use_nvenc: bool, gpu_encoder: str) -> List[str]:
    """Fold each chosen boundary's two segment files into one xfade chunk.

    Returns the concat file list with every chosen (i, i+1) pair replaced by a
    single combined chunk (len_a + len_b frames); untouched segments pass
    through verbatim and still stream-copy. The chunk re-encode is the only
    boundary-chunk re-encode the roadmap contract calls for. On a chunk
    failure the extended A side is truncated back to its planned length so the
    hard-cut fallback can't drift the timeline."""
    assembly: List[str] = []
    n = len(clip_files)
    i = 0
    made = 0
    while i < n:
        spec = xfade_boundaries.get(i)
        if spec and i + 1 < n:
            len_a = int(segment_frames[i])
            len_b = int(segment_frames[i + 1])
            d_frames = int(spec['frames'])
            combined = os.path.join(temp_dir, f"xfade_chunk_{i:05d}_{uuid.uuid4().hex}.mp4")
            ok = build_crossfade_chunk(
                clip_files[i], clip_files[i + 1], combined, len_a, len_b,
                d_frames, fps, spec['transition'], use_nvenc, gpu_encoder)
            if ok:
                assembly.append(combined)
                made += 1
                i += 2
                continue
            # Fallback: the A side holds len_a + D frames; trim it back to
            # len_a so a plain hard cut keeps the timeline frame-exact.
            print(f"   ⚠️  Crossfade chunk {i} failed; falling back to a hard cut")
            trimmed = os.path.join(temp_dir, f"xfade_trim_{i:05d}_{uuid.uuid4().hex}.mp4")
            if truncate_segment_to_frames(clip_files[i], trimmed, len_a, fps,
                                          use_nvenc, gpu_encoder):
                assembly.append(trimmed)
            else:
                raise RuntimeError(
                    f"Crossfade boundary {i} failed and its extended segment could "
                    f"not be trimmed back — refusing to deliver a drifted timeline."
                )
            i += 1
            continue
        assembly.append(clip_files[i])
        i += 1
    print(f"   🎞 Crossfades: {made} calm boundary/boundaries dissolved "
          f"({len(clip_files)} segments → {len(assembly)} concat entries)")
    return assembly


@dataclass
class RenderContext:
    """Everything the render phases share, resolved once up front.

    Built by _resolve_render_config; planned_clip_sequence is filled in by
    _plan_visuals. render_info aliases beat_info['render_info'] so stats
    written here surface in the GUI summary exactly as before.
    """
    # Call arguments
    audio_file: str
    video_files: VideoList
    output_file: str
    start_time: float
    end_time: Optional[float]
    beat_info: Optional[dict]
    lossless_mode: bool
    gpu_encoder: str
    # Resolved configuration
    fps: float
    session_temp_dir: str
    use_nvenc: bool
    max_workers: int
    render_info: Dict
    video_creation_started: float
    # Settings-derived style flags
    fit_mode: str
    effect_style: str
    effect_intensity: float
    effect_mode: str
    effect_palette: Optional[List[str]]
    effect_seed: int
    look_cube: Optional[str]
    text_entries: Optional[List[str]]
    text_position: str
    text_scale: float
    text_style: str
    text_accent: str
    text_font: str
    variety: float
    semantic_variety: float
    media_aware: bool
    semantic_fx: bool
    speed_ramps: bool
    split_screen: bool
    crossfades: bool
    # Frame-locked cut timeline (the duration authority)
    selected_beats: List[float]
    segment_frames: List[int]
    segment_durations: List[float]
    total_clips: int
    target_size: Tuple[int, int]
    # Filled by _plan_visuals (None = legacy random sampling fallback)
    planned_clip_sequence: Optional[List[Dict]] = None


def _resolve_render_config(audio_file: str, video_files: VideoList,
                           beat_times: BeatTimes, output_file: str,
                           start_time: float, end_time: float,
                           max_workers: int, beat_info: dict,
                           lossless_mode: bool, use_gpu: bool,
                           gpu_encoder: str, fps: float, fit_mode: str,
                           output_format: str, effect_style: str,
                           effect_intensity: float, effect_mode: str,
                           effect_palette: List[str], effect_seed: int,
                           look_cube: str, text_entries: List[str],
                           text_position: str, text_scale: float,
                           variety: float, speed_ramps: bool,
                           split_screen: bool, crossfades: bool,
                           settings: Dict, text_style: str = 'classic',
                           text_accent: str = '#FF4D8D',
                           text_font: str = '') -> RenderContext:
    """Merge settings, detect fps, wipe the temp dir, resolve encoder and
    workers, init render_info, and build the frame-locked cut timeline."""
    # A settings dict (GUI path) overrides the individual style kwargs — the
    # positional chain grew past the point where order mistakes are survivable.
    if settings:
        fit_mode = settings.get('fit_mode', fit_mode)
        output_format = settings.get('output_format', output_format)
        effect_style = settings.get('effect_style', effect_style)
        effect_intensity = settings.get('effect_intensity', effect_intensity)
        effect_mode = settings.get('effect_mode', effect_mode)
        effect_palette = settings.get('effect_palette', effect_palette)
        effect_seed = settings.get('effect_seed', effect_seed)
        look_cube = settings.get('look_cube', look_cube)
        text_entries = settings.get('text_entries', text_entries)
        text_position = settings.get('text_position', text_position)
        text_scale = settings.get('text_scale', text_scale)
        text_style = settings.get('text_style', text_style)
        text_accent = settings.get('text_accent', text_accent)
        text_font = settings.get('text_font', text_font)
        variety = settings.get('variety', variety)
        speed_ramps = settings.get('speed_ramps', speed_ramps)
        split_screen = settings.get('split_screen', split_screen)
        crossfades = settings.get('crossfades', crossfades)

    # Visual variety only travels via the settings dict (no positional kwarg
    # on this function) — read it alongside the other settings, defaulting to
    # the same 0.4 the GUI slider ships with.
    semantic_variety = float((settings or {}).get('semantic_variety', 0.4))
    # Media-aware planning and content-aware effects travel the same
    # settings-dict-only route as semantic_variety; both default False so
    # every existing plan stays byte-identical.
    media_aware = bool((settings or {}).get('media_aware', False))
    semantic_fx = bool((settings or {}).get('semantic_fx', False))

    video_creation_started = time.perf_counter()

    if max_workers is None:
        max_workers = PARALLEL_WORKERS

    # Determine FPS to use
    if fps is None:
        # Auto-detect from the first real video file. Stills have no timebase
        # of their own and GIF display-duration timing must not set the render
        # clock (contributes_render_fps), so neither can decide the fps.
        try:
            fps_source = next((f for f in video_files if contributes_render_fps(f)), None)
            if fps_source is None:
                fps = 30.0
                print(f"🎞️ No real-video source (stills/GIFs only); using default FPS: {fps}")
            else:
                fps = get_video_fps(fps_source)
                print(f"🎞️ Auto-detected FPS from input video: {fps}")
        except Exception as e:
            fps = 30.0
            print(f"⚠️ Could not detect FPS, using default: {fps}")
    else:
        print(f"🎞️ Using custom FPS: {fps}")

    # Use fixed processing directory
    session_temp_dir = get_processing_dir()
    
    # 🧹 CLEANUP: Clear processing directory for a fresh start
    try:
        if os.path.exists(session_temp_dir):
            for item in os.listdir(session_temp_dir):
                item_path = os.path.join(session_temp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
        os.makedirs(session_temp_dir, exist_ok=True)
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clear processing directory: {e}")
        
    print(f"📁 Processing directory: {session_temp_dir}")

    # Determine processing mode
    use_nvenc = (not lossless_mode) and gpu_encoder != 'none' and hw_encoder_available(gpu_encoder)
    requested_workers = max_workers
    max_workers = _effective_clip_workers(max_workers, use_nvenc)
    render_info = beat_info.setdefault("render_info", {}) if isinstance(beat_info, dict) else {}
    
    gpu_status = "⚡ GPU" if use_gpu and GPU_AVAILABLE else "💻 CPU"
    encoder_status = f"⚡ {gpu_encoder.upper()}" if use_nvenc else ("🎯 Frame-Perfect ProRes" if lossless_mode else "💻 CPU (libx264)")
    render_info.update({
        "clip_workers": int(max_workers),
        "requested_workers": int(requested_workers),
        "encoder": gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if lossless_mode else "H264_CPU"),
        "output_fps": float(fps),
    })
    # Determine mode name
    mode_name = beat_info.get('mode', 'unknown') if beat_info else 'unknown'
    
    print(f"🎬 Processing Settings:")
    if max_workers != requested_workers:
        print(f"   • Parallel clip workers: {max_workers} (requested {requested_workers}; NVENC contention cap)")
    else:
        print(f"   • Parallel clip workers: {max_workers}")
    print(f"   • Video processing: {encoder_status}")
    print(f"   • Output FPS: {fps}")
    print(f"   • Frame-accurate mode: ENABLED (zero drift)")
    print(f"   • Generation mode: {mode_name}")
    if lossless_mode:
        print(f"   • Lossless Mode: ENABLED (ProRes 422 Proxy)")
        print(f"   • Precision Mode: Frame-perfect (re-encodes all segments)")
        print(f"   • Export Format: Apple ProRes 422 Proxy (.mov)")
    else:
        print(f"   • Export Format: H.264/H.265 (.mkv)")

    # Get audio duration using ffprobe
    audio_duration = get_video_duration(audio_file)
    if end_time and end_time > start_time:
        audio_duration = end_time - start_time
    elif start_time > 0:
        audio_duration = audio_duration - start_time
    render_info["audio_duration"] = float(audio_duration)
    
    print(f"🎵 Audio duration: {audio_duration:.2f} seconds")


    # Build one frame-locked output timeline before creating clips.
    selected_beats, segment_frames, segment_durations, dropped_boundaries = build_frame_aligned_cut_timeline(
        beat_times, audio_duration, fps
    )
    total_clips = len(segment_durations)
    render_info.update({
        "render_cuts": int(total_clips),
        "timeline_boundaries": int(len(selected_beats)),
        "timeline_frames": int(sum(segment_frames)),
    })

    print(f"🎬 Creating video with {total_clips} frame-locked cuts")
    print(f"⏱️  Cut timeline: {len(selected_beats)} boundaries, {sum(segment_frames)} frames")
    if dropped_boundaries:
        print(f"   ⚠️  Dropped {dropped_boundaries} duplicate/too-close cut boundaries after frame quantization")
    # Resolve the output canvas before planning: the planner needs it to decide
    # duo pairing orientation (portrait panes on a landscape canvas, or the
    # converse). Both render branches below reuse this same resolution.
    target_size = resolve_target_resolution(output_format, video_files)
    render_info["target_resolution"] = f"{target_size[0]}x{target_size[1]}"

    return RenderContext(
        audio_file=audio_file, video_files=video_files,
        output_file=output_file, start_time=start_time, end_time=end_time,
        beat_info=beat_info, lossless_mode=lossless_mode,
        gpu_encoder=gpu_encoder, fps=fps,
        session_temp_dir=session_temp_dir, use_nvenc=use_nvenc,
        max_workers=max_workers, render_info=render_info,
        video_creation_started=video_creation_started,
        fit_mode=fit_mode, effect_style=effect_style,
        effect_intensity=effect_intensity, effect_mode=effect_mode,
        effect_palette=effect_palette, effect_seed=effect_seed,
        look_cube=look_cube, text_entries=text_entries,
        text_position=text_position, text_scale=text_scale,
        text_style=text_style, text_accent=text_accent,
        text_font=text_font,
        variety=variety, semantic_variety=semantic_variety,
        media_aware=media_aware, semantic_fx=semantic_fx,
        speed_ramps=speed_ramps, split_screen=split_screen,
        crossfades=crossfades, selected_beats=selected_beats,
        segment_frames=segment_frames, segment_durations=segment_durations,
        total_clips=total_clips, target_size=target_size,
    )


def _plan_visuals(ctx: RenderContext) -> None:
    """Annotate candidates with visual embeddings (optional backend) and run
    the stage6 planner; stores the plan on ctx.planned_clip_sequence."""
    beat_info = ctx.beat_info
    render_info = ctx.render_info
    lossless_mode = ctx.lossless_mode
    semantic_variety = ctx.semantic_variety
    variety = ctx.variety
    video_files = ctx.video_files
    selected_beats = ctx.selected_beats
    segment_durations = ctx.segment_durations
    fps = ctx.fps
    speed_ramps = ctx.speed_ramps
    split_screen = ctx.split_screen
    target_size = ctx.target_size

    # Visual variety: cluster visually-similar candidates via DINOv2 embeddings
    # so stage6 can avoid back-to-back similar-looking shots. Lazy/guarded
    # import — the pipeline must keep running if the embeddings module or its
    # model weights aren't installed. Never runs in ProRes precise mode (that
    # branch stays untouched footage, no planner variety games either).
    if semantic_variety > 0 and not lossless_mode:
        candidates = ((beat_info or {}).get('video_analysis') or {}).get('candidates')
        if candidates:
            try:
                from visual_embeddings import annotate_candidates_with_embeddings
                embed_stats = annotate_candidates_with_embeddings(
                    candidates, sim_threshold=0.82)
                if embed_stats.get('available'):
                    print(f"   🎨 Visual embeddings: {embed_stats.get('embedded', 0)} candidates, "
                          f"{embed_stats.get('clusters', 0)} clusters "
                          f"({embed_stats.get('seconds', 0.0):.1f}s)")
            except ImportError:
                print("   ⚠️  Visual embeddings module not available; skipping visual variety")
            except Exception as e:
                print(f"   ⚠️  Visual embeddings failed, continuing un-annotated: {e}")

    planned_clip_sequence = build_planned_clip_sequence(
        cut_times=selected_beats,
        segment_durations=segment_durations,
        beat_info=beat_info,
        video_files=video_files,
        variety=variety,
        # Nulled in ProRes: the session analysis cache shares candidate dicts
        # by reference, so a prior H.264 render may already have annotated them
        # with embedding/visual_cluster keys — without this, a same-session
        # ProRes plan would pick up semantic penalties a fresh session wouldn't.
        semantic_variety=0.0 if lossless_mode else semantic_variety,
        speed_ramps=speed_ramps,
        lossless=lossless_mode,
        fps=fps,
        # Belt-and-braces on top of the planner's own lossless gate: ProRes
        # precise mode never pairs clips, so don't even ask for duos there.
        split_screen=split_screen and not lossless_mode,
        target_size=target_size,
        # Media-aware auction adjustments (planner picks only — like variety,
        # they choose WHICH source serves a segment, never how it is rendered,
        # so ProRes precise mode keeps them too).
        media_aware=ctx.media_aware,
    )
    if planned_clip_sequence:
        plan_summary = summarize_clip_plan(
            planned_clip_sequence, video_files=video_files,
            candidates=((beat_info or {}).get('video_analysis') or {}).get('candidates'))
        if beat_info is not None:
            beat_info['clip_plan_summary'] = plan_summary
            render_info["plan_summary"] = plan_summary
        print(f"🧠 Auto visual planner: {plan_summary['clip_count']} planned clips")
        print(f"   Sources used: {plan_summary.get('source_count', 0)} (variety {variety:.2f})")
        print(f"   Targets: {plan_summary.get('targets', {})}")
        print(f"   AI-tagged source moments used: {plan_summary.get('ai_tagged', 0)}")
        if plan_summary.get('retimes'):
            print(f"   Speed ramps: {plan_summary['retimes']}")
        usage_hist = plan_summary.get('source_usage') or {}
        if usage_hist:
            print("   Source usage: " + ", ".join(
                f"{name}×{count}" for name, count in usage_hist.items()))
        if plan_summary.get('sources_never_selected'):
            print("   ⚠️  Never selected (had candidates): "
                  + ", ".join(plan_summary['sources_never_selected']))
        if plan_summary.get('sources_without_candidates'):
            print("   ⚠️  No usable candidates found: "
                  + ", ".join(plan_summary['sources_without_candidates']))
    else:
        print("🎲 Visual planner fallback: source moments will use legacy random sampling")

    ctx.planned_clip_sequence = planned_clip_sequence


def _convert_prores_group(indices: List[int], convert_sources: List[str],
                          prores_dir: str, prores_fps: float,
                          target_size: Tuple[int, int],
                          fit_mode: str) -> List[Tuple[int, str]]:
    """Convert one basename-sharing group of sources to ProRes, serially in
    input order (was a closure inside the lossless branch; sources sharing a
    proxy stem must not run concurrently — last one wins, as the old serial
    loop guaranteed)."""
    return [(idx, convert_to_prores_proxy(
                convert_sources[idx], prores_dir, prores_fps,
                target_size=target_size, fit_mode=fit_mode))
            for idx in indices]


def _render_lossless(ctx: RenderContext) -> str:
    """LOSSLESS MODE - ProRes workflow with FRAME-PERFECT precision."""
    audio_file = ctx.audio_file
    video_files = ctx.video_files
    output_file = ctx.output_file
    start_time = ctx.start_time
    end_time = ctx.end_time
    render_info = ctx.render_info
    fps = ctx.fps
    session_temp_dir = ctx.session_temp_dir
    max_workers = ctx.max_workers
    fit_mode = ctx.fit_mode
    segment_durations = ctx.segment_durations
    total_clips = ctx.total_clips
    target_size = ctx.target_size
    planned_clip_sequence = ctx.planned_clip_sequence

    print(f"\n{'='*60}")
    print(f"🎯 LOSSLESS MODE: Converting videos to ProRes 422 Proxy")
    print(f"{'='*60}")
    
    # Create ProRes conversion directory
    prores_dir = os.path.join(session_temp_dir, 'prores')
    os.makedirs(prores_dir, exist_ok=True)
    
    # Use detected FPS for ProRes conversion
    prores_fps = fps
    print(f"🎞️ Using FPS: {prores_fps} (for frame-perfect precision)")

    # Mixed-resolution sources must not reach concat stream-copy: normalize
    # every proxy to one target frame (resolved above, before planning) so
    # all segment streams are identical. Planned clips in this branch only
    # feed extract_prores_segment_random (source + start): partner dicts
    # never reach the ProRes path.

    # Convert input videos to ProRes (video only, no audio).
    #
    # When a plan exists we only ever look up its DISTINCT sources in
    # prores_map (line ~782), so converting the whole library would burn
    # real-time re-encodes on files no segment references (e.g. ~700 dead
    # conversions for a 300-cut plan over a 1000-file library). Convert
    # only the referenced subset: each entry's primary video_file plus,
    # defensively, any duo partner's — the planner excludes duos from
    # lossless, but we don't lean on that. The map-miss fallback pool
    # (prores_files) is seeded from this same subset. It preserves the
    # original video_files order (filtered to the subset) so the fallback
    # RNG iterates a stable ordering; in a consistent plan-exists run every
    # planned source is in the subset, so line ~782 always hits the map and
    # the map-miss branch never fires — output stays byte-identical.
    #
    # With NO plan (legacy pure-random path) any source can be sampled, so
    # convert all of them exactly as before — byte-for-byte identical.
    if planned_clip_sequence:
        referenced = set()
        for entry in planned_clip_sequence:
            primary = entry.get('video_file')
            if primary:
                referenced.add(os.path.abspath(primary))
            partner = entry.get('partner')
            if isinstance(partner, dict):
                partner_file = partner.get('video_file')
                if partner_file:
                    referenced.add(os.path.abspath(partner_file))
        convert_sources = [vf for vf in video_files
                           if os.path.abspath(vf) in referenced]
        print(f"🎯 Plan references {len(convert_sources)} of {len(video_files)} "
              f"sources; converting only those to ProRes")
    else:
        convert_sources = list(video_files)

    # Proxy conversions are independent whole-source re-encodes, so they
    # go through the worker pool. One caveat: the proxy filename derives
    # from the source BASENAME, so two sources sharing a basename share
    # one output path (the old serial loop simply let the later conversion
    # overwrite the earlier one). Those must not run concurrently — group
    # conversions by proxy stem and convert each group serially in input
    # order inside a single pooled task, preserving the serial
    # last-one-wins file content. prores_files/prores_map are filled by
    # original index, so list order (which the seeded fallback choice
    # depends on) never depends on completion order.
    prores_files: List[Optional[str]] = [None] * len(convert_sources)
    prores_map = {}

    conversion_groups: Dict[str, List[int]] = {}
    for idx, video_file in enumerate(convert_sources):
        stem = os.path.splitext(os.path.basename(video_file))[0]
        conversion_groups.setdefault(stem, []).append(idx)

    completed_conversions = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        conversion_futures = [
            executor.submit(_convert_prores_group, indices, convert_sources,
                            prores_dir, prores_fps, target_size, fit_mode)
            for indices in conversion_groups.values()]
        for future in as_completed(conversion_futures):
            for idx, prores_file in future.result():
                prores_files[idx] = prores_file
                prores_map[os.path.abspath(convert_sources[idx])] = prores_file
                completed_conversions += 1
                print(f"   ✓ Converted {completed_conversions}/{len(convert_sources)}")

    print(f"✓ All videos converted to ProRes 422 Proxy (video only)")
    
    # Create segments from ProRes files with FRAME-PERFECT precision
    print(f"\n{'='*60}")
    print(f"✂️  EXTRACTING SEGMENTS (FRAME-PERFECT PRECISION)")
    print(f"{'='*60}")
    print(f"   Mode: Frame-accurate re-encoding")
    print(f"   Method: Exact frame count calculation")
    print(f"   Playback: forward")
    print(f"   Audio: Stripped (will add music at the end)")
    print(f"   FPS: {prores_fps} (fixed)")
    
    segment_files: List[Optional[str]] = [None] * len(segment_durations)
    segments_dir = os.path.join(session_temp_dir, 'segments')
    os.makedirs(segments_dir, exist_ok=True)

    # Phase 1 (serial): resolve every segment's source and start time in
    # index order, so all seeded draws happen in exactly the order the old
    # serial loop made them. Phase 2 then only runs ffmpeg jobs, which
    # never touch the RNG streams.
    extraction_jobs: List[Tuple[int, str, float, float]] = []
    for i, exact_duration in enumerate(segment_durations):
        # Duration comes from the absolute frame-locked timeline.
        planned_clip = planned_clip_sequence[i] if planned_clip_sequence else None

        if planned_clip:
            source_video = os.path.abspath(planned_clip.get('video_file', ''))
            prores_file = prores_map.get(source_video)
            if prores_file is None:
                # A silent substitute here would mask a path-normalization
                # bug between the planner and the proxy map.
                print(f"   ⚠️  Planned source missing from ProRes map: {source_video}; "
                      f"using deterministic fallback source")
                prores_file = _stable_rng('prores_fallback', i).choice(prores_files)
            segment_start = float(planned_clip.get('start_time', 0.0))
        else:
            # Seeded ProRes source + start so precise mode renders the same
            # video for the same inputs even without a visual plan.
            prores_file = _stable_rng('prores_fallback', i).choice(prores_files)
            prores_duration = get_cached_video_duration(prores_file)
            max_start = max(0.0, prores_duration - float(exact_duration))
            segment_start = _stable_rng('prores_start', i, prores_file).uniform(0.0, max_start)

        extraction_jobs.append((i, prores_file, exact_duration, segment_start))

    # Phase 2 (pooled): each extraction is an independent ffmpeg run
    # writing its own segment_<index>.mov; results are collected by index
    # so completion order can never reorder the timeline.
    completed_segments = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                extract_prores_segment_random, prores_file, exact_duration,
                prores_fps, segments_dir, i, start_time=segment_start): i
            for i, prores_file, exact_duration, segment_start in extraction_jobs
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                segment_files[idx] = future.result()
            except Exception as e:
                # The serial loop let extraction errors propagate; keep
                # that contract (a missing segment would silently drift
                # every later cut against the audio).
                raise RuntimeError(
                    f"ProRes segment {idx + 1}/{total_clips} extraction failed: {e}"
                ) from e
            completed_segments += 1
            if completed_segments % 10 == 0:
                print(f"   ✓ Extracted {completed_segments}/{total_clips} segments (frame-perfect)")
    
    print(f"✓ Extracted all {len(segment_files)} segments (frame-perfect, video only)")
    
    # Concatenate and add audio
    print(f"\n{'='*60}")
    print(f"🔗 LOSSLESS CONCATENATION + MUSIC")
    print(f"{'='*60}")
    
    concatenate_videos_ffmpeg(
        video_files=segment_files,
        output_file=output_file,
        audio_file=audio_file,
        start_time=start_time,
        end_time=end_time,
        use_nvenc=False,  # ProRes uses stream copy
        fps=prores_fps,
        temp_dir=session_temp_dir,
        total_frames=int(render_info.get("timeline_frames") or 0)
    )
    _assert_output_frames(output_file, render_info)

    print(f"\n{'='*60}")
    print(f"✅ LOSSLESS VIDEO CREATION COMPLETE!")
    print(f"   Output: {output_file}")
    print(f"   Method: Frame-perfect re-encoding + lossless concatenation")
    print(f"   Quality: ProRes 422 Proxy (lossless)")
    print(f"   Audio: Music track from input file")
    print(f"   FPS: {prores_fps} (fixed)")
    print(f"   Total Segments: {len(segment_files)}")
    print(f"   Timing Precision: Frame-perfect (zero drift)")
    print(f"{'='*60}\n")
    
    # Cleanup
    print(f"🧹 Cleaning up temporary files...")
    if os.name == 'nt':
        # Windows can hold file handles briefly after ffmpeg exits; give
        # the OS a beat before deleting. POSIX has no such lag.
        time.sleep(1.0)
    gc.collect()
    
    for segment_file in segment_files:
        try:
            if os.path.exists(segment_file):
                os.remove(segment_file)
        except Exception:
            pass
    
    for prores_file in prores_files:
        try:
            if os.path.exists(prores_file):
                os.remove(prores_file)
        except Exception:
            pass
    
    try:
        if os.path.exists(segments_dir):
            shutil.rmtree(segments_dir, ignore_errors=True)
        if os.path.exists(prores_dir):
            shutil.rmtree(prores_dir, ignore_errors=True)
    except Exception:
        pass
    
    print(f"✓ Cleanup complete")
    
    return output_file


def _render_standard(ctx: RenderContext) -> str:
    """STANDARD MODE - Direct parallel processing (NO BATCHES)."""
    audio_file = ctx.audio_file
    video_files = ctx.video_files
    output_file = ctx.output_file
    start_time = ctx.start_time
    end_time = ctx.end_time
    beat_info = ctx.beat_info
    render_info = ctx.render_info
    fps = ctx.fps
    session_temp_dir = ctx.session_temp_dir
    use_nvenc = ctx.use_nvenc
    gpu_encoder = ctx.gpu_encoder
    max_workers = ctx.max_workers
    video_creation_started = ctx.video_creation_started
    fit_mode = ctx.fit_mode
    effect_style = ctx.effect_style
    effect_intensity = ctx.effect_intensity
    effect_mode = ctx.effect_mode
    effect_palette = ctx.effect_palette
    effect_seed = ctx.effect_seed
    look_cube = ctx.look_cube
    text_entries = ctx.text_entries
    text_position = ctx.text_position
    text_scale = ctx.text_scale
    text_style = ctx.text_style
    text_accent = ctx.text_accent
    text_font = ctx.text_font or None  # '' = auto (BEATSYNC_FONT / candidates)
    crossfades = ctx.crossfades
    selected_beats = ctx.selected_beats
    segment_frames = ctx.segment_frames
    segment_durations = ctx.segment_durations
    total_clips = ctx.total_clips
    target_size = ctx.target_size
    planned_clip_sequence = ctx.planned_clip_sequence

    # Output canvas (target_size) was resolved in _resolve_render_config,
    # before planning.
    print(f"\n{'='*60}")
    print(f"🎬 PROCESSING ALL CLIPS (No batch processing with FFmpeg)")
    print(f"   Total clips: {total_clips}")
    print(f"   Parallel workers: {max_workers}")
    print(f"   Frame-accurate: ENABLED")
    if use_nvenc:
        vendor = 'Apple' if 'videotoolbox' in gpu_encoder else 'NVIDIA'
        print(f"   Encoder: ⚡ {vendor} {gpu_encoder.upper()} (GPU-accelerated)")
    else:
        print(f"   Encoder: 💻 libx264 (CPU)")
    print(f"{'='*60}\n")
    
    text_plan = {}
    if text_entries:
        entries = parse_text_entries(text_entries)
        # Motion-style progress widgets need windows at least as long as
        # their 'duration:Ns' (plus fades) or the bar completes early; the
        # availability check runs BEFORE planning so a cairosvg-less fallback
        # to classic also gets the classic window math. min_durations=None
        # (classic) keeps planning bit-for-bit identical to the historic path.
        styled_ready = False
        min_durations = None
        if text_style != 'classic':
            try:
                import cairosvg  # noqa: F401 — probe only; render imports it again
                if os.environ.get('BEATSYNC_DISABLE_STYLEDTEXT'):
                    raise RuntimeError('disabled via BEATSYNC_DISABLE_STYLEDTEXT')
                from styled_text import parse_entry_tags
                styled_ready = True
                durs = [parse_entry_tags(t)[2] for t, _ in entries]
                min_durations = [d if d and d > 0 else None for d in durs]
                if not any(min_durations):
                    min_durations = None
            except Exception as e:
                print(f"   ⚠️  Styled text unavailable ({e}) — using classic rendering")
        seg_map, schedule = plan_text_windows(
            entries, selected_beats,
            beat_times=(beat_info or {}).get('times'),
            planned_clip_sequence=planned_clip_sequence,
            min_durations=min_durations,
        )
        styled_patterns = None
        styled_band_y = 0
        if styled_ready and seg_map:
            # Styled path: same planner, per-frame SVG sequences instead of
            # one static PNG. Any failure (no font, rasterizer error)
            # degrades to the classic renderer with one log line.
            try:
                from styled_text import build_styled_sequences
                styled_patterns, styled_band_y = build_styled_sequences(
                    entries=entries, schedule=schedule, seg_map=seg_map,
                    cut_times=selected_beats, segment_frames=segment_frames,
                    fps=fps, target_size=target_size,
                    audio_file=ctx.audio_file, temp_dir=session_temp_dir,
                    position=text_position, scale=text_scale,
                    accent=text_accent, font_path=text_font)
            except Exception as e:
                print(f"   ⚠️  Styled text unavailable ({e}) — using classic rendering")
                styled_patterns = None
                if min_durations is not None:
                    # The widget-widened windows only make sense for the styled
                    # renderer; the classic fallback re-plans with the classic
                    # window math so captions keep their historic pacing.
                    seg_map, schedule = plan_text_windows(
                        entries, selected_beats,
                        beat_times=(beat_info or {}).get('times'),
                        planned_clip_sequence=planned_clip_sequence,
                        min_durations=None,
                    )
        if styled_patterns is not None:
            for seg_idx, (text, fade_in_start, fade_in_duration, fade_out_start) in seg_map.items():
                pattern = styled_patterns.get(seg_idx)
                if pattern:
                    # 5th element = overlay y for the band-cropped sequence
                    # (see styled_text._compute_band). Classic entries stay
                    # 4-tuples so the historic command strings are untouched;
                    # ffmpeg_processing unpacks both shapes.
                    text_plan[seg_idx] = (pattern, fade_in_start,
                                          fade_in_duration, fade_out_start,
                                          styled_band_y)
        else:
            png_cache = {}
            for seg_idx, (text, fade_in_start, fade_in_duration, fade_out_start) in seg_map.items():
                if text not in png_cache:
                    # A styled render that degraded to classic must not burn
                    # raw '[style:…]' tags into the video; explicit Classic
                    # style keeps its text verbatim (byte-identity).
                    draw_text = _strip_entry_tags(text) if text_style != 'classic' else text
                    png = None
                    if draw_text:
                        png_path = os.path.join(session_temp_dir, f"text_{len(png_cache):03d}.png")
                        png = render_text_png(draw_text, target_size, png_path,
                                              position=text_position, scale=text_scale,
                                              font_path=text_font)
                    # Cache failures (None) too, so a missing font warns once
                    # instead of once per segment.
                    png_cache[text] = png
                png = png_cache[text]
                if png:
                    text_plan[seg_idx] = (png, fade_in_start, fade_in_duration, fade_out_start)
        for text, ws, we in schedule:
            print(f"   Text overlay: 📝 {ws:6.2f}s–{we:6.2f}s  {text[:60]!r}")
        skipped = len(entries) - len(schedule)
        text_summary = (f"Text: placed {len(schedule)}/{len(entries)} entries, "
                        f"{skipped} skipped" + (" (see render log)" if skipped else ""))
        print(f"   📝 {text_summary}")
        # Handoff for the curated UI status line: the console logger lives in
        # the orchestrator, which reads render_info after the render returns
        # (render_log._stage6_summary). Stashing the line here lets that
        # summary surface it in the on-screen status box.
        render_info['text_summary'] = text_summary

    # Interior beat offsets per segment (in each segment's local clock) so
    # beat-locked effects fire on real beats, not a tempo approximation.
    segment_beats: Dict[int, List[float]] = {}
    beat_grid = np.asarray((beat_info or {}).get('times', []), dtype=float)
    beat_grid = beat_grid[np.isfinite(beat_grid)]
    if beat_grid.size:
        for i in range(total_clips):
            seg_start, seg_end = selected_beats[i], selected_beats[i + 1]
            local = beat_grid[(beat_grid >= seg_start - 1e-6) & (beat_grid < seg_end - 1e-6)] - seg_start
            if local.size:
                segment_beats[i] = [round(float(b), 4) for b in local[:8]]

    render_opts = {
        'fit_mode': fit_mode,
        'effect_style': effect_style,
        'effect_intensity': effect_intensity,
        'effect_mode': effect_mode,
        'effect_palette': effect_palette,
        'effect_seed': effect_seed,
        'look_cube': look_cube,
        'tempo': (beat_info or {}).get('tempo'),
        'text_plan': text_plan,
        'segment_beats': segment_beats,
        # Content-aware effect selection (veto matrix + impact-weighted
        # firing) inside build_effect_filters; False = historical engine.
        'semantic_fx': ctx.semantic_fx,
        # Split transitions ride the effects engine, so they follow the
        # style: any non-clean style gets them.
        'transitions': bool(effect_style and effect_style != 'clean'),
    }
    if effect_style and effect_style != 'clean':
        print(f"   Effects: 🎨 {effect_style} (intensity {effect_intensity:.2f}) | Frame fit: {fit_mode}")
    else:
        print(f"   Frame fit: {fit_mode}")

    # Opt-in crossfades on calm boundaries (never in ProRes precise mode —
    # this whole branch is the standard path). Selection is deterministic
    # over the finished plan; the A side of each chosen boundary renders D
    # extra tail frames, which the boundary-chunk re-encode dissolves into
    # B during assembly. crossfades off (default) → xfade_boundaries empty
    # → every ClipJob carries extend 0 and the assembly is untouched.
    xfade_boundaries: Dict[int, Dict] = {}
    if crossfades and planned_clip_sequence:
        xfade_boundaries = _select_crossfade_boundaries(
            planned_clip_sequence, segment_frames, fps)
        if xfade_boundaries:
            print(f"   🎞 Crossfades: {len(xfade_boundaries)} calm boundary/boundaries "
                  f"selected to dissolve")

    clip_args = []
    for i, final_duration in enumerate(segment_durations):
        # Duration comes from the absolute frame-locked cut timeline.
        planned_clip = planned_clip_sequence[i] if planned_clip_sequence else None
        video_file = (planned_clip.get('video_file') if planned_clip
                      else _stable_rng('source_fallback', i).choice(video_files))
        xfade_extend = int(xfade_boundaries.get(i, {}).get('frames', 0))
        clip_args.append(ClipJob(
            index=i, video_file=video_file, final_duration=final_duration,
            target_size=target_size, use_nvenc=use_nvenc, gpu_encoder=gpu_encoder,
            temp_dir=session_temp_dir, fps=fps,
            planned_clip=planned_clip, render_opts=render_opts,
            xfade_extend_frames=xfade_extend))
    
    clip_files = [None] * len(clip_args)
    clip_timings: List[float] = []
    clip_stage_started = time.perf_counter()
    
    # Process all clips in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(create_clip_parallel, args): idx 
            for idx, args in enumerate(clip_args)
        }
        
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                i, clip_path, new_target_size, temp_path, error, clip_elapsed = future.result()

                if clip_elapsed:
                    clip_timings.append(float(clip_elapsed))
                
                if error:
                    src_name = os.path.basename(clip_args[idx].video_file or '?')
                    print(f"⚠️  Warning: Clip {i+1} failed after {_fmt_seconds(clip_elapsed)}: "
                          f"{error} (source: {src_name})")
                    continue
                
                if clip_path is not None:
                    clip_files[idx] = clip_path
                    
                    completed += 1
                    if completed % 10 == 0 or completed == len(clip_args):
                        progress = (completed / len(clip_args)) * 100
                        elapsed = time.perf_counter() - clip_stage_started
                        rate = completed / max(0.001, elapsed)
                        print(
                            f"   ⚡ Progress: {completed}/{len(clip_args)} clips ({progress:.1f}%) "
                            f"[{_fmt_seconds(elapsed)}, {rate:.2f} clips/s]"
                        )
                
            except Exception as e:
                print(f"⚠️  Warning: Error processing clip: {str(e)}")
                continue
    
    clip_stage_seconds = time.perf_counter() - clip_stage_started
    _summarize_clip_timings(clip_timings, clip_stage_seconds)

    # Never drop a failed clip from the timeline: that would compress the
    # output and make every later cut drift against the audio. Instead,
    # rescue the segment — re-extract the same time slot (same frame count)
    # from seeded fallback sources. Runs only when something already failed,
    # so clean renders stay byte-identical; given the same failure set, the
    # rescue itself is deterministic.
    failed_indices = [idx for idx, f in enumerate(clip_files) if f is None]
    for idx in failed_indices:
        job = clip_args[idx]
        rescued = False
        for attempt in range(1, _CLIP_RESCUE_ATTEMPTS + 1):
            pool = [v for v in video_files if v != job.video_file] or list(video_files)
            fallback = _stable_rng('clip_rescue', idx, attempt).choice(pool)
            print(f"   🔁 Rescue: re-extracting clip {idx + 1} from fallback source "
                  f"{os.path.basename(fallback)} (attempt {attempt}/{_CLIP_RESCUE_ATTEMPTS})")
            rescue_job = ClipJob(
                index=job.index, video_file=fallback,
                final_duration=job.final_duration, target_size=job.target_size,
                use_nvenc=job.use_nvenc, gpu_encoder=job.gpu_encoder,
                temp_dir=job.temp_dir, fps=job.fps,
                # Planless path: seeded start time, exact frame count. The
                # original plan's start/retime/partner belong to the failed
                # source and cannot transfer.
                planned_clip=None,
                render_opts=job.render_opts,
                xfade_extend_frames=job.xfade_extend_frames)
            _, clip_path, _, _, error, clip_elapsed = create_clip_parallel(rescue_job)
            if clip_path is not None and not error:
                clip_files[idx] = clip_path
                rescued = True
                break
            print(f"   ⚠️  Rescue attempt {attempt} failed after "
                  f"{_fmt_seconds(clip_elapsed)}: {error}")
        if not rescued:
            raise RuntimeError(
                f"Clip {idx + 1} failed and all {_CLIP_RESCUE_ATTEMPTS} rescue attempts "
                f"failed; refusing to concatenate an incomplete timeline."
            )

    if not clip_files:
        raise ValueError('No valid video clips could be created')

    # Fold chosen calm boundaries into crossfade chunks (one boundary-chunk
    # re-encode each; every other segment still stream-copies). The
    # combined chunk holds len_a + len_b frames, so the concat list just
    # has fewer, longer entries — the total-frames bound is unchanged.
    assembly_files = clip_files
    if xfade_boundaries:
        assembly_files = _assemble_crossfade_chunks(
            clip_files, xfade_boundaries, segment_frames, fps,
            session_temp_dir, use_nvenc, gpu_encoder)

    print(f"\n{'='*60}")
    print(f"🎬 FINAL ASSEMBLY: Concatenating {len(assembly_files)} clips")
    print(f"{'='*60}\n")

    # Concatenate all clips and add audio
    assembly_started = time.perf_counter()
    concatenate_videos_ffmpeg(
        video_files=assembly_files,
        output_file=output_file,
        audio_file=audio_file,
        start_time=start_time,
        end_time=end_time,
        use_nvenc=use_nvenc,
        gpu_encoder=gpu_encoder,
        fps=fps,
        temp_dir=session_temp_dir,
        total_frames=int(render_info.get("timeline_frames") or 0)
    )
    assembly_seconds = time.perf_counter() - assembly_started
    render_info["final_assembly_seconds"] = float(assembly_seconds)
    print(f"   ⏱ Final assembly total: {_fmt_seconds(assembly_seconds)}")
    _assert_output_frames(output_file, render_info)
 
    print(f"\n🧹 Cleaning up resources...")
    
    # Cleanup clip files
    for clip_file in clip_files:
        try:
            if os.path.exists(clip_file):
                os.remove(clip_file)
        except Exception as e:
            print(f"⚠️  Warning: Could not delete clip file: {e}")
    
    # Clean up processing directory
    try:
        if os.path.exists(session_temp_dir):
            shutil.rmtree(session_temp_dir, ignore_errors=True)
            print(f"✓ Cleaned up processing directory")
    except Exception as e:
        print(f"⚠️  Warning: Could not delete processing directory: {e}")
    
    gc.collect()
 
    print(f"\n{'='*60}")
    print(f"✅ VIDEO CREATION COMPLETE!")
    print(f"   Output: {output_file}")
    print(f"   FPS: {fps} (frame-accurate)")
    print(f"   Total Cuts: {total_clips}")
    print(f"   Zero Drift: Absolute frame-locked cut timeline")
    print(f"   Total video creation time: {_fmt_seconds(time.perf_counter() - video_creation_started)}")
    print(f"{'='*60}\n")
    
    return output_file


def create_music_video(audio_file: str, video_files: VideoList, beat_times: BeatTimes,
                      output_file: str = 'output_music_video.mkv',
                      start_time: float = 0.0, end_time: float = None,
                      max_workers: int = None,
                      beat_info: dict = None,
                      lossless_mode: bool = False, use_gpu: bool = False,
                      gpu_encoder: str = 'h264_nvenc', fps: float = None,
                      fit_mode: str = 'crop',
                      output_format: str = DEFAULT_OUTPUT_FORMAT,
                      effect_style: str = 'clean',
                      effect_intensity: float = 0.7,
                      effect_mode: str = 'curated', effect_palette: List[str] = None,
                      effect_seed: int = 0, look_cube: str = None,
                      text_entries: List[str] = None, text_position: str = 'bottom',
                      text_scale: float = 1.0, variety: float = 0.4,
                      speed_ramps: bool = False,
                      split_screen: bool = True,
                      crossfades: bool = False,
                      settings: Dict = None) -> str:
    """
    Creates a music video with video clips cut to detected beats.
    
    **PURE FFMPEG IMPLEMENTATION - FRAME-ACCURATE**
    
    ✅ NO BATCH PROCESSING: FFmpeg handles memory independently
    ✅ FRAME-ACCURATE: Uses exact frame counts for zero drift
    ✅ NO CUMULATIVE ERROR: Each segment is precisely timed
    
    Args:
        audio_file: Path to audio file
        video_files: List of video file paths
        beat_times: Array of beat times (already processed by mode)
        output_file: Output file path
        start_time: Audio start time
        end_time: Audio end time
        max_workers: Number of parallel workers
        beat_info: Beat information dictionary
        lossless_mode: Use ProRes 422 Proxy mode
        use_gpu: Use GPU acceleration
        gpu_encoder: GPU encoder to use
        fps: Output FPS
    
    Returns:
        Path to output video file
    """
    if len(beat_times) == 0:
        raise ValueError("No beats were detected. Cannot create video.")

    ctx = _resolve_render_config(
        audio_file=audio_file, video_files=video_files, beat_times=beat_times,
        output_file=output_file, start_time=start_time, end_time=end_time,
        max_workers=max_workers, beat_info=beat_info,
        lossless_mode=lossless_mode, use_gpu=use_gpu,
        gpu_encoder=gpu_encoder, fps=fps, fit_mode=fit_mode,
        output_format=output_format, effect_style=effect_style,
        effect_intensity=effect_intensity, effect_mode=effect_mode,
        effect_palette=effect_palette, effect_seed=effect_seed,
        look_cube=look_cube, text_entries=text_entries,
        text_position=text_position, text_scale=text_scale, variety=variety,
        speed_ramps=speed_ramps, split_screen=split_screen,
        crossfades=crossfades, settings=settings)

    _plan_visuals(ctx)

    if ctx.lossless_mode:
        return _render_lossless(ctx)
    return _render_standard(ctx)
 
 
def main() -> None:
    args = parse_arguments()
 
    if not os.path.exists(args.mp3_file):
        raise FileNotFoundError('Audio file not found: ' + args.mp3_file)
 
    if not os.path.isdir(args.video_directory):
        raise NotADirectoryError('Video directory not found: ' + args.video_directory)
 
    # Enable GPU mode if requested
    if args.gpu:
        if GPU_AVAILABLE:
            set_gpu_mode(True)
            print(f"⚡ GPU acceleration ENABLED: {gpu_info}")
        else:
            print(f"⚠️  GPU requested but CuPy not available, using CPU")
            args.gpu = False
 
    print(f"\n{'='*60}")
    print(f"🎵 BEATSYNC ENGINE - AUTO MODE")
    print(f"   Audio Analysis: {'⚡ GPU' if args.gpu else '💻 CPU'}")
    if hw_encoder_available(args.gpu_encoder) and not args.lossless:
        print(f"   Video Encoding: ⚡ {args.gpu_encoder.upper()}")
    else:
        print(f"   Video Encoding: 💻 CPU")
    if args.fps:
        print(f"   FPS: {args.fps} (custom)")
    else:
        print(f"   FPS: Auto-detect from input video")
    print(f"   Mode: AUTO")
    if args.lossless:
        print(f"   Export: 🎯 Lossless/Precise (ProRes 422 Proxy - Frame Perfect)")
    else:
        print(f"   Export: 📹 H.264/H.265 (.mkv) - Frame Accurate")
    print(f"{'='*60}\n")
    
    print(f'📁 Audio file: {args.mp3_file}')
    video_files = get_video_files(args.video_directory)
    print(f'✓ Found {len(video_files)} video files')
 
    print(f"🤖 Using AUTO mode")
    selected_beats, beat_info = analyze_beats_auto(
        args.mp3_file,
        start_time=args.start_time,
        end_time=args.end_time,
        use_gpu=args.gpu,
        video_files=video_files,
    )
 
    print(f'✓ Selected {len(selected_beats)} cuts for video')
 
    output_file = args.output
    if args.lossless and not output_file.lower().endswith('.mov'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mov'
        print(f'📝 Changed output to .mov for Lossless mode: {output_file}')
    elif not args.lossless and not output_file.lower().endswith('.mkv'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mkv'
        print(f'📝 Changed output to .mkv: {output_file}')
 
    print(f'\n🎬 Starting video creation (frame-accurate)...\n')
    output_file = create_music_video(
        args.mp3_file,
        video_files,
        selected_beats,  # Pass pre-processed beats
        output_file=output_file,
        start_time=args.start_time,
        end_time=args.end_time,
        max_workers=PARALLEL_WORKERS,
        beat_info=beat_info,
        lossless_mode=args.lossless,
        use_gpu=args.gpu,
        gpu_encoder=args.gpu_encoder,
        fps=args.fps
    )
 
    print(f'✅ Music video created successfully: {output_file}')
 
 
if __name__ == '__main__':
    main()
