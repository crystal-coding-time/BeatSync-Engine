#!/usr/bin/env python3
"""
FFmpeg Processing Module - Frame-Accurate Video Operations
- Frame-perfect segment extraction
- Zero timing drift
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import hashlib
import math
import subprocess
import time
import json
import random
import shutil
import uuid
import re
from typing import Tuple, List


from logger import (
    setup_environment,
    FFMPEG_EXE as FFMPEG_PATH,
    check_nvenc as check_nvenc_support,
)
from gpu_cpu_utils import MAX_THREADS

# Initialize environment
setup_environment()

# Set up FFPROBE_PATH based on FFMPEG_PATH
_FFMPEG_DIR, _FFMPEG_NAME = os.path.split(FFMPEG_PATH)
FFPROBE_PATH = os.path.join(_FFMPEG_DIR, _FFMPEG_NAME.replace('ffmpeg', 'ffprobe'))


NVENC_QUALITY_CQ = '1'
NVENC_LOOKAHEAD = '32'
NVENC_AQ_STRENGTH = '12'


def _run_media_command(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run an FFmpeg/FFprobe command with consistent capture settings."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _safe_remove_file(path: str | None) -> None:
    """Best-effort removal for temporary media files."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            print(f"   ⚠️  Could not remove temporary file {os.path.basename(path)}: {e}")


def _short_ffmpeg_error(stderr: str, max_chars: int = 2200) -> str:
    """Return the useful tail of FFmpeg stderr without flooding the UI/log."""
    if not stderr:
        return ""
    text = stderr.strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _fmt_seconds(seconds: float) -> str:
    try:
        value = float(seconds)
    except Exception:
        value = 0.0
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _extract_scene_times(stderr: str) -> List[float]:
    """Parse showinfo pts_time values from FFmpeg stderr."""
    scene_changes: List[float] = []
    for line in stderr.split('\n'):
        if 'pts_time:' in line:
            match = re.search(r'pts_time:([\d.]+)', line)
            if match:
                try:
                    scene_changes.append(float(match.group(1)))
                except ValueError:
                    pass
    scene_changes = sorted(set(round(t, 3) for t in scene_changes if t >= 0.0))
    cleaned: List[float] = []
    for t in scene_changes:
        if not cleaned or t - cleaned[-1] >= 0.08:
            cleaned.append(t)
    return cleaned


def get_nvenc_quality_args(gpu_encoder: str, include_pix_fmt: bool = True) -> List[str]:
    """Return high-quality NVENC settings for H.264/HEVC exports."""
    args = [
        '-c:v', gpu_encoder,
        '-preset', 'p7',
        '-tune', 'uhq' if gpu_encoder == 'hevc_nvenc' else 'hq',
        '-rc', 'vbr',
        '-b:v', '0',
        '-cq', NVENC_QUALITY_CQ,
        '-multipass', 'fullres',
        '-rc-lookahead', NVENC_LOOKAHEAD,
        '-spatial_aq', '1',
        '-temporal_aq', '1',
        '-aq-strength', NVENC_AQ_STRENGTH,
        '-b_ref_mode', 'middle',
    ]

    if gpu_encoder == 'h264_nvenc':
        args.extend(['-profile:v', 'high'])
    elif gpu_encoder == 'hevc_nvenc':
        args.extend(['-profile:v', 'main'])

    if include_pix_fmt:
        args.extend(['-pix_fmt', 'yuv420p'])

    return args


def get_videotoolbox_quality_args(gpu_encoder: str, include_pix_fmt: bool = True) -> List[str]:
    """Return high-quality Apple VideoToolbox settings for H.264/HEVC exports."""
    args = [
        '-c:v', gpu_encoder,
        '-q:v', '65',
        '-allow_sw', '1',
    ]
    if gpu_encoder == 'hevc_videotoolbox':
        args.extend(['-tag:v', 'hvc1'])
    if include_pix_fmt:
        args.extend(['-pix_fmt', 'yuv420p'])
    return args


def get_gpu_quality_args(gpu_encoder: str, include_pix_fmt: bool = True) -> List[str]:
    """Return settings for whichever hardware encoder was selected."""
    if 'videotoolbox' in gpu_encoder:
        return get_videotoolbox_quality_args(gpu_encoder, include_pix_fmt)
    return get_nvenc_quality_args(gpu_encoder, include_pix_fmt)


def get_hwaccel_args(use_hw_encoder: bool, gpu_encoder: str) -> List[str]:
    """CUDA decode assist only applies to NVENC; everything else probes safely."""
    if use_hw_encoder and 'nvenc' in gpu_encoder:
        return ['-hwaccel', 'cuda']
    return ['-hwaccel', 'auto']


FIT_MODES = ('crop', 'blur', 'pad', 'stretch')


def get_fit_filters(target_size: Tuple[int, int], fit_mode: str) -> List[str]:
    """Aspect-ratio handling for sources that don't match the target frame.

    crop    – scale to fill, center-crop overflow (no distortion, default)
    pad     – letterbox/pillarbox with black bars
    stretch – legacy distorting scale
    blur    – handled separately (needs a filter graph, see caller)
    """
    w, h = target_size
    if fit_mode == 'stretch':
        return [f"scale={w}:{h}", "setsar=1"]
    if fit_mode == 'pad':
        return [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
        ]
    return [
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
        "setsar=1",
    ]


def build_blur_fit_graph(pre_filters: List[str], target_size: Tuple[int, int],
                         post_filters: List[str], out_label: str = 'outv') -> str:
    """Filter graph for blur fit: blurred fill in back, undistorted fit in front."""
    w, h = target_size
    pre = ",".join(pre_filters)
    post = ("," + ",".join(post_filters)) if post_filters else ""
    return (
        f"[0:v]{pre},split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=16[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,setsar=1{post}[{out_label}]"
    )


def build_text_overlay_graph(base_graph: str, fade_in_start: float,
                             fade_in_duration: float, fade_out_start: float,
                             fade: float = 0.35) -> str:
    """Composite input 1 (a looped transparent PNG) over the [basev] stream.

    Text is rendered by Pillow (see text_overlay.py) because this ffmpeg
    build has no drawtext; overlay/fade/format are core filters. Fade
    timings are in the segment's local clock (text_overlay.plan_text_windows
    projects the global text window onto each segment). A zero fade-in
    duration means the fade completed in an earlier segment, so the filter
    is omitted (fade rejects st<0 and d=0); a fade-in with st>0 also keeps
    the text invisible before st, handling windows that open mid-segment.
    """
    txt_chain = ["format=rgba"]
    if fade_in_duration > 0.001:
        txt_chain.append(f"fade=t=in:st={max(0.0, fade_in_start):.4f}:d={fade_in_duration:.4f}:alpha=1")
    txt_chain.append(f"fade=t=out:st={max(0.0, fade_out_start):.4f}:d={fade:.4f}:alpha=1")
    return (
        f"{base_graph};"
        f"[1:v]{','.join(txt_chain)}[txt];"
        f"[basev][txt]overlay=0:0[outv]"
    )


def get_cpu_h264_quality_args(include_pix_fmt: bool = True) -> List[str]:
    """Return lossless CPU H.264 settings."""
    args = [
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '0',
    ]
    if include_pix_fmt:
        args.extend(['-pix_fmt', 'yuv420p'])
    args.extend(['-threads', str(MAX_THREADS)])
    return args


_SOURCE_DURATION_CACHE: dict = {}


def get_cached_video_duration(video_file: str) -> float:
    """Duration lookup with a per-run cache (segments hit the same sources repeatedly)."""
    cached = _SOURCE_DURATION_CACHE.get(video_file)
    if cached is None:
        cached = _probe_video_duration(video_file)
        if cached is None:
            # Don't cache failures: a transient ffprobe error must not pin the
            # fallback duration for the rest of the run.
            return 10.0
        _SOURCE_DURATION_CACHE[video_file] = cached
    return cached


def get_loop_input_args(video_file: str, start_time: float, duration: float) -> Tuple[List[str], float]:
    """Loop args for sources shorter than the requested window (GIFs, short clips).

    Returns (['-stream_loop', N] or [], adjusted_start_time). With looping the
    demuxer presents the source repeated N+1 times. ffmpeg gotcha: an input-side
    -ss combined with -stream_loop re-seeks to the offset on EVERY loop
    iteration (each pass yields only [start, EOF]), so when these args are used
    with a nonzero start the caller must seek in the filter chain (trim=start=)
    instead of with -ss.
    """
    src_duration = get_cached_video_duration(video_file)
    if src_duration <= 0.05:
        return [], start_time
    if start_time + duration <= src_duration - 0.02:
        return [], start_time
    if start_time >= src_duration:
        start_time = start_time % src_duration
    loops = max(1, math.ceil((start_time + duration) / src_duration))
    return ['-stream_loop', str(loops)], start_time


def _probe_video_duration(video_file: str) -> float | None:
    """Duration via ffprobe; None on failure so callers can avoid caching it."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_file
        ]

        result = _run_media_command(probe_cmd, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(_short_ffmpeg_error(result.stderr, 300) or "ffprobe failed")
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"   ⚠️  Could not get video duration with ffprobe ({os.path.basename(video_file)}): {e}")
        return None


def get_video_duration(video_file: str) -> float:
    """Get the duration of a video file using ffprobe."""
    duration = _probe_video_duration(video_file)
    return 10.0 if duration is None else duration  # Default fallback


def get_video_fps(video_file: str) -> float:
    """Get the FPS of a video file using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_file
        ]
        
        result = _run_media_command(probe_cmd, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(_short_ffmpeg_error(result.stderr, 300) or "ffprobe failed")
        fps_str = result.stdout.strip()

        # Parse fraction (e.g., "30000/1001" or "30/1")
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return fps
    except Exception as e:
        print(f"   ⚠️  Could not get video FPS with ffprobe ({os.path.basename(video_file)}): {e}")
        return 30.0  # Default fallback


def get_video_resolution(video_file: str) -> Tuple[int, int]:
    """Get the resolution (width, height) of a video file using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_file
        ]
        
        result = _run_media_command(probe_cmd, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(_short_ffmpeg_error(result.stderr, 300) or "ffprobe failed")
        data = json.loads(result.stdout)

        width = data['streams'][0]['width']
        height = data['streams'][0]['height']

        return (width, height)
    except Exception as e:
        print(f"   ⚠️  Could not get video resolution with ffprobe ({os.path.basename(video_file)}): {e}")
        return (1920, 1080)  # Default fallback


def seconds_to_frame_count(seconds: float, fps: float) -> int:
    """
    Convert seconds to exact frame count.
    This ensures frame-accurate timing with no drift.
    """
    return int(round(seconds * fps))


def frame_count_to_seconds(frames: int, fps: float) -> float:
    """
    Convert frame count back to exact seconds.
    This is the EXACT duration for the frame count.
    """
    return frames / fps


def convert_to_prores_proxy(video_file: str, output_dir: str, fps: float = None,
                            target_size: Tuple[int, int] = None,
                            fit_mode: str = 'crop') -> str:
    """
    Convert video to ProRes 422 Proxy for lossless editing.
    All frames are I-frames (keyframes) for frame-accurate cutting.
    STRIPS AUDIO - we'll add the music track at the end.

    target_size normalizes every proxy to one frame size so the later concat
    stream-copy sees identical streams (mixed-resolution sources would
    otherwise produce an invalid output). None keeps the source resolution.
    """
    filename = os.path.basename(video_file)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}_prores.mov")

    print(f"   📹 Converting to ProRes 422 Proxy: {filename}")

    # Detect FPS if not provided
    if fps is None:
        fps = get_video_fps(video_file)

    # Build FFmpeg command for ProRes 422 Proxy (NO AUDIO).
    # ProRes encode/decode is CPU-native in FFmpeg; forcing hwaccel auto can make
    # FFmpeg pick Vulkan/D3D paths that are slower or unstable for ProRes.
    cmd = [
        FFMPEG_PATH,
        '-nostdin',
        '-hide_banner',
        '-i', video_file,
        '-map', '0:v:0',
    ]
    if target_size:
        # Blur fit needs a filter graph; proxies use get_fit_filters' default
        # crop chain for it instead (proxy normalization only needs identical
        # dimensions, not the blurred-background look).
        cmd.extend(['-vf', ','.join(get_fit_filters(target_size, fit_mode))])
    cmd.extend([
        '-c:v', 'prores',  # ProRes encoder
        '-profile:v', '0',  # Proxy quality (0=Proxy, 1=LT, 2=Standard, 3=HQ)
        '-vendor', 'apl0',
        '-pix_fmt', 'yuv422p10le',
        '-an',  # ✅ STRIP AUDIO - we'll add music at the end
        '-sn',
        '-dn',
        '-r', str(fps),  # Set frame rate
        '-threads', str(MAX_THREADS),
        '-y',
        output_file
    ])
    
    try:
        result = _run_media_command(cmd, timeout=600)  # 10 minute timeout
        
        if result.returncode != 0:
            print(f"   ⚠️  FFmpeg error: {result.stderr}")
            raise Exception(f"ProRes conversion failed for {filename}")
        
        print(f"   ✓ ProRes conversion complete: {name}_prores.mov (video only, no audio)")
        return output_file
        
    except subprocess.TimeoutExpired:
        raise Exception(f"ProRes conversion timeout for {filename}")
    except Exception as e:
        raise Exception(f"ProRes conversion error: {str(e)}")


def build_segment_pre_filters(exact_duration: float, fps: float,
                              trim_start: float = 0.0) -> List[str]:
    """Shared trim/setpts/fps head of every segment's filter chain.

    Trim first so each extracted segment has exact timing; effects come after
    fps so their time expressions see the final frame timing. A nonzero
    trim_start seeks in the filter chain (used with -stream_loop, where an
    input-side -ss re-seeks on every loop iteration).
    """
    if trim_start > 0:
        trim_filter = f"trim=start={trim_start:.6f}:duration={exact_duration}"
    else:
        trim_filter = f"trim=duration={exact_duration}"
    return [trim_filter, "setpts=PTS-STARTPTS", f"fps={fps}"]


def extract_clip_segment_ffmpeg(video_file: str, start_time: float, duration: float,
                                output_file: str, fps: float, target_size: Tuple[int, int],
                                use_nvenc: bool,
                                gpu_encoder: str = 'h264_nvenc',
                                fit_mode: str = 'crop',
                                extra_filters: List[str] = None,
                                text_overlay: Tuple[str, float, float, float] = None) -> bool:
    """
    Extract a video segment using FFmpeg with FRAME-ACCURATE timing.
    
    ✅ FRAME-ACCURATE: Uses exact frame counts instead of floating-point seconds
    ✅ ZERO DRIFT: No cumulative timing errors
    """
    try:
        # ✅ FRAME-ACCURATE: Calculate exact source and output frame counts.
        source_frame_count = max(1, seconds_to_frame_count(duration, fps))
        exact_source_duration = frame_count_to_seconds(source_frame_count, fps)
        output_frame_count = source_frame_count

        # Loop sources shorter than the segment (GIFs, short clips) so the
        # frame count stays exact instead of drifting. ffmpeg gotcha: with
        # -stream_loop, an input-side -ss re-seeks to the offset on EVERY loop
        # iteration (each pass yields only [start, EOF] instead of wrapping),
        # so looped seeks happen in the filter chain via trim=start= instead.
        loop_args, start_time = get_loop_input_args(video_file, start_time, exact_source_duration)
        filter_seek = bool(loop_args) and start_time > 0

        pre_filters = build_segment_pre_filters(
            exact_source_duration, fps, trim_start=start_time if filter_seek else 0.0)
        post_filters = list(extra_filters or [])

        # text_overlay: (png_path, fade_in_start, fade_in_duration,
        # fade_out_start) in this segment's local clock — see
        # text_overlay.plan_text_windows.
        use_blur_graph = bool(target_size) and fit_mode == 'blur'
        use_graph = use_blur_graph or bool(text_overlay)
        filter_graph = None
        filter_complex = None
        base_label = 'basev' if text_overlay else 'outv'
        if use_blur_graph:
            filter_graph = build_blur_fit_graph(pre_filters, target_size, post_filters,
                                                out_label=base_label)
        else:
            filters = list(pre_filters)
            if target_size:
                filters.extend(get_fit_filters(target_size, fit_mode))
            filters.extend(post_filters)
            filter_complex = ",".join(filters)
            if use_graph:
                filter_graph = f"[0:v]{filter_complex}[{base_label}]"
        if text_overlay:
            _, fade_in_start, fade_in_duration, fade_out_start = text_overlay
            filter_graph = build_text_overlay_graph(filter_graph, fade_in_start,
                                                    fade_in_duration, fade_out_start)
        
        # Build FFmpeg command
        cmd = [FFMPEG_PATH]
        
        # Hardware acceleration
        cmd.extend(get_hwaccel_args(use_nvenc, gpu_encoder))

        cmd.extend(loop_args)

        if filter_seek:
            # Looped seek: no input-side -ss/-t (see gotcha above); trim=start=
            # in the filter chain positions the window and -vframes caps output.
            # Looped sources are short by definition, so the extra decode from
            # 0 to start is negligible.
            cmd.extend(['-i', video_file])
        else:
            # ✅ FRAME-ACCURATE INPUT SEEKING (input -ss decodes forward from
            # the preceding keyframe and discards, so it stays frame-accurate)
            cmd.extend([
                '-ss', str(start_time),
                '-t', str(exact_source_duration),
                '-i', video_file
            ])

        if text_overlay:
            cmd.extend(['-loop', '1', '-i', text_overlay[0]])

        # Video filters
        if use_graph:
            cmd.extend(['-filter_complex', filter_graph, '-map', '[outv]'])
        else:
            cmd.extend(['-vf', filter_complex])

        # ✅ FRAME-ACCURATE DURATION: Use -vframes instead of -t
        cmd.extend(['-vframes', str(output_frame_count)])
        
        # Video encoding
        if use_nvenc:
            cmd.extend(get_gpu_quality_args(gpu_encoder, include_pix_fmt=True))
        else:
            cmd.extend(get_cpu_h264_quality_args(include_pix_fmt=True))
        
        # No audio, frame-accurate settings
        cmd.extend([
            '-an',
            '-fps_mode', 'cfr',  # Constant frame rate
            '-r', str(fps),   # Exact output FPS
            '-fflags', '+genpts',
            '-movflags', '+faststart',
            '-y',
            output_file
        ])
        
        result = _run_media_command(cmd, timeout=120)
        
        if result.returncode != 0:
            print(f"   ⚠️  FFmpeg error: {result.stderr}")
            return False
        
        # Verify output exists and has content
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            return False
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Error extracting clip: {e}")
        return False


def extract_prores_segment_random(video_file: str, duration: float, fps: float,
                                  temp_dir: str, segment_index: int,
                                  start_time: float = None) -> str:
    """
    Extract a segment from a ProRes proxy with frame-perfect precision.

    Important stability fix:
    - Do NOT use '-hwaccel auto' here. FFmpeg can select Vulkan/D3D hwaccel for
      ProRes, which is unnecessary for ProRes and can fail on some Windows GPU
      driver combinations.
    - Use a CPU-native ProRes path, exact frame count, and a safe retry command.
    """
    output_file = os.path.join(temp_dir, f"segment_{segment_index:05d}.mov")

    video_duration = get_cached_video_duration(video_file)
    if video_duration <= 0:
        raise Exception(f"Invalid ProRes source duration: {video_file}")

    loop_args: List[str] = []
    if video_duration >= duration:
        max_start = max(0.0, video_duration - duration)
        if start_time is None:
            # Same seeded-RNG idiom as effects.py: identical inputs must pick
            # identical source moments (deterministic renders).
            seed_raw = f"prores_start|{segment_index}|{os.path.basename(video_file)}"
            seed = int(hashlib.sha1(seed_raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
            start_time = random.Random(seed).uniform(0.0, max_start)
        else:
            start_time = max(0.0, min(float(start_time), max_start))
    else:
        # Source is shorter than the segment: loop it so precise mode keeps
        # its exact frame count instead of emitting a short clip. start_time
        # stays 0 here, so pairing -stream_loop with the input -ss below is
        # safe (the per-iteration re-seek gotcha only bites for start > 0).
        start_time = 0.0
        loop_args, start_time = get_loop_input_args(video_file, start_time, duration)

    frame_count = max(1, seconds_to_frame_count(duration, fps))
    exact_duration = frame_count_to_seconds(frame_count, fps)

    filter_complex = ",".join(build_segment_pre_filters(exact_duration, fps))

    def build_cmd(fast_seek: bool) -> List[str]:
        cmd = [FFMPEG_PATH, '-nostdin', '-hide_banner']
        cmd.extend(loop_args)
        if fast_seek:
            # ProRes proxy is intra-frame, so input-side seeking remains accurate
            # while being much faster for long sources.
            cmd.extend(['-ss', f'{start_time:.6f}', '-i', video_file])
        else:
            # Ultra-safe fallback. Slower on long sources, but avoids muxer/seek
            # edge cases if a specific FFmpeg build rejects the fast path.
            cmd.extend(['-i', video_file, '-ss', f'{start_time:.6f}'])

        cmd.extend([
            '-map', '0:v:0',
            '-vf', filter_complex,
            '-vframes', str(frame_count),
            '-c:v', 'prores',
            '-profile:v', '0',
            '-vendor', 'apl0',
            '-pix_fmt', 'yuv422p10le',
            '-r', str(fps),
            '-fps_mode', 'cfr',
            '-fflags', '+genpts',
            '-an',
            '-sn',
            '-dn',
            '-threads', str(MAX_THREADS),
            '-y',
            output_file,
        ])
        return cmd

    last_error = ""
    for attempt_name, fast_seek, timeout in [
        ('fast intra-frame seek', True, 180),
        ('safe accurate seek retry', False, 360),
    ]:
        _safe_remove_file(output_file)
        result = _run_media_command(build_cmd(fast_seek), timeout=timeout)
        if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return output_file
        last_error = _short_ffmpeg_error(result.stderr)
        print(f"   ⚠️  ProRes extraction failed on {attempt_name}: {last_error}")

    raise Exception(f"ProRes segment extraction error: {last_error}")

def concatenate_videos_ffmpeg(video_files: List[str], output_file: str, 
                              audio_file: str = None, start_time: float = 0.0,
                              end_time: float = None, use_nvenc: bool = False,
                              gpu_encoder: str = 'h264_nvenc', fps: float = 30.0,
                              temp_dir: str = None) -> str:
    """
    Concatenate video files using FFmpeg concat demuxer.
    
    ✅ FRAME-ACCURATE: Maintains precise timing through concatenation
    """
    if temp_dir is None:
        temp_dir = os.path.dirname(output_file)
    
    # Create concat file
    concat_file = os.path.join(temp_dir, f'concat_list_{uuid.uuid4().hex}.txt')
    with open(concat_file, 'w', encoding='utf-8') as f:
        for video_file in video_files:
            # Concat demuxer quoting: a literal ' inside the single-quoted
            # path must be written as '\'' or the list fails to parse.
            escaped_path = video_file.replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    is_prores = output_file.lower().endswith('.mov')
    temp_video = None
    temp_audio = None

    def add_audio_input(cmd: List[str]) -> None:
        if not audio_file:
            return
        if end_time and end_time > start_time:
            cmd.extend(['-ss', str(start_time), '-t', str(end_time - start_time), '-i', audio_file])
        elif start_time > 0:
            cmd.extend(['-ss', str(start_time), '-i', audio_file])
        else:
            cmd.extend(['-i', audio_file])

    try:
        if is_prores:
            # ProRes: concat with stream copy (lossless)
            print(f"   🔗 Concatenating {len(video_files)} segments (lossless stream copy)...")
            
            temp_video = os.path.join(temp_dir, f'video_only_{uuid.uuid4().hex}.mov')
            
            cmd = [
                FFMPEG_PATH,
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                '-y',
                temp_video
            ]
            
            result = _run_media_command(cmd, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Concatenation failed: {result.stderr}")
            
            # Add audio if provided
            if audio_file:
                print(f"   🎵 Adding music track...")
                
                temp_audio = os.path.join(temp_dir, f'music_{uuid.uuid4().hex}.wav')
                
                audio_cmd = [FFMPEG_PATH]
                add_audio_input(audio_cmd)

                audio_cmd.extend([
                    '-acodec', 'pcm_s24le',
                    '-ar', '48000',
                    '-ac', '2',
                    '-y',
                    temp_audio
                ])
                
                result = _run_media_command(audio_cmd, timeout=120)
                
                if result.returncode != 0:
                    raise Exception(f"Audio extraction failed: {result.stderr}")
                
                # Combine video + audio with AUDIO as master timeline
                cmd = [
                    FFMPEG_PATH,
                    '-i', temp_video,
                    '-i', temp_audio,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'pcm_s24le',
                    '-ar', '48000',
                    '-shortest',  # Use shortest stream (audio)
                    '-y',
                    output_file
                ]
                
                result = _run_media_command(cmd, timeout=300)
                
                if result.returncode != 0:
                    raise Exception(f"Audio merging failed: {result.stderr}")
                
                _safe_remove_file(temp_video)
                _safe_remove_file(temp_audio)
            else:
                shutil.move(temp_video, output_file)
        
        else:
            # H.264/H.265 standard path. Temp clips were already encoded with
            # matching FPS/resolution/codec settings, so the fastest safe path is
            # stream-copy concatenation plus audio mux. If a specific codec/container
            # combination rejects stream copy, fall back to the old re-encode path.
            fast_concat_enabled = _env_flag('BEATSYNC_FAST_CONCAT_COPY', True)

            if fast_concat_enabled:
                print(f"   🔗 Fast final assembly: concat stream-copy video + mux audio...")
                copy_started = time.perf_counter()
                cmd = [
                    FFMPEG_PATH,
                    '-nostdin',
                    '-hide_banner',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                ]
                add_audio_input(cmd)
                if audio_file:
                    cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
                else:
                    cmd.extend(['-map', '0:v:0'])
                cmd.extend(['-c:v', 'copy'])
                if audio_file:
                    cmd.extend(['-c:a', 'pcm_s24le', '-ar', '48000', '-shortest'])
                cmd.extend(['-fflags', '+genpts'])
                if output_file.lower().endswith(('.mp4', '.mov', '.m4v')):
                    cmd.extend(['-movflags', '+faststart'])
                cmd.extend(['-y', output_file])

                result = _run_media_command(cmd, timeout=300)
                if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"   ✓ Fast concat-copy complete in {_fmt_seconds(time.perf_counter() - copy_started)}")
                    return output_file
                print(
                    f"   ⚠️  Fast concat-copy failed in {_fmt_seconds(time.perf_counter() - copy_started)}; "
                    f"falling back to re-encode. {_short_ffmpeg_error(result.stderr, 900)}"
                )

            # Fallback/original behavior: H.264/H.265 full re-encode.
            print(f"   🔗 Concatenating and encoding {len(video_files)} segments...")
            encode_started = time.perf_counter()
            cmd = [FFMPEG_PATH]
            cmd.extend(get_hwaccel_args(use_nvenc, gpu_encoder))
            
            cmd.extend([
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file
            ])
            
            add_audio_input(cmd)
            if audio_file:
                cmd.extend(['-map', '0:v', '-map', '1:a'])
            
            if use_nvenc:
                cmd.extend(get_gpu_quality_args(gpu_encoder, include_pix_fmt=True))
            else:
                cmd.extend(get_cpu_h264_quality_args(include_pix_fmt=True))
            
            if audio_file:
                cmd.extend([
                    '-c:a', 'pcm_s24le',
                    '-ar', '48000',
                    '-shortest',
                ])
            
            cmd.extend([
                '-fps_mode', 'cfr',
                '-r', str(fps),
                '-y',
                output_file
            ])
            
            result = _run_media_command(cmd, timeout=600)
            
            if result.returncode != 0:
                raise Exception(f"Encoding failed: {result.stderr}")
            print(f"   ✓ Full final re-encode complete in {_fmt_seconds(time.perf_counter() - encode_started)}")
        
        return output_file
    finally:
        _safe_remove_file(concat_file)
        _safe_remove_file(temp_audio)
        _safe_remove_file(temp_video)


def detect_video_scene_changes(video_path: str, threshold: float = 0.28,
                               use_gpu: bool = False,
                               analysis_fps: float = 8.0,
                               analysis_width: int = 384) -> List[float]:
    """
    Detect likely montage/scene changes using FFmpeg's scene score.

    GPU note:
    FFmpeg's `scene` comparison itself is a CPU video filter, but when GPU mode
    is enabled this uses CUDA decode + CUDA resize, then downloads a small
    analysis frame for the CPU scene score. That makes the expensive decode/scale
    stage GPU-assisted while preserving the same semantic scene-cut signal.
    If CUDA decode is unsupported for a source codec, it automatically falls
    back to the CPU analysis path.
    """
    try:
        print(f"   🎬 Analyzing scene changes: {os.path.basename(video_path)}")
        print(f"   Threshold: {threshold}")
        print(f"   Analysis: {analysis_fps:g} fps @ {analysis_width}px wide")

        filter_cpu = (
            f"fps={analysis_fps},"
            f"scale={analysis_width}:-2:flags=fast_bilinear,"
            f"select='gt(scene,{threshold})',showinfo"
        )
        filter_gpu = (
            f"scale_cuda={analysis_width}:-2,"
            f"hwdownload,format=nv12,"
            f"fps={analysis_fps},"
            f"select='gt(scene,{threshold})',showinfo"
        )

        commands = []
        if use_gpu:
            commands.append((
                'GPU-assisted CUDA decode/scale',
                [
                    FFMPEG_PATH,
                    '-nostdin',
                    '-hide_banner',
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda',
                    '-i', video_path,
                    '-vf', filter_gpu,
                    '-an', '-sn', '-dn',
                    '-f', 'null',
                    '-'
                ],
            ))

        commands.append((
            'CPU fast scene score',
            [
                FFMPEG_PATH,
                '-nostdin',
                '-hide_banner',
                '-threads', str(MAX_THREADS),
                '-i', video_path,
                '-vf', filter_cpu,
                '-an', '-sn', '-dn',
                '-f', 'null',
                '-'
            ],
        ))

        last_error = ''
        for label, cmd in commands:
            if label.startswith('GPU'):
                print("   ⚡ GPU scene analysis: CUDA decode/scale + CPU scene score")
            else:
                if use_gpu:
                    print("   ↪ GPU scene analysis unavailable/failed, using CPU fallback")
                else:
                    print("   💻 CPU scene analysis")

            command_started = time.perf_counter()
            result = _run_media_command(cmd, timeout=300)
            command_elapsed = time.perf_counter() - command_started
            if result.returncode == 0:
                cleaned = _extract_scene_times(result.stderr)
                print(
                    f"   ✓ Found {len(cleaned)} visual scene changes ({label}) "
                    f"in {command_elapsed:.1f}s"
                )
                return cleaned

            print(f"   ⚠️  Scene command failed after {command_elapsed:.1f}s ({label})")
            last_error = _short_ffmpeg_error(result.stderr, max_chars=1200)
            if label.startswith('GPU'):
                print(f"   ⚠️  GPU scene analysis failed, fallback enabled: {last_error}")

        print(f"   ⚠️  Warning: Scene detection failed: {last_error}")
        return []

    except subprocess.TimeoutExpired:
        print(f"   ⚠️  Warning: Scene detection timeout")
        return []
    except Exception as e:
        print(f"   ⚠️  Warning: Could not analyze scene changes: {e}")
        return []

def detect_video_keyframes(video_path: str, min_interval: float = 0.20) -> List[float]:
    """
    Detect codec keyframes/I-frames with ffprobe.

    Keyframes are not always creative scene cuts, but they are useful as a
    fallback signal for footage that was encoded with keyframes at montage cuts.
    """
    try:
        print(f"   🔑 Reading codec keyframes: {os.path.basename(video_path)}")
        cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-skip_frame', 'nokey',
            '-show_entries', 'frame=best_effort_timestamp_time,pkt_pts_time,pts_time',
            '-of', 'json',
            video_path,
        ]
        result = _run_media_command(cmd, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
        keyframes = []
        for frame in data.get('frames', []):
            ts = frame.get('best_effort_timestamp_time') or frame.get('pkt_pts_time') or frame.get('pts_time')
            if ts is None:
                continue
            try:
                t = float(ts)
            except (TypeError, ValueError):
                continue
            if t >= 0.0:
                keyframes.append(round(t, 3))

        keyframes = sorted(set(keyframes))
        cleaned = []
        for t in keyframes:
            if not cleaned or t - cleaned[-1] >= min_interval:
                cleaned.append(t)

        print(f"   ✓ Found {len(cleaned)} codec keyframes")
        return cleaned
    except subprocess.TimeoutExpired:
        print("   ⚠️  Warning: Keyframe detection timeout")
        return []
    except Exception as e:
        print(f"   ⚠️  Warning: Could not read keyframes: {e}")
        return []
