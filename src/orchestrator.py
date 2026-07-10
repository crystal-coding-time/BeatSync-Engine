"""Render orchestration: the headless entry point and the Gradio worker.

Split out of gui.py so the render pipeline (settings model, analysis, encode,
status assembly) lives independently of the Gradio UI construction. gui.py
re-exports `_process_video_impl`/`process_video` from here, so the documented
smoke entry point `gui._process_video_impl(...)` keeps working unchanged.

Import ordering note: this module imports the CUDA/FFmpeg-dependent pipeline
modules, so its importer (gui.py) must have already called
`logger.setup_environment()` before importing it — gui does exactly that.
"""

import os
import sys
import tempfile
import shutil
import datetime
import queue
import re
import subprocess
import threading
import time
import contextlib
from dataclasses import dataclass, fields as dataclass_fields, replace as dataclass_replace
from typing import Callable, Iterator, TypeAlias, Tuple, Dict, List

from ffmpeg_processing import get_video_fps, get_video_resolution, is_image_source, FFMPEG_PATH

from gpu_cpu_utils import (
    PARALLEL_WORKERS,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    VIDEOTOOLBOX_AVAILABLE,
    HW_ENCODERS,
    hw_encoder_available,
    set_gpu_mode,
)
from paths import GRADIO_TEMP_DIR, get_output_dir

from video_processor import create_music_video, DEFAULT_OUTPUT_FORMAT
from effects import resolve_effect_palette
from auto_mode import analyze_beats_auto

from ui_content import get_success_message_auto
from render_log import (
    RenderLogConsole,
    StageConsoleLogger,
    _stage5_summary,
    _stage6_summary,
)

VideoFilesInput: TypeAlias = List[str]
StatusResult: TypeAlias = Tuple[str, str, Dict]


# Human-readable status shown in the UI for each pipeline stage. The pipeline
# internally still emits "Stage N is processing." strings (auto_mode._notify_progress);
# process_video's progress_callback maps either form to the friendly label below
# while keeping the console-logger stage signal intact.
STAGE_LABELS: dict[int, str] = {
    1: "Analyzing audio (beats & tempo)…",
    2: "Reading per-beat features…",
    3: "Finding song sections…",
    4: "Choosing cuts on the beat grid…",
    5: "Tagging visuals…",
    6: "Planning the edit & rendering…",
}
_LABEL_TO_STAGE: dict[str, int] = {label: n for n, label in STAGE_LABELS.items()}


def _stage_status(stage_number: int) -> str:
    return STAGE_LABELS.get(stage_number, f"Working… (stage {stage_number})")


def _as_existing_source_path(file_path: str | None) -> str | None:
    """Use the selected source file directly instead of copying it locally."""
    if not file_path:
        return None
    try:
        path = os.path.abspath(os.fspath(file_path))
    except TypeError:
        return None
    return path if os.path.isfile(path) else None


def _as_existing_source_paths(file_paths: VideoFilesInput) -> list[str]:
    if not file_paths:
        return []
    return [path for path in (_as_existing_source_path(p) for p in file_paths) if path]


@dataclass(frozen=True)
class RenderSettings:
    """Single source of truth for the 16 per-render settings.

    Field order and defaults are canonical: the headless kwargs of
    `_process_video_impl`, the GUI settings dict, and the Gradio positional
    boundary (`SETTINGS_KEYS` / `settings_components`) all derive from here.
    """

    fit_mode: str = 'crop'
    output_format: str = DEFAULT_OUTPUT_FORMAT
    effect_style: str = 'clean'
    effect_intensity: float = 0.7
    effect_mode: str = 'curated'
    effect_palette: List[str] | None = None
    effect_seed: float = 0
    look_cube: str = ''
    variety: float = 0.4
    semantic_variety: float = 0.4
    speed_ramps: bool = False
    split_screen: bool = True
    crossfades: bool = False
    text_entries: str = ''
    text_position: str = 'bottom'
    text_scale: float = 1.0
    text_style: str = 'classic'
    text_accent: str = '#FF4D8D'
    text_font: str = ''       # font picker dropdown ('' = auto)
    text_font_path: str = ''  # custom file path box; wins over the dropdown

    @classmethod
    def from_dict(cls, d: dict | None, base: 'RenderSettings | None' = None) -> 'RenderSettings':
        """Overlay dict values on `base` (or the defaults).

        A key present in the dict wins — even with an explicitly falsy/None
        value — and unknown keys are ignored, exactly matching the old
        per-key `settings.get(key, kwarg)` override block.
        """
        base = base if base is not None else cls()
        if not d:
            return base
        known = {f.name for f in dataclass_fields(cls)}
        overrides = {k: v for k, v in d.items() if k in known}
        return dataclass_replace(base, **overrides) if overrides else base

    def to_settings_dict(self, *, is_prores: bool,
                         palette_ids, resolved_seed) -> dict:
        """Produce the resolved settings dict `create_music_video` consumes.

        Carries the per-render transforms the old inline repack applied:
        the resolved palette/seed from `resolve_effect_palette`, the ProRes
        look gate (ProRes stays ungraded), and text entries split into
        stripped non-empty lines.
        """
        return {
            'fit_mode': self.fit_mode,
            'output_format': self.output_format,
            'effect_style': self.effect_style,
            'effect_intensity': self.effect_intensity,
            'effect_mode': self.effect_mode,
            'effect_palette': palette_ids,
            'effect_seed': resolved_seed,
            'look_cube': (None if is_prores else (self.look_cube or None)),
            'variety': self.variety,
            'semantic_variety': self.semantic_variety,
            'speed_ramps': bool(self.speed_ramps),
            'split_screen': bool(self.split_screen),
            'crossfades': bool(self.crossfades),
            'text_entries': [line.strip() for line in (self.text_entries or '').splitlines() if line.strip()],
            'text_position': self.text_position,
            'text_scale': self.text_scale,
            'text_style': self.text_style,
            'text_accent': self.text_accent,
            # Two UI controls, one downstream knob: the custom path box wins
            # over the dropdown; '' means the historic auto lookup.
            'text_font': (self.text_font_path or '').strip() or self.text_font,
        }


# Canonical key order for the Gradio boundary: process_video's positional
# settings parameters and create_ui's settings_components list both follow
# RenderSettings field order, and dict(zip(...)) marries them.
SETTINGS_KEYS: tuple[str, ...] = tuple(f.name for f in dataclass_fields(RenderSettings))

_RS_DEFAULTS = RenderSettings()


def _process_video_impl(audio_files: VideoFilesInput, video_files: VideoFilesInput,
                       output_filename: str, processing_mode: str,
                       custom_fps: float, session_state: dict,
                       fit_mode: str = _RS_DEFAULTS.fit_mode,
                       output_format: str = _RS_DEFAULTS.output_format,
                       effect_style: str = _RS_DEFAULTS.effect_style,
                       effect_intensity: float = _RS_DEFAULTS.effect_intensity,
                       effect_mode: str = _RS_DEFAULTS.effect_mode,
                       effect_palette: List[str] | None = _RS_DEFAULTS.effect_palette,
                       effect_seed: float = _RS_DEFAULTS.effect_seed,
                       look_cube: str = _RS_DEFAULTS.look_cube,
                       variety: float = _RS_DEFAULTS.variety,
                       semantic_variety: float = _RS_DEFAULTS.semantic_variety,
                       speed_ramps: bool = _RS_DEFAULTS.speed_ramps,
                       split_screen: bool = _RS_DEFAULTS.split_screen,
                       crossfades: bool = _RS_DEFAULTS.crossfades,
                       text_entries: str = _RS_DEFAULTS.text_entries,
                       text_position: str = _RS_DEFAULTS.text_position,
                       text_scale: float = _RS_DEFAULTS.text_scale,
                       text_style: str = _RS_DEFAULTS.text_style,
                       text_accent: str = _RS_DEFAULTS.text_accent,
                       text_font: str = _RS_DEFAULTS.text_font,
                       text_font_path: str = _RS_DEFAULTS.text_font_path,
                       settings: dict | None = None,
                       progress_callback: Callable[[str], None] | None = None,
                       console_logger: StageConsoleLogger | None = None) -> StatusResult:
    total_started = time.perf_counter()
    try:
        # The GUI passes one settings dict; the individual kwargs remain for
        # the headless/smoke-test entry point. The dict wins where present.
        rs = RenderSettings(
            fit_mode=fit_mode, output_format=output_format,
            effect_style=effect_style, effect_intensity=effect_intensity,
            effect_mode=effect_mode, effect_palette=effect_palette,
            effect_seed=effect_seed, look_cube=look_cube,
            variety=variety, semantic_variety=semantic_variety,
            speed_ramps=speed_ramps, split_screen=split_screen,
            crossfades=crossfades, text_entries=text_entries,
            text_position=text_position, text_scale=text_scale,
            text_style=text_style, text_accent=text_accent,
            text_font=text_font, text_font_path=text_font_path,
        )
        rs = RenderSettings.from_dict(settings, base=rs)
        parallel_workers = PARALLEL_WORKERS

        # Initialize session state if needed
        if 'original_audio_paths' not in session_state:
            session_state['original_audio_paths'] = None
            session_state['original_video_paths'] = []
        if 'session_dir' not in session_state or not os.path.isdir(session_state['session_dir']):
            session_state['session_dir'] = tempfile.mkdtemp(prefix='beatsync_', dir=GRADIO_TEMP_DIR)
        session_dir = session_state['session_dir']

        # Audio may now be one song or several, ordered. Normalize to a
        # deduped, order-preserving list of selections. A single-song
        # selection (len == 1) flows through exactly as before: the cache
        # value and every downstream value stay byte-identical to the old
        # single-path behavior.
        if isinstance(audio_files, (str, bytes)):
            audio_files = [audio_files] if audio_files else []
        audio_selection: list[str] = []
        _seen_audio: set = set()
        for _a in (audio_files or []):
            if _a and _a not in _seen_audio:
                _seen_audio.add(_a)
                audio_selection.append(_a)
        if not audio_selection:
            return None, '❌ Error: No audio file selected', session_state

        # Reference the selected file paths directly; the cache key is now the
        # ordered list. _as_existing_source_paths drops anything unreadable, so
        # a length mismatch means at least one selected file is gone.
        if audio_selection != session_state.get('original_audio_paths'):
            local_audio_paths = _as_existing_source_paths(audio_selection)
            if local_audio_paths and len(local_audio_paths) == len(audio_selection):
                session_state['local_audio_paths'] = local_audio_paths
                session_state['original_audio_paths'] = audio_selection
            else:
                return None, '❌ Error: Could not access audio file', session_state
        else:
            local_audio_paths = session_state.get('local_audio_paths')

        # Handle videos by referencing selected file paths directly.
        if video_files:
            if video_files != session_state.get('original_video_paths'):
                local_video_paths = _as_existing_source_paths(video_files)
                if local_video_paths:
                    session_state['local_video_paths'] = local_video_paths
                    session_state['original_video_paths'] = video_files
                else:
                    return None, '❌ Error: Could not access video files', session_state
            else:
                local_video_paths = session_state.get('local_video_paths')
        else:
            return None, '❌ Error: No video files selected', session_state

        # Verify files exist
        if not local_audio_paths or not all(p and os.path.exists(p) for p in local_audio_paths):
             return None, f"❌ Error: Audio file is missing or inaccessible.", session_state
        if not local_video_paths or not all(p and os.path.exists(p) for p in local_video_paths):
             return None, f"❌ Error: Video files are missing or inaccessible.", session_state

        # Set GPU mode
        use_gpu = GPU_AVAILABLE
        set_gpu_mode(use_gpu)

        # Determine processing mode
        is_prores = processing_mode == 'prores_proxy'
        use_nvenc = (processing_mode in HW_ENCODERS) and hw_encoder_available(processing_mode)
        gpu_encoder = processing_mode if use_nvenc else 'none'

        # Determine FPS
        if custom_fps is not None and custom_fps > 0:
            output_fps = custom_fps
        else:
            # Follow the fps of the highest-resolution source (the source that
            # also decides the canvas in legacy "match best source" mode) — a
            # 10fps GIF that happens to be first in the list must not drag the
            # whole render down to 10fps. Still images have no real fps (probe
            # returns a flat 30), so they can't win this pick even when they
            # win the resolution.
            fps_candidates = [p for p in local_video_paths if not is_image_source(p)]
            if fps_candidates:
                best_path = max(
                    fps_candidates,
                    key=lambda p: (lambda wh: wh[0] * wh[1])(get_video_resolution(p)),
                )
                output_fps = get_video_fps(best_path)
            else:
                output_fps = 30.0

        # Prepare output paths
        output_folder = get_output_dir()
        os.makedirs(output_folder, exist_ok=True)
        name, _ = os.path.splitext(output_filename)
        ext = '.mov' if is_prores else '.mp4'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        output_path = os.path.join(output_folder, filename)
        temp_output = os.path.join(session_dir, filename)

        _audio_console_cb = (lambda stage, message:
                             console_logger.stage_line(stage, message) if console_logger else None)
        # Stages 1-5 are deterministic and depend only on the media selection,
        # so reuse them verbatim when the same songs+videos are re-rendered with
        # different render settings (variety, effects, text…) — the tuning loop.
        # The full render stays byte-identical: identical selected_beats/beat_info
        # in → identical plan+encode out; only the minutes of re-analysis are
        # skipped. Cache is per-Gradio-session and the key invalidates it the
        # moment the audio or video selection changes.
        analysis_key = (tuple(local_audio_paths), tuple(local_video_paths), bool(use_gpu))
        _cached = session_state.get('analysis_cache')
        if (isinstance(_cached, dict) and _cached.get('key') == analysis_key
                and _cached.get('local_audio_path')
                and os.path.exists(_cached['local_audio_path'])):
            local_audio_path = _cached['local_audio_path']
            selected_beats = _cached['selected_beats']
            beat_info = _cached['beat_info']
            print("♻️  Reusing cached analysis (media unchanged); skipping stages 1-5")
        else:
            if len(local_audio_paths) == 1:
                # Single song: byte-identical to the pre-multisong path.
                local_audio_path = local_audio_paths[0]
                selected_beats, beat_info = analyze_beats_auto(
                    local_audio_path,
                    use_gpu=use_gpu,
                    video_files=local_video_paths,
                    progress_callback=progress_callback,
                    console_callback=_audio_console_cb,
                )
            else:
                # Multi-song: concatenate the ordered tracks into one continuous
                # wav and analyze the whole timeline. Lazy import keeps the
                # single-song path free of any dependency on the module; a missing
                # module raises a clear, user-facing error only here, where more
                # than one song was actually requested. The returned trio maps 1:1
                # onto the single-song values: (concat wav, cut beats, beat_info).
                try:
                    import multisong
                except ImportError:
                    return (None,
                            '❌ Error: Multiple songs selected, but the multi-song '
                            'module (src/multisong.py) is unavailable. Select a '
                            'single song, or install/enable multi-song support.',
                            session_state)
                # work_dir is the per-session temp dir, NOT the processing dir:
                # create_music_video clears the processing dir at render start,
                # which would delete the concat wav before the audio mux reads it.
                # session_dir lives under GRADIO_TEMP_DIR and is cleaned on app
                # startup, so the concat wav has the right lifecycle.
                local_audio_path, selected_beats, beat_info = multisong.analyze_and_concat(
                    local_audio_paths,
                    session_dir,
                    video_files=local_video_paths,
                    use_gpu=use_gpu,
                    enable_qwen_semantics=True,
                    qwen_model_path=None,
                    progress_callback=progress_callback,
                    console_callback=_audio_console_cb,
                )
                _total_audio = float(beat_info.get('audio_duration') or 0.0)
                print(f"🎶 Multi-song: {len(local_audio_paths)} tracks → total "
                      f"{int(_total_audio) // 60}:{int(_total_audio) % 60:02d}")
            session_state['analysis_cache'] = {
                'key': analysis_key,
                'local_audio_path': local_audio_path,
                'selected_beats': selected_beats,
                'beat_info': beat_info,
            }
        beat_times = beat_info.get('times', selected_beats)
        _stage5_summary(console_logger, beat_info.get("video_analysis"))

        if progress_callback:
            progress_callback(_stage_status(6))

        # Resolve the effect palette once per render; the recipe line makes a
        # look reproducible (it lands in the render log via redirected stdout).
        # The Shuffle effect seed derives from the song filename when the seed
        # is 0. For multi-song it derives from the FIRST song (deterministic —
        # the concat wav's name is not stable across runs, whereas the first
        # song is); the derivation itself is unchanged. local_audio_paths[0]
        # equals local_audio_path in the single-song case, so single-song
        # renders are byte-identical.
        palette_ids, resolved_seed, recipe_line = resolve_effect_palette(
            rs.effect_mode, rs.effect_palette, rs.effect_seed, local_audio_paths[0])
        if rs.effect_style and rs.effect_style != 'clean':
            print(f"   🎛 {recipe_line}")

        # Create video. One resolved-settings dict; looks grade H.264/HEVC
        # renders only — ProRes stays pristine for external editing, matching
        # effects and text.
        resolved_settings = rs.to_settings_dict(
            is_prores=is_prores, palette_ids=palette_ids,
            resolved_seed=resolved_seed)
        result_path = create_music_video(
            local_audio_path, local_video_paths, selected_beats,
            output_file=temp_output, max_workers=parallel_workers,
            beat_info=beat_info, lossless_mode=is_prores,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, fps=output_fps,
            settings=resolved_settings,
        )

        # Move to output folder
        shutil.move(result_path, output_path)

        # Create preview for ProRes if needed
        preview_path = output_path
        if is_prores:
            preview_filename = f"{name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(session_dir, preview_filename)
            # Only input options (like -hwaccel) may appear before -i; the
            # encoder settings are output options and must come after it.
            preview_cmd = [FFMPEG_PATH, '-nostdin', '-hide_banner',
                           '-hwaccel', 'auto', '-i', output_path]
            if NVENC_AVAILABLE:
                preview_cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'])
            elif VIDEOTOOLBOX_AVAILABLE:
                preview_cmd.extend(['-c:v', 'h264_videotoolbox', '-q:v', '55', '-allow_sw', '1'])
            else:
                preview_cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23'])
            preview_cmd.extend(['-pix_fmt', 'yuv420p', '-y', preview_path])
            preview_result = subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)
            if (preview_result.returncode != 0 or not os.path.exists(preview_path)
                    or os.path.getsize(preview_path) == 0):
                print(f"   ⚠️  Preview transcode failed, showing ProRes file directly: "
                      f"{(preview_result.stderr or '').strip()[-500:]}")
                preview_path = output_path
        _stage6_summary(console_logger, beat_info)

        # Generate status message based on mode
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"

        if is_prores:
            encoder_info = "🎯 Lossless Concatenation"
        elif use_nvenc:
            encoder_info = f"⚡ {gpu_encoder.upper()}"
        else:
            encoder_info = "💻 libx264"

        total_cuts = len(selected_beats) - 1
        sections_info = beat_info.get('selection_info', [])
        total_processing_seconds = time.perf_counter() - total_started
        processing_label = gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if is_prores else "H264_CPU")

        status_msg = get_success_message_auto(
            total_cuts, len(beat_times),
            beat_info.get('tempo', 120), sections_info,
            encoder_info, fps_info, filename,
            audio_duration=beat_info.get('audio_duration'),
            output_fps=output_fps,
            total_processing_seconds=total_processing_seconds,
            processing_label=processing_label
        )
        # Return preview path for display, keep session_state intact
        return preview_path, status_msg, session_state

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg, session_state


def process_video(audio_files: VideoFilesInput, video_files: VideoFilesInput,
                 output_filename: str, processing_mode: str,
                 custom_fps: float, fit_mode: str, output_format: str,
                 effect_style: str,
                 effect_intensity: float, effect_mode: str,
                 effect_palette: List[str], effect_seed: float,
                 look_cube: str, variety: float, semantic_variety: float, speed_ramps: bool,
                 split_screen: bool, crossfades: bool,
                 text_entries: str, text_position: str,
                 text_scale: float, text_style: str, text_accent: str,
                 text_font: str, text_font_path: str,
                 session_state: dict) -> Iterator[StatusResult]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[StatusResult] = queue.Queue(maxsize=1)
    initial_status = _stage_status(1)
    # mirror=status_queue.put streams the curated per-stage content lines the
    # console logger already produces (source counts, beats/BPM, cut counts,
    # planner summary) to the UI status box, not just the render log.
    console_logger = StageConsoleLogger(sys.__stdout__, mirror=status_queue.put)
    render_console = RenderLogConsole(get_output_dir())

    # Single settings dict from here down: the style/effect/text parameter
    # chain is order-coupled positional at the Gradio boundary only, and
    # SETTINGS_KEYS (RenderSettings field order) is the one place that
    # defines that order. RenderSettings.to_settings_dict() bool()-coerces
    # the checkbox values downstream, so raw values pass through here.
    render_settings = dict(zip(SETTINGS_KEYS, (
        fit_mode, output_format, effect_style, effect_intensity,
        effect_mode, effect_palette, effect_seed, look_cube,
        variety, semantic_variety, speed_ramps, split_screen, crossfades,
        text_entries, text_position, text_scale, text_style, text_accent,
        text_font, text_font_path,
    ), strict=True))

    def progress_callback(message: str) -> None:
        # Accept either the pipeline's legacy "Stage N is processing." string or
        # an already-friendly label, resolve the stage number for the console
        # logger, and surface the friendly label to the UI. Non-stage messages
        # (should not occur today) pass through unchanged.
        stage_n = _LABEL_TO_STAGE.get(message)
        if stage_n is None:
            match = re.search(r"Stage (\d+) is processing", message)
            stage_n = int(match.group(1)) if match else None
        if stage_n is not None:
            console_logger.start_stage(stage_n)
            status_queue.put(_stage_status(stage_n))
        else:
            status_queue.put(message)

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(render_console), contextlib.redirect_stderr(render_console):
                result = _process_video_impl(
                    audio_files=audio_files,
                    video_files=video_files,
                    output_filename=output_filename,
                    processing_mode=processing_mode,
                    custom_fps=custom_fps,
                    settings=render_settings,
                    session_state=session_state,
                    progress_callback=progress_callback,
                    console_logger=console_logger,
                )
        except Exception as e:
            console_logger.line(f"Error: {e}")
            result = None, f"❌ Error: {e}", session_state
        finally:
            console_logger.finish()
            render_console.close()
        # Point failures at the captured pipeline log (FFmpeg stderr etc.).
        if result[1].startswith('❌') and os.path.exists(render_console.log_path):
            result = result[0], f"{result[1]}\n📄 Full log: {render_console.log_path}", result[2]
        result_queue.put(result)
        status_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    console_logger.start_stage(1)
    thread.start()

    last_status = initial_status
    yield None, initial_status, session_state

    while True:
        message = status_queue.get()
        if message is None:
            break
        if message != last_status:
            last_status = message
            yield None, message, session_state

    thread.join()
    yield result_queue.get()
