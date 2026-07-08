import os
import sys
import contextlib
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _install_windows_asyncio_connection_reset_filter() -> None:
    """Hide benign Windows asyncio pipe resets after browser/subprocess shutdown.

    On Windows, asyncio's Proactor transport can log a scary traceback when a
    local socket or subprocess pipe is closed by the other side after the real
    work is already complete. The app result is not affected, so suppress only
    that exact WinError 10054 callback and let all other async errors through.
    """
    if os.name != "nt" or getattr(asyncio, "_beatsync_win10054_filter", False):
        return
    asyncio._beatsync_win10054_filter = True

    def is_benign_reset(exc: BaseException | None) -> bool:
        if not isinstance(exc, ConnectionResetError):
            return False
        winerror = getattr(exc, "winerror", None)
        errno_value = getattr(exc, "errno", None)
        return winerror == 10054 or errno_value == 10054 or "WinError 10054" in str(exc)

    # Directly patch the noisy Proactor pipe cleanup callback when available.
    try:
        from asyncio import proactor_events

        transport_cls = getattr(proactor_events, "_ProactorBasePipeTransport", None)
        original_call_lost = getattr(transport_cls, "_call_connection_lost", None)
        if transport_cls is not None and original_call_lost is not None:
            def quiet_call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
                try:
                    return original_call_lost(self, exc)
                except ConnectionResetError as reset_exc:
                    if is_benign_reset(reset_exc):
                        return None
                    raise

            transport_cls._call_connection_lost = quiet_call_connection_lost
    except Exception:
        pass

    # Fallback for the same exception if it still reaches the loop logger.
    original_exception_handler = asyncio.BaseEventLoop.call_exception_handler

    def quiet_exception_handler(self, context):  # type: ignore[no-untyped-def]
        exc = context.get("exception") if isinstance(context, dict) else None
        handle = str(context.get("handle", "")) if isinstance(context, dict) else ""
        if is_benign_reset(exc) and "_ProactorBasePipeTransport._call_connection_lost" in handle:
            return None
        return original_exception_handler(self, context)

    asyncio.BaseEventLoop.call_exception_handler = quiet_exception_handler


_install_windows_asyncio_connection_reset_filter()

from logger import (
    setup_environment,
    USING_PORTABLE_PYTHON, USING_PORTABLE_CUDA, USING_CUPY_CTK, FFMPEG_FOUND
)

# Initialize environment
setup_environment()
# NOW import other modules (after CUDA environment is set)
import gradio as gr
import tempfile
import shutil
import datetime
import multiprocessing
import queue
import re
import subprocess
import threading
import time
import socket
from typing import Callable, Iterator, TypeAlias, Tuple, Dict, List

# Import FFmpeg processing module
from ffmpeg_processing import get_video_fps, get_video_resolution, is_image_source, FFMPEG_PATH
from looks import ensure_look_cubes, list_looks

# Shared runtime settings
from gpu_cpu_utils import (
    CPU_COUNT,
    MAX_THREADS,
    PARALLEL_WORKERS,
    GPU_INFO,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    VIDEOTOOLBOX_AVAILABLE,
    HW_ENCODERS,
    hw_encoder_available,
    set_gpu_mode,
)
from paths import (
    GRADIO_TEMP_DIR,
    get_input_dir,
    get_audio_input_dir,
    get_video_input_dir,
    get_output_dir,
)

gpu_data = GPU_INFO
gpu_info = f"{gpu_data['name']} ({gpu_data['cuda_version']})" if gpu_data['available'] else "CPU Mode"

from video_processor import create_music_video, OUTPUT_FORMATS, DEFAULT_OUTPUT_FORMAT
from effects import list_effect_choices, resolve_effect_palette

from auto_mode import analyze_beats_auto

# Import UI content
from ui_content import *

# Set environment variable for Gradio
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

VideoFilesInput : TypeAlias = List[str]
StatusResult : TypeAlias = Tuple[str, str, Dict]

STATUS_BOX_CSS = """
#status-output-box {
    min-height: 238px !important;
}

#status-output-box textarea {
    height: 186px !important;
    min-height: 186px !important;
    max-height: 186px !important;
    overflow-y: auto !important;
    resize: none !important;
}
"""


def _stage_status(stage_number: int) -> str:
    return f"Stage {stage_number} is processing. Please wait."


class RenderLogConsole:
    """Capture legacy verbose prints to a log file while the Gradio worker runs.

    The pipeline's diagnostics (including FFmpeg stderr on failures) used to be
    discarded entirely, leaving "N clip(s) failed" with no way to see why. Now
    everything lands in a per-run render log the error message can point at.
    """

    def __init__(self, log_dir: str, prefix: str = 'render'):
        self.log_path = os.path.join(
            log_dir, f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._file = None
        self._failed = False

    def _ensure_file(self):
        if self._file is None and not self._failed:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._file = open(self.log_path, 'a', encoding='utf-8', errors='replace')
            except OSError:
                self._failed = True
        return self._file

    def write(self, text: str) -> int:
        f = self._ensure_file()
        if f is not None:
            try:
                f.write(text)
                f.flush()
            except OSError:
                self._failed = True
        return len(text)

    def flush(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


class StageConsoleLogger:
    """Small CMD logger: stage start, up to 5 useful lines, stage end."""

    def __init__(self, stream, max_lines_per_stage: int = 5):
        self.stream = stream
        self.max_lines_per_stage = max(1, int(max_lines_per_stage))
        self.stage_number: int | None = None
        self.stage_started = 0.0
        self.stage_line_count = 0
        self.total_started = time.perf_counter()

    def start_stage(self, stage_number: int) -> None:
        if self.stage_number == stage_number:
            return
        self.end_stage()
        self.stage_number = stage_number
        self.stage_started = time.perf_counter()
        self.stage_line_count = 0
        self._write(f"Stage {stage_number} processing started:\n")

    def stage_line(self, stage_number: int, message: str) -> None:
        if self.stage_number != stage_number:
            self.start_stage(stage_number)
        self.line(message)

    def line(self, message: str) -> None:
        if self.stage_number is None:
            return
        if self.stage_line_count >= self.max_lines_per_stage:
            return
        message = self._clean(message)
        if message:
            self._write(f"  {message}\n")
            self.stage_line_count += 1

    def end_stage(self) -> None:
        if self.stage_number is None:
            return
        elapsed = int(round(time.perf_counter() - self.stage_started))
        self._write(f"Stage {self.stage_number} ended in {elapsed} seconds.\n\n")
        self.stage_number = None
        self.stage_started = 0.0
        self.stage_line_count = 0

    def finish(self) -> None:
        self.end_stage()
        elapsed = int(round(time.perf_counter() - self.total_started))
        self._write(f"Total time processing: {elapsed} seconds\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clean(self, text: str) -> str:
        text = str(text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", text).strip()


def _fmt_stage_seconds(seconds: float | int | None) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return "0.0s"


def _short_model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return os.path.basename(str(model_id).rstrip("/\\")) or str(model_id)


def _stage5_summary(console_logger: StageConsoleLogger | None, video_analysis: Dict | None) -> None:
    if console_logger is None or not isinstance(video_analysis, dict):
        return

    source_count = int(video_analysis.get("source_count") or len(video_analysis.get("videos") or []))
    worker_count = int(video_analysis.get("worker_count") or 1)
    cache_hits = int(video_analysis.get("cache_hits") or 0)
    ai_enabled = bool(video_analysis.get("ai_enabled"))
    qwen_total = int(video_analysis.get("qwen_frame_count") or 0)
    qwen_tags = int(video_analysis.get("qwen_tag_count") or 0)
    model_id = _short_model_name(video_analysis.get("qwen_model_id"))
    batch_size = int(video_analysis.get("qwen_concurrency") or 0)

    console_logger.line(f"Source videos: {source_count}, visual workers: {worker_count}")
    if ai_enabled:
        qwen_bits = ["Qwen: enabled"]
        if model_id:
            qwen_bits.append(f"model {model_id}")
        if batch_size:
            qwen_bits.append(f"batch {batch_size}")
        console_logger.line(", ".join(qwen_bits))
        if qwen_total:
            inference_seconds = float(video_analysis.get("qwen_inference_seconds") or video_analysis.get("qwen_seconds") or 0.0)
            qwen_rate = (qwen_total / inference_seconds) if inference_seconds > 0 else 0.0
            peak_vram = float(video_analysis.get("qwen_peak_vram_gb") or 0.0)
            perf_bits = []
            if batch_size:
                perf_bits.append(f"batch {batch_size}")
            if peak_vram > 0:
                perf_bits.append(f"~{peak_vram:.2f} GB VRAM")
            if qwen_rate > 0:
                perf_bits.append(f"{qwen_rate:.2f} candidates/s")
            if perf_bits:
                console_logger.line(f"Qwen performance: {', '.join(perf_bits)}")
            console_logger.line(
                f"Qwen tags: {qwen_tags}/{qwen_total} in {_fmt_stage_seconds(video_analysis.get('qwen_seconds'))}"
            )
    else:
        console_logger.line("Qwen: disabled")

    summary = video_analysis.get("summary")
    if summary:
        console_logger.line(f"Visual library: {summary}")
    console_logger.line(
        f"Analysis time: {_fmt_stage_seconds(video_analysis.get('analysis_seconds'))}, cache {cache_hits}/{source_count}"
    )


def _stage6_summary(console_logger: StageConsoleLogger | None, beat_info: Dict | None) -> None:
    if console_logger is None or not isinstance(beat_info, dict):
        return

    render_info = beat_info.get("render_info") or {}
    if not render_info:
        return

    cuts = int(render_info.get("render_cuts") or 0)
    frames = int(render_info.get("timeline_frames") or 0)
    fps = render_info.get("output_fps")
    if cuts or frames:
        fps_text = f" @ {float(fps):.1f} FPS" if fps is not None else ""
        console_logger.line(f"Render timeline: {cuts} cuts, {frames} frames{fps_text}")

    clip_workers = render_info.get("clip_workers")
    requested_workers = render_info.get("requested_workers")
    worker_text = ""
    if clip_workers:
        worker_text = f", workers {clip_workers}"
        if requested_workers and requested_workers != clip_workers:
            worker_text += f"/{requested_workers}"
    encoder = render_info.get("encoder")
    if encoder or worker_text:
        console_logger.line(f"Encoder: {encoder or 'unknown'}{worker_text}")

    if render_info.get("audio_duration") is not None:
        console_logger.line(f"Audio duration: {float(render_info['audio_duration']):.2f} seconds")

    plan_summary = render_info.get("plan_summary") or {}
    if plan_summary:
        console_logger.line(
            "Planner: "
            f"{int(plan_summary.get('clip_count') or 0)} clips, "
            f"{int(plan_summary.get('source_count') or 0)} sources, "
            f"AI moments {int(plan_summary.get('ai_tagged') or 0)}"
        )

    final_bits = []
    if render_info.get("target_resolution"):
        final_bits.append(f"resolution {render_info['target_resolution']}")
    if render_info.get("final_assembly_seconds") is not None:
        final_bits.append(f"assembly {_fmt_stage_seconds(render_info['final_assembly_seconds'])}")
    if final_bits:
        console_logger.line("Final: " + ", ".join(final_bits))

def find_launch_port(default_port: int = 7860, search_limit: int = 20) -> int:
    """Prefer the default Gradio port, then step forward if it is busy."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            print(f"⚠️ Invalid GRADIO_SERVER_PORT={env_port!r}; using auto port search.")

    for port in range(default_port, default_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                if port != default_port:
                    print(f"⚠️ Port {default_port} is busy. Starting BeatSync on port {port}.")
                return port

    raise OSError(f"Cannot find empty port in range: {default_port}-{default_port + search_limit - 1}")


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


def _process_video_impl(audio_file: str, video_files: VideoFilesInput,
                       output_filename: str, processing_mode: str,
                       custom_fps: float, session_state: dict,
                       fit_mode: str = 'crop',
                       output_format: str = DEFAULT_OUTPUT_FORMAT,
                       effect_style: str = 'clean',
                       effect_intensity: float = 0.7,
                       effect_mode: str = 'curated',
                       effect_palette: List[str] | None = None,
                       effect_seed: float = 0,
                       look_cube: str = '',
                       variety: float = 0.4,
                       semantic_variety: float = 0.4,
                       speed_ramps: bool = False,
                       split_screen: bool = True,
                       text_entries: str = '', text_position: str = 'bottom',
                       text_scale: float = 1.0,
                       settings: dict | None = None,
                       progress_callback: Callable[[str], None] | None = None,
                       console_logger: StageConsoleLogger | None = None) -> StatusResult:
    total_started = time.perf_counter()
    try:
        # The GUI passes one settings dict; the individual kwargs remain for
        # the headless/smoke-test entry point. The dict wins where present.
        if settings:
            fit_mode = settings.get('fit_mode', fit_mode)
            output_format = settings.get('output_format', output_format)
            effect_style = settings.get('effect_style', effect_style)
            effect_intensity = settings.get('effect_intensity', effect_intensity)
            effect_mode = settings.get('effect_mode', effect_mode)
            effect_palette = settings.get('effect_palette', effect_palette)
            effect_seed = settings.get('effect_seed', effect_seed)
            look_cube = settings.get('look_cube', look_cube)
            variety = settings.get('variety', variety)
            semantic_variety = settings.get('semantic_variety', semantic_variety)
            speed_ramps = settings.get('speed_ramps', speed_ramps)
            split_screen = settings.get('split_screen', split_screen)
            text_entries = settings.get('text_entries', text_entries)
            text_position = settings.get('text_position', text_position)
            text_scale = settings.get('text_scale', text_scale)
        parallel_workers = PARALLEL_WORKERS

        # Initialize session state if needed
        if 'original_audio_path' not in session_state:
            session_state['original_audio_path'] = None
            session_state['original_video_paths'] = []
        if 'session_dir' not in session_state or not os.path.isdir(session_state['session_dir']):
            session_state['session_dir'] = tempfile.mkdtemp(prefix='beatsync_', dir=GRADIO_TEMP_DIR)
        session_dir = session_state['session_dir']

        # Handle audio by referencing the selected file path directly.
        if audio_file:
            if audio_file != session_state.get('original_audio_path'):
                local_audio_path = _as_existing_source_path(audio_file)
                if local_audio_path:
                    session_state['local_audio_path'] = local_audio_path
                    session_state['original_audio_path'] = audio_file
                else:
                    return None, '❌ Error: Could not access audio file', session_state
            else:
                local_audio_path = session_state.get('local_audio_path')
        else:
            return None, '❌ Error: No audio file selected', session_state

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
        if not local_audio_path or not os.path.exists(local_audio_path):
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
        
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "CuPy CTK" if USING_CUPY_CTK else ("Portable" if USING_PORTABLE_CUDA else "System/None")

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

        selected_beats, beat_info = analyze_beats_auto(
            local_audio_path,
            use_gpu=use_gpu,
            video_files=local_video_paths,
            progress_callback=progress_callback,
            console_callback=lambda stage, message: console_logger.stage_line(stage, message) if console_logger else None,
        )
        beat_times = beat_info.get('times', selected_beats)
        _stage5_summary(console_logger, beat_info.get("video_analysis"))

        if progress_callback:
            progress_callback(_stage_status(6))

        # Resolve the effect palette once per render; the recipe line makes a
        # look reproducible (it lands in the render log via redirected stdout).
        palette_ids, resolved_seed, recipe_line = resolve_effect_palette(
            effect_mode, effect_palette, effect_seed, local_audio_path)
        if effect_style and effect_style != 'clean':
            print(f"   🎛 {recipe_line}")

        # Create video. One resolved-settings dict; looks grade H.264/HEVC
        # renders only — ProRes stays pristine for external editing, matching
        # effects and text.
        resolved_settings = {
            'fit_mode': fit_mode,
            'output_format': output_format,
            'effect_style': effect_style,
            'effect_intensity': effect_intensity,
            'effect_mode': effect_mode,
            'effect_palette': palette_ids,
            'effect_seed': resolved_seed,
            'look_cube': (None if is_prores else (look_cube or None)),
            'variety': variety,
            'semantic_variety': semantic_variety,
            'speed_ramps': bool(speed_ramps),
            'split_screen': bool(split_screen),
            'text_entries': [line.strip() for line in (text_entries or '').splitlines() if line.strip()],
            'text_position': text_position,
            'text_scale': text_scale,
        }
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
        gpu_info = f"⚡ GPU: {GPU_INFO}" if use_gpu else "💻 CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)"
        
        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "🎯 Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4)"
            encoder_info = f"⚡ {gpu_encoder.upper()}"
        else:
            codec_info = "H.264 (.mp4)"
            encoder_info = "💻 libx264"

        total_cuts = len(selected_beats) - 1
        sections_info = beat_info.get('selection_info', [])
        total_processing_seconds = time.perf_counter() - total_started
        processing_label = gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if is_prores else "H264_CPU")
        
        status_msg = get_success_message_auto(
            total_cuts, len(beat_times),
            beat_info.get('tempo', 120), sections_info,
            python_str, cuda_str, MAX_THREADS, CPU_COUNT,
            parallel_workers, gpu_info, encoder_info,
            codec_info, fps_info, filename, audio_info,
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


def process_video(audio_file: str, video_files: VideoFilesInput,
                 output_filename: str, processing_mode: str,
                 custom_fps: float, fit_mode: str, output_format: str,
                 effect_style: str,
                 effect_intensity: float, effect_mode: str,
                 effect_palette: List[str], effect_seed: float,
                 look_cube: str, variety: float, semantic_variety: float, speed_ramps: bool,
                 split_screen: bool,
                 text_entries: str, text_position: str,
                 text_scale: float, session_state: dict) -> Iterator[StatusResult]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[StatusResult] = queue.Queue(maxsize=1)
    initial_status = _stage_status(1)
    console_logger = StageConsoleLogger(sys.__stdout__)
    render_console = RenderLogConsole(get_output_dir())

    # Single settings dict from here down: the style/effect/text parameter
    # chain is order-coupled positional at the Gradio boundary only.
    render_settings = {
        'fit_mode': fit_mode,
        'output_format': output_format,
        'effect_style': effect_style,
        'effect_intensity': effect_intensity,
        'effect_mode': effect_mode,
        'effect_palette': effect_palette,
        'effect_seed': effect_seed,
        'look_cube': look_cube,
        'variety': variety,
        'semantic_variety': semantic_variety,
        'speed_ramps': bool(speed_ramps),
        'split_screen': bool(split_screen),
        'text_entries': text_entries,
        'text_position': text_position,
        'text_scale': text_scale,
    }

    def progress_callback(message: str) -> None:
        status_queue.put(message)
        match = re.search(r"Stage (\d+) is processing", message)
        if match:
            console_logger.start_stage(int(match.group(1)))

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(render_console), contextlib.redirect_stderr(render_console):
                result = _process_video_impl(
                    audio_file=audio_file,
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


def cleanup_on_startup():
    """
    Clean temporary runtime files on script start while preserving user inputs
    and the persistent video analysis cache.
    """
    input_base = get_input_dir()
    protected_dirs = {'audio', 'video', 'video_analysis_cache'}

    try:
        os.makedirs(get_audio_input_dir(), exist_ok=True)
        os.makedirs(get_video_input_dir(), exist_ok=True)
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

        if os.path.exists(input_base):
            for item in os.listdir(input_base):
                item_path = os.path.join(input_base, item)

                # Keep the latest user input files across restarts.
                if item in protected_dirs:
                    continue

                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f"   ⚠️  Could not clean {item}: {e}")

        # Recreate runtime temp upload folder after cleanup.
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

    except Exception as e:
        print(f"   ⚠️  Warning during startup cleanup: {e}")


def create_ui() -> gr.Blocks:
    # These definitions are needed within the function's scope
    python_status = "✅ Portable (bin/python-3.13.14-embed-amd64/)" if USING_PORTABLE_PYTHON else "⚠️  System Python"
    if USING_CUPY_CTK:
        cuda_status = "✅ CuPy CTK (Python wheel libraries)"
    elif USING_PORTABLE_CUDA:
        cuda_status = "✅ Portable (bin/CUDA/v13.3)"
    else:
        cuda_status = "⚠️  System CUDA (or not available)"
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 📁 Input Files')
                audio_input = gr.File(label=LABEL_AUDIO_FILE, file_types=[t for ext in ['.mp3', '.wav', '.flac'] for t in (ext, ext.upper())], type='filepath', elem_id='audio-file-input')
                # Include uppercase variants: Gradio's drag-drop filter is case-sensitive
                # (gradio#10746), unlike its file picker.
                # Still images ride along with the video pipeline (Ken Burns
                # duration synthesis lands with the image-source support).
                video_file_types = [t for ext in ['.mp4', '.mkv', '.mov', '.webm', '.m4v', '.avi', '.gif',
                                                  '.jpg', '.jpeg', '.png', '.webp', '.bmp'] for t in (ext, ext.upper())]

                # Gradio's File component ignores drops once it holds files
                # (gradio#10325), so the dropzone never keeps its value: uploads
                # accumulate in video_state and render in the list below, which
                # keeps the zone permanently droppable.
                video_dropzone = gr.File(label=LABEL_VIDEO_FILES, file_count='multiple', file_types=video_file_types, type='filepath', elem_id='video-files-input', height=110)
                video_state = gr.State([])
                video_list = gr.CheckboxGroup(choices=[], value=[], label='🎬 Loaded videos (0)', info='Tick files to remove them', visible=False)
                with gr.Row():
                    remove_videos_btn = gr.Button('🗑 Remove selected', size='sm', visible=False)
                    clear_videos_btn = gr.Button('♻️ Clear all', size='sm', visible=False)

                def _video_list_updates(files):
                    shown = len(files) > 0
                    return (
                        gr.update(choices=[(os.path.basename(f), f) for f in files],
                                  value=[], label=f'🎬 Loaded videos ({len(files)})', visible=shown),
                        gr.update(visible=shown),
                        gr.update(visible=shown),
                    )

                def _add_videos(new_files, files):
                    files = list(files or [])
                    for f in (new_files or []):
                        if f not in files:
                            files.append(f)
                    return (files, None, *_video_list_updates(files))

                def _remove_videos(selected, files):
                    selected = set(selected or [])
                    files = [f for f in (files or []) if f not in selected]
                    return (files, *_video_list_updates(files))

                def _clear_videos():
                    return ([], *_video_list_updates([]))

                video_dropzone.upload(
                    _add_videos,
                    inputs=[video_dropzone, video_state],
                    outputs=[video_state, video_dropzone, video_list, remove_videos_btn, clear_videos_btn],
                )
                remove_videos_btn.click(
                    _remove_videos,
                    inputs=[video_list, video_state],
                    outputs=[video_state, video_list, remove_videos_btn, clear_videos_btn],
                )
                clear_videos_btn.click(
                    _clear_videos,
                    inputs=None,
                    outputs=[video_state, video_list, remove_videos_btn, clear_videos_btn],
                )

                # Tier 1 — creative intents only. Mechanical knobs live in the
                # Advanced accordion below; the engine's defaults are the real UI.
                with gr.Group():
                    gr.Markdown('### 🎨 Create')
                    effect_style_input = gr.Radio(
                        choices=[('Minimal (clean cuts)', 'clean'), ('Music video (AMV)', 'amv'), ('Hype', 'hype')],
                        value='clean', label='Style',
                        info='How energetic the edit feels. Beat-aware effects: zooms and flashes on drops, saturation pulses on the beat. Minimal disables effects in every mode.')
                    look_input = gr.Dropdown(
                        choices=list_looks(), value='', label='Look',
                        info='Color grade for the whole video (baked LUTs: film, warm/cool, day-for-night). H.264/HEVC modes only; ProRes stays ungraded.')
                    output_format_input = gr.Dropdown(
                        choices=[(label, key) for key, (label, _) in OUTPUT_FORMATS.items()],
                        value=DEFAULT_OUTPUT_FORMAT, label='Output canvas',
                        info='The frame every render targets. Fixed canvases keep one odd portrait clip from flipping the whole video; "Match best source" is the old behavior (highest-resolution source decides).')

                with gr.Group():
                    gr.Markdown('### 📝 Text Overlays')
                    text_entries_input = gr.Textbox(
                        label='Text entries (one per line)', lines=4, value='',
                        placeholder='Leave empty for no text.\nEach line appears once, spread evenly across the video.\nPin an entry to a time with @: "@15 Finish strong" or "@1:23 Halfway"',
                        info='Quotes, captions, titles — any text. Every line gets its own beat-snapped time window; @ pins one to a timestamp.')

                # Tier 2 — deliberate overrides of decisions the engine already
                # makes well. Everything keeps its variable name and values;
                # only the container (and some labels) changed.
                with gr.Accordion('⚙️ Advanced', open=False):
                    gr.Markdown('**Effects**')
                    effect_mode_input = gr.Radio(
                        choices=[('Curated', 'curated'), ('Custom', 'custom'), ('Surprise shuffle', 'shuffle')],
                        value='curated', label='Effect mode',
                        info='Curated = the classic style presets. Custom = pick your own palette. Shuffle = a seeded random palette (same seed, same video).')
                    effect_palette_input = gr.CheckboxGroup(
                        choices=list_effect_choices(), value=[], visible=False,
                        label='Effect palette',
                        info='Picked effects still land where the music calls for them (drops, builds, calm parts).')
                    effect_seed_input = gr.Number(
                        value=0, precision=0, visible=False, label='Shuffle seed',
                        info='0 = derived from the song. Change it to re-roll the palette; renders stay reproducible.')
                    effect_intensity_input = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                                       label='Effect intensity')

                    def _effect_mode_updates(mode):
                        return (
                            gr.update(visible=mode == 'custom'),
                            gr.update(visible=mode == 'shuffle'),
                        )

                    effect_mode_input.change(
                        _effect_mode_updates,
                        inputs=[effect_mode_input],
                        outputs=[effect_palette_input, effect_seed_input],
                    )

                    gr.Markdown('**Editing**')
                    variety_input = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.05, label='Source variety',
                        info='0 = pure quality picks (some uploads may never appear). Higher guarantees every source at least one moment and spreads usage more evenly.')
                    semantic_variety_input = gr.Slider(
                        0.0, 1.0, value=0.4, step=0.05, label='Visual variety',
                        info='Avoid runs of visually similar shots (needs the DINOv2 model — scripts/fetch_dinov2.py).')
                    speed_ramps_input = gr.Checkbox(
                        value=False, label='Speed ramps (experimental)',
                        info='Beat-aware retiming: slow-mo drifts on calm parts, rushes through builds, decel ramps and freeze hits on drops. Frame counts stay exact; H.264/HEVC modes only.')
                    split_screen_input = gr.Checkbox(
                        value=True, label='Pair vertical clips (split screen)',
                        info='Renders some high-energy segments as two vertical clips side by side. Needs two or more vertical sources; fires on hard cuts only. H.264/HEVC modes only.')

                    gr.Markdown('**Framing**')
                    fit_mode_input = gr.Radio(
                        choices=[('Auto (smart)', 'crop'), ('Blurred background', 'blur'), ('Letterbox', 'pad')],
                        value='crop', label='Frame fit',
                        info='How sources with a different aspect ratio fill the frame. Auto picks per clip: a subject-tracked crop for small mismatches (trims at most ~15%), a graded blur fill for bigger ones, and a slow scanning pan for extreme ones (e.g. vertical phone clips). Blurred background and Letterbox force that single look on every clip.')

                    gr.Markdown('**Output**')
                    if NVENC_AVAILABLE:
                        processing_mode = gr.Radio(choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC (H.265)', 'hevc_nvenc'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_nvenc', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_nvenc())
                    elif VIDEOTOOLBOX_AVAILABLE:
                        processing_mode = gr.Radio(choices=[('Apple VideoToolbox H.264', 'h264_videotoolbox'), ('Apple VideoToolbox HEVC (H.265)', 'hevc_videotoolbox'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_videotoolbox', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_videotoolbox())
                    else:
                        processing_mode = gr.Radio(choices=[('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='cpu', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_cpu())
                    gr.Markdown('*ProRes Precise Mode keeps footage pristine for external editing: effects, text overlays, looks and speed ramps are **not** applied there.*')
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)
                    output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)
                    with gr.Row():
                        text_position_input = gr.Radio(
                            choices=[('Lower third', 'bottom'), ('Center', 'center'), ('Top', 'top')],
                            value='bottom', label='Text position')
                        text_scale_input = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label='Text size')

                process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')

            with gr.Column(scale=1):
                gr.Markdown('### 📺 Output')
                status_output = gr.Textbox(label='Status', interactive=False, value=get_ready_status(python_status, cuda_status, MAX_THREADS, CPU_COUNT, ffmpeg_status, GPU_AVAILABLE, gpu_info, NVENC_AVAILABLE), lines=4, max_lines=4, elem_id='status-output-box')
                video_output = gr.Video(label='Generated Music Video', interactive=False, elem_id='generated-video-output')
                
        # The button is disabled for the whole render and re-enabled by a
        # final .then() step, which Gradio runs on success AND on error — a
        # second click mid-render would clear the processing directory out
        # from under the run in flight.
        process_btn.click(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=[process_btn],
        ).then(
            fn=process_video,
            inputs=[
                audio_input, video_state,
                output_filename, processing_mode, custom_fps,
                fit_mode_input, output_format_input,
                effect_style_input, effect_intensity_input,
                effect_mode_input, effect_palette_input, effect_seed_input,
                look_input, variety_input, semantic_variety_input, speed_ramps_input,
                split_screen_input,
                text_entries_input, text_position_input, text_scale_input,
                session_state
            ],
            outputs=[video_output, status_output, session_state],
            show_progress='hidden'
        ).then(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[process_btn],
        )

    return app

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Clean up old files only on startup
    cleanup_on_startup()

    # Derive .cube LUTs from the committed look PNGs (looks/*.cube is
    # gitignored — ~7 MB each, cheap to regenerate).
    ensure_look_cubes()
    
    app = create_ui()
    launch_port = find_launch_port()
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=True,
        show_error=True
    )
