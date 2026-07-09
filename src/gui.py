import os
import sys
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

from logger import setup_environment

# Initialize environment
setup_environment()
# NOW import other modules (after CUDA environment is set)
import gradio as gr
import multiprocessing
import shutil
import socket

from looks import ensure_look_cubes, list_looks

# Shared runtime settings — only the hardware-encoder flags that pick the
# processing-mode radio's choices are read here.
from gpu_cpu_utils import (
    NVENC_AVAILABLE,
    VIDEOTOOLBOX_AVAILABLE,
)
from paths import (
    GRADIO_TEMP_DIR,
    get_input_dir,
    get_audio_input_dir,
    get_video_input_dir,
)

from video_processor import OUTPUT_FORMATS, DEFAULT_OUTPUT_FORMAT
from effects import list_effect_choices

# Import UI content (only the names this module renders).
from ui_content import (
    UI_TITLE, UI_MAIN_DESCRIPTION,
    LABEL_AUDIO_FILE, LABEL_VIDEO_FILES,
    LABEL_CUSTOM_FPS, INFO_CUSTOM_FPS,
    LABEL_PROCESSING_MODE,
    LABEL_OUTPUT_FILENAME, INFO_OUTPUT_FILENAME,
    get_ready_status,
    get_processing_mode_info_nvenc,
    get_processing_mode_info_videotoolbox,
    get_processing_mode_info_cpu,
)

# Render orchestration (settings model, headless pipeline, Gradio worker) lives
# in orchestrator.py. Re-exported here so the documented smoke entry point
# `gui._process_video_impl(...)` and the button callback keep working unchanged.
# Imported after setup_environment() so the CUDA/FFmpeg-dependent pipeline
# modules it pulls in initialize with the environment already configured.
from orchestrator import (
    _process_video_impl,
    process_video,
    RenderSettings,
    SETTINGS_KEYS,
)

# Set environment variable for Gradio
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

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
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 📁 Input Files')
                # Multi-song: the audio input mirrors the video dropzone
                # pattern below (accumulating gr.State + list; see the
                # gradio#10325 note there). Order matters — songs play in the
                # order they were added, so the list is numbered and removal
                # keeps the remaining order intact.
                audio_dropzone = gr.File(label=LABEL_AUDIO_FILE, file_count='multiple', file_types=[t for ext in ['.mp3', '.wav', '.flac'] for t in (ext, ext.upper())], type='filepath', elem_id='audio-file-input', height=110)
                audio_state = gr.State([])
                audio_list = gr.CheckboxGroup(choices=[], value=[], label='🎵 Loaded songs (0)', info='Songs play in this order. Tick files to remove them', visible=False)
                with gr.Row():
                    remove_audio_btn = gr.Button('🗑 Remove selected', size='sm', visible=False)
                    clear_audio_btn = gr.Button('♻️ Clear all', size='sm', visible=False)

                def _audio_list_updates(files):
                    shown = len(files) > 0
                    return (
                        gr.update(choices=[(f"{n + 1}. {os.path.basename(f)}", f) for n, f in enumerate(files)],
                                  value=[], label=f'🎵 Loaded songs ({len(files)})', visible=shown),
                        gr.update(visible=shown),
                        gr.update(visible=shown),
                    )

                def _add_audio(new_files, files):
                    files = list(files or [])
                    for f in (new_files or []):
                        if f not in files:
                            files.append(f)
                    return (files, None, *_audio_list_updates(files))

                def _remove_audio(selected, files):
                    selected = set(selected or [])
                    files = [f for f in (files or []) if f not in selected]
                    return (files, *_audio_list_updates(files))

                def _clear_audio():
                    return ([], *_audio_list_updates([]))

                audio_dropzone.upload(
                    _add_audio,
                    inputs=[audio_dropzone, audio_state],
                    outputs=[audio_state, audio_dropzone, audio_list, remove_audio_btn, clear_audio_btn],
                )
                remove_audio_btn.click(
                    _remove_audio,
                    inputs=[audio_list, audio_state],
                    outputs=[audio_state, audio_list, remove_audio_btn, clear_audio_btn],
                )
                clear_audio_btn.click(
                    _clear_audio,
                    inputs=None,
                    outputs=[audio_state, audio_list, remove_audio_btn, clear_audio_btn],
                )

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
                    crossfades_input = gr.Checkbox(
                        value=False, label='Crossfade calm cuts',
                        info='Dissolves ~1 in 3 calm (soft/flow) boundaries instead of hard-cutting. Re-encodes those boundary chunks; hard cuts stay the fast default. H.264/HEVC modes only.')

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

                # --- ProRes Precise Mode conflict guard ----------------------
                # ProRes voids every creative control below (orchestrator's
                # to_settings_dict look gate + the lossless_mode branches in
                # video_processor). Greying them out is COSMETIC ONLY: disabled
                # inputs still submit their values and the pipeline still
                # nullifies them, so the resolved settings dict is byte-identical.
                # This just makes the silent dependency visible. fit_mode and
                # variety stay live — both are honored on ProRes proxies.
                _prores_disabled_controls = [
                    effect_style_input, look_input,
                    effect_mode_input, effect_palette_input, effect_seed_input,
                    effect_intensity_input, semantic_variety_input,
                    speed_ramps_input, split_screen_input, crossfades_input,
                    text_entries_input, text_position_input, text_scale_input,
                ]
                _prores_base_labels = [c.label for c in _prores_disabled_controls]
                _PRORES_NOTE = ' — disabled in ProRes Precise Mode'

                def _prores_conflict_guard(mode):
                    prores = mode == 'prores_proxy'
                    return [
                        gr.update(interactive=not prores,
                                  label=(base + _PRORES_NOTE) if prores else base)
                        for base in _prores_base_labels
                    ]

                processing_mode.change(
                    _prores_conflict_guard,
                    inputs=[processing_mode],
                    outputs=_prores_disabled_controls,
                )

                # --- Render recipe readout -----------------------------------
                # Plain-English echo of the current control state so the mode's
                # effect is visible before clicking. Read-only sugar: NOT a
                # settings field, NOT wired into process_video's outputs.
                _STYLE_LABELS = {'clean': 'Minimal', 'amv': 'Music video', 'hype': 'Hype'}
                _MODE_LABELS = {
                    'h264_nvenc': 'H.264 (NVENC)', 'hevc_nvenc': 'HEVC (NVENC)',
                    'h264_videotoolbox': 'H.264', 'hevc_videotoolbox': 'HEVC',
                    'cpu': 'H.264 (CPU)', 'prores_proxy': 'ProRes Precise',
                }

                def _render_recipe(mode, style, look, variety, split, ramps, xfades, text):
                    text_lines = [ln for ln in (text or '').splitlines() if ln.strip()]
                    if mode == 'prores_proxy':
                        parts = ['**ProRes Precise**', 'effects OFF', 'look OFF',
                                 'text OFF', 'no retimes/split/crossfade', 'untouched footage']
                        return '🎬 ' + '  ·  '.join(parts)
                    parts = [f'**{_STYLE_LABELS.get(style, style)}**']
                    if look:
                        parts.append(f'{look} look')
                    if style != 'clean':
                        parts.append('effects ON')
                    parts.append(f'variety {variety:.2g}')
                    if text_lines:
                        parts.append(f'text ×{len(text_lines)}')
                    if split:
                        parts.append('split-screen ON')
                    if ramps:
                        parts.append('speed ramps ON')
                    if xfades:
                        parts.append('crossfades ON')
                    parts.append(_MODE_LABELS.get(mode, mode))
                    return '🎬 ' + '  ·  '.join(parts)

                _recipe_inputs = [
                    processing_mode, effect_style_input, look_input, variety_input,
                    split_screen_input, speed_ramps_input, crossfades_input, text_entries_input,
                ]
                recipe_readout = gr.Markdown('', elem_id='recipe-readout')
                for _c in _recipe_inputs:
                    _c.change(_render_recipe, inputs=_recipe_inputs, outputs=recipe_readout)
                app.load(_render_recipe, inputs=_recipe_inputs, outputs=recipe_readout)

                process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')

            with gr.Column(scale=1):
                gr.Markdown('### 📺 Output')
                status_output = gr.Textbox(label='Status', interactive=False, value=get_ready_status(), lines=4, max_lines=4, elem_id='status-output-box')
                video_output = gr.Video(label='Generated Music Video', interactive=False, elem_id='generated-video-output')

        # One component per RenderSettings field, in SETTINGS_KEYS order —
        # Gradio hands these to process_video positionally, so this list is
        # the only place the component-to-field pairing is spelled out.
        settings_components = [
            fit_mode_input, output_format_input,
            effect_style_input, effect_intensity_input,
            effect_mode_input, effect_palette_input, effect_seed_input,
            look_input, variety_input, semantic_variety_input,
            speed_ramps_input, split_screen_input, crossfades_input,
            text_entries_input, text_position_input, text_scale_input,
        ]
        assert len(SETTINGS_KEYS) == len(settings_components)

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
                audio_state, video_state,
                output_filename, processing_mode, custom_fps,
                *settings_components,
                session_state
            ],
            outputs=[video_output, status_output, session_state],
            # 'minimal' shows a small spinner so a multi-minute render never
            # looks frozen; the streamed status text carries the real detail.
            show_progress='minimal'
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
