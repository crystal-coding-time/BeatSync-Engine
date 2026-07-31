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
import math
import multiprocessing
import re
import shutil
import socket
import subprocess

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


# --- Font discovery for the Text tab picker ----------------------------------

_FONT_DIRS = [
    os.path.expanduser('~/Library/Fonts'),
    '/Library/Fonts',
    '/System/Library/Fonts/Supplemental',
    '/usr/share/fonts/truetype',      # Linux parity
    'C:/Windows/Fonts',               # Windows parity
]


def _discover_font_choices() -> list:
    """(label, path) choices for the font dropdown: 'Auto' plus every
    .ttf/.otf found in the standard font directories, deduped by filename
    (first hit wins — user fonts shadow system ones). Scan only; nothing is
    loaded, so a corrupt file can't break UI construction."""
    choices = [('Auto (Arial Bold / $BEATSYNC_FONT)', '')]
    env_font = os.environ.get('BEATSYNC_FONT')
    if env_font and os.path.exists(env_font):
        choices.append((f'$BEATSYNC_FONT — {os.path.basename(env_font)}', env_font))
    seen = set()
    try:
        for font_dir in _FONT_DIRS:
            if not os.path.isdir(font_dir):
                continue
            for name in sorted(os.listdir(font_dir)):
                if not name.lower().endswith(('.ttf', '.otf')):
                    continue
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                choices.append((os.path.splitext(name)[0], os.path.join(font_dir, name)))
    except Exception as e:
        print(f"   ⚠️  Font scan failed ({e}) — dropdown offers Auto only")
    return choices


# --- Text schedule preview (read-only helpers) -------------------------------
# The timeline planner silently skips entries that don't fit; until now the
# only trace was a render-log warning discovered AFTER a multi-minute render.
# These helpers power the Text tab's "Preview schedule" button. They are
# strictly read-only and total: never start analysis, never run ffmpeg
# renders, never mutate session state, never raise into Gradio.

_TEXT_TAG_RE = re.compile(r'\[\s*(?:style|widget)\s*:', re.IGNORECASE)
_TEXT_PREVIEW_HEADERS = ['#', 'Entry', 'Outcome', 'Notes']
_CLASSIC_TAG_WARNING = ('⚠️ [tags] only render in Motion style — '
                        "they'll appear as literal text in Classic")


def _fmt_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    return f"{total // 60}:{total % 60:02d}"


def _classic_tag_lint_message(text_value, style_value):
    """U4 lint: [style:…]/[widget:…] tags are Motion-only; in Classic they
    burn into the video as literal text. Returns the warning string or None."""
    try:
        if (style_value or 'classic') == 'classic' and _TEXT_TAG_RE.search(text_value or ''):
            return _CLASSIC_TAG_WARNING
    except Exception:
        pass
    return None


def _probe_audio_duration(paths) -> float | None:
    """Total duration of the selected songs via ffprobe (metadata read only,
    no decode — well under a second). None when nothing is probeable."""
    try:
        from ffmpeg_processing import FFPROBE_PATH  # bundled bin/, else PATH
    except Exception:
        return None
    total, found = 0.0, False
    for p in (paths or []):
        if not p or not os.path.exists(p):
            continue
        try:
            proc = subprocess.run(
                [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', p],
                capture_output=True, text=True, timeout=10)
            dur = float(proc.stdout.strip().splitlines()[0])
            if math.isfinite(dur) and dur > 0:
                total += dur
                found = True
        except Exception:
            continue
    return total if found else None


def _preview_timeline(audio_files, session_state):
    """Best cut grid recoverable WITHOUT running analysis.

    Beat-exact path: orchestrator caches stages 1-5 per Gradio session as
    session_state['analysis_cache'] = {key, local_audio_path, selected_beats,
    beat_info}. It is trusted only while the current audio selection still
    matches session_state['original_audio_paths'] AND the cache key's audio
    tuple matches the resolved local paths (same staleness rule the render
    uses; a changed video selection keeps the audio-driven beat grid valid).
    The cut grid is rebuilt the way build_frame_aligned_cut_timeline does
    ([0] + interior beats + [audio_duration]) minus frame quantization and
    the 1-frame cut-lead — a sub-frame delta, invisible at M:SS granularity.

    Fallback: an even 2s grid over the ffprobe'd total duration (labelled an
    estimate; ~2s is a typical cut cadence, and the grid only needs enough
    segments that entries don't falsely clash). No duration at all →
    (None, None, 'none').

    Returns (cut_times, beat_times, kind), kind ∈ {'beat-exact', 'estimate',
    'none'}.
    """
    st = session_state if isinstance(session_state, dict) else {}
    selection, _seen = [], set()
    for a in (audio_files or []):
        if a and a not in _seen:
            _seen.add(a)
            selection.append(a)
    try:
        cache = st.get('analysis_cache')
        if (isinstance(cache, dict) and selection
                and selection == st.get('original_audio_paths')
                and tuple(st.get('local_audio_paths') or ())
                == tuple((cache.get('key') or ((),))[0])):
            beat_info = cache.get('beat_info') or {}
            beats = sorted(float(b) for b in (cache.get('selected_beats') or ())
                           if math.isfinite(float(b)))
            duration = float(beat_info.get('audio_duration') or 0.0)
            if duration <= 0 and beats:
                duration = beats[-1]
            if beats and duration > 0.5:
                cut_times = [0.0] + [b for b in beats if 0.0 < b < duration] + [duration]
                return cut_times, beat_info.get('times'), 'beat-exact'
    except Exception:
        pass
    duration = _probe_audio_duration(selection)
    if duration and duration > 0.5:
        step = 2.0
        cut_times = [i * step for i in range(int(duration / step) + 1)]
        if duration - cut_times[-1] > 0.01:
            cut_times.append(duration)
        if len(cut_times) >= 2:
            return cut_times, None, 'estimate'
    return None, None, 'none'


def _build_text_preview(text_value, style_value, audio_files, session_state):
    """U1: one row per entry line, in original line order — its time window,
    'SKIPPED — no room', or a lint note. Returns (note_markdown, rows).

    May raise on truly unexpected input; the Gradio wrapper catches and shows
    a friendly one-row table instead.
    """
    raw_lines = [ln for ln in (text_value or '').splitlines() if ln.strip()]
    if not raw_lines:
        return ('Nothing to preview — add a text entry above first.', [])
    from text_overlay import parse_text_entries, plan_text_windows

    # Parse line-by-line so every raw line keeps its identity even when the
    # parser drops it. parse_text_entries is line-wise, so the concatenation
    # equals parsing all lines at once and the planner sees the same entries
    # the render will.
    per_line = [parse_text_entries([ln]) for ln in raw_lines]
    entries = [e for parsed in per_line for e in parsed]

    cut_times, beat_times, kind = _preview_timeline(audio_files, session_state)

    # Motion style widens windows for '[widget:… duration:Ns]' entries —
    # mirror the render's min_durations so the preview reflects the widened
    # plan. Same filter as the render: only finite positive durations count
    # (the -1.0 'span window' sentinel stays None).
    is_classic = (style_value or 'classic') == 'classic'
    min_durations = None
    if not is_classic and entries:
        try:
            from styled_text import parse_entry_tags
            durs = [parse_entry_tags(t)[2] for t, _ in entries]
            min_durations = [d if (d and d > 0) else None for d in durs]
            if not any(min_durations):
                min_durations = None
        except Exception:
            min_durations = None

    schedule = []
    planned = bool(entries) and cut_times is not None and len(cut_times) >= 2
    if planned:
        _seg_map, schedule = plan_text_windows(
            entries, cut_times, beat_times=beat_times,
            min_durations=min_durations)

    remaining = list(schedule)  # placed (text, start, end), sorted by start
    timeline_end = cut_times[-1] if cut_times else None

    rows = []
    entry_idx = 0
    placed_count = 0
    for line_no, (raw, parsed) in enumerate(zip(raw_lines, per_line), start=1):
        raw = raw.strip()
        display = raw if len(raw) <= 50 else raw[:49] + '…'
        notes = []
        if is_classic and _TEXT_TAG_RE.search(raw):
            notes.append('[tags] only render in Motion style — literal text in Classic')
        if not parsed:
            # The parser dropped the whole line. Distinguish a readable stamp
            # with no text ('@15') from an unreadable stamp by re-parsing with
            # a placeholder word appended.
            probe = parse_text_entries([raw + ' placeholder'])
            if probe and probe[0][1] is not None:
                notes.append('no text after the @-stamp')
            elif raw.startswith('@'):
                notes.append('@-stamp unreadable')
            else:
                notes.append('empty after parsing')
            rows.append([str(line_no), display, 'NOT RENDERED', '; '.join(notes)])
            continue
        text, pin = entries[entry_idx]
        want = min_durations[entry_idx] if min_durations else None
        entry_idx += 1
        if pin is None and raw.startswith('@'):
            # Wording adapts to the parser's fallback: stamp stripped →
            # auto-placed clean text; stamp kept → the '@…' renders literally.
            if text == raw:
                notes.append("@-stamp unreadable — whole line (incl. '@…') renders as literal text")
            else:
                notes.append('@-stamp unreadable — placed automatically (no pin)')
        if want:
            notes.append(f'widget widens window to ≥{want:g}s')
        if pin is not None and timeline_end is not None and pin > timeline_end + 0.01:
            notes.append(f'pin @{_fmt_mmss(pin)} is past the end of the audio '
                         f'({_fmt_mmss(timeline_end)})')
        if planned:
            # Match placements back to entries by text, consuming in original
            # entry order so duplicate lines pair up deterministically.
            window = None
            for i, (t, ws, we) in enumerate(remaining):
                if t == text:
                    window = remaining.pop(i)
                    break
            if window is not None:
                placed_count += 1
                outcome = f'{_fmt_mmss(window[1])}–{_fmt_mmss(window[2])}'
            else:
                outcome = 'SKIPPED — no room on the timeline'
        else:
            outcome = (f'@{_fmt_mmss(pin)} (pinned)' if pin is not None
                       else 'auto — spread evenly')
        rows.append([str(line_no), display, outcome, '; '.join(notes)])

    if kind == 'beat-exact':
        note = ("**Beat-exact preview** — windows come from this session's cached "
                'beat analysis, the same grid the next render will use.')
    elif kind == 'estimate':
        note = ('**Estimate** (even grid over the probed song duration '
                f'{_fmt_mmss(timeline_end)}) — render once to get beat-exact windows.')
    else:
        note = ('**Estimate — render once to get beat-exact windows.** No song '
                'duration recoverable yet (load audio first); showing entry order only.')
    if planned:
        skipped = sum(1 for r in rows if r[2].startswith('SKIPPED'))
        note += f'  \nPlaced {placed_count}/{len(entries)}'
        if skipped:
            note += f' · **{skipped} skipped**'
    return note, rows


def create_ui() -> gr.Blocks:
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                gr.Markdown('### 1 · Media')
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

            # Zone 2 — every creative decision, grouped by what you're deciding.
            # The engine's defaults are the real UI; purpose tabs hold the full
            # control set without a catch-all "Advanced" bucket. Each control
            # keeps its variable name/values so the settings contract is intact.
            with gr.Column(scale=3):
                gr.Markdown('### 2 · Direct the edit')
                gr.Markdown("*Defaults already look good — tweak only what you want. Each control's help text says what it does.*")
                with gr.Tabs():
                    with gr.Tab('Vibe'):
                        effect_style_input = gr.Radio(
                            choices=[('Minimal (clean cuts)', 'clean'), ('Music video (AMV)', 'amv'), ('Hype', 'hype')],
                            value='hype', label='Style',
                            info='How energetic the edit feels. Beat-aware effects: zooms and flashes on drops, saturation pulses on the beat. Minimal disables effects in every mode.')
                        look_input = gr.Dropdown(
                            choices=list_looks(), value='', label='Look',
                            info='Color grade for the whole video (baked LUTs: film, warm/cool, day-for-night). H.264/HEVC modes only; ProRes stays ungraded.')
                        effect_intensity_input = gr.Slider(
                            0.0, 1.0, value=0.7, step=0.05, label='Effect intensity',
                            info='How hard the beat-aware effects hit. Lower = subtle pulses; higher = punchier zooms and flashes on the drops.')
                        # Defaults here must match RenderSettings in
                        # orchestrator.py exactly, or the UI and the headless
                        # entry point would render differently.
                        semantic_fx_input = gr.Checkbox(
                            value=True, label='Content-aware effects',
                            info='Effects check the footage before firing: no glitch or shake on serene clips, no mirror effects on face close-ups, and effects concentrate where the music hits hardest. Untagged clips fall back to measured motion; missing data means the effect behaves as before.')
                        still_motion_input = gr.Checkbox(
                            value=True, label='Smart photo motion',
                            info='Photos move with the music instead of the generic drift: slow pans toward the subject on calm parts, push-ins on builds, decisive punch-ins on drops. Uses the detected face/subject as the camera target when one is found.')
                        with gr.Accordion('Customize effects (optional)', open=False):
                            effect_mode_input = gr.Radio(
                                choices=[('Curated', 'curated'), ('Custom', 'custom'), ('Surprise shuffle', 'shuffle')],
                                value='curated', label='Effect mode',
                                info='Curated = the classic style presets. Custom = pick your own palette. Shuffle = a seeded random palette (same seed, same video).')
                            # Reveal the palette only for Custom, the seed only
                            # for Shuffle. Both are created visible=True so they
                            # mount in the DOM immediately — a control created
                            # hidden inside a gr.Tab isn't mounted until forced,
                            # so its first show no-ops (the gradio tab quirk that
                            # made this need a double-toggle). app.load then hides
                            # whichever the default mode doesn't use; hiding an
                            # already-mounted control is reliable, and the panel
                            # is collapsed so there's no flash on load.
                            effect_palette_input = gr.CheckboxGroup(
                                choices=list_effect_choices(), value=[],
                                label='Effect palette',
                                info='Picked effects still land where the music calls for them (drops, builds, calm parts).')
                            effect_seed_input = gr.Number(
                                value=0, precision=0, label='Shuffle seed',
                                info='0 = derived from the song. Change it to re-roll the palette; renders stay reproducible.')

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
                            # Set the correct initial visibility for the default
                            # mode (Curated → both hidden) once they're mounted.
                            app.load(
                                _effect_mode_updates,
                                inputs=[effect_mode_input],
                                outputs=[effect_palette_input, effect_seed_input],
                            )

                    with gr.Tab('The Cut'):
                        variety_input = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05, label='Source variety',
                            info='0 = pure quality picks (some uploads may never appear). Anything above 0 gives each upload with usable footage at least one moment; higher spreads usage more evenly.')
                        semantic_variety_input = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05, label='Visual variety',
                            info='Avoid runs of visually similar shots. Breaks near-ties; never overrides a clearly better-fitting clip. Needs the DINOv2 model (scripts/fetch_dinov2.py) and onnxruntime.')
                        media_aware_input = gr.Checkbox(
                            value=True, label='Media-aware selection',
                            info='Smarter picks for mixed uploads: low-resolution clips are less likely to be blown up onto the canvas, GIFs prefer segments they can cover in one pass (seamless loops are exempt), and GIF motion is measured at its native speed.')
                        split_screen_input = gr.Checkbox(
                            value=True, label='Pair vertical clips (split screen)',
                            info='Renders some high-energy segments as two cross-orientation clips (side by side on a landscape canvas, stacked on portrait). Needs two or more such sources; duo boundaries are never crossfaded. H.264/HEVC modes only.')
                        crossfades_input = gr.Checkbox(
                            value=False, label='Crossfade calm cuts',
                            info='Dissolves ~1 in 3 eligible calm (soft/flow) boundaries instead of hard-cutting — retimed, paired, or very short segments are skipped. Re-encodes those boundary chunks; hard cuts stay the fast default. H.264/HEVC modes only.')
                        speed_ramps_input = gr.Checkbox(
                            value=False, label='Speed ramps (experimental)',
                            info='Beat-aware retiming: slow-mo drifts on soft parts, rushes through builds, decel ramps and freeze hits on drops. Frame counts stay exact; H.264/HEVC modes only.')

                    with gr.Tab('Framing'):
                        output_format_input = gr.Dropdown(
                            choices=[(label, key) for key, (label, _) in OUTPUT_FORMATS.items()],
                            value=DEFAULT_OUTPUT_FORMAT, label='Output canvas',
                            info='The frame every render targets. Fixed canvases keep one odd portrait clip from flipping the whole video; "Match best source" is the old behavior (highest-resolution source decides).')
                        fit_mode_input = gr.Radio(
                            choices=[('Auto (smart)', 'crop'), ('Blurred background', 'blur'), ('Letterbox', 'pad')],
                            value='crop', label='Frame fit',
                            info='How sources with a different aspect ratio fill the frame. Auto picks per clip: a subject-tracked crop for small mismatches (trims at most ~15%), a graded blur fill for bigger ones, and a slow scanning pan for extreme ones (e.g. vertical phone clips). Blurred background and Letterbox force that single look on every clip.')

                    with gr.Tab('Text'):
                        text_entries_input = gr.Textbox(
                            label='Text entries (one per line)', lines=4, value='',
                            placeholder='Leave empty for no text.\nEach line appears once, spread evenly across the video.\nPin an entry to a time with @: "@15 Finish strong" or "@1:23 Halfway"\nMotion style adds tags: "[style:glitch] Drop!" or "Wait [widget:progress duration:4s]"',
                            info='Quotes, captions, titles — any text. Every line gets its own beat-snapped time window; @ pins one to a timestamp.')
                        text_tag_lint = gr.Markdown('', visible=False)
                        with gr.Row():
                            tpl_caption_btn = gr.Button('+ Caption', size='sm')
                            tpl_pinned_btn = gr.Button('+ Pinned @time', size='sm')
                            tpl_loading_btn = gr.Button('+ Loading bar', size='sm')
                            tpl_glitch_btn = gr.Button('+ Glitch', size='sm')
                        with gr.Row():
                            text_position_input = gr.Radio(
                                choices=[('Lower third', 'bottom'), ('Center', 'center'), ('Top', 'top')],
                                value='bottom', label='Text position')
                            text_scale_input = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label='Text size')
                        with gr.Row():
                            text_style_input = gr.Radio(
                                choices=[('Classic', 'classic'), ('Motion', 'motion')],
                                value='classic', label='Text style',
                                info='Classic: static white captions (the original look). Motion: beat-reactive pulse + halo, with [style:glitch] and [widget:progress duration:4s] tags per line.')
                            text_accent_input = gr.ColorPicker(
                                value='#FF4D8D', label='Accent color',
                                info='Motion style only: glitch tint and progress-bar fill.')
                        with gr.Row():
                            text_font_input = gr.Dropdown(
                                choices=_discover_font_choices(),
                                value='', label='Font',
                                info='Fonts found on this Mac (~/Library/Fonts, /Library/Fonts, system Supplemental). Auto = Arial Bold or $BEATSYNC_FONT. Applies to Classic and Motion.')
                            text_font_path_input = gr.Textbox(
                                value='', label='Custom font file (overrides the dropdown)',
                                placeholder='/path/to/YourFont-Bold.ttf — any .ttf/.otf, no install needed')

                        # Quick-add template buttons: append a ready-made line to
                        # the entries box. Templates that use Motion-only tags also
                        # flip the style radio — otherwise a Classic render would
                        # burn the literal "[widget:...]" text into the video.
                        def _make_template_appender(template: str, needs_motion: bool):
                            def _append(current: str, style: str):
                                current = (current or '').rstrip()
                                text = (current + '\n' if current else '') + template
                                return text, ('motion' if needs_motion else style)
                            return _append

                        for _btn, _tpl, _motion in (
                            (tpl_caption_btn, 'Your caption here', False),
                            (tpl_pinned_btn, '@15 Your text at 15s', False),
                            (tpl_loading_btn, 'Loading… [widget:progress duration:4s]', True),
                            (tpl_glitch_btn, '[style:glitch] YOUR DROP LINE', True),
                        ):
                            _btn.click(
                                _make_template_appender(_tpl, _motion),
                                inputs=[text_entries_input, text_style_input],
                                outputs=[text_entries_input, text_style_input])

                        # U4: live Classic-tag lint. .change also fires on
                        # programmatic updates, so the template buttons' style
                        # flip refreshes the warning with no extra wiring.
                        def _refresh_tag_lint(text_value, style_value):
                            msg = _classic_tag_lint_message(text_value, style_value)
                            return gr.update(value=msg or '', visible=bool(msg))

                        for _lint_src in (text_entries_input, text_style_input):
                            _lint_src.change(
                                _refresh_tag_lint,
                                inputs=[text_entries_input, text_style_input],
                                outputs=[text_tag_lint])

                        # U1: read-only schedule preview — shows per line WHERE
                        # it lands (or that the planner would skip it) before a
                        # multi-minute render. Display-only sugar: none of these
                        # components joins settings_components/SETTINGS_KEYS.
                        preview_schedule_btn = gr.Button('📋 Preview schedule', size='sm')
                        text_preview_note = gr.Markdown('', visible=False)
                        text_preview_table = gr.Dataframe(
                            headers=list(_TEXT_PREVIEW_HEADERS), value=[],
                            interactive=False, visible=False, wrap=True,
                            column_widths=['6%', '42%', '28%', '24%'],
                            label='Planned text schedule')

                        def _on_text_preview(text_value, style_value, audio_files, session_st):
                            try:
                                note, rows = _build_text_preview(
                                    text_value, style_value, audio_files, session_st)
                            except Exception as e:  # total: never raise into Gradio
                                note = '⚠️ Preview unavailable — the render itself is unaffected.'
                                rows = [['', '(preview error)',
                                         f'{type(e).__name__}: {e}'[:120],
                                         'try again after loading audio']]
                            return (gr.update(value=note, visible=True),
                                    gr.update(value=rows, visible=bool(rows)))

                        preview_schedule_btn.click(
                            _on_text_preview,
                            inputs=[text_entries_input, text_style_input,
                                    audio_state, session_state],
                            outputs=[text_preview_note, text_preview_table])

                    with gr.Tab('Export'):
                        if NVENC_AVAILABLE:
                            processing_mode = gr.Radio(choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC (H.265)', 'hevc_nvenc'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_nvenc', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_nvenc())
                        elif VIDEOTOOLBOX_AVAILABLE:
                            processing_mode = gr.Radio(choices=[('Apple VideoToolbox H.264', 'h264_videotoolbox'), ('Apple VideoToolbox HEVC (H.265)', 'hevc_videotoolbox'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_videotoolbox', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_videotoolbox())
                        else:
                            processing_mode = gr.Radio(choices=[('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='cpu', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_cpu())
                        gr.Markdown('*ProRes Precise Mode keeps footage pristine for external editing: effects, text overlays, looks and speed ramps are **not** applied there.*')
                        custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)
                        output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

            # Zone 3 — commit and watch. The recipe line + conflict guard put the
            # current settings' real effect right next to the render button.
            with gr.Column(scale=2):
                gr.Markdown('### 3 · Render & result')

                # --- ProRes Precise Mode conflict guard ----------------------
                # ProRes voids every creative control it lists (orchestrator's
                # to_settings_dict look gate + the lossless_mode branches in
                # video_processor). Greying them out is COSMETIC ONLY: disabled
                # inputs still submit their values and the pipeline still
                # nullifies them, so the resolved settings dict is byte-identical.
                # This just makes the silent dependency visible. fit_mode and
                # variety stay live — both are honored on ProRes proxies. All the
                # referenced controls live in the Zone 2 tabs, built above.
                _prores_disabled_controls = [
                    effect_style_input, look_input,
                    effect_mode_input, effect_palette_input, effect_seed_input,
                    effect_intensity_input, semantic_variety_input,
                    semantic_fx_input, still_motion_input,
                    speed_ramps_input, split_screen_input, crossfades_input,
                    text_entries_input, text_position_input, text_scale_input,
                    text_style_input, text_accent_input,
                    text_font_input, text_font_path_input,
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
            semantic_fx_input,
            speed_ramps_input, split_screen_input, crossfades_input,
            text_entries_input, text_position_input, text_scale_input,
            text_style_input, text_accent_input,
            text_font_input, text_font_path_input,
            media_aware_input,
            still_motion_input,
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
