#!/usr/bin/env python3
"""Styled (motion-graphics) text rendering — the opt-in upgrade to text_overlay.

text_overlay.py stays the timing authority: entries are parsed and planned
there (beat-snapped windows, drop avoidance, per-segment projection with
fades). This module replaces only the RENDERING when the Text tab's style is
not 'classic': instead of one static Pillow PNG per entry, each text-bearing
segment gets a per-frame transparent PNG sequence rasterized from SVG via
cairosvg, composited by the same overlay+fade graph in ffmpeg_processing.
Frames rasterize band-cropped (viewBox crop to the one horizontal band the
text can touch; ffmpeg overlays it at the returned band_y) — full-frame
raster is the automatic fallback whenever containment can't be guaranteed.

Per-frame features: an audio-onset envelope (librosa, computed once on the
render audio) drives a subtle text pulse and a stacked-stroke halo glow;
inline tags add per-line looks — '[style:glitch]' (envelope-gated bursts of
jittered RGB-split copies, plain caption between bursts) and
'[widget:progress duration:4s]' (a progress bar under the text, timed from
the entry's window start; the fill starts once the fade-in completes).

Determinism: no rng anywhere — the envelope is a pure function of the audio,
glitch jitter is a bit-mixed hash of the global frame index held for 2-3
frames, and mapped outputs are quantized (pulse 2dp on an eased envelope,
halo 0.5px, bar fill 2px) so identical runs produce byte-identical PNGs (and
the quantization lets the sha1 dedup + hardlink fan-out collapse repeated
frames). Frames the fade filter provably holds at alpha 0 are emitted as the
blank SVG so they all dedup to one file. Unique frames rasterize on a
process pool (spawn-safe); any pool failure degrades to the serial path.
cairosvg is optional: callers catch any exception from
build_styled_sequences and fall back to the classic path with one log line
(kill switch: BEATSYNC_DISABLE_STYLEDTEXT=1).
"""

import ctypes
import hashlib
import math
import os
import re
import shutil
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
from PIL import ImageFont

from text_overlay import FADE_SECONDS, _wrap_text, find_font

_PULSE_K = 0.06          # text scales up to +6% on a full-strength onset
_HALO_BASE, _HALO_K = 2.0, 6.0   # stroked under-copy spread, px (cairosvg has no blur filters)
_GLITCH_GATE = 0.55      # envelope level above which glitch bursts fire
_MAX_WIDGET_SECONDS = 60.0
_TAG_OPEN, _TAG_CLOSE = '[', ']'
_KCT_SCOPE_PROCESS = 1
_POOL_THRESHOLD = 24     # below this many pending frames, worker spawn+import costs more than it saves
_MAX_BAND_FRAC = 0.70    # a band taller than this fraction of the frame buys nothing — use full frame


class StyledTextError(RuntimeError):
    """Raised when the styled path cannot run; callers fall back to classic."""


def _sanitize_color(value: str, default: str = '#FF4D8D') -> str:
    """Coerce a UI color value to safe hex. Gradio's ColorPicker can emit
    'rgba(r, g, b, a)' strings, and cairosvg's color parser hard-crashes on
    anything it can't read — so only clean hex ever reaches the SVG."""
    value = (value or '').strip()
    if re.fullmatch(r'#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})', value):
        return value
    m = re.fullmatch(
        r'rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
        r'\s*(?:,\s*[\d.]+\s*)?\)', value)
    if m:
        r, g, b = (min(255, int(float(m.group(i)))) for i in (1, 2, 3))
        return f'#{r:02X}{g:02X}{b:02X}'
    return default


# ---------------------------------------------------------------------------
# Entry tags

# Only the kinds we define are tag syntax; any other bracketed text
# ('[note: call mom]') is user content and must survive into the caption.
_TAG_RE = re.compile(r'\[\s*(style|widget)\s*:\s*([^\]]*?)\s*\]', re.IGNORECASE)
_DURATION_RE = re.compile(r'duration\s*:\s*(\d+(?:\.\d+)?)\s*s?\b', re.IGNORECASE)
_warned_multi_widget: set = set()


def parse_entry_tags(text: str) -> Tuple[str, str, Optional[float]]:
    """Split '[style:glitch]'/'[widget:progress duration:4s]' tags from text.

    Returns (clean_text, style, widget_duration_seconds_or_None); -1.0 is
    the sentinel for a widget that spans the whole planned window. Only the
    known kinds 'style' and 'widget' are consumed (whitespace around the
    colon and inside the brackets is tolerated); unknown bracketed text
    stays in the caption. Durations take an optional 's' suffix and clamp
    to 60s; extra widget tags on a line are dropped with one warning.
    """
    style = 'default'
    widget_duration: Optional[float] = None
    extra_widgets = False

    def _consume(m: 're.Match[str]') -> str:
        nonlocal style, widget_duration, extra_widgets
        kind = m.group(1).lower()
        body = m.group(2).strip()
        if kind == 'style' and body:
            style = body.split()[0].lower()
        elif kind == 'widget' and [w.lower() for w in body.split()[:1]] == ['progress']:
            if widget_duration is not None:
                extra_widgets = True
                return ''
            dm = _DURATION_RE.search(body)
            widget_duration = (min(float(dm.group(1)), _MAX_WIDGET_SECONDS)
                               if dm else -1.0)  # -1.0: span the whole window
        return ''

    clean = _TAG_RE.sub(_consume, text)
    if extra_widgets and text not in _warned_multi_widget:
        _warned_multi_widget.add(text)
        print("   ⚠️  Text tags: multiple [widget:...] on one line — keeping the first")
    return ' '.join(clean.split()), style, widget_duration


# ---------------------------------------------------------------------------
# Fonts — cairosvg resolves by FAMILY NAME through the platform font system
# (Quartz/CoreText on macOS, fontconfig on Linux), never by file path.


def _register_font_coretext(font_file: str) -> None:
    """darwin: register a font file process-scoped so CoreText can resolve a
    non-installed family (no-op if already installed)."""
    cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    ct = ctypes.CDLL('/System/Library/Frameworks/CoreText.framework/CoreText')
    cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
    ct.CTFontManagerRegisterFontsForURL.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    raw = os.path.abspath(font_file).encode('utf-8')
    url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)
    if url:
        ct.CTFontManagerRegisterFontsForURL(url, _KCT_SCOPE_PROCESS, None)
        cf.CFRelease(url)


def _resolve_font(scale: float, height: int,
                  preferred: Optional[str] = None) -> Tuple[str, str, 'ImageFont.FreeTypeFont', int]:
    """Pick the render font: (family, weight, pil_font_for_measuring, size_px).

    Same file lookup and sizing as text_overlay.render_text_png, so classic
    and styled agree on which font a machine uses. Pillow reads the family
    name out of the file; the file is registered with CoreText on macOS so
    the SVG's family reference resolves even for non-installed fonts.
    """
    font_path = find_font(preferred)
    if not font_path:
        raise StyledTextError('no usable font found (set BEATSYNC_FONT)')
    size_px = max(14, int(height / 14 * max(0.2, float(scale))))
    pil_font = ImageFont.truetype(font_path, size_px)
    family, subfamily = pil_font.getname()
    weight = 'bold' if 'bold' in (subfamily or '').lower() else 'normal'
    if sys.platform == 'darwin':
        _register_font_coretext(font_path)
    return family, weight, pil_font, size_px


# ---------------------------------------------------------------------------
# Audio-onset envelope (per video frame, [0, 1])

_env_cache: Dict[Tuple[str, float, int], np.ndarray] = {}


def _onset_envelope(audio_file: str, fps: float, n_frames: int,
                    attack: float = 0.55, decay: float = 0.90) -> np.ndarray:
    """Onset strength resampled onto the video frame grid, with a fast-attack
    slow-decay smoother so transients pulse instead of flickering. Pure
    function of the audio (98th-percentile normalization, silence-guarded);
    cached per (path, fps, n_frames) for the duo of planning passes."""
    key = (audio_file, float(fps), int(n_frames))
    cached = _env_cache.get(key)
    if cached is not None:
        return cached
    import librosa
    y, sr = librosa.load(audio_file, sr=22050, mono=True)
    hop = 512
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_times = librosa.times_like(strength, sr=sr, hop_length=hop)
    frame_times = np.arange(n_frames, dtype=np.float64) / fps
    env = np.interp(frame_times, env_times, strength)
    peak = float(np.percentile(env, 98))
    env = np.clip(env / peak, 0.0, 1.0) if peak > 1e-9 else np.zeros_like(env)
    out = np.empty_like(env)
    level = 0.0
    for i, x in enumerate(env):
        level = level + attack * (x - level) if x > level else level * decay
        out[i] = level
    _env_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Band geometry — rasterize only the horizontal band the text can touch.
# Full-frame cairosvg spends most of its time on empty pixels (the audit
# measured ~102ms/frame at 1920x1080 vs ~37ms at 1920x400), so each render
# computes ONE band covering every entry's possible extent and the SVGs are
# cropped to it via viewBox; ffmpeg composites the band at overlay y=band_top.


def _compute_band(metas: Sequence[Tuple[List[str], str, Optional[float]]],
                  pil_font: 'ImageFont.FreeTypeFont', size_px: int,
                  width: int, height: int, position: str) -> Tuple[int, int]:
    """(band_top, band_h): the even-aligned vertical band that provably
    contains everything _svg_for_frame can draw for ANY of the entries.

    Mirrors the layout math there exactly (y0 per position, baselines at
    y0+(li+0.83)*line_h) and adds conservative margins for every mechanism
    that can push ink past the glyph box: font ascent/descent (PIL metrics),
    the halo shadow dy (+3) plus half the widest glow stroke, glitch copy dy
    (±2), the max pulse scale (about the block center; stroke widths scale
    with it too), the progress bar below the block, and a pad for glyph
    overshoot beyond the font's nominal metrics + antialiasing spill.
    Pure function of its inputs (determinism). Returns (0, height) — the
    full frame — whenever the band can't be guaranteed to contain the
    content or wouldn't save enough (taller than _MAX_BAND_FRAC).
    """
    ascent, descent = pil_font.getmetrics()
    line_h = round(size_px * 1.25)
    pulse_max = 1.0 + _PULSE_K
    # Widest halo layer stroke is halo*3.5 with halo <= _HALO_BASE+_HALO_K;
    # half of it extends past the glyph outline. Applied per-side together
    # with the worst dy in each direction (over-covers: the stroked layers
    # only shift down, the glitch copies up — summing both is safe).
    stroke_half = (_HALO_BASE + _HALO_K) * 3.5 / 2.0
    dy_up, dy_down = 2.0, 3.0            # glitch copy up / halo shadow down
    pad = max(4.0, size_px * 0.15)       # metric overshoot + AA spill
    top: Optional[float] = None
    bottom: Optional[float] = None
    for wrapped, _style, widget_duration in metas:
        n = len(wrapped)
        if n == 0 and widget_duration is None:
            continue  # same skip as the sequence loop — draws nothing
        block_h = line_h * n
        if position == 'top':
            y0 = height * 0.08
        elif position == 'center':
            y0 = (height - block_h) / 2
        else:  # bottom (lower third)
            y0 = height * 0.88 - block_h
        extents: List[float] = []
        if n:
            cy = y0 + block_h / 2   # pulse scales about the block center
            raw_top = y0 + 0.83 * line_h - ascent - dy_up - stroke_half
            raw_bot = (y0 + (n - 1 + 0.83) * line_h + descent
                       + dy_down + stroke_half)
            extents.append(cy + (raw_top - cy) * pulse_max)
            extents.append(cy + (raw_bot - cy) * pulse_max)
        if widget_duration is not None:  # bar sits OUTSIDE the pulse <g>
            by = y0 + block_h + line_h * 0.4
            extents.extend([by, by + max(4, round(height / 90))])
        top = min(extents) if top is None else min(top, min(extents))
        bottom = max(extents) if bottom is None else max(bottom, max(extents))
    if top is None:
        return 0, height
    # Even pixel bounds: the overlay y must stay chroma-aligned on yuv420.
    band_top = max(0, int(math.floor((top - pad) / 2.0)) * 2)
    band_bot = min(height, int(math.ceil((bottom + pad) / 2.0)) * 2)
    band_h = band_bot - band_top
    if band_h <= 0 or band_h % 2 or band_h > height * _MAX_BAND_FRAC:
        return 0, height
    return band_top, band_h


# ---------------------------------------------------------------------------
# SVG generation


def _glitch_offset(global_frame: int) -> int:
    """Deterministic, rng-free jitter in [-2, 2], held for 2-3 frames.

    Each 5-frame stretch splits into a 2-frame and a 3-frame hold block; the
    block index is bit-mixed BEFORE the mod. (The old per-frame form
    `(g * 2654435761) % 5 - 2` was a no-op multiply — 2654435761 ≡ 1 mod 5 —
    yielding a smooth sawtooth sweep instead of a glitch.)"""
    block = (global_frame // 5) * 2 + (1 if global_frame % 5 >= 2 else 0)
    return ((block * 2654435761) >> 13) % 5 - 2


def _svg_for_frame(lines: List[str], style: str, family: str, weight: str,
                   size_px: int, width: int, height: int, position: str,
                   accent: str, env_value: float, global_frame: int,
                   progress: Optional[float],
                   band: Tuple[int, int]) -> str:
    """One frame's SVG. Everything env-driven is quantized so repeated
    visual states produce identical strings (dedup relies on this).

    All coordinates stay in FULL-FRAME terms; `band` = (band_top, band_h)
    from _compute_band crops the canvas via viewBox, so the raster is only
    band_h tall and ffmpeg overlays it at y=band_top. (0, height) = the
    uncropped full frame — the viewBox is then the identity."""
    e = min(max(float(env_value), 0.0), 1.0)
    # e**1.5 eases the response so mid-level onsets don't keep the text
    # constantly breathing; 2dp / 0.5px quantization collapses near-identical
    # frames into one rasterization without visible stepping.
    pulse = round(1.0 + _PULSE_K * e ** 1.5, 2)
    halo = round((_HALO_BASE + _HALO_K * e) * 2) / 2
    line_h = round(size_px * 1.25)
    block_h = line_h * len(lines)
    if position == 'top':
        y0 = height * 0.08
    elif position == 'center':
        y0 = (height - block_h) / 2
    else:  # bottom (lower third)
        y0 = height * 0.88 - block_h
    cx, cy = width / 2, y0 + block_h / 2

    def text_lines(cls: str, dx: int = 0, dy: int = 0, extra: str = '') -> str:
        parts = []
        for li, line in enumerate(lines):
            y = y0 + (li + 0.83) * line_h
            parts.append(
                f'<text x="{cx + dx:.1f}" y="{y + dy:.1f}" text-anchor="middle" '
                f'class="{cls}" font-size="{size_px}px"{extra}>{escape(line)}</text>')
        return ''.join(parts)

    # Halo = three stacked strokes at decreasing opacity — a cheap glow
    # (cairosvg has no blur); a pure function of `halo`, so it dedups
    # exactly like the old single stroke did.
    halo_layers = (
        text_lines('halo-glow', dy=3,
                   extra=f' stroke-width="{halo * 3.5:.2f}" stroke-opacity="0.08"')
        + text_lines('halo-glow', dy=3,
                     extra=f' stroke-width="{halo * 2:.1f}" stroke-opacity="0.18"')
        + text_lines('halo', dy=3, extra=f' stroke-width="{halo:.1f}"'))

    # Glitch bursts only on strong onsets (plain caption between bursts);
    # the offset holds 2-3 frames so it reads as displacement, not shimmer.
    glitch = ''
    if style == 'glitch' and e > _GLITCH_GATE:
        j = _glitch_offset(global_frame)
        glitch = (text_lines('glitch-a', dx=-3 + j, dy=-2)
                  + text_lines('glitch-b', dx=3 - j, dy=2))

    widget = ''
    if progress is not None:
        p = min(max(progress, 0.0), 1.0)
        track_w = round(width * 0.30)
        bar_h = max(4, round(height / 90))
        bx = (width - track_w) / 2
        by = y0 + block_h + line_h * 0.4
        fill_w = min(track_w, int(round(track_w * p / 2.0)) * 2)  # 2px steps: dedup-friendly
        widget = (
            f'<rect class="track" x="{bx:.1f}" y="{by:.1f}" rx="{bar_h/2:.1f}" '
            f'width="{track_w}" height="{bar_h}"/>'
            f'<rect class="fill" x="{bx:.1f}" y="{by:.1f}" rx="{bar_h/2:.1f}" '
            f'width="{fill_w}" height="{bar_h}"/>')

    band_top, band_h = band
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{band_h}"'
        f' viewBox="0 {band_top} {width} {band_h}">'
        f'<style>'
        f'.cap {{ font-family: "{escape(family)}"; font-weight: {weight}; fill: #ffffff; }}'
        f'.halo {{ font-family: "{escape(family)}"; font-weight: {weight};'
        f' fill: #000000; fill-opacity: 0.55; stroke: #000000; stroke-opacity: 0.35; }}'
        f'.halo-glow {{ font-family: "{escape(family)}"; font-weight: {weight};'
        f' fill: none; stroke: #000000; }}'
        f'.glitch-a {{ font-family: "{escape(family)}"; font-weight: {weight};'
        f' fill: {accent}; fill-opacity: 0.65; }}'
        f'.glitch-b {{ font-family: "{escape(family)}"; font-weight: {weight};'
        f' fill: #ffffff; fill-opacity: 0.45; }}'
        f'.track {{ fill: #ffffff; fill-opacity: 0.25; }}'
        f'.fill {{ fill: {accent}; }}'
        f'</style>'
        f'<g transform="translate({cx:.1f} {cy:.1f}) scale({pulse}) '
        f'translate({-cx:.1f} {-cy:.1f})">'
        + halo_layers + glitch + text_lines('cap')
        + '</g>' + widget + '</svg>')


def _blank_svg(width: int, band_h: int) -> str:
    """Fully transparent frame at the BAND size, so every PNG in a sequence
    has identical dimensions (ffmpeg's image2 demuxer assumes that)."""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{band_h}"/>'


# ---------------------------------------------------------------------------
# Rasterization — process pool over unique SVGs


def _rasterize_svg(svg: str, path: str) -> None:
    """Rasterize one SVG to a content-addressed PNG (atomic write, so a
    killed worker can't leave a truncated file that later runs would trust).
    cairosvg is imported HERE so spawn-started pool workers initialize it
    fresh in their own interpreter."""
    if os.path.exists(path):
        return
    import cairosvg
    data = cairosvg.svg2png(bytestring=svg.encode('utf-8'), background_color=None)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)


def _rasterize_job(job: Tuple[str, str]) -> None:
    """Picklable pool entry point: job = (svg_string, out_path)."""
    _rasterize_svg(*job)


@contextmanager
def _spawn_safe_main():
    """multiprocessing 'spawn' (the macOS/Windows default) re-executes the
    parent's __main__ script in every worker unless it can import main by
    module name. Our launchers (the Gradio script, headless smoke harnesses)
    do real render work at import time, so a re-run would recurse the whole
    pipeline. Temporarily pointing __main__.__spec__ at this module makes
    spawn children initialize by importing 'styled_text' instead — which is
    import-safe by design (no work at module import time)."""
    main = sys.modules.get('__main__')
    if main is None or getattr(getattr(main, '__spec__', None), 'name', None):
        yield  # already spawn-safe (python -m, embedded, or worker context)
        return
    import importlib.util
    old_spec = getattr(main, '__spec__', None)
    main.__spec__ = importlib.util.spec_from_loader('styled_text', loader=None)
    try:
        yield
    finally:
        main.__spec__ = old_spec


def _rasterize_all(jobs: List[Tuple[str, str]]) -> None:
    """Rasterize the unique SVGs, in parallel when there are enough pending
    to pay for worker startup (each worker imports cairosvg once). Every job
    writes its own content-addressed path, so scheduling order cannot affect
    output bytes; any pool failure falls back to the serial path."""
    pending = [(svg, path) for svg, path in jobs if not os.path.exists(path)]
    if not pending:
        return
    max_workers = min(8, os.cpu_count() or 1)
    if len(pending) >= _POOL_THRESHOLD and max_workers >= 2:
        import concurrent.futures
        import multiprocessing
        try:
            # Explicit spawn on every platform: the caller may hold live
            # threads (Gradio server) and CoreText state, both unsafe to fork.
            with _spawn_safe_main(), concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                list(pool.map(_rasterize_job, pending, chunksize=8))
            return
        except Exception as exc:
            print(f"   ⚠️  Styled text raster pool failed ({exc}) — rasterizing serially")
    for svg, path in pending:
        _rasterize_svg(svg, path)


# ---------------------------------------------------------------------------
# Sequence build (public entry point)


def build_styled_sequences(entries: Sequence[Tuple[str, Optional[float]]],
                           schedule: List[Tuple[str, float, float]],
                           seg_map: Dict[int, Tuple[str, float, float, float]],
                           cut_times: Sequence[float],
                           segment_frames: Sequence[int],
                           fps: float, target_size: Tuple[int, int],
                           audio_file: str, temp_dir: str,
                           position: str, scale: float,
                           accent: str,
                           font_path: Optional[str] = None) -> Tuple[Dict[int, str], int]:
    """Render one per-frame PNG sequence per text-bearing segment.

    Returns ({segment_index: '%06d'-pattern path}, band_y): the PNGs are
    cropped to the one horizontal band all entries can touch (see
    _compute_band) and must be composited at overlay y=band_y — 0 whenever
    the band math fell back to the full frame. Each sequence covers the
    segment's full frame count (the existing fade filters gate visibility,
    exactly as they did for the looped static PNG). Raises StyledTextError /
    ImportError on any missing optional dependency — the caller logs one
    line and falls back to classic rendering.
    """
    if os.environ.get('BEATSYNC_DISABLE_STYLEDTEXT'):
        raise StyledTextError('disabled via BEATSYNC_DISABLE_STYLEDTEXT')
    import cairosvg  # noqa: F401 — optional dependency probe; ImportError falls back to classic

    width, height = int(target_size[0]), int(target_size[1])
    accent = _sanitize_color(accent)
    family, weight, pil_font, size_px = _resolve_font(scale, height, font_path)
    cut_times = [float(t) for t in cut_times]
    total_frames = int(round(cut_times[-1] * fps)) + 2
    env = _onset_envelope(audio_file, fps, total_frames)

    # Per-entry render meta, keyed the same way seg_map keys text (raw string).
    meta: Dict[str, Tuple[List[str], str, Optional[float]]] = {}
    for raw_text, _pin in entries:
        clean, style, widget_duration = parse_entry_tags(raw_text)
        wrapped = _wrap_text(clean, pil_font, max_width=width * 0.86) if clean else []
        meta[raw_text] = (wrapped, style, widget_duration)

    # One band per render (not per entry): every sequence shares it, so all
    # PNGs have one size and one overlay y. Any surprise in the band math
    # degrades to the full frame — never to the classic fallback.
    try:
        band = _compute_band(list(meta.values()), pil_font, size_px,
                             width, height, position)
    except Exception as exc:
        print(f"   ⚠️  Styled text band math failed ({exc}) — full-frame raster")
        band = (0, height)
    band_top, band_h = band

    def _window_for(raw_text: str, seg_start: float, seg_end: float) -> Tuple[float, float]:
        """The schedule window this segment actually sits in, for widget
        progress timing. A repeated entry text is placed many times, so the
        window must be matched by TIME OVERLAP with the segment (same >0.01s
        rule as text_overlay._project_onto_segments; last-wins mirrors its
        projection) — a text-keyed lookup would always hit the final
        occurrence and blank the widget everywhere else."""
        found = (seg_start, seg_end)
        for s_text, s_ws, s_we in schedule:
            if s_text == raw_text and min(s_we, seg_end) - max(s_ws, seg_start) > 0.01:
                found = (s_ws, s_we)
        return found

    # Pass 1: compute every frame's SVG string, collecting unique SVGs with
    # content-addressed target paths (dedup happens here, before any raster).
    unique_dir = os.path.join(temp_dir, 'styled_text', '_unique')
    os.makedirs(unique_dir, exist_ok=True)
    unique: Dict[str, str] = {}          # svg -> unique png path
    seg_frame_svgs: Dict[int, List[str]] = {}

    for seg_idx, (raw_text, fi_start, fi_dur, _fo_start) in seg_map.items():
        wrapped, style, widget_duration = meta.get(raw_text, ([], 'default', None))
        if not wrapped and widget_duration is None:
            continue
        seg_start = cut_times[seg_idx]
        ws, we = _window_for(raw_text, seg_start, cut_times[seg_idx + 1])
        # The bar starts filling when the fade-in ENDS (the overlay is
        # invisible before that), and still completes by the window end.
        p_start = ws + FADE_SECONDS
        g0 = int(round(seg_start * fps))
        n_frames = int(segment_frames[seg_idx])
        frame_svgs: List[str] = []
        for f in range(n_frames):
            # Provably invisible frames dedup to the one blank PNG:
            # build_text_overlay_graph applies fade=t=in:st=fi_start
            # whenever fi_dur > 0.001, and ffmpeg's fade holds alpha at 0
            # for pts < st. The sequence's pts are exactly f/fps
            # (-framerate input); the 1e-3 margin covers st's 4dp rounding.
            if fi_dur > 0.001 and f / fps < fi_start - 1e-3:
                frame_svgs.append(_blank_svg(width, band_h))
                continue
            g = min(g0 + f, total_frames - 1)
            t = seg_start + f / fps
            progress = None
            if widget_duration is not None:
                if t > we:
                    progress = 1.0
                elif t >= ws:
                    wd = ((we - p_start) if widget_duration <= 0
                          else min(widget_duration, max(we - p_start, 0.0)))
                    progress = (min(max((t - p_start) / wd, 0.0), 1.0)
                                if wd > 1e-6 else 1.0)
            frame_svgs.append(
                _svg_for_frame(wrapped, style, family, weight, size_px,
                               width, height, position, accent,
                               float(env[g]), g, progress, band)
                if wrapped or progress is not None
                else _blank_svg(width, band_h))
        seg_frame_svgs[seg_idx] = frame_svgs
        for svg in frame_svgs:
            if svg not in unique:
                unique[svg] = os.path.join(
                    unique_dir,
                    hashlib.sha1(svg.encode('utf-8')).hexdigest() + '.png')

    # Pass 2: rasterize the unique frames (process pool when it pays off).
    _rasterize_all(list(unique.items()))

    # Pass 3: hardlink the unique PNGs into per-segment '%06d' sequences.
    patterns: Dict[int, str] = {}
    for seg_idx, frame_svgs in seg_frame_svgs.items():
        seq_dir = os.path.join(temp_dir, 'styled_text', f'seq_{seg_idx:03d}')
        os.makedirs(seq_dir, exist_ok=True)
        for f, svg in enumerate(frame_svgs):
            src = unique[svg]
            dst = os.path.join(seq_dir, f'{f:06d}.png')
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        patterns[seg_idx] = os.path.join(seq_dir, '%06d.png')

    band_note = (f"{width}x{band_h} band @y={band_top}"
                 if band_h < height else "full frame")
    print(f"   🎨 Styled text: {len(patterns)} segment sequence(s), "
          f"{len(unique)} unique frames rasterized ({band_note})")
    return patterns, band_top
