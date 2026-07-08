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

# Still images accepted as sources. HEIC is deliberately absent: this
# machine's Homebrew ffmpeg build ships no HEIF demuxer/decoder.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Stills have no timeline of their own; the probe layer reports this synthetic
# duration so the planner can place a still anywhere, and extraction serves it
# with -loop 1 for exactly the frames each segment needs.
SYNTHETIC_IMAGE_DURATION = 3600.0

# ProRes proxies rendered from stills get a fixed runway comfortably longer
# than the longest possible segment hold.
IMAGE_PRORES_SECONDS = 60.0


def is_image_source(path: str) -> bool:
    """True for still-image sources (extension check; cheap and deterministic)."""
    return os.path.splitext(str(path))[1].lower() in IMAGE_EXTENSIONS


def count_video_frames(video_file: str) -> int | None:
    """Frame count of the first video stream; None if it can't be determined.

    Uses -count_packets (demux only, no decode — one packet per frame for
    video streams), so this is cheap enough to run on every segment. It is
    the runtime enforcement of the zero-drift contract: -vframes caps output
    but never pads it, so a short source window would otherwise produce a
    short segment that silently shifts every later cut against the audio.
    """
    try:
        probe_cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_file,
        ]
        result = _run_media_command(probe_cmd, timeout=30)
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except Exception:
        return None


def _verify_segment_frames(output_file: str, expected: int) -> bool:
    """Post-extract guard: the segment must hold exactly the planned frames."""
    actual = count_video_frames(output_file)
    if actual is None:
        print(f"   ⚠️  Could not verify frame count of {os.path.basename(output_file)}; keeping it")
        return True
    if actual != expected:
        print(
            f"   ❌ Frame-count mismatch in {os.path.basename(output_file)}: "
            f"expected {expected}, got {actual} — rejecting segment (timeline would drift)"
        )
        return False
    return True

# Smart crop caps how much of a source the fill-and-center-crop may discard.
# Mild mismatches keep the classic full-frame crop (best-looking, loses
# little); beyond the cap the source is scaled so at most this fraction of
# the cropped axis is lost and the rest of the frame is filled with the
# blurred background instead (a vertical 9:16 source in a 16:9 target would
# otherwise lose ~69% of its content).
MAX_CROP_PER_AXIS = 0.15

# Above this crop_loss (fraction of the source a fill-crop would discard) the
# limited-crop hybrid still wastes most of the frame on blur fill, so — when
# the segment duration is known — the fit engine switches to scan-fit: fill
# the target's short axis fully and sweep the crop window along the long axis
# over the segment (smoothstep-eased crop expression, not zoompan).
SCAN_CROP_LOSS = 0.40

# Motion-sickness guard for scan-fit: the sweep's PEAK speed may not exceed
# this fraction of the overflow per second (smoothstep peaks at 1.5x the
# average speed). Too-short segments shrink the travel instead.
SCAN_MAX_SPEED_FRAC = 0.40

# Split-screen duo panes deliberately crop past MAX_CROP_PER_AXIS (an 8:9
# pane of a 9:16 source loses ~37% of its height) — acceptable only because
# the window is subject-anchored/tracked (design doc Option A). Beyond this
# cap the pane falls back to the echo blur fit at pane size instead of a
# blind extreme crop (rare: a landscape source landing in a pane).
PANE_MAX_CROP = 0.40

# Anchors below this confidence are ignored (treated as "no anchor" →
# centered behavior identical to the pre-anchor engine).
ANCHOR_MIN_CONFIDENCE = 0.2

# --- Tracked pan (subject-following crop) -----------------------------------
# PAN_PATH_KEY is the trigger: stage6 rebases the analysis subject path onto
# the segment's local clock and stores it under this key as
# [[t_seg_seconds, cx, cy], ...] (t_seg = 0 at the segment's first frame,
# cx/cy normalized 0..1). Anchors carrying only the raw candidate-relative
# "path" (wave-5 shape, external callers) never animate — they keep the exact
# static offset crop, so an unrebased path can't pan on a wrong clock.
PAN_PATH_KEY = 'path_seg'
# Cap on emitted knots so the crop expression stays compact.
PAN_MAX_KNOTS = 6
# Centered moving-average window over path samples (in samples).
PAN_SMOOTH_WINDOW = 3
# Consecutive smoothed samples closer than this (normalized distance) merge.
PAN_DUP_EPS = 0.01
# Below this total travel (fraction of the frame on a pannable axis) the
# static crop is visually indistinguishable and cheaper → static fallback.
PAN_MIN_TRAVEL = 0.03
# Motion-sickness guard: the crop window may move at most this fraction of
# the axis crop headroom (scaled_dim − crop_dim) per second; faster hops get
# pulled toward the previous knot.
PAN_MAX_SPEED = 0.25
# Knots closer in time than this merge (also guards near-zero lerp slopes).
PAN_MIN_KNOT_DT = 0.05

# Echo blur fill: background overscan (fraction of cover) that gives the
# drifting crop window room to move, and how far it drifts over the segment
# (fraction of the target dimension).
ECHO_BG_OVERSCAN = 1.10
ECHO_DRIFT_TRAVEL = 0.03

# Slow foreground push-in for echo-fill composites (limited-crop hybrid and
# pure blur-mode foregrounds): total zoom travel over the segment as a
# fraction of 1.0 (0.035 = 3.5%). Uses zoompan (crop/scale can't animate
# w/h), d=1 so the frame count is untouched. 0 disables and reproduces the
# pre-zoom chains byte-for-byte.
HYBRID_FG_ZOOM = 0.035

# Beat-reactive echo margins: on each beat the background grade briefly
# lifts saturation/brightness (the _fx_sat_pulse idiom — windowed
# between(t,b,b+WINDOW) terms inside the eq expressions, eval=frame).
# Margin seasoning, not a strobe: the lift rides ON TOP of the echo grade
# (saturation 0.6 → 0.85, brightness -0.08 → -0.05) and only on the
# background branch, before the foreground overlay.
ECHO_PULSE_SAT = 0.25
ECHO_PULSE_BRIGHT = 0.03
ECHO_PULSE_WINDOW = 0.18
ECHO_PULSE_MAX_BEATS = 8


def _resolve_anchor(anchor) -> object:
    """(cx, cy) in 0..1 from an analysis anchor dict, or None when unusable.

    Anchor contract: {"cx": 0..1, "cy": 0..1, "confidence": 0..1, ...} in
    normalized source coordinates (scale-safe, so the same fractions apply
    before or after any scale). None / low confidence / malformed → None,
    which every consumer maps to the exact legacy centered behavior.
    """
    if not isinstance(anchor, dict):
        return None
    try:
        if float(anchor.get('confidence', 0.0)) < ANCHOR_MIN_CONFIDENCE:
            return None
        cx = min(1.0, max(0.0, float(anchor['cx'])))
        cy = min(1.0, max(0.0, float(anchor['cy'])))
    except (KeyError, TypeError, ValueError):
        return None
    return cx, cy


def _anchored_crop(w: int, h: int, anchor=None) -> str:
    """crop=w:h centered on the anchor (clamped to frame bounds), or the
    plain centered crop when there is no usable anchor.

    The x/y expressions position the crop window's CENTER at (cx·iw, cy·ih);
    clip() keeps the window inside the frame, so an anchor near an edge
    degrades gracefully to an edge-aligned crop. Anchor=None emits the exact
    legacy `crop=w:h` string so unchanged inputs stay byte-identical.
    """
    pt = _resolve_anchor(anchor)
    if pt is None:
        return f"crop={w}:{h}"
    cx, cy = pt
    return (
        f"crop={w}:{h}"
        f":x='clip(iw*{cx:.6f}-ow/2\\,0\\,iw-ow)'"
        f":y='clip(ih*{cy:.6f}-oh/2\\,0\\,ih-oh)'"
    )


def _pan_knots(anchor, duration: float, headroom_x: float, headroom_y: float):
    """Usable tracked-pan knots [(t, cx, cy), ...] or None (→ static crop).

    None whenever any gate fails: no rebased path under PAN_PATH_KEY,
    confidence below ANCHOR_MIN_CONFIDENCE, unknown duration, no crop
    headroom on either axis, fewer than 2 knots after clamp/smooth/merge, or
    total travel under PAN_MIN_TRAVEL. Pure deterministic math on the anchor
    dict — same inputs always yield the same knots.
    """
    if _resolve_anchor(anchor) is None:
        return None
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    hx = max(0.0, float(headroom_x))
    hy = max(0.0, float(headroom_y))
    if hx <= 1e-6 and hy <= 1e-6:
        return None
    raw = anchor.get(PAN_PATH_KEY)
    if not isinstance(raw, (list, tuple)):
        return None

    # Clamp samples to the segment window (rebase already dropped far-out
    # samples; slight overhang clamps to the edges).
    samples = []
    for item in raw:
        try:
            t = float(item[0])
            cx = float(item[1])
            cy = float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(t) and math.isfinite(cx) and math.isfinite(cy)):
            continue
        samples.append((min(duration, max(0.0, t)),
                        min(1.0, max(0.0, cx)),
                        min(1.0, max(0.0, cy))))
    if len(samples) < 2:
        return None
    samples.sort(key=lambda s: s[0])

    # Moving-average smoothing (window PAN_SMOOTH_WINDOW, centered, edges
    # shrink) knocks single-sample detector jitter out of the pan.
    half = PAN_SMOOTH_WINDOW // 2
    smoothed = []
    for i in range(len(samples)):
        lo, hi = max(0, i - half), min(len(samples), i + half + 1)
        n = hi - lo
        smoothed.append((samples[i][0],
                         sum(s[1] for s in samples[lo:hi]) / n,
                         sum(s[2] for s in samples[lo:hi]) / n))

    # Collapse near-duplicates (in time or position) into the earlier knot.
    knots = [smoothed[0]]
    for t, cx, cy in smoothed[1:]:
        pt, px, py = knots[-1]
        if t - pt < PAN_MIN_KNOT_DT or math.hypot(cx - px, cy - py) < PAN_DUP_EPS:
            continue
        knots.append((t, cx, cy))
    if len(knots) < 2:
        return None

    # Cap the knot count (always keeping the first and last).
    if len(knots) > PAN_MAX_KNOTS:
        idx = sorted({round(i * (len(knots) - 1) / (PAN_MAX_KNOTS - 1))
                      for i in range(PAN_MAX_KNOTS)})
        knots = [knots[i] for i in idx]

    # Speed cap: pull each knot toward its predecessor so the window never
    # moves faster than PAN_MAX_SPEED of the axis headroom per second.
    capped = [knots[0]]
    for t, cx, cy in knots[1:]:
        pt, px, py = capped[-1]
        dt = max(PAN_MIN_KNOT_DT, t - pt)
        max_dx = PAN_MAX_SPEED * hx * dt
        max_dy = PAN_MAX_SPEED * hy * dt
        cx = px + max(-max_dx, min(max_dx, cx - px))
        cy = py + max(-max_dy, min(max_dy, cy - py))
        capped.append((t, cx, cy))
    knots = capped

    travel_x = (max(k[1] for k in knots) - min(k[1] for k in knots)) if hx > 1e-6 else 0.0
    travel_y = (max(k[2] for k in knots) - min(k[2] for k in knots)) if hy > 1e-6 else 0.0
    if max(travel_x, travel_y) < PAN_MIN_TRAVEL:
        return None
    return knots


def _pan_axis_expr(dim_var: str, out_var: str, knots, axis: int,
                   static_frac: float, animate: bool) -> str:
    """clip()-wrapped crop position expression for one axis.

    animate=False emits the exact static expression _anchored_crop uses (the
    anchor's cx/cy), so a non-panning axis stays byte-identical to the static
    crop. The animated form is piecewise-linear via summed clip() ramps:
    c(t) = c0 + Σ Δc_i·clip((t−t_i)/(t_{i+1}−t_i), 0, 1) — monotonic knot
    times make each ramp contribute only inside its own span. Commas are
    escaped (\\,) for -vf parsing; the caller single-quotes the whole
    expression, same idiom as _anchored_crop and the scan-fit sweep.
    """
    if not animate:
        return f"clip({dim_var}*{static_frac:.6f}-{out_var}/2\\,0\\,{dim_var}-{out_var})"
    terms = [f"{knots[0][axis]:.6f}"]
    for a, b in zip(knots, knots[1:]):
        dc = b[axis] - a[axis]
        if abs(dc) < 1e-6:
            continue
        dt = max(PAN_MIN_KNOT_DT, b[0] - a[0])
        terms.append(f"{dc:+.6f}*clip((t-{a[0]:.4f})/{dt:.4f}\\,0\\,1)")
    pos = "".join(terms)
    return f"clip({dim_var}*({pos})-{out_var}/2\\,0\\,{dim_var}-{out_var})"


def _tracked_crop(w: int, h: int, anchor, duration: float,
                  headroom_x: float, headroom_y: float):
    """crop=w:h whose window follows the rebased subject path over the
    segment, or None when the static _anchored_crop should be used instead.

    headroom_x/y are the crop slack per axis as fractions of the SCALED
    frame ((scaled_dim − crop_dim) / scaled_dim); they gate which axes may
    animate and feed the PAN_MAX_SPEED cap. The emitted expressions keep the
    same clip(dim*frac − out/2, 0, dim − out) shape as the static crop, so
    edge anchors still degrade to edge-aligned windows.
    """
    knots = _pan_knots(anchor, duration, headroom_x, headroom_y)
    if knots is None:
        return None
    cx, cy = _resolve_anchor(anchor)  # non-None: _pan_knots gated on it
    animate_x = (headroom_x > 1e-6
                 and max(k[1] for k in knots) - min(k[1] for k in knots) >= 1e-6)
    animate_y = (headroom_y > 1e-6
                 and max(k[2] for k in knots) - min(k[2] for k in knots) >= 1e-6)
    if not (animate_x or animate_y):
        return None
    x_expr = _pan_axis_expr('iw', 'ow', knots, 1, cx, animate_x)
    y_expr = _pan_axis_expr('ih', 'oh', knots, 2, cy, animate_y)
    return f"crop={w}:{h}:x='{x_expr}':y='{y_expr}'"


def _fill_crop_headroom(display_size: Tuple[float, float],
                        target_size: Tuple[int, int]) -> Tuple[float, float]:
    """Crop slack of the plain fill-and-crop, per axis, as fractions of the
    scaled frame — how far the crop window can slide. (0, 0) on bad input."""
    sw, sh = display_size
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return 0.0, 0.0
    f_fill = max(tw / sw, th / sh)
    scaled_w = sw * f_fill
    scaled_h = sh * f_fill
    return (max(0.0, (scaled_w - tw) / scaled_w),
            max(0.0, (scaled_h - th) / scaled_h))


def _fit_filters_with_pan(video_file: str, target_size: Tuple[int, int],
                          fit_mode: str, anchor: dict,
                          duration: float) -> List[str]:
    """get_fit_filters, upgraded to a tracked pan when the anchor carries a
    usable rebased path (PAN_PATH_KEY) and the segment duration is known.

    Every fallback path returns exactly what get_fit_filters emits today, so
    anchors without a rebased path (or failing any _pan_knots gate) stay
    byte-identical to the static engine.
    """
    if (fit_mode == 'crop' and target_size and duration
            and isinstance(anchor, dict) and anchor.get(PAN_PATH_KEY)):
        info = get_cached_display_info(video_file)
        if info:
            hx, hy = _fill_crop_headroom((info[0], info[1]), target_size)
            w, h = target_size
            tracked = _tracked_crop(w, h, anchor, duration, hx, hy)
            if tracked:
                return [
                    f"scale={w}:{h}:force_original_aspect_ratio=increase",
                    tracked,
                    "setsar=1",
                ]
    return get_fit_filters(target_size, fit_mode, anchor=anchor)


def get_fit_filters(target_size: Tuple[int, int], fit_mode: str, *,
                    anchor: dict = None) -> List[str]:
    """Aspect-ratio handling for sources that don't match the target frame.

    crop    – scale to fill, center-crop overflow (no distortion, default;
              callers with a probed source size upgrade big mismatches to the
              limited-crop hybrid via plan_source_fit). An anchor moves WHERE
              the crop window sits, never how much is cropped.
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
        _anchored_crop(w, h, anchor),
        "setsar=1",
    ]


_SOURCE_DISPLAY_INFO_CACHE: dict = {}


def get_cached_display_info(video_file: str):
    """(display_w, display_h, sar) with a per-run cache; None if the probe fails.

    Display dimensions fold the sample aspect ratio in (storage width × SAR),
    which is what fit decisions must compare — scale's
    force_original_aspect_ratio only looks at storage dimensions.
    """
    cached = _SOURCE_DISPLAY_INFO_CACHE.get(video_file)
    if cached is not None:
        return cached
    try:
        probe_cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,sample_aspect_ratio',
            '-of', 'json',
            video_file,
        ]
        result = _run_media_command(probe_cmd, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(_short_ffmpeg_error(result.stderr, 300) or "ffprobe failed")
        stream = json.loads(result.stdout)['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        sar = 1.0
        raw_sar = str(stream.get('sample_aspect_ratio') or '')
        if ':' in raw_sar:
            num, den = raw_sar.split(':', 1)
            if float(num) > 0 and float(den) > 0:
                sar = float(num) / float(den)
        info = (width * sar, float(height), sar)
        _SOURCE_DISPLAY_INFO_CACHE[video_file] = info
        return info
    except Exception as e:
        print(f"   ⚠️  Could not probe display size ({os.path.basename(video_file)}): {e}")
        return None


def _crop_loss(display_size: Tuple[float, float],
               target_size: Tuple[int, int]) -> float:
    """Fraction of the source a fill-and-crop would discard (0 = perfect fit).

    Negative/zero dimensions return 0.0 so callers fall back to the plain
    chain without special-casing.
    """
    sw, sh = display_size
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return 0.0
    f_fit = min(tw / sw, th / sh)
    f_fill = max(tw / sw, th / sh)
    return 1.0 - f_fit / f_fill


def plan_smart_crop(display_size: Tuple[float, float],
                    target_size: Tuple[int, int]):
    """Foreground geometry for the limited-crop hybrid, or None to keep the
    plain fill-and-center-crop chain.

    Returns (fg_w, fg_h, crop_w, crop_h): the source is scaled to fg_w×fg_h
    (losing at most MAX_CROP_PER_AXIS of the overflowing axis to the crop —
    centered by default, anchor-offset when the caller has one) and
    composited over the blurred fill.
    """
    sw, sh = display_size
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return None
    f_fit = min(tw / sw, th / sh)
    f_fill = max(tw / sw, th / sh)
    crop_loss = 1.0 - f_fit / f_fill
    if crop_loss <= MAX_CROP_PER_AXIS:
        return None
    f = min(f_fill, f_fit / (1.0 - MAX_CROP_PER_AXIS))
    fg_w = max(2, int(round(sw * f / 2)) * 2)
    fg_h = max(2, int(round(sh * f / 2)) * 2)
    return fg_w, fg_h, min(fg_w, tw), min(fg_h, th)


def plan_scan_fit(display_size: Tuple[float, float],
                  target_size: Tuple[int, int], duration: float,
                  anchor: dict = None):
    """Filter list for scan-fit, or None when it doesn't apply.

    Scan-fit fills the target's short axis completely and sweeps the crop
    window along the overflowing axis over the segment with smoothstep easing
    (pos = travel·(3u²−2u³), u = t/duration) — a pure per-frame crop
    expression, cheaper and frame-exact where zoompan is not. With an anchor
    the sweep ENDS centered on the subject (starting from the far side);
    without one it runs top→bottom / left→right. SCAN_MAX_SPEED_FRAC caps the
    peak sweep speed; segments too short for the full travel shrink it and
    center the covered range on the anchor (or frame center).
    """
    sw, sh = display_size
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return None
    if not duration or duration <= 0:
        return None
    src_ar = sw / sh
    tgt_ar = tw / th
    if src_ar < tgt_ar:
        # Portrait-ish source in landscape-ish target: fill width, sweep
        # vertically along the overflowing height.
        scale_w = tw
        scale_h = max(th, int(round(sh * (tw / sw) / 2)) * 2)
        overflow = scale_h - th
        sweep_y = True
    elif src_ar > tgt_ar:
        # The converse: fill height, sweep horizontally.
        scale_h = th
        scale_w = max(tw, int(round(sw * (th / sh) / 2)) * 2)
        overflow = scale_w - tw
        sweep_y = False
    else:
        return None
    if overflow < 2:
        return None

    pt = _resolve_anchor(anchor)
    if pt is not None:
        frac = pt[1] if sweep_y else pt[0]
        scaled_dim = scale_h if sweep_y else scale_w
        out_dim = th if sweep_y else tw
        focus = min(float(overflow), max(0.0, frac * scaled_dim - out_dim / 2.0))
        end = focus
        start = float(overflow) if end <= overflow / 2.0 else 0.0
    else:
        focus = overflow / 2.0
        start, end = 0.0, float(overflow)

    # Peak smoothstep speed is 1.5·travel/duration; keep it under the cap.
    travel_max = SCAN_MAX_SPEED_FRAC * overflow * duration / 1.5
    travel = abs(end - start)
    if travel > travel_max:
        travel = travel_max
        lo = min(float(overflow) - travel, max(0.0, focus - travel / 2.0))
        if end >= start:
            start, end = lo, lo + travel
        else:
            start, end = lo + travel, lo
    delta = end - start

    pos_expr = (
        f"'{start:.3f}+{delta:.3f}*"
        f"(3*pow(clip(t/{duration:.6f}\\,0\\,1)\\,2)"
        f"-2*pow(clip(t/{duration:.6f}\\,0\\,1)\\,3))'"
    )
    if sweep_y:
        crop = f"crop={tw}:{th}:0:{pos_expr}"
    else:
        crop = f"crop={tw}:{th}:{pos_expr}:0"
    return [f"scale={scale_w}:{scale_h}", crop, "setsar=1"]


def plan_source_fit(video_file: str, target_size: Tuple[int, int],
                    fit_mode: str, *, anchor: dict = None,
                    duration: float = None) -> Tuple[List[str], object]:
    """Per-source fit decisions: (sar_fix_filters, fit_plan).

    sar_fix resamples anamorphic sources to square pixels before the fit
    chain — force_original_aspect_ratio compares storage dimensions and the
    fit chains end in setsar=1, so non-square SAR would render distorted.

    fit_plan is (unchanged defaults keep this identical to the pre-anchor
    engine):
      None ................................. plain get_fit_filters chain
      (fg_w, fg_h, crop_w, crop_h) tuple ... limited-crop hybrid over blur
                                             fill (crop_loss in
                                             (MAX_CROP_PER_AXIS, SCAN_CROP_LOSS])
      {'mode': 'scan', 'filters': [...]} ... scan-fit sweep (crop_loss >
                                             SCAN_CROP_LOSS and duration
                                             known; never chosen with the
                                             default duration=None)
    """
    sar_fix: List[str] = []
    fit_plan = None
    if not target_size or fit_mode not in ('crop', 'blur', 'pad'):
        return sar_fix, fit_plan
    info = get_cached_display_info(video_file)
    if not info:
        return sar_fix, fit_plan
    disp_w, disp_h, sar = info
    if abs(sar - 1.0) > 0.01:
        sar_fix = ["scale=iw*sar:ih", "setsar=1"]
    if fit_mode == 'crop':
        fit_plan = plan_smart_crop((disp_w, disp_h), target_size)
        if (fit_plan is not None and duration
                and _crop_loss((disp_w, disp_h), target_size) > SCAN_CROP_LOSS):
            scan = plan_scan_fit((disp_w, disp_h), target_size, duration,
                                 anchor=anchor)
            if scan:
                fit_plan = {'mode': 'scan', 'filters': scan}
            # else: keep the hybrid — never a static extreme crop.
    return sar_fix, fit_plan


# Drift directions for the echo background, indexed by a stable hash of the
# source path — deterministic across runs (double-run framemd5 idiom), varied
# across sources.
_ECHO_DRIFT_DIRECTIONS = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)


def _echo_drift_direction(source_key: str) -> Tuple[int, int]:
    """Deterministic drift direction from a stable hash of the source path."""
    digest = hashlib.sha1((source_key or '').encode('utf-8')).hexdigest()
    return _ECHO_DRIFT_DIRECTIONS[int(digest[:8], 16) % len(_ECHO_DRIFT_DIRECTIONS)]


def _pulse_beats(local_beats) -> List[float]:
    """Sanitized beat offsets for the margin pulse: finite, >= 0, sorted,
    capped at ECHO_PULSE_MAX_BEATS. Empty/None/garbage → [] (no pulse)."""
    if not local_beats:
        return []
    out = []
    for b in local_beats:
        try:
            t = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and t >= 0.0:
            out.append(t)
    return sorted(out)[:ECHO_PULSE_MAX_BEATS]


def _echo_grade(local_beats=None) -> str:
    """The echo background grade, beat-pulsed when local beat offsets are
    known. local_beats=None (or unusable) emits the exact legacy static
    grade string, so pulse-free chains stay byte-identical.

    The pulse is the _fx_sat_pulse idiom: a sum of between(t,b,b+WINDOW)
    window terms modulating the eq values per frame — outside every window
    the expression collapses to the static graded values exactly. t here is
    the segment-local output clock (pre_filters end in setpts=PTS-STARTPTS
    + fps before the split), the same clock segment_beats are rebased to.
    """
    beats = _pulse_beats(local_beats)
    if not beats:
        return "eq=brightness=-0.08:saturation=0.6"
    pulses = '+'.join(
        f"between(t,{b:.4f},{b + ECHO_PULSE_WINDOW:.4f})" for b in beats)
    return (
        f"eq=brightness='-0.08+{ECHO_PULSE_BRIGHT:.3f}*({pulses})'"
        f":saturation='0.6+{ECHO_PULSE_SAT:.3f}*({pulses})':eval=frame"
    )


def _echo_bg_filters(target_size: Tuple[int, int], duration: float = None,
                     source_key: str = None,
                     local_beats: List[float] = None) -> List[str]:
    """The 'echo' background: blurred overscanned fill with a deliberate
    grade (darkened, desaturated, subtle vignette) and — when the segment
    duration is known — a slow linear drift of the crop window (~3% of the
    frame over the segment; steadier and cheaper than zoompan).

    duration=None keeps today's static cover framing (grade still applies).
    The vignette runs after the drifting crop so it stays centered on the
    visible frame instead of wandering with the window. local_beats (segment
    -local beat offsets) makes the grade pulse on each beat; None keeps the
    static grade byte-identical.
    """
    w, h = target_size
    grade = _echo_grade(local_beats)
    vignette = "vignette=angle=PI/8"
    if not duration or duration <= 0:
        return [
            f"scale={w}:{h}:force_original_aspect_ratio=increase",
            f"crop={w}:{h}",
            "gblur=sigma=16",
            grade,
            vignette,
        ]
    bw = max(w + 2, int(round(w * ECHO_BG_OVERSCAN / 2)) * 2)
    bh = max(h + 2, int(round(h * ECHO_BG_OVERSCAN / 2)) * 2)
    dx, dy = _echo_drift_direction(source_key)
    travel_x = min(bw - w, int(round(w * ECHO_DRIFT_TRAVEL))) * dx
    travel_y = min(bh - h, int(round(h * ECHO_DRIFT_TRAVEL))) * dy

    def _axis_expr(margin: int, travel: int) -> str:
        if travel == 0:
            return f"{margin / 2.0:.3f}"
        start = margin / 2.0 - travel / 2.0
        return f"'{start:.3f}+{travel:.3f}*clip(t/{duration:.6f}\\,0\\,1)'"

    x_expr = _axis_expr(bw - w, travel_x)
    y_expr = _axis_expr(bh - h, travel_y)
    return [
        f"scale={bw}:{bh}:force_original_aspect_ratio=increase",
        f"crop={bw}:{bh}",
        "gblur=sigma=16",
        grade,
        f"crop={w}:{h}:{x_expr}:{y_expr}",
        vignette,
    ]


def _blur_fit_chain(target_size: Tuple[int, int], fg_filters: List[str] = None,
                    *, duration: float = None, source_key: str = None,
                    label_suffix: str = '',
                    local_beats: List[float] = None) -> str:
    """Single-input chain: echo blur fill in back, foreground centered on top.

    Works in both -vf and -filter_complex (a linear chain with an internal
    split). The default foreground is the undistorted full fit (pure blur
    mode); the limited-crop hybrid passes its own scale+crop chain (which may
    carry an anchor offset — the fg compositing itself stays centered).
    duration/source_key feed the background drift; both default to the
    static-background behavior. label_suffix keeps the internal labels
    unique when the chain appears more than once in one filter_complex
    (split-screen panes); the default '' keeps solo graphs byte-identical.
    local_beats feeds the background's beat pulse (margins only — the fg
    branch is untouched); the default None keeps the static grade.
    """
    w, h = target_size
    if fg_filters is None:
        fg_filters = [f"scale={w}:{h}:force_original_aspect_ratio=decrease"]
    bg = ",".join(_echo_bg_filters(target_size, duration, source_key,
                                   local_beats))
    ls = label_suffix
    return (
        f"split=2[bg{ls}][fg{ls}];"
        f"[bg{ls}]{bg}[bgb{ls}];"
        f"[fg{ls}]{','.join(fg_filters)}[fgs{ls}];"
        f"[bgb{ls}][fgs{ls}]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,setsar=1"
    )


def build_blur_fit_graph(pre_filters: List[str], target_size: Tuple[int, int],
                         post_filters: List[str], out_label: str = 'outv',
                         fg_filters: List[str] = None, *,
                         duration: float = None, source_key: str = None,
                         local_beats: List[float] = None) -> str:
    """Filter graph for blur fit: echo blur fill in back, foreground in front."""
    pre = ",".join(pre_filters)
    post = ("," + ",".join(post_filters)) if post_filters else ""
    chain = _blur_fit_chain(target_size, fg_filters, duration=duration,
                            source_key=source_key, local_beats=local_beats)
    return f"[0:v]{pre},{chain}{post}[{out_label}]"


def _fg_zoom_filter(fg_w: int, fg_h: int, duration: float, fps: float):
    """Slow push-in for an echo-fill foreground, or None when gated off.

    zoompan (crop/scale can't animate w/h) with d=1 emits exactly one output
    frame per input frame — the fg branch's frame count is untouched and
    -vframes stays the authority. s= is locked to the fg pane dims, so the
    composited rectangle never moves; the CONTENT inside it scales up
    linearly to 1+HYBRID_FG_ZOOM by the last frame (z is a pure expression
    of the output frame counter `on` — deterministic, no rng). fps= keeps
    the output timestamps on the segment clock so downstream t-based effect
    expressions stay beat-aligned. Runs AFTER the anchored/tracked crop, so
    a tracked pan composes with the zoom instead of fighting it.

    Gates (any → None, chain byte-identical to the pre-zoom engine):
    HYBRID_FG_ZOOM <= 0 (kill switch), unknown/invalid duration or fps,
    degenerate pane dims.
    """
    if not HYBRID_FG_ZOOM or HYBRID_FG_ZOOM <= 0:
        return None
    try:
        duration = float(duration)
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    if duration <= 0 or fps <= 0 or fg_w < 2 or fg_h < 2:
        return None
    n = max(2, int(round(duration * fps)))
    return (
        f"zoompan=z='1+{HYBRID_FG_ZOOM:.4f}*on/{n}':d=1"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={fg_w}x{fg_h}:fps={fps}"
    )


def _hybrid_fg_filters(hybrid_fg, anchor: dict = None, *,
                       duration: float = None,
                       fps: float = None) -> List[str]:
    """Foreground scale+crop for the limited-crop hybrid; the crop window
    honors the anchor (offset, never enlarged) when one is usable, and — when
    the anchor carries a rebased subject path (PAN_PATH_KEY) and the segment
    duration is known — follows it as a tracked pan. Fallbacks emit the exact
    static _anchored_crop, and the foreground dims are known here, so the
    pan headroom is exact.

    With duration AND fps known (and HYBRID_FG_ZOOM > 0) the branch gains
    the slow zoompan push-in after the crop (pan first, then zoom). The
    default fps=None keeps existing callers (ProRes proxy conversion via
    build_source_fit_chain) byte-identical."""
    fg_w, fg_h, crop_w, crop_h = hybrid_fg
    crop = None
    if duration and isinstance(anchor, dict) and anchor.get(PAN_PATH_KEY):
        hx = max(0.0, (fg_w - crop_w) / float(fg_w))
        hy = max(0.0, (fg_h - crop_h) / float(fg_h))
        crop = _tracked_crop(crop_w, crop_h, anchor, duration, hx, hy)
    if crop is None:
        crop = _anchored_crop(crop_w, crop_h, anchor)
    filters = [f"scale={fg_w}:{fg_h}", crop]
    zoom = _fg_zoom_filter(crop_w, crop_h, duration, fps)
    if zoom:
        filters.append(zoom)
    return filters


def _blur_mode_fg_filters(video_file: str, target_size: Tuple[int, int], *,
                          duration: float = None, fps: float = None):
    """Foreground filters for pure blur mode WITH the slow push-in, or None
    to keep the legacy default fg (scale=...decrease, no zoom).

    zoompan needs a fixed s=, so the undistorted fit is computed here in
    Python (even-rounded, never above target — the same rounding idiom as
    plan_smart_crop) and emitted as an explicit scale instead of
    force_original_aspect_ratio=decrease. Only taken when the zoom actually
    fires; every gate (kill switch, unknown duration/fps, failed probe)
    returns None so the chain stays byte-identical to the current engine.
    """
    if not HYBRID_FG_ZOOM or HYBRID_FG_ZOOM <= 0:
        return None
    if not duration or not fps:
        return None
    info = get_cached_display_info(video_file)
    if not info:
        return None
    sw, sh = info[0], info[1]
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return None
    f = min(tw / sw, th / sh)
    fw = min(tw, max(2, int(round(sw * f / 2)) * 2))
    fh = min(th, max(2, int(round(sh * f / 2)) * 2))
    zoom = _fg_zoom_filter(fw, fh, duration, fps)
    if not zoom:
        return None
    return [f"scale={fw}:{fh}", zoom]


def build_source_fit_chain(video_file: str, target_size: Tuple[int, int],
                           fit_mode: str, *, anchor: dict = None,
                           duration: float = None) -> str:
    """-vf chain fitting a whole source to target_size (SAR normalization plus
    blur/limited-crop/scan handling) — used by ProRes proxy conversion so
    lossless mode fits sources the same way the standard pipeline does.
    duration/anchor are optional upgrades: duration enables the echo drift
    and scan-fit, anchor offsets the crop windows; the defaults keep the
    pre-anchor framing (plus the echo grade on blur fills)."""
    sar_fix, fit_plan = plan_source_fit(video_file, target_size, fit_mode,
                                        anchor=anchor, duration=duration)
    scan_plan = fit_plan if isinstance(fit_plan, dict) else None
    hybrid_fg = fit_plan if fit_plan is not None and scan_plan is None else None
    if scan_plan:
        chain = ",".join(scan_plan['filters'])
    elif fit_mode == 'blur' or hybrid_fg:
        fg_filters = (_hybrid_fg_filters(hybrid_fg, anchor, duration=duration)
                      if hybrid_fg else None)
        chain = _blur_fit_chain(target_size, fg_filters, duration=duration,
                                source_key=video_file)
    else:
        chain = ",".join(_fit_filters_with_pan(video_file, target_size,
                                               fit_mode, anchor, duration))
    return ",".join(sar_fix + [chain]) if sar_fix else chain


# --- Split-screen duo panes (design doc: docs/DESIGN_split_screen.md) --------

def _pane_sizes(target_size: Tuple[int, int]):
    """((w0, h0), (w1, h1), 'hstack'|'vstack') pane geometry for a duo.

    Landscape canvas → side-by-side panes (hstack); portrait/square →
    stacked (vstack). The first pane takes the even-rounded half of the
    split axis, the second takes the remainder (may differ by 2 px —
    stacking only requires the shared axis to match).
    """
    tw, th = target_size
    if tw >= th:
        w0 = max(2, (tw // 2) // 2 * 2)
        return (w0, th), (tw - w0, th), 'hstack'
    h0 = max(2, (th // 2) // 2 * 2)
    return (tw, h0), (tw, th - h0), 'vstack'


def _pane_fit_chain(video_file: str, pane_size: Tuple[int, int], anchor: dict,
                    duration: float, label_suffix: str) -> str | None:
    """Filter chain fitting one duo input into its pane, or None when the
    source can't be probed (the caller drops the partner and renders solo —
    never a blind pane crop; the 15%-crop lesson, pane edition).

    Pane policy (Option A): SAR fix, then fill-and-crop at pane size with
    the anchored/tracked window as long as crop_loss stays within
    PANE_MAX_CROP; beyond that (rare — a landscape source in a pane) the
    pane falls back to the echo blur fit at pane size. No scan inside panes.
    """
    info = get_cached_display_info(video_file)
    if not info:
        return None
    disp_w, disp_h, sar = info
    parts: List[str] = []
    if abs(sar - 1.0) > 0.01:
        parts.extend(["scale=iw*sar:ih", "setsar=1"])
    pw, ph = pane_size
    if _crop_loss((disp_w, disp_h), pane_size) > PANE_MAX_CROP:
        parts.append(_blur_fit_chain(pane_size, None, duration=duration,
                                     source_key=video_file,
                                     label_suffix=label_suffix))
        return ",".join(parts)
    crop = None
    if duration and isinstance(anchor, dict) and anchor.get(PAN_PATH_KEY):
        hx, hy = _fill_crop_headroom((disp_w, disp_h), pane_size)
        crop = _tracked_crop(pw, ph, anchor, duration, hx, hy)
    if crop is None:
        crop = _anchored_crop(pw, ph, anchor)
    parts.extend([
        f"scale={pw}:{ph}:force_original_aspect_ratio=increase",
        crop,
        "setsar=1",
    ])
    return ",".join(parts)


def _duo_input_plan(video_file: str, start_time: float, exact_duration: float,
                    fps: float, anchor: dict, pane_size: Tuple[int, int],
                    label_suffix: str):
    """(input_args, branch_chain) for one duo input, or None if unplannable.

    Mirrors the solo input logic exactly: each input gets its own loop/seek
    decision (get_loop_input_args; a looped nonzero start seeks via
    trim=start= in the chain, never input-side -ss — the -stream_loop
    gotcha) and its own build_segment_pre_filters head, then the pane fit.
    """
    try:
        start_time = max(0.0, float(start_time))
    except (TypeError, ValueError):
        return None
    pane_fit = _pane_fit_chain(video_file, pane_size, anchor, exact_duration,
                               label_suffix)
    if pane_fit is None:
        return None
    loop_args, start_time = get_loop_input_args(video_file, start_time,
                                                exact_duration)
    filter_seek = bool(loop_args) and start_time > 0
    pre_filters = build_segment_pre_filters(
        exact_duration, fps, trim_start=start_time if filter_seek else 0.0)
    input_args = list(loop_args)
    if filter_seek:
        input_args.extend(['-i', video_file])
    else:
        input_args.extend(['-ss', str(start_time), '-t', str(exact_duration),
                           '-i', video_file])
    return input_args, ",".join(pre_filters + [pane_fit])


def _plan_duo_render(primary_file: str, primary_start: float, primary_anchor,
                     partner: dict, exact_duration: float, fps: float,
                     target_size: Tuple[int, int]):
    """Plan the two-input split-screen render.

    Returns ([input_args0, input_args1], [branch0, branch1],
    'hstack'|'vstack'), or None — with a printed warning — when either input
    can't be planned; the caller then renders the primary solo. Never
    raises: a bad partner must not kill a segment. Deterministic: pure
    probe-and-math, no rng.
    """
    try:
        pane0, pane1, stack = _pane_sizes(target_size)
        plan0 = _duo_input_plan(primary_file, primary_start, exact_duration,
                                fps, primary_anchor, pane0, '0')
        plan1 = _duo_input_plan(partner.get('video_file'),
                                partner.get('start_time', 0.0),
                                exact_duration, fps,
                                partner.get('subject_anchor'), pane1, '1')
        if plan0 is None or plan1 is None:
            raise ValueError("pane planning failed (probe or start time)")
        return [plan0[0], plan1[0]], [plan0[1], plan1[1]], stack
    except Exception as e:
        pname = os.path.basename(str((partner or {}).get('video_file') or '?'))
        print(f"   ⚠️  Partner dropped ({pname}): {e} — rendering solo")
        return None


def build_text_overlay_graph(base_graph: str, fade_in_start: float,
                             fade_in_duration: float, fade_out_start: float,
                             fade: float = 0.35,
                             text_input_index: int = 1) -> str:
    """Composite the looped transparent PNG input over the [basev] stream.

    Text is rendered by Pillow (see text_overlay.py) because this ffmpeg
    build has no drawtext; overlay/fade/format are core filters. Fade
    timings are in the segment's local clock (text_overlay.plan_text_windows
    projects the global text window onto each segment). A zero fade-in
    duration means the fade completed in an earlier segment, so the filter
    is omitted (fade rejects st<0 and d=0); a fade-in with st>0 also keeps
    the text invisible before st, handling windows that open mid-segment.

    text_input_index is the PNG's input position: 1 for solo segments
    (default, byte-identical to the historic graph), 2 for split-screen
    duos where inputs 0 and 1 are the two video sources.
    """
    txt_chain = ["format=rgba"]
    if fade_in_duration > 0.001:
        txt_chain.append(f"fade=t=in:st={max(0.0, fade_in_start):.4f}:d={fade_in_duration:.4f}:alpha=1")
    txt_chain.append(f"fade=t=out:st={max(0.0, fade_out_start):.4f}:d={fade:.4f}:alpha=1")
    return (
        f"{base_graph};"
        f"[{text_input_index}:v]{','.join(txt_chain)}[txt];"
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


def _cached_video_duration(video_file: str) -> Tuple[float, bool]:
    """(duration, known) with a per-run cache (segments hit sources repeatedly).

    known=False flags a probe failure: the returned value is a safe 0.0
    sentinel, NOT a real duration. Callers that clamp a start against it stay
    safe (max_start -> 0, so no seek past a real EOF), and get_loop_input_args
    reads the flag to force the loop path instead of trusting the sentinel.
    A transient ffprobe error is never cached, so a later segment can recover.
    """
    if is_image_source(video_file):
        # ffprobe reports one frame (~0.04s) for a still; the synthetic
        # duration lets stills fill any segment (extraction loops them).
        return SYNTHETIC_IMAGE_DURATION, True
    cached = _SOURCE_DURATION_CACHE.get(video_file)
    if cached is not None:
        return cached, True
    probed = _probe_video_duration(video_file)
    if probed is None:
        # Don't cache failures: a transient ffprobe error must not pin the
        # fallback for the rest of the run.
        return 0.0, False
    _SOURCE_DURATION_CACHE[video_file] = probed
    return probed, True


def get_cached_video_duration(video_file: str) -> float:
    """Duration lookup with a per-run cache (segments hit the same sources repeatedly).

    Returns 0.0 on probe failure (uncached). 0.0 is safe through every clamp
    consumer (start clamps to 0) and routes get_loop_input_args onto the loop
    path so an unknown length can never seek past a real EOF.
    """
    duration, _known = _cached_video_duration(video_file)
    return duration


def get_loop_input_args(video_file: str, start_time: float, duration: float) -> Tuple[List[str], float]:
    """Loop args for sources shorter than the requested window (GIFs, short clips).

    Returns (['-stream_loop', N] or [], adjusted_start_time). With looping the
    demuxer presents the source repeated N+1 times. ffmpeg gotcha: an input-side
    -ss combined with -stream_loop re-seeks to the offset on EVERY loop
    iteration (each pass yields only [start, EOF]), so when these args are used
    with a nonzero start the caller must seek in the filter chain (trim=start=)
    instead of with -ss.
    """
    src_duration, known = _cached_video_duration(video_file)
    if not known:
        # Unknown source length (probe failed): force an infinite stream_loop
        # from the start. This can never seek past a real EOF and never skips
        # looping a source that might be shorter than the window. The
        # downstream -t / -vframes cap always bounds the output, so an infinite
        # loop cannot run away; start_time resets to 0 so we take the -ss 0
        # (not filter-seek) path where the bound is guaranteed present.
        return ['-stream_loop', '-1'], 0.0
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
    if is_image_source(video_file):
        return SYNTHETIC_IMAGE_DURATION
    duration = _probe_video_duration(video_file)
    return 10.0 if duration is None else duration  # Default fallback


_SOURCE_FPS_CACHE: dict = {}


def get_cached_video_fps(video_file: str) -> float:
    """fps lookup with a per-run cache (the planner probes per segment).

    Mirrors get_cached_video_duration: a probe failure returns the 30.0
    fallback WITHOUT caching it, so one transient ffprobe error can't pin
    fps=30.0 (indistinguishable from a real 30fps source) for the whole run
    and skew stage6's retime gating (source_fps < 24 skip, >= 50 slow-mo gate).
    """
    if is_image_source(video_file):
        # A still has no timebase; ffprobe would report the image2 demuxer
        # default (25) which must not masquerade as a real source fps.
        # Callers detecting output fps should skip image sources entirely.
        return 30.0
    cached = _SOURCE_FPS_CACHE.get(video_file)
    if cached is None:
        cached = _probe_video_fps(video_file)
        if cached is None:
            # Don't cache failures: a transient ffprobe error must not pin the
            # 30fps fallback for the rest of the run.
            return 30.0
        _SOURCE_FPS_CACHE[video_file] = cached
    return cached


def _probe_video_fps(video_file: str) -> float | None:
    """fps via ffprobe; None on failure so callers can avoid caching it."""
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
        return None


def get_video_fps(video_file: str) -> float:
    """Get the FPS of a video file using ffprobe."""
    if is_image_source(video_file):
        # A still has no timebase; ffprobe would report the image2 demuxer
        # default (25) which must not masquerade as a real source fps.
        # Callers detecting output fps should skip image sources entirely.
        return 30.0
    fps = _probe_video_fps(video_file)
    return 30.0 if fps is None else fps  # Default fallback


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
    ]
    if is_image_source(video_file):
        # A still becomes a fixed-length looped clip so precise mode's segment
        # extraction and concat see a normal video stream.
        cmd.extend(['-loop', '1', '-framerate', str(fps), '-t', str(IMAGE_PRORES_SECONDS)])
    cmd.extend([
        '-i', video_file,
        '-map', '0:v:0',
    ])
    if target_size:
        # Same fit behavior as the standard pipeline (SAR normalization,
        # blur fit, limited-crop hybrid) — -vf accepts the internal-split
        # graph because it stays single-input/single-output.
        cmd.extend(['-vf', build_source_fit_chain(video_file, target_size, fit_mode)])
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


RETIME_MIN_SPEED = 0.4
RETIME_MAX_SPEED = 2.5
# Over-provision every retimed source window by this many OUTPUT frames of
# source time. -vframes truncates the excess for free, while a window that
# comes up even one source frame short yields a short segment (audit repro:
# 59/60 and 23/60) that would silently drift the whole timeline.
RETIME_SLACK_FRAMES = 2


def _retime_speeds(retime: dict) -> Tuple[float, float]:
    """(speed_start, speed_end) of a retime spec, clamped to the safe band."""
    kind = retime.get('kind')
    if kind == 'ramp':
        s0 = float(retime.get('speed_start', 1.0))
        s1 = float(retime.get('speed_end', 1.0))
    else:
        s0 = s1 = float(retime.get('speed', 1.0))
    clamp = lambda s: max(RETIME_MIN_SPEED, min(RETIME_MAX_SPEED, s))
    return clamp(s0), clamp(s1)


def retime_source_window(output_duration: float, retime: dict | None,
                         fps: float) -> float:
    """Source seconds a segment must decode, slack included.

    Shared by the planner (runway gating), the clip worker (start clamping)
    and extraction (trim/-t) so the three can never disagree about how much
    source a retimed segment consumes.
    """
    if not retime:
        return output_duration
    if retime.get('kind') == 'freeze':
        out_frames = max(1, seconds_to_frame_count(output_duration, fps))
        freeze = min(int(retime.get('freeze_frames', 0)), out_frames - 1)
        content_frames = max(1, out_frames - freeze)
        return content_frames / fps + RETIME_SLACK_FRAMES / fps
    s0, s1 = _retime_speeds(retime)
    avg = (s0 + s1) / 2.0
    slack = (RETIME_SLACK_FRAMES / fps) * max(1.0, s0, s1)
    return output_duration * avg + slack


def _retime_filters(retime: dict, output_duration: float, fps: float) -> List[str]:
    """Filter snippet realizing a retime spec.

    Sits BETWEEN setpts=PTS-STARTPTS and the final fps= (see
    build_segment_pre_filters): everything downstream of fps — effects,
    transitions, text fades, zoompan, -vframes — sees the OUTPUT clock, so
    beat-locked timing is untouched by how fast the content underneath runs.
    """
    kind = retime.get('kind')
    if kind == 'freeze':
        # loop repeats the last content frame but duplicates its PTS, which
        # fps would then drop — re-linearize with setpts=N/FR/TB after it. The
        # leading fps= normalizes to the output rate so frame indexes are in
        # output units; the frame-exact trim guarantees nothing follows the
        # frozen tail (the decode slack would otherwise leak in after the
        # repeats), and looping slack extra repeats keeps -vframes fed even
        # if the decode came up a frame short.
        out_frames = max(1, seconds_to_frame_count(output_duration, fps))
        freeze = min(int(retime.get('freeze_frames', 0)), out_frames - 1)
        if freeze <= 0:
            return []
        content_frames = max(1, out_frames - freeze)
        return [
            f"fps={fps}",
            f"trim=end_frame={content_frames}",
            "setpts=PTS-STARTPTS",
            f"loop=loop={freeze + RETIME_SLACK_FRAMES}:size=1:start={content_frames - 1}",
            f"setpts=N/{fps}/TB",
        ]
    s0, s1 = _retime_speeds(retime)
    if abs(s1 - s0) < 0.01:
        return [f"setpts=PTS/{s0:.6f}"]
    # Variable ramp: source time τ(t) = s0·t + a·t²/2 with a=(s1-s0)/T, so the
    # output timestamp is the inverse t(τ) = (sqrt(s0²+2aτ) − s0)/a. Monotonic
    # for any positive speeds, which fps= requires.
    T = max(0.05, float(output_duration))
    a = (s1 - s0) / T
    # max(0, ...) guards the sqrt domain: for a decelerating ramp the radicand
    # goes negative past tau_zero, and ffmpeg's sqrt of a negative yields NOPTS
    # which stalls the downstream fps stage (segment comes up short). Unreachable
    # via the sole call site today (stage6 gates leave >=0.077s margin), so this
    # is a no-op for every reachable input — it only clamps an out-of-domain tail.
    expr = (
        f"(sqrt(max(0,{s0 * s0:.8f}+{2.0 * a:.8f}*PTS*TB))-{s0:.6f})/{a:.8f}/TB"
    )
    return [f"setpts='{expr}'"]


def build_segment_pre_filters(exact_duration: float, fps: float,
                              trim_start: float = 0.0,
                              retime: dict | None = None,
                              output_duration: float = None) -> List[str]:
    """Shared trim/setpts/fps head of every segment's filter chain.

    Trim first so each extracted segment has exact timing; effects come after
    fps so their time expressions see the final frame timing. A nonzero
    trim_start seeks in the filter chain (used with -stream_loop, where an
    input-side -ss re-seeks on every loop iteration).

    ORDERING INVARIANT: retime filters live between setpts=PTS-STARTPTS and
    the final fps=. Moving them after fps would put every time-based effect
    expression, segment_beats gate and zoompan counter on the RETIMED clock
    and silently break beat alignment. exact_duration is the SOURCE window
    (equal to the output duration when retime is None); output_duration is
    what -vframes will enforce.
    """
    if trim_start > 0:
        trim_filter = f"trim=start={trim_start:.6f}:duration={exact_duration}"
    else:
        trim_filter = f"trim=duration={exact_duration}"
    filters = [trim_filter, "setpts=PTS-STARTPTS"]
    if retime:
        filters.extend(_retime_filters(retime, output_duration or exact_duration, fps))
    # Hold the last frame if the demuxer under-delivers. GIFs (and some VFR
    # sources) carry a long display duration on their FINAL frame — container
    # duration 3.75s but last packet at pts 2.5 — and fps= stops at the last
    # packet instead of honoring that trailing display time, so a window that
    # ends inside the gap comes up short and the frame guard kills the render
    # (reproduced: 3-frame GIF, 75/105 frames). Cloning the last frame is the
    # correct rendering for display-duration sources; a no-op whenever the
    # input actually covers the window. Every consumer of this chain caps
    # output with -vframes, which bounds the open-ended pad.
    filters.append("tpad=stop_mode=clone:stop=-1")
    filters.append(f"fps={fps}")
    return filters


def build_ken_burns_filter(rng, target_size: Tuple[int, int], fps: float,
                           frame_count: int) -> str:
    """Gentle deterministic pan/zoom so a still reads as footage.

    Runs after the fit chain (frames are already target-sized), so zoompan's
    output size matches the input and only the virtual camera moves. Callers
    seed rng per segment; the same idiom as effects.py keeps renders
    reproducible.
    """
    w, h = target_size
    n = max(2, int(frame_count))
    dz = 0.06 + 0.06 * rng.random()  # 6-12% zoom travel over the segment
    if rng.random() < 0.5:
        zoom_expr = f"1+{dz:.4f}*on/{n}"
    else:
        zoom_expr = f"{1 + dz:.4f}-{dz:.4f}*on/{n}"
    pan = rng.choice(('lr', 'rl', 'tb', 'bt', 'center'))
    x_expr = {
        'lr': f"(iw-iw/zoom)*on/{n}",
        'rl': f"(iw-iw/zoom)*(1-on/{n})",
    }.get(pan, "(iw-iw/zoom)/2")
    y_expr = {
        'tb': f"(ih-ih/zoom)*on/{n}",
        'bt': f"(ih-ih/zoom)*(1-on/{n})",
    }.get(pan, "(ih-ih/zoom)/2")
    return (
        f"zoompan=z='{zoom_expr}':d=1:x='{x_expr}':y='{y_expr}'"
        f":s={w}x{h}:fps={fps}"
    )


def _lut3d_filter(cube_path: str) -> str:
    """lut3d with the path escaped for filter-arg parsing (\\ : ' are special)."""
    escaped = (cube_path.replace('\\', '/')
               .replace(':', '\\:')
               .replace("'", "\\'"))
    return f"lut3d=file='{escaped}'"


def extract_clip_segment_ffmpeg(video_file: str, start_time: float, duration: float,
                                output_file: str, fps: float, target_size: Tuple[int, int],
                                use_nvenc: bool,
                                gpu_encoder: str = 'h264_nvenc',
                                fit_mode: str = 'crop',
                                extra_filters: List[str] = None,
                                text_overlay: Tuple[str, float, float, float] = None,
                                look_cube: str = None,
                                retime: dict = None,
                                *, anchor: dict = None,
                                partner: dict = None,
                                local_beats: List[float] = None) -> bool:
    """
    Extract a video segment using FFmpeg with FRAME-ACCURATE timing.

    ✅ FRAME-ACCURATE: Uses exact frame counts instead of floating-point seconds
    ✅ ZERO DRIFT: No cumulative timing errors

    duration is the OUTPUT duration (what the timeline reserved). With a
    retime spec the source window differs — retime_source_window() computes
    it, over-provisioned so -vframes always has enough frames to cap.

    anchor (optional, from the analysis stage): {"cx","cy","confidence",...}
    normalized subject center; offsets fit-crop windows and biases the
    scan-fit sweep. None or low confidence keeps centered framing. When the
    planner attached a segment-local subject path under "path_seg" (see
    PAN_PATH_KEY), the offset-crop and hybrid-foreground crop windows follow
    it as a tracked pan; anchors without it (or failing any pan gate) keep
    the exact static behavior. The scan-fit tier never animates on the path
    (its own sweep would compound with the pan).

    partner (optional, from the stage6 duo planner): {"video_file",
    "start_time", "source_duration", "candidate_id", "source_name",
    "subject_anchor"} — a second source rendered split-screen beside the
    primary (hstack on a landscape canvas, vstack on portrait). Each input
    keeps its own loop/seek/pre-filter logic and gets a pane-sized
    anchored/tracked crop (PANE_MAX_CROP); effects/look/text apply to the
    composed frame. Duos never combine with retimes or still images, and
    any partner planning or render failure drops the partner and renders
    the primary solo — a partner can never fail a segment. partner=None is
    byte-identical to the pre-duo engine.

    local_beats (optional, from the effects planner's segment_beats): beat
    offsets on this segment's local output clock (seconds from the first
    frame, <= 8 entries). Only consumed when the echo blur fill is on
    screen (fit_mode='blur' or the limited-crop hybrid): the background
    grade pulses briefly on each beat (ECHO_PULSE_*), background branch
    only, before the fg overlay. local_beats=None (the default, and what
    Minimal/clean renders pass) keeps every chain byte-identical to the
    pulse-free engine. Duo panes never pulse (their echo fill is a rare
    fallback and the composed frame already carries the beat effects).
    """
    try:
        # ✅ FRAME-ACCURATE: Calculate exact output frame count first; the
        # source window derives from it (and the retime spec, if any).
        output_frame_count = max(1, seconds_to_frame_count(duration, fps))
        exact_output_duration = frame_count_to_seconds(output_frame_count, fps)
        exact_source_duration = retime_source_window(exact_output_duration, retime, fps)

        # Still images have no timeline: -loop 1 serves the single frame for
        # exactly the segment window, so seeking, -stream_loop and retiming
        # don't apply (retiming a static frame is a no-op with extra risk).
        image_source = is_image_source(video_file)
        if retime and image_source:
            retime = None
            exact_source_duration = exact_output_duration

        post_filters = list(extra_filters or [])
        if look_cube:
            # Color grade last so the look sits on top of the effects; text
            # overlays composite after this, so text stays ungraded (white
            # text keeps reading white on a day-for-night grade).
            post_filters.append(_lut3d_filter(look_cube))

        # Split-screen duo guards — the planner already enforces all of
        # these; the strips below are belt-and-braces for external callers,
        # mirroring the retime strips. A dropped partner NEVER fails the
        # segment: the primary just renders solo through the normal ladder.
        if partner is not None and (not isinstance(partner, dict)
                                    or not partner.get('video_file')
                                    or not target_size):
            print(f"   ⚠️  Partner dropped for {os.path.basename(video_file)}: invalid partner spec")
            partner = None
        if partner is not None and retime:
            # A warped clock breaks the per-pane tracked pan and doubles the
            # runway math; duos live on hard cuts at natural speed.
            print(f"   ⚠️  Partner dropped for {os.path.basename(video_file)}: retime and split-screen never combine")
            partner = None
        if partner is not None and (image_source
                                    or is_image_source(partner['video_file'])):
            print(f"   ⚠️  Partner dropped for {os.path.basename(video_file)}: still images render solo")
            partner = None

        if partner is not None:
            duo_plan = _plan_duo_render(video_file, start_time, anchor,
                                        partner, exact_output_duration, fps,
                                        target_size)
            if duo_plan is None:
                partner = None  # warning printed inside; render solo below
            else:
                duo_inputs, duo_branches, duo_stack = duo_plan
                base_label = 'basev' if text_overlay else 'outv'
                post = ("," + ",".join(post_filters)) if post_filters else ""
                filter_graph = (
                    f"[0:v]{duo_branches[0]}[pane0];"
                    f"[1:v]{duo_branches[1]}[pane1];"
                    f"[pane0][pane1]{duo_stack},setsar=1{post}[{base_label}]"
                )
                if text_overlay:
                    _, fade_in_start, fade_in_duration, fade_out_start = text_overlay
                    filter_graph = build_text_overlay_graph(
                        filter_graph, fade_in_start, fade_in_duration,
                        fade_out_start, text_input_index=2)
                cmd = [FFMPEG_PATH]
                cmd.extend(get_hwaccel_args(use_nvenc, gpu_encoder))
                cmd.extend(duo_inputs[0])
                cmd.extend(duo_inputs[1])
                if text_overlay:
                    cmd.extend(['-loop', '1', '-i', text_overlay[0]])
                cmd.extend(['-filter_complex', filter_graph, '-map', '[outv]'])
                # -vframes caps the COMPOSED stream; hstack/vstack pad a
                # briefly-short branch by repeating its last frame
                # (framesync default), so the cap — not the shortest branch —
                # stays the frame-count authority.
                cmd.extend(['-vframes', str(output_frame_count)])
                if use_nvenc:
                    cmd.extend(get_gpu_quality_args(gpu_encoder, include_pix_fmt=True))
                else:
                    cmd.extend(get_cpu_h264_quality_args(include_pix_fmt=True))
                cmd.extend([
                    '-an',
                    '-fps_mode', 'cfr',
                    '-r', str(fps),
                    '-fflags', '+genpts',
                    '-movflags', '+faststart',
                    '-y',
                    output_file
                ])
                result = _run_media_command(cmd, timeout=120)
                if (result.returncode == 0
                        and os.path.exists(output_file)
                        and os.path.getsize(output_file) > 0
                        and _verify_segment_frames(output_file, output_frame_count)):
                    return True
                detail = (_short_ffmpeg_error(result.stderr, 400)
                          if result.returncode != 0 else "output verification failed")
                print(f"   ⚠️  Duo render failed for {os.path.basename(output_file)} ({detail}) — retrying solo")
                partner = None

        # Loop sources shorter than the segment (GIFs, short clips) so the
        # frame count stays exact instead of drifting. ffmpeg gotcha: with
        # -stream_loop, an input-side -ss re-seeks to the offset on EVERY loop
        # iteration (each pass yields only [start, EOF] instead of wrapping),
        # so looped seeks happen in the filter chain via trim=start= instead.
        if image_source:
            loop_args, start_time = [], 0.0
        else:
            loop_args, start_time = get_loop_input_args(video_file, start_time, exact_source_duration)
        if retime and loop_args:
            # Ramps never loop: a loop seam mid-retime is jarring, and the
            # planner already gates on runway — this is the deterministic
            # belt-and-braces strip for anything that slipped through.
            print(f"   ⚠️  Retime dropped for {os.path.basename(video_file)}: window would need looping")
            retime = None
            exact_source_duration = exact_output_duration
            loop_args, start_time = get_loop_input_args(video_file, start_time, exact_source_duration)
        filter_seek = bool(loop_args) and start_time > 0

        pre_filters = build_segment_pre_filters(
            exact_source_duration, fps, trim_start=start_time if filter_seek else 0.0,
            retime=retime, output_duration=exact_output_duration)

        # Per-source fit decisions: SAR normalization for anamorphic inputs,
        # the limited-crop hybrid when plain Smart crop would discard more
        # than MAX_CROP_PER_AXIS of the source, and the scan-fit sweep beyond
        # SCAN_CROP_LOSS (the segment duration drives the sweep and the echo
        # background drift; an anchor offsets every crop window).
        sar_fix, fit_plan = plan_source_fit(video_file, target_size, fit_mode,
                                            anchor=anchor,
                                            duration=exact_output_duration)
        scan_plan = fit_plan if isinstance(fit_plan, dict) else None
        hybrid_fg = fit_plan if fit_plan is not None and scan_plan is None else None
        pre_filters.extend(sar_fix)

        # text_overlay: (png_path, fade_in_start, fade_in_duration,
        # fade_out_start) in this segment's local clock — see
        # text_overlay.plan_text_windows.
        use_blur_graph = bool(target_size) and (fit_mode == 'blur' or hybrid_fg is not None)
        use_graph = use_blur_graph or bool(text_overlay)
        filter_graph = None
        filter_complex = None
        base_label = 'basev' if text_overlay else 'outv'
        # Tracked pan clock guard: the crop expressions run on the OUTPUT
        # clock (after setpts/fps), which only matches the rebased path's
        # source-local clock when the segment is not retimed. The planner
        # already drops the path from retimed clips; this belt-and-braces
        # covers retimes attached by external callers.
        pan_duration = None if retime else exact_output_duration
        # Foreground push-in clock guard: same retime reasoning as the pan
        # (zoompan's counter runs on the retimed clock), plus still images
        # already get Ken Burns motion downstream — don't compound two zooms.
        fg_zoom_duration = None if (retime or image_source) else exact_output_duration
        if use_blur_graph:
            if hybrid_fg:
                fg_filters = _hybrid_fg_filters(hybrid_fg, anchor,
                                                duration=pan_duration,
                                                fps=fps if fg_zoom_duration else None)
            else:
                fg_filters = _blur_mode_fg_filters(video_file, target_size,
                                                   duration=fg_zoom_duration,
                                                   fps=fps)
            filter_graph = build_blur_fit_graph(pre_filters, target_size, post_filters,
                                                out_label=base_label, fg_filters=fg_filters,
                                                duration=exact_output_duration,
                                                source_key=video_file,
                                                local_beats=local_beats)
        else:
            filters = list(pre_filters)
            if scan_plan:
                filters.extend(scan_plan['filters'])
            elif target_size:
                filters.extend(_fit_filters_with_pan(video_file, target_size,
                                                     fit_mode, anchor,
                                                     pan_duration))
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

        if image_source:
            cmd.extend([
                '-loop', '1',
                '-framerate', str(fps),
                '-t', str(exact_source_duration),
                '-i', video_file,
            ])
        elif filter_seek:
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

        # Frame-count guard on every segment: returncode 0 does not prove the
        # window held enough source (-vframes truncates a short window without
        # complaint), and one short segment drifts every later cut.
        if not _verify_segment_frames(output_file, output_frame_count):
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
                              temp_dir: str = None,
                              total_frames: int = None) -> str:
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

    def add_audio_end_bound(cmd: List[str]) -> None:
        """Trim the audio at the video's end WITHOUT -shortest.

        The frame-locked timeline rounds round(audio_duration*fps), so the
        video can legitimately run up to half a frame past the audio; with
        -shortest the muxer then drops the final video packet (observed:
        4561/4562 frames). Bounding the output half a frame past the video
        end keeps every video packet and still cuts the audio at the video
        boundary.
        """
        if total_frames and fps > 0:
            cmd.extend(['-t', f'{(int(total_frames) + 0.5) / float(fps):.6f}'])
        else:
            cmd.extend(['-shortest'])

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
                ]
                add_audio_end_bound(cmd)
                cmd.extend([
                    '-y',
                    output_file
                ])
                
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
                    cmd.extend(['-c:a', 'pcm_s24le', '-ar', '48000'])
                    add_audio_end_bound(cmd)
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
                ])
                add_audio_end_bound(cmd)
            
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
