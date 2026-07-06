#!/usr/bin/env python3
"""Beat-aware per-segment effect chains.

Builds ffmpeg -vf snippets from the stage 6 planner's segment metadata
(target: drop/build/rhythm/soft/flow, impact, wave). Segments start at t=0
after setpts=PTS-STARTPTS, so time-based expressions are automatically
aligned to the cut (and therefore to the beat).

Everything is deterministic: the same inputs and settings always produce
the same video (same seeding approach as the stage 6 planner).
"""

import hashlib
import random
from typing import Dict, List, Optional, Tuple

EFFECT_STYLES = ('clean', 'amv', 'hype')


def _stable_rng(*parts) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    return random.Random(seed)


def _clamp01(value, default=0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def build_effect_filters(planned_clip: Optional[Dict], style: str, intensity: float,
                         tempo_bpm: Optional[float], segment_index: int,
                         target_size: Optional[Tuple[int, int]],
                         fps: float = 30.0) -> List[str]:
    """Return -vf snippets for one segment. Empty list = no effects."""
    if not style or style == 'clean':
        return []
    k = _clamp01(intensity, 0.0)
    if k <= 0.0:
        return []

    clip = planned_clip or {}
    target = str(clip.get('target', 'flow'))
    energy = _clamp01(clip.get('impact', clip.get('wave', 0.5)))
    rng = _stable_rng('fx', segment_index, clip.get('video_file', ''), style)
    hype = style == 'hype'
    filters: List[str] = []

    # Punch-in zoom decaying from the cut: the visual "hit" on beat-aligned
    # cuts. crop can't animate w/h, so zoompan (1 output frame per input
    # frame, time reconstructed from the output frame counter) does the zoom.
    if (target in ('drop', 'rhythm') or energy > 0.75) and target_size:
        amp = (0.16 if hype else 0.10) * k * (0.6 + 0.4 * energy)
        zoom_expr = f"1+{amp:.4f}*exp(-(on/{fps:.4f})*9)"
        filters.append(
            f"zoompan=z='{zoom_expr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={target_size[0]}x{target_size[1]}:fps={fps}"
        )

    # Camera shake, hype drops only.
    if hype and target == 'drop':
        a = max(2, int(round(8 * k)))
        filters.append(
            f"crop=w=in_w-{2 * a}:h=in_h-{2 * a}"
            f":x='{a}+{a}*sin(t*41)*exp(-t*3)':y='{a}+{a}*cos(t*57)*exp(-t*3)'"
        )

    # White flash right at drop cuts.
    if target == 'drop':
        amp = (0.55 if hype else 0.35) * k
        filters.append(f"eq=brightness='{amp:.3f}*exp(-t*14)':eval=frame")

    # Saturation pulsing at the beat frequency on high-energy segments.
    if energy > 0.6 and tempo_bpm:
        bps = max(0.5, min(4.0, float(tempo_bpm) / 60.0))
        amp = (0.35 if hype else 0.20) * k * energy
        filters.append(f"eq=saturation='1+{amp:.3f}*sin(2*PI*t*{bps:.4f})':eval=frame")

    # Occasional chromatic aberration on hard segments.
    if (hype or target == 'drop') and rng.random() < 0.35 * k:
        shift = 3 if hype else 2
        filters.append(f"rgbashift=rh={shift}:bh=-{shift}")

    # Hype look: vignette always, grain sometimes.
    if hype:
        filters.append("vignette=PI/5")
        if rng.random() < 0.5 * k:
            filters.append("noise=alls=5:allf=t")

    # Zoom/shake crops shrink the frame; restore exact target dimensions so
    # concat sees identical streams.
    if filters and target_size:
        filters.append(f"scale={target_size[0]}:{target_size[1]}")

    return filters
