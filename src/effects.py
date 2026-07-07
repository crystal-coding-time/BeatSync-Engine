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

    # Recipe pack: capped so hype doesn't turn into soup. All entries are
    # single-stream and frame-count-safe; shuffleframes is deliberately
    # excluded (drops trailing frames when its pattern doesn't divide the
    # segment frame count) and elbg is excluded (single-threaded, very slow).
    # Commas inside enable='...' are protected by the filter-arg quoting.
    pack_cap = 3 if hype else 2
    pack = 0

    # Pixelize burst right at hype drop cuts, decaying quickly.
    if hype and target == 'drop' and pack < pack_cap and rng.random() < 0.45 * k:
        block = max(8, int(round(12 + 20 * k)))
        filters.append(f"pixelize=w={block}:h={block}:enable='lt(t,0.25)'")
        pack += 1

    # Directional smear standing in for zoom blur on drop cuts.
    if target == 'drop' and pack < pack_cap and rng.random() < 0.30 * k:
        radius = max(4, int(round(10 * k * (0.5 + 0.5 * energy))))
        filters.append(f"dblur=angle=90:radius={radius}:enable='lt(t,0.3)'")
        pack += 1

    # Negative-flash strobe, hype drops with strong impacts only: two
    # inverted frames out of every eight, and only in the first 0.6s.
    if hype and target == 'drop' and energy > 0.75 and pack < pack_cap and rng.random() < 0.25 * k:
        filters.append("negate=enable='lt(mod(n,8),2)*lt(t,0.6)'")
        pack += 1

    # Motion trails on calmer segments: frame-mix echo, or lagfun
    # light-paint for hype.
    if target in ('flow', 'soft') and pack < pack_cap and rng.random() < 0.30 * k:
        if hype and rng.random() < 0.5:
            filters.append(f"lagfun=decay={0.9 + 0.05 * k:.3f}")
        else:
            frames = 6 if energy > 0.5 else 4
            weights = ' '.join(str(w) for w in range(frames, 0, -1))
            filters.append(f"tmix=frames={frames}:weights='{weights}'")
        pack += 1

    # Slow hue sweep through build-ups.
    if hype and target == 'build' and pack < pack_cap and rng.random() < 0.35 * k:
        rate = 30 + int(round(60 * k))
        filters.append(f"hue=h={rate}*t")
        pack += 1

    # Rare subtle fisheye bulge for the hype look.
    if hype and pack < pack_cap and rng.random() < 0.12 * k:
        filters.append(f"lenscorrection=k1={-0.15 * k:.3f}:k2=-0.05:i=bilinear")
        pack += 1

    # Rare posterize flash on hype drops (8 luma levels, first 0.3s).
    if hype and target == 'drop' and pack < pack_cap and rng.random() < 0.15 * k:
        filters.append("lutyuv=y='floor(val/32)*32+16':enable='lt(t,0.3)'")
        pack += 1

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
