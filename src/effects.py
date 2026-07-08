#!/usr/bin/env python3
"""Beat-aware per-segment effect chains.

Builds ffmpeg -vf snippets from the stage 6 planner's segment metadata
(target: drop/build/rhythm/soft/flow, impact, wave). Segments start at t=0
after setpts=PTS-STARTPTS, so time-based expressions are automatically
aligned to the cut (and therefore to the beat).

Effects are registered primitives so the GUI can offer three modes:
- curated: the classic Clean/AMV/Hype presets. Builders replicate the
  pre-registry gating (including the order rng draws are consumed in), so
  curated output is character-identical to the historical styles.
- custom: the user picks the eligible palette; style gates are relaxed and
  fire probabilities boosted so picks actually appear, but each primitive
  keeps its target affinity so effects still land where the music does.
- shuffle: a seeded random palette; same seed always re-rolls the same set.

Everything is deterministic: the same inputs and settings always produce
the same video (same seeding approach as the stage 6 planner).
"""

import hashlib
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

EFFECT_STYLES = ('clean', 'amv', 'hype')
EFFECT_MODES = ('curated', 'custom', 'shuffle')


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


def _loudness_gain(ctx) -> float:
    """Amplitude multiplier for impact effects from the segment's loudness.

    'loudness' is an optional stage-6 planner key (EBU momentary loudness of
    the segment's music, 0..1, 0.5 = average). Missing/None (every plan
    before wave-13, and fallback segments) returns 1.0 — an exact no-op
    multiplier (x*1.0 == x bit-for-bit in IEEE754) — so every filter chain
    that doesn't carry the key stays byte-identical to today's output. This
    is a pure amplitude modulator: callers must only multiply it into a
    HOW-HARD parameter (zoom/brightness/pixel amplitudes), never use it to
    decide WHETHER an effect fires — that decision is owned entirely by each
    builder's own rng stream, and loudness must never perturb rng draws.
    """
    loudness = ctx['clip'].get('loudness')
    if loudness is None:
        return 1.0
    return 0.75 + 0.5 * _clamp01(loudness)


# ---------------------------------------------------------------------------
# Primitive builders. Each decides for itself whether it fires and appends to
# ctx['filters']. In curated mode the conditions (and therefore the exact
# sequence of rng draws) replicate the pre-registry code path; custom/shuffle
# relax style gates and boost probabilities but keep target affinity.
# ---------------------------------------------------------------------------


def _fx_punch_fill(ctx) -> None:
    # Punch to full-bleed: when the source sits in the fit ladder's hybrid
    # tier (crop_loss in (MAX_CROP_PER_AXIS, SCAN_CROP_LOSS] — e.g. 4:3 in a
    # 16:9 target), the fitted frame shows the clip over blurred echo margins.
    # The composed frame reaches this chain AFTER fitting, so a zoompan that
    # STARTS at the fill ratio on the cut and decays back to 1 makes the real
    # image swallow the margins exactly on the beat, then breathe back out to
    # the framed view — the letterbox gap becomes a rhythmic device. Same
    # decay curve as _fx_punch_zoom, just inverted intent: punch_zoom adds a
    # small hit on top of a full frame; punch_fill spends its amplitude
    # closing the margins.
    #
    # IMPORTANT: like dutch_tilt, this builder never touches ctx['rng'] — it
    # draws only from its own 'fx_punchfill' stable stream, so inserting it
    # into the curated sequence cannot shift the historical styles' draws.
    if not ctx['target_size'] or ctx['target'] != 'drop':
        return
    if ctx['clip'].get('partner'):
        return  # duo panes have no margins to punch through
    if any('zoompan' in f for f in ctx['filters']):
        return  # one zoompan per segment (the Ken Burns gate idiom)
    video_file = ctx['clip'].get('video_file', '')
    if not video_file:
        return
    # Import inside the builder (the stage6 convention for cross-module use):
    # effects.py stays import-light and the fit-ladder constants/probes can
    # never drift out of lockstep with the geometry this primitive mirrors.
    from ffmpeg_processing import (MAX_CROP_PER_AXIS, SCAN_CROP_LOSS,
                                   get_cached_display_info, plan_smart_crop)
    info = get_cached_display_info(video_file)
    if not info:
        return
    disp_w, disp_h, _sar = info
    tw, th = ctx['target_size']
    if disp_w <= 0 or disp_h <= 0 or tw <= 0 or th <= 0:
        return
    f_fit = min(tw / disp_w, th / disp_h)
    f_fill = max(tw / disp_w, th / disp_h)
    crop_loss = 1.0 - f_fit / f_fill
    if not (MAX_CROP_PER_AXIS < crop_loss <= SCAN_CROP_LOSS):
        return  # outside the hybrid band: no margins (or scan-fit) — no-op
    seed_parts = ['fx_punchfill', ctx['segment_index'], video_file, ctx['style']]
    if not ctx['curated']:
        seed_parts.extend([ctx['mode'], ctx['palette_seed']])
    fill_rng = _stable_rng(*seed_parts)
    # Rare in curated mode (~1 in 4 eligible hybrid-tier drops; only AMV/Hype
    # ever reach the builders); boosted in custom/shuffle, keeping the drop
    # affinity, so a ticked palette entry actually shows up.
    chance = 0.25 if ctx['curated'] else 0.6 * ctx['k']
    if fill_rng.random() >= chance:
        return
    # f_fill/f_fit is the zoom that takes the letterboxed fit to full-bleed.
    # The hybrid foreground is already over-scaled (it traded MAX_CROP_PER_AXIS
    # of crop for smaller margins), so cap at 10% past ITS true full-bleed
    # zoom — max(tw/crop_w, th/crop_h) from the ladder's own geometry — and at
    # 1.6 absolute for sanity.
    ratio = min(f_fill / f_fit, 1.6)
    hybrid = plan_smart_crop((disp_w, disp_h), (tw, th))
    if hybrid:
        _fg_w, _fg_h, crop_w, crop_h = hybrid
        bleed = max(tw / max(2, crop_w), th / max(2, crop_h))
        ratio = min(ratio, 1.1 * bleed)
    base_amp = (ratio - 1.0) * ctx['k']  # intensity blends the punch toward 1
    # loudness gain scales the hit but never past the full-bleed+10% cap
    # established above (ratio - 1.0): min() is a no-op that always resolves
    # to base_amp when gain==1.0 (base_amp <= ratio - 1.0 since ratio >= 1
    # and k <= 1.0), so the absent-'loudness' chain is bit-for-bit unchanged.
    amp = min(ratio - 1.0, base_amp * _loudness_gain(ctx))
    if amp < 0.005:
        return
    fps = ctx['fps']
    zoom_expr = f"1+{amp:.4f}*exp(-(on/{fps:.4f})*9)"
    ctx['filters'].append(
        f"zoompan=z='{zoom_expr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={tw}x{th}:fps={fps}"
    )


def _fx_punch_zoom(ctx) -> None:
    # Punch-in zoom decaying from the cut: the visual "hit" on beat-aligned
    # cuts. crop can't animate w/h, so zoompan (1 output frame per input
    # frame, time reconstructed from the output frame counter) does the zoom.
    if not ((ctx['target'] in ('drop', 'rhythm') or ctx['energy'] > 0.75) and ctx['target_size']):
        return
    if any('zoompan' in f for f in ctx['filters']):
        # One zoompan per segment: punch_fill (which runs first and is the
        # bigger hit) replaces the plain punch on the drops where it fired.
        # Nothing before this builder emitted zoompan historically, so this
        # guard leaves every pre-punch_fill render byte-identical.
        return
    hype, k, fps = ctx['hype'], ctx['k'], ctx['fps']
    amp = (0.16 if hype else 0.10) * k * (0.6 + 0.4 * ctx['energy']) * _loudness_gain(ctx)
    zoom_expr = f"1+{amp:.4f}*exp(-(on/{fps:.4f})*9)"
    ctx['filters'].append(
        f"zoompan=z='{zoom_expr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={ctx['target_size'][0]}x{ctx['target_size'][1]}:fps={fps}"
    )


def _fx_push_pull_zoom(ctx) -> None:
    # Sustained push-in through builds / pull-out on calm releases, ~linear
    # over the whole segment. Needs the planner's segment duration to know the
    # frame count, so it is unavailable for unplanned fallback segments.
    # Custom/shuffle only (never fires in curated mode).
    if ctx['curated'] or not ctx['target_size']:
        return
    if any('zoompan' in f for f in ctx['filters']):
        # One zoompan per segment: punch_fill/punch_zoom (which run earlier in
        # _CUSTOM_SEQUENCE) win over the sustained push/pull if either fired.
        return
    push_selected = 'push_in' in ctx['palette']
    pull_selected = 'pull_out' in ctx['palette']
    duration = 0.0
    try:
        duration = float(ctx['clip'].get('final_duration') or ctx['clip'].get('source_duration') or 0.0)
    except (TypeError, ValueError):
        pass
    if duration <= 0.4 or ctx['energy'] > 0.75:
        return
    direction = None
    if push_selected and ctx['target'] in ('build', 'rhythm'):
        direction = 'in'
    elif pull_selected and ctx['target'] in ('soft', 'flow'):
        direction = 'out'
    if direction is None:
        return
    frames = max(2, int(round(duration * ctx['fps'])))
    amp = 0.12 * ctx['k'] * (0.5 + 0.5 * ctx['energy'])
    if direction == 'in':
        zoom_expr = f"1+{amp:.4f}*on/{frames}"
    else:
        # max() keeps z >= 1 on the final frame despite float rounding.
        zoom_expr = f"max(1,1+{amp:.4f}*(1-on/{frames}))"
    ctx['filters'].append(
        f"zoompan=z='{zoom_expr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={ctx['target_size'][0]}x{ctx['target_size'][1]}:fps={ctx['fps']}"
    )


def _fx_dutch_tilt(ctx) -> None:
    # Dutch tilt: rotate the already-fit frame a few degrees and overscan so
    # no corners/background show — reads as extra energy on hard cuts. The
    # overscan scale is the closed-form cover bound for a WxH frame rotated
    # by angle a: s = cos(a) + max(W/H, H/W)*sin(a) satisfies both axes
    # (W*s >= W*cos+H*sin and H*s >= H*cos+W*sin), so one factor covers
    # landscape and portrait canvases alike. scale/rotate/crop are all 1:1
    # per-frame filters, so the segment frame count is untouched.
    #
    # IMPORTANT: this builder never touches ctx['rng'] — it draws from its
    # own stable rng so inserting it into the curated sequence cannot shift
    # the historical styles' draw order (see _CURATED_SEQUENCE note).
    if not ctx['target_size'] or ctx['pack'] >= ctx['pack_cap']:
        return
    if ctx['target'] not in ('drop', 'rhythm'):
        return
    seed_parts = ['fx_dutch', ctx['segment_index'], ctx['clip'].get('video_file', ''), ctx['style']]
    if not ctx['curated']:
        seed_parts.extend([ctx['mode'], ctx['palette_seed']])
    tilt_rng = _stable_rng(*seed_parts)
    # Rare in curated mode (~1 in 6 eligible drop/rhythm segments, and only
    # AMV/Hype ever reach the builders); boosted in custom/shuffle so a
    # ticked palette entry actually shows up.
    chance = (1.0 / 6.0) if ctx['curated'] else 0.5 * ctx['k']
    if tilt_rng.random() >= chance:
        return
    # 4-8 degrees at full intensity, scaled by k like the other primitives'
    # amplitudes; floored (shake-style) so a fired tilt is never invisible.
    deg = max(1.5, tilt_rng.uniform(4.0, 8.0) * ctx['k'])
    rad = math.radians(deg)
    sign = 1.0 if ctx['segment_index'] % 2 == 0 else -1.0  # alternate by parity
    w, h = ctx['target_size']
    cover = math.cos(rad) + max(w / h, h / w) * math.sin(rad)
    cover *= 1.005  # rounding margin
    sw = int(math.ceil(w * cover / 2.0)) * 2
    sh = int(math.ceil(h * cover / 2.0)) * 2
    beats = ctx['beats']
    if len(beats) >= 2:
        # Beat-alternating flavor: the sign flips every inter-beat interval
        # (same span construction as beat_flip). rotate evaluates 'a' per
        # frame, so this stays a pure expression — no frame-count risk.
        spans = []
        for j in range(0, len(beats), 2):
            span_end = beats[j + 1] if j + 1 < len(beats) else beats[j] + (beats[j] - beats[j - 1])
            spans.append(f"between(t,{beats[j]:.4f},{span_end:.4f})")
        a_expr = f"{sign * rad:.6f}*(1-2*({'+'.join(spans)}))"
    else:
        a_expr = f"{sign * rad:.6f}"
    ctx['filters'].append(
        f"scale={sw}:{sh},rotate=a='{a_expr}':c=black,crop={w}:{h}"
    )
    ctx['pack'] += 1


def _fx_shake(ctx) -> None:
    # Camera shake on drops (hype-only in curated mode).
    if ctx['curated']:
        fire = ctx['hype'] and ctx['target'] == 'drop'
    else:
        fire = ctx['target'] == 'drop'
    if not fire:
        return
    a = max(2, int(round(8 * ctx['k'] * _loudness_gain(ctx))))
    ctx['filters'].append(
        f"crop=w=in_w-{2 * a}:h=in_h-{2 * a}"
        f":x='{a}+{a}*sin(t*41)*exp(-t*3)':y='{a}+{a}*cos(t*57)*exp(-t*3)'"
    )
    ctx['needs_scale_restore'] = True  # crop shrank the frame below target_size


def _fx_white_flash(ctx) -> None:
    # White flash right at drop cuts.
    if ctx['target'] != 'drop':
        return
    amp = (0.55 if ctx['hype'] else 0.35) * ctx['k'] * _loudness_gain(ctx)
    ctx['filters'].append(f"eq=brightness='{amp:.3f}*exp(-t*14)':eval=frame")


def _fx_sat_pulse(ctx) -> None:
    # Saturation pulsing on the beat, on high-energy segments. With real
    # interior beat offsets the pulse fires exactly on each beat; without
    # them it falls back to a sine at the tempo frequency.
    if not (ctx['energy'] > 0.6 and (ctx['tempo'] or ctx['beats'])):
        return
    amp = (0.35 if ctx['hype'] else 0.20) * ctx['k'] * ctx['energy']
    beats = ctx['beats']
    if beats:
        pulses = '+'.join(f"between(t,{b:.4f},{b + 0.18:.4f})" for b in beats)
        ctx['filters'].append(f"eq=saturation='1+{amp:.3f}*({pulses})':eval=frame")
        return
    bps = max(0.5, min(4.0, float(ctx['tempo']) / 60.0))
    ctx['filters'].append(f"eq=saturation='1+{amp:.3f}*sin(2*PI*t*{bps:.4f})':eval=frame")


def _fx_chroma_shift(ctx) -> None:
    # Occasional chromatic aberration on hard segments.
    if ctx['curated']:
        fire = (ctx['hype'] or ctx['target'] == 'drop') and ctx['rng'].random() < 0.35 * ctx['k']
    else:
        fire = ctx['target'] in ('drop', 'rhythm', 'build') and ctx['rng'].random() < 0.5 * ctx['k']
    if not fire:
        return
    shift = 3 if ctx['hype'] else 2
    ctx['filters'].append(f"rgbashift=rh={shift}:bh=-{shift}")


def _fx_pixelize_burst(ctx) -> None:
    # Pixelize burst right at drop cuts, decaying quickly.
    if ctx['curated']:
        fire = (ctx['hype'] and ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.45 * ctx['k'])
    else:
        fire = (ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.6 * ctx['k'])
    if not fire:
        return
    block = max(8, int(round(12 + 20 * ctx['k'])))
    ctx['filters'].append(f"pixelize=w={block}:h={block}:enable='lt(t,0.25)'")
    ctx['pack'] += 1


def _fx_zoom_blur(ctx) -> None:
    # Directional smear standing in for zoom blur on drop cuts.
    if ctx['curated']:
        fire = (ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.30 * ctx['k'])
    else:
        fire = (ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.5 * ctx['k'])
    if not fire:
        return
    radius = max(4, int(round(10 * ctx['k'] * (0.5 + 0.5 * ctx['energy']) * _loudness_gain(ctx))))
    ctx['filters'].append(f"dblur=angle=90:radius={radius}:enable='lt(t,0.3)'")
    ctx['pack'] += 1


def _fx_strobe(ctx) -> None:
    # Negative-flash strobe: two inverted frames out of every eight, and only
    # in the first 0.6s (hype drops with strong impacts in curated mode).
    if ctx['curated']:
        fire = (ctx['hype'] and ctx['target'] == 'drop' and ctx['energy'] > 0.75
                and ctx['pack'] < ctx['pack_cap'] and ctx['rng'].random() < 0.25 * ctx['k'])
    else:
        fire = (ctx['target'] == 'drop' and ctx['energy'] > 0.6
                and ctx['pack'] < ctx['pack_cap'] and ctx['rng'].random() < 0.5 * ctx['k'])
    if not fire:
        return
    ctx['filters'].append("negate=enable='lt(mod(n,8),2)*lt(t,0.6)'")
    ctx['pack'] += 1


def _fx_trails(ctx) -> None:
    # Motion trails on calmer segments: frame-mix echo, or lagfun light-paint.
    if ctx['curated']:
        fire = (ctx['target'] in ('flow', 'soft') and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.30 * ctx['k'])
    else:
        fire = (ctx['target'] in ('flow', 'soft') and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.55 * ctx['k'])
    if not fire:
        return
    lag = (ctx['hype'] if ctx['curated'] else True) and ctx['rng'].random() < 0.5
    if lag:
        ctx['filters'].append(f"lagfun=decay={0.9 + 0.05 * ctx['k']:.3f}")
    else:
        frames = 6 if ctx['energy'] > 0.5 else 4
        weights = ' '.join(str(w) for w in range(frames, 0, -1))
        ctx['filters'].append(f"tmix=frames={frames}:weights='{weights}'")
    ctx['pack'] += 1


def _fx_hue_sweep(ctx) -> None:
    # Slow hue sweep through build-ups.
    if ctx['curated']:
        fire = (ctx['hype'] and ctx['target'] == 'build' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.35 * ctx['k'])
    else:
        fire = (ctx['target'] == 'build' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.6 * ctx['k'])
    if not fire:
        return
    rate = 30 + int(round(60 * ctx['k']))
    ctx['filters'].append(f"hue=h={rate}*t")
    ctx['pack'] += 1


def _fx_fisheye(ctx) -> None:
    # Rare subtle fisheye bulge.
    if ctx['curated']:
        fire = ctx['hype'] and ctx['pack'] < ctx['pack_cap'] and ctx['rng'].random() < 0.12 * ctx['k']
    else:
        fire = ctx['pack'] < ctx['pack_cap'] and ctx['rng'].random() < 0.25 * ctx['k']
    if not fire:
        return
    ctx['filters'].append(f"lenscorrection=k1={-0.15 * ctx['k']:.3f}:k2=-0.05:i=bilinear")
    ctx['pack'] += 1


def _fx_posterize_flash(ctx) -> None:
    # Posterize flash on drops (8 luma levels, first 0.3s).
    if ctx['curated']:
        fire = (ctx['hype'] and ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.15 * ctx['k'])
    else:
        fire = (ctx['target'] == 'drop' and ctx['pack'] < ctx['pack_cap']
                and ctx['rng'].random() < 0.35 * ctx['k'])
    if not fire:
        return
    ctx['filters'].append("lutyuv=y='floor(val/32)*32+16':enable='lt(t,0.3)'")
    ctx['pack'] += 1


def _fx_half_mirror(ctx) -> None:
    # Left-right symmetry: left half reflected onto the right. Emits a small
    # branch-and-merge subgraph, which the ffmpeg graph parser accepts inside
    # a comma-joined chain. Custom/shuffle only; one mirror-family effect per
    # segment (they all restructure the whole frame).
    if ctx['curated'] or ctx['mirror_used'] or ctx['pack'] >= ctx['pack_cap']:
        return
    if not (ctx['target'] in ('rhythm', 'flow', 'build') and ctx['rng'].random() < 0.4 * ctx['k']):
        return
    ctx['filters'].append(
        "crop=iw/2:ih:0:0,split[mfa][mfb];[mfb]hflip[mfbf];[mfa][mfbf]hstack"
    )
    ctx['mirror_used'] = True
    ctx['pack'] += 1


def _fx_kaleido_quad(ctx) -> None:
    # Four-way kaleidoscope: top-left quarter mirrored across both axes.
    if ctx['curated'] or ctx['mirror_used'] or ctx['pack'] >= ctx['pack_cap']:
        return
    if not (ctx['target'] in ('drop', 'build') and ctx['rng'].random() < 0.25 * ctx['k']):
        return
    ctx['filters'].append(
        "crop=iw/2:ih/2:0:0,split[kqa][kqb];[kqb]hflip[kqbf];[kqa][kqbf]hstack,"
        "split[kqc][kqd];[kqd]vflip[kqdf];[kqc][kqdf]vstack"
    )
    ctx['mirror_used'] = True
    ctx['pack'] += 1


def _fx_beat_flip(ctx) -> None:
    # Horizontal flip toggling on the beat (hflip supports timeline enable).
    # With real interior beats the flip holds for every other inter-beat
    # interval; without them it approximates from the tempo.
    if ctx['curated'] or ctx['mirror_used'] or ctx['pack'] >= ctx['pack_cap']:
        return
    if not (ctx['target'] in ('rhythm', 'drop') and ctx['rng'].random() < 0.35 * ctx['k']):
        return
    beats = ctx['beats']
    if len(beats) >= 2:
        spans = []
        for j in range(0, len(beats), 2):
            span_end = beats[j + 1] if j + 1 < len(beats) else beats[j] + (beats[j] - beats[j - 1])
            spans.append(f"between(t,{beats[j]:.4f},{span_end:.4f})")
        ctx['filters'].append(f"hflip=enable='{'+'.join(spans)}'")
    else:
        if ctx['tempo']:
            frames_per_beat = max(4, int(round(ctx['fps'] * 60.0 / max(40.0, float(ctx['tempo'])))))
        else:
            frames_per_beat = 15
        ctx['filters'].append(
            f"hflip=enable='lt(mod(n,{2 * frames_per_beat}),{frames_per_beat})'"
        )
    ctx['mirror_used'] = True
    ctx['pack'] += 1


def _fx_vignette_grain(ctx) -> None:
    # Vignette always, grain sometimes (the hype finishing look).
    if ctx['curated']:
        fire = ctx['hype']
    else:
        fire = True
    if not fire:
        return
    ctx['filters'].append("vignette=PI/5")
    if ctx['rng'].random() < 0.5 * ctx['k']:
        ctx['filters'].append("noise=alls=5:allf=t")


# ---------------------------------------------------------------------------
# Split transitions. Planned by stage 6 as boundary specs; each renders as an
# out-chain on segment A's tail plus an in-chain on segment B's head, so the
# concat stream-copy assembly is untouched. Every filter used here is
# frame-count neutral (crop dims are constant; the rest are enable-gated).
# ---------------------------------------------------------------------------

_TRANSITION_WINDOW = 0.2


def _transition_filters(spec: Dict, side: str, duration: float, k: float,
                        target_size: Optional[Tuple[int, int]],
                        ctx: Optional[Dict] = None) -> List[str]:
    """Build the tail ('transition_out') or head ('transition_in') chain."""
    kind = str(spec.get('type', ''))
    if duration <= 0.3:
        return []
    w = min(_TRANSITION_WINDOW, 0.4 * duration)
    s = duration - w  # window start for out-chains
    k = max(0.4, k)   # transitions are planned; keep them readable at low k
    out_side = side == 'transition_out'
    filters: List[str] = []

    if kind == 'whip_pan':
        # Shake-idiom pan: constant inward crop with an eased x sweep, sold by
        # staggered directional blur. crop has no timeline support, so the
        # small overscan applies to the whole segment; the restore scale at
        # the chain end brings dimensions back to target.
        if not target_size:
            return []
        a = max(12, int(round(target_size[0] * 0.06)))
        rightward = str(spec.get('direction', 'right')) == 'right'
        if out_side:
            prog = f"pow(min(1,max(0,(t-{s:.4f})/{w:.4f})),2)"
            x_expr = f"{a}+{a}*{prog}" if rightward else f"{a}-{a}*{prog}"
            gates = (f"gt(t,{s:.4f})", f"gt(t,{s + w / 2:.4f})")
        else:
            prog = f"pow(1-min(1,t/{w:.4f}),2)"
            x_expr = f"{a}-{a}*{prog}" if rightward else f"{a}+{a}*{prog}"
            gates = (f"lt(t,{w:.4f})", f"lt(t,{w / 2:.4f})")
        radius = max(6, int(round(10 * k)))
        filters.append(f"crop=w=in_w-{2 * a}:h=in_h-{2 * a}:x='{x_expr}':y={a}")
        filters.append(f"dblur=angle=0:radius={radius}:enable='{gates[0]}'")
        filters.append(f"dblur=angle=0:radius={radius * 2}:enable='{gates[1]}'")
        if ctx is not None:
            ctx['needs_scale_restore'] = True  # crop shrank the frame below target_size
    elif kind == 'glitch_cut':
        shift = 6 + int(round(4 * k))
        noise = 14 + int(round(10 * k))
        block = max(8, int(round(10 + 14 * k)))
        if out_side:
            g_full, g_half = f"gt(t,{s:.4f})", f"gt(t,{s + w / 2:.4f})"
        else:
            g_full, g_half = f"lt(t,{w:.4f})", f"lt(t,{w / 2:.4f})"
        filters.append(f"rgbashift=rh={shift}:bh=-{shift}:edge=wrap:enable='{g_full}'")
        filters.append(f"noise=alls={noise}:allf=t:enable='{g_full}'")
        filters.append(f"pixelize=w={block}:h={block}:enable='{g_half}'")
    elif kind in ('dip_black', 'dip_flash'):
        color = ':color=white' if kind == 'dip_flash' else ''
        if out_side:
            filters.append(f"fade=t=out:st={s:.4f}:d={w:.4f}{color}")
        else:
            filters.append(f"fade=t=in:st=0:d={w:.4f}{color}")
    return filters


# Canonical order: curated iterates the classic subset in the historical
# order (rng draw order must not change); custom/shuffle iterate the full
# list. push/pull share one builder so only one zoompan direction fires.
_PRIMITIVES = [
    ('punch_fill', 'Punch to full-bleed (hybrid-fit drops)', _fx_punch_fill),
    ('punch_zoom', 'Punch-in zoom (drop hits)', _fx_punch_zoom),
    ('push_in', 'Slow push-in (builds)', _fx_push_pull_zoom),
    ('pull_out', 'Slow pull-out (calm releases)', _fx_push_pull_zoom),
    ('dutch_tilt', 'Dutch tilt (hard cuts)', _fx_dutch_tilt),
    ('shake', 'Camera shake (drops)', _fx_shake),
    ('white_flash', 'White flash (drop cuts)', _fx_white_flash),
    ('sat_pulse', 'Saturation pulse (on the beat)', _fx_sat_pulse),
    ('chroma_shift', 'Chromatic aberration', _fx_chroma_shift),
    ('pixelize_burst', 'Pixelize burst (drop cuts)', _fx_pixelize_burst),
    ('zoom_blur', 'Zoom-blur smear (drop cuts)', _fx_zoom_blur),
    ('strobe', 'Negative strobe (hard drops)', _fx_strobe),
    ('trails', 'Motion trails (calm segments)', _fx_trails),
    ('hue_sweep', 'Hue sweep (builds)', _fx_hue_sweep),
    ('fisheye', 'Subtle fisheye', _fx_fisheye),
    ('posterize_flash', 'Posterize flash (drop cuts)', _fx_posterize_flash),
    ('half_mirror', 'Half mirror symmetry', _fx_half_mirror),
    ('kaleido_quad', 'Kaleidoscope quad', _fx_kaleido_quad),
    ('beat_flip', 'Beat-flipped mirror', _fx_beat_flip),
    ('vignette_grain', 'Vignette + grain', _fx_vignette_grain),
]

EFFECT_REGISTRY = {pid: {'label': label, 'builder': builder} for pid, label, builder in _PRIMITIVES}

# The classic style pipeline, in its exact historical order. dutch_tilt and
# punch_fill are later insertions, but each draws only from its own stable
# rng (never ctx['rng']), so the historical draw order below is unchanged.
# punch_fill must precede punch_zoom: it emits the segment's one zoompan and
# punch_zoom's guard then yields to it.
_CURATED_SEQUENCE = (
    'punch_fill',
    'punch_zoom', 'dutch_tilt', 'shake', 'white_flash', 'sat_pulse', 'chroma_shift',
    'pixelize_burst', 'zoom_blur', 'strobe', 'trails', 'hue_sweep',
    'fisheye', 'posterize_flash', 'vignette_grain',
)

# push_in/pull_out share a builder; run it once under the 'push_in' slot.
_CUSTOM_SEQUENCE = tuple(
    pid for pid, _, _ in _PRIMITIVES if pid != 'pull_out'
)


def list_effect_choices() -> List[Tuple[str, str]]:
    """(label, id) pairs for the GUI palette picker."""
    return [(entry[1], entry[0]) for entry in _PRIMITIVES]


def resolve_effect_palette(mode: str, selected_ids: Optional[Sequence[str]],
                           seed, audio_file: str = '') -> Tuple[Optional[List[str]], int, str]:
    """Resolve the effect palette for a render.

    Returns (palette or None for curated, resolved_seed, recipe_line). A seed
    of 0 derives from the audio filename so the default is stable per song;
    any other seed is a deterministic re-roll.
    """
    mode = mode if mode in EFFECT_MODES else 'curated'
    try:
        seed = int(seed or 0)
    except (TypeError, ValueError):
        seed = 0
    if seed == 0:
        seed = int(hashlib.sha1(
            os.path.basename(str(audio_file or '')).encode('utf-8', errors='ignore')
        ).hexdigest()[:8], 16) or 1

    all_ids = [pid for pid, _, _ in _PRIMITIVES]
    if mode == 'custom':
        palette = [pid for pid in all_ids if pid in set(selected_ids or [])]
        recipe = f"effects: {','.join(palette) or '(none selected)'} | mode custom"
        return palette, seed, recipe
    if mode == 'shuffle':
        rng = _stable_rng('palette', seed)
        size = 5 + rng.randrange(4)
        palette = sorted(rng.sample(all_ids, min(size, len(all_ids))))
        recipe = f"effects: {','.join(palette)} @ seed {seed} | mode shuffle"
        return palette, seed, recipe
    return None, seed, "effects: curated style presets"


def build_effect_filters(planned_clip: Optional[Dict], style: str, intensity: float,
                         tempo_bpm: Optional[float], segment_index: int,
                         target_size: Optional[Tuple[int, int]],
                         fps: float = 30.0,
                         mode: str = 'curated',
                         palette: Optional[Sequence[str]] = None,
                         palette_seed: int = 0,
                         local_beats: Optional[Sequence[float]] = None,
                         transitions: bool = True) -> List[str]:
    """Return -vf snippets for one segment. Empty list = no effects.

    local_beats are the segment's interior beat times in its own clock; when
    provided, beat-locked primitives gate on the real beats instead of a
    tempo-frequency approximation.
    """
    if not style or style == 'clean':
        return []
    k = _clamp01(intensity, 0.0)
    if k <= 0.0:
        return []

    mode = mode if mode in EFFECT_MODES else 'curated'
    if mode == 'custom' and not palette:
        # Custom with nothing ticked means no effects — falling back to the
        # curated preset here would contradict the "(none selected)" recipe
        # line in the render log.
        return []
    curated = mode == 'curated' or not palette

    clip = planned_clip or {}
    target = str(clip.get('target', 'flow'))
    energy = _clamp01(clip.get('impact', clip.get('wave', 0.5)))
    hype = style == 'hype'

    if curated:
        # Seed parts must stay exactly as the historical styles used them.
        rng = _stable_rng('fx', segment_index, clip.get('video_file', ''), style)
        sequence = _CURATED_SEQUENCE
        palette_set = None
    else:
        rng = _stable_rng('fx', segment_index, clip.get('video_file', ''), style,
                          mode, palette_seed, ','.join(sorted(palette)))
        sequence = _CUSTOM_SEQUENCE
        palette_set = set(palette)

    beats = []
    for b in (local_beats or []):
        try:
            b = float(b)
        except (TypeError, ValueError):
            continue
        if b >= 0.0:
            beats.append(round(b, 4))
    beats = beats[:8]  # cap enable-expression length; segments are short

    ctx = {
        'clip': clip,
        'target': target,
        'energy': energy,
        'k': k,
        'rng': rng,
        'hype': hype,
        'style': style,
        'mode': mode,
        'segment_index': segment_index,
        'palette_seed': palette_seed,
        'tempo': tempo_bpm,
        'beats': beats,
        'target_size': target_size,
        'fps': fps,
        'curated': curated,
        'palette': palette_set or set(),
        'filters': [],
        # Cap so busy segments don't turn into soup. All entries are
        # single-stream and frame-count-safe; shuffleframes is deliberately
        # excluded (drops trailing frames when its pattern doesn't divide the
        # segment frame count) and elbg is excluded (single-threaded, very
        # slow). Commas inside enable='...' are protected by the quoting.
        'pack_cap': 3 if (hype or not curated) else 2,
        'pack': 0,
        'mirror_used': False,
        # Set True only by filters that shrink the frame below target_size
        # (shake's crop; the whip_pan transition's crop) so the trailing
        # restore scale runs only when it's actually needed.
        'needs_scale_restore': False,
    }

    for pid in sequence:
        if palette_set is not None:
            # push_in/pull_out share one builder invoked under 'push_in'.
            if pid == 'push_in':
                if not ({'push_in', 'pull_out'} & palette_set):
                    continue
            elif pid not in palette_set:
                continue
        EFFECT_REGISTRY[pid]['builder'](ctx)

    filters = ctx['filters']

    # Split transitions ride the planned clip (stage 6 labels boundaries) and
    # append after the regular chain: head-of-segment in-chain first, then the
    # tail out-chain.
    if transitions:
        seg_duration = 0.0
        try:
            seg_duration = float(clip.get('final_duration') or 0.0)
        except (TypeError, ValueError):
            pass
        for side in ('transition_in', 'transition_out'):
            spec = clip.get(side)
            if isinstance(spec, dict) and spec.get('type'):
                filters.extend(_transition_filters(spec, side, seg_duration, k, target_size, ctx))

    # Only shake's crop and the whip_pan transition's crop actually shrink the
    # frame below target_size (zoompan sets an explicit s=WxH; dutch_tilt and
    # the mirror/kaleido chains crop-and-recombine back to the input size;
    # everything else is a 1:1 per-pixel filter) — restore exact target
    # dimensions only when one of those fired, so concat sees identical
    # streams without a redundant same-size swscale pass on every segment.
    if filters and target_size and ctx['needs_scale_restore']:
        filters.append(f"scale={target_size[0]}:{target_size[1]}")

    return filters
