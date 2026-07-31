#!/usr/bin/env python3
"""Stage 4: deliberate rhythmic cut selection and final cleanup."""

import math
from typing import Dict, List, Tuple
import numpy as np

from . import AutoWaveConfig
from . import _normalize, _safe_percentile, _unique_sorted
from .contracts import BeatFeatures, Section

# ---------------------------------------------------------------------------
# Pacing dial (cfg.cut_density, 0..1)
# ---------------------------------------------------------------------------
# Everything below is inert at density 0.0: each consumer branches to the
# legacy code path, so a 0.0 render is byte-identical to the pre-dial engine.
#
# The dial exists because three independent mechanisms flattened cut density
# against the music:
#   P1  adaptive_beat_step gated fast cutting on a CONJUNCTION of two
#       independently top-decile GLOBAL statistics (chorus step=1 needed
#       wave > 0.88 AND impact > 0.78 -- jointly ~3% of beats), so per-beat
#       cutting was effectively unreachable. Replaced by a continuous map on a
#       SECTION-LOCAL percentile.
#   P2  final_wave_cleanup capped cuts GLOBALLY and pruned by
#       impact + 0.55*phrase_anchor + 0.32*bar_anchor -- bonuses that dwarf
#       impact's 0..1 range, so the prune stripped non-anchor cuts everywhere
#       and dragged the edit onto a uniform bar grid. Now per-section, with
#       anchor bonuses that shrink as density rises.
#   P3  peak_energy_min_interval = 0.30 was an absolute floor, so at 130 BPM
#       (0.46s beat) per-beat cutting was legal but half-beat was impossible.
#       The cleanup floor is now tempo-relative and per-section.

# How hard a section type WANTS to be cut, before measured energy corrects it.
_SECTION_LABEL_PRIOR: Dict[str, float] = {
    "intro": 0.0, "outro": 0.0, "breakdown": 0.0,
    "verse": 0.35, "bridge": 0.35, "hook": 0.35,
    "chorus": 0.75,
    "drop": 1.00, "finale": 1.00,
}

# Absolute safety floor on any cut interval, in seconds. 6 frames at 30fps;
# the legacy engine already permitted 0.30s (9 frames).
_ABSOLUTE_MIN_GAP = 0.20


def _density(cfg: AutoWaveConfig) -> float:
    """The pacing dial, clamped. 0.0 => every legacy path stays selected."""
    return min(1.0, max(0.0, float(getattr(cfg, "cut_density", 0.0) or 0.0)))


def _raw_intensity(section: Section) -> float:
    """Absolute "how hot is this section", 0..1.

    Deliberately NOT the section label alone. The label comes from the
    structure backend and is not reliable -- on a synthetic click track with
    unmistakable loud/quiet blocks, all-in-one-mlx labelled every one of them
    "intro" while the measured `energy` (section mean wave) tracked the blocks
    exactly (0.02 / 0.31 / 0.78 / 0.16 / 0.84). Keying pacing on the label
    alone would make the whole dial inert whenever the backend guesses wrong,
    and would silently do nothing at all on the librosa fallback path.
    Energy is measured straight off the audio, so it carries the larger
    weight; the label survives as a prior because a quiet chorus should still
    cut like a chorus.
    """
    prior = _SECTION_LABEL_PRIOR.get(section.get("type", "verse"), 0.35)
    energy = min(1.0, max(0.0, float(section.get("energy", 0.5))))
    return min(1.0, max(0.0, 0.35 * prior + 0.65 * energy))


# A track whose hottest section still only reaches this raw intensity is
# treated as genuinely calm rather than scaled up to full density.
_INTENSITY_REFERENCE = 0.55


def _section_intensity(section: Section, sections: List[Section] | None = None) -> float:
    """Section intensity NORMALIZED against the track's own peak.

    Same reasoning as P1's section-local beat percentile, one level up: a
    track's drop should be dense relative to THAT track. Raw intensity does
    not travel, because it depends on how the structure backend labelled the
    section and on how the track's energy happened to normalize -- on the
    smoke track the hottest section reaches only 0.55 raw, which left the
    cleanup imposing a 1.2-beat floor on a drop the selector had just cut per
    beat. Dividing by the track peak makes the tables mean the same thing on
    every track. The _INTENSITY_REFERENCE floor on the divisor stops a
    uniformly calm track from having its quietest peak promoted to "drop".
    """
    raw = _raw_intensity(section)
    if not sections:
        return raw
    peak = max((_raw_intensity(s) for s in sections), default=0.0)
    return min(1.0, max(0.0, raw / max(_INTENSITY_REFERENCE, peak)))


def _step_range(intensity: float) -> Tuple[int, int]:
    """(fastest, slowest) beat step allowed at this section intensity."""
    fast = 2.0 - 1.0 * intensity          # calm 2 -> peak 1
    slow = 4.0 - 1.0 * intensity          # calm 4 -> peak 3
    return int(math.floor(fast + 0.5)), int(math.floor(slow + 0.5))


def _cap_peak(intensity: float) -> float:
    """Cut-count ceiling as a fraction of the section's beats, at full density."""
    return 0.34 + intensity * (1.00 - 0.34)


def _min_scale(intensity: float) -> float:
    """Scaling of the 'do not go too sparse' floor, at full density."""
    return 0.70 + intensity * (1.40 - 0.70)


def _gap_beats(intensity: float, density: float) -> float:
    """Cleanup min-gap in BEAT PERIODS.

    Expressed in beats rather than seconds so half-beat cutting is reachable
    at any tempo instead of only above ~100 BPM (P3).

    The (1-intensity)**1.5 shaping matters: with a plain linear map the floor
    only reached a half beat at intensity EXACTLY 1.0, so a drop sitting at
    0.92 got a 0.55-beat floor -- fractionally wider than the half-beat
    micro-cuts add_rare_micro_cuts had just placed, and the cleanup deleted
    every one of them. The curve concentrates the reduction in the top of the
    intensity range, which is also where relentless cutting belongs.
    """
    shape = (1.0 - intensity) ** 1.5
    lo = 0.95 + 0.95 * shape              # density 0+: peak 0.95 -> calm 1.90
    hi = 0.40 + 1.20 * shape              # density 1:  peak 0.40 -> calm 1.60
    return lo + density * (hi - lo)


def _median_beat_period(beat_times: np.ndarray) -> float:
    diffs = np.diff(np.asarray(beat_times, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.median(diffs)) if diffs.size else 0.5


def _section_local_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] of each value WITHIN its section.

    This is the heart of P1: ranking against the section instead of the whole
    track means a chorus is dense relative to the chorus. A track whose every
    beat is loud no longer reads as uniformly "top decile" and therefore
    uniformly slow. Ties resolve by stable sort order, so it is deterministic.
    """
    n = int(values.size)
    if n <= 1:
        return np.full(n, 0.5, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / float(n - 1)


def _section_of(sections: List[Section], t: float) -> Section | None:
    for section in sections:
        if float(section["start"]) <= t < float(section["end"]):
            return section
    return None


def select_wave_cuts(beat_times: np.ndarray, sections: List[Section], features: BeatFeatures,
                     tempo: float, audio_duration: float,
                     cfg: AutoWaveConfig) -> Tuple[np.ndarray, List[Dict]]:
    selected: List[float] = []
    info: List[Dict] = []

    for section in sections:
        beat_indices = np.where((beat_times >= section["start"]) & (beat_times < section["end"]))[0]
        if beat_indices.size == 0:
            continue

        section_selected = select_section_wave_cuts(beat_indices, beat_times, features, section,
                                                    cfg, sections=sections)
        selected.extend(section_selected)
        info.append({
            "section": section,
            "selected_count": len(section_selected),
            "beat_count": int(beat_indices.size),
            "density": len(section_selected) / max(1, int(beat_indices.size)),
        })

    # Rare micro-cuts only after the stable main grid is built.
    selected_arr = np.asarray(selected, dtype=float)
    selected_arr = add_rare_micro_cuts(selected_arr, beat_times, features, audio_duration, cfg,
                                       sections=sections)

    # Cleanup: fewer cuts, exact rhythm, no jitter. Global at density 0,
    # per-section once the dial is up (P2).
    selected_arr = final_wave_cleanup(selected_arr, beat_times, features, audio_duration, cfg,
                                      sections=sections)
    return selected_arr, info


def select_section_wave_cuts(beat_indices: np.ndarray, beat_times: np.ndarray,
                             features: BeatFeatures, section: Section,
                             cfg: AutoWaveConfig,
                             sections: List[Section] | None = None) -> List[float]:
    section_type = section.get("type", "verse")
    pattern = section.get("dominant_pattern", "mixed")
    selected: List[float] = []

    scores = compute_cut_scores(beat_indices, features, section, cfg)
    score_map = {int(idx): float(score) for idx, score in zip(beat_indices, scores)}

    density = _density(cfg)
    beat_period = _median_beat_period(beat_times)
    intensity = _section_intensity(section, sections)

    # P1: rank this section's beats against EACH OTHER, so the step map reads a
    # section-local percentile instead of two global top-decile thresholds.
    rank_map: Dict[int, float] = {}
    if density > 0.0:
        drive = (0.60 * np.asarray(features["wave"][beat_indices], dtype=float)
                 + 0.40 * np.asarray(features["impact_score"][beat_indices], dtype=float))
        rank_map = {int(idx): float(r)
                    for idx, r in zip(beat_indices, _section_local_rank(drive))}

    # The "let the shot breathe" gate below suppresses cuts on any beat scoring
    # under a fixed 0.42. That threshold cannot mean the same thing in every
    # section: compute_cut_scores adds the anchor bonuses (0.32 bar / 0.48
    # phrase) BEFORE _normalize, so anchors saturate near 1.0 and every
    # non-anchor beat is squashed toward 0. The fixed gate therefore rejects
    # essentially all non-anchor beats, and cuts only happen when max_hold
    # forces one -- which lands on the bar grid. That, not the step map, is
    # what pinned the edit to a uniform grid. On the density path the gate
    # becomes a section-local percentile instead, so it always means "the
    # weakest N% of THIS section" and N shrinks as the dial rises.
    if density > 0.0:
        weak_pct = 40.0 * (1.0 - 0.88 * density)      # 40% -> ~5%
        weak_score_gate = _safe_percentile(scores, weak_pct, 0.0)
    else:
        weak_score_gate = 0.42

    current_pos = 0
    last_cut_time = -999.0
    max_hold = max_hold_for_section(section, cfg, sections)

    # First cut in a section: use section start if there is a good downbeat nearby.
    first_idx = choose_best_nearby(beat_indices, 0, radius=1, scores=score_map, features=features)
    if first_idx is not None:
        selected.append(float(beat_times[first_idx]))
        last_cut_time = float(beat_times[first_idx])
        current_pos = max(0, int(np.where(beat_indices == first_idx)[0][0]))

    while current_pos < beat_indices.size - 1:
        local_idx = int(beat_indices[current_pos])
        wave = float(features["wave"][local_idx])
        impact = float(features["impact_score"][local_idx])
        if density > 0.0:
            step = adaptive_beat_step_density(
                rank_map.get(local_idx, 0.5), intensity, pattern, density)
        else:
            step = adaptive_beat_step(wave, impact, section_type, pattern)

        target_pos = min(beat_indices.size - 1, current_pos + step)
        target_idx = choose_best_nearby(
            beat_indices,
            target_pos,
            radius=1 if step <= 2 else 2,
            scores=score_map,
            features=features,
            # Never re-select the beat we just cut on (see choose_best_nearby).
            min_pos=current_pos + 1 if density > 0.0 else 0,
            # Snapping every target onto the nearest bar/phrase anchor is the
            # other half of the bar-grid magnet; ease it off as density rises
            # so fast steps can land on off-anchor beats.
            anchor_pull=1.0 - 0.6 * density,
        )
        if target_idx is None:
            break

        target_time = float(beat_times[target_idx])
        min_gap = min_interval_for_wave(float(features["wave"][target_idx]), section_type, cfg,
                                        beat_period=beat_period,
                                        intensity=intensity if density > 0.0 else None)

        # If too close, move one more rhythmic step forward instead of cutting fast.
        if target_time - last_cut_time < min_gap:
            current_pos = min(beat_indices.size - 1, target_pos + 1)
            continue

        # If the target score is weak and we are not exceeding max hold, let the
        # shot breathe until the next cleaner beat.
        score = score_map.get(int(target_idx), 0.0)
        if score < weak_score_gate and target_time - last_cut_time < max_hold:
            current_pos = target_pos
            continue

        selected.append(target_time)
        last_cut_time = target_time
        current_pos = int(np.where(beat_indices == target_idx)[0][0])

        # Safety: if we somehow hold too long, force a clean bar/downbeat.
        # This scans the WHOLE future after every cut and jumps to the first
        # beat that would breach max_hold, so on the density path it was
        # leapfrogging the step map -- with every step requesting 1 it still
        # produced 8,9,12,13 instead of 8,9,10,11. It is only a safety net for
        # slow cutting, so at step 1-2 (where a hold cannot run long) skip it.
        skip_forced = density > 0.0 and step <= 2
        if current_pos < beat_indices.size - 1 and not skip_forced:
            future = beat_indices[current_pos + 1:]
            if future.size:
                future_times = beat_times[future]
                too_far = np.where(future_times - last_cut_time >= max_hold)[0]
                if too_far.size and (future_times[too_far[0]] - last_cut_time) > max_hold * 1.15:
                    forced_pos = current_pos + 1 + int(too_far[0])
                    forced_idx = choose_best_nearby(beat_indices, forced_pos, radius=2, scores=score_map, features=features)
                    if forced_idx is not None:
                        forced_time = float(beat_times[forced_idx])
                        if forced_time - last_cut_time >= min_gap:
                            selected.append(forced_time)
                            last_cut_time = forced_time
                            current_pos = int(np.where(beat_indices == forced_idx)[0][0])

    return selected


def adaptive_beat_step(wave: float, impact: float, section_type: str, pattern: str) -> int:
    """Map musical energy to beat-step spacing. Bigger wave => faster, but still grid-like."""
    # Base steps are intentionally calmer than V3/V3.1.
    if section_type in {"intro", "outro", "breakdown"}:
        if wave > 0.82 and impact > 0.76:
            return 2
        if wave > 0.55:
            return 3
        return 4

    if section_type in {"verse", "bridge", "hook"}:
        if wave > 0.84 and impact > 0.72:
            return 2
        if wave > 0.56:
            return 3
        return 4

    if section_type in {"chorus"}:
        if wave > 0.88 and impact > 0.78:
            return 1
        if wave > 0.58:
            return 2
        return 3

    if section_type in {"drop", "finale"}:
        if wave > 0.90 and impact > 0.82 and pattern in {"kick_clap", "kick", "bass"}:
            return 1
        if wave > 0.62:
            return 2
        return 3

    return 3


def adaptive_beat_step_density(rank: float, intensity: float, pattern: str,
                               density: float) -> int:
    """P1: continuous section-local percentile -> beat step.

    ``rank`` is the beat's percentile within its own section (0 = calmest,
    1 = most driving); ``intensity`` is how hard the section as a whole should
    be cut. Two things vary with the dial: the SLOPE (how much of the
    section's rank range is spent reaching the fast end) and the OFFSET (how
    far up the curve the whole section sits). At density 1 a drop's top ~40%
    of beats all reach step 1, which is the relentless per-beat cutting the
    legacy conjunction made unreachable.
    """
    fast, slow = _step_range(intensity)
    # The dial's effect is itself scaled by intensity. Without this the offset
    # term lifts EVERY section toward its fast end, so a breakdown ends up
    # cutting almost as hard as the drop and the contrast the wave exists to
    # create collapses at exactly the setting meant to maximise it.
    eff = density * (0.30 + 0.70 * intensity)
    drive = rank * (0.75 + 0.85 * eff) + 0.35 * eff
    drive = min(1.0, max(0.0, drive))
    step = slow - drive * (slow - fast)

    # Percussion-led sections earn fast cutting; a hi-hat or melodic bed alone
    # should not, which is the one gate worth keeping from the legacy ladder.
    if pattern in {"hihat", "mixed"}:
        step += 0.5 * density

    return int(min(slow, max(fast, math.floor(step + 0.5))))


def choose_best_nearby(beat_indices: np.ndarray, target_pos: int, radius: int,
                       scores: Dict[int, float], features: BeatFeatures,
                       min_pos: int = 0, anchor_pull: float = 1.0) -> int | None:
    """Pick the best-scoring beat near ``target_pos``.

    ``min_pos`` clamps the search to beats at or after it. It exists for
    step=1 cutting: with the default window the candidates are
    [current_pos, current_pos+2], so the beat we are ALREADY sitting on is
    eligible -- and since we only just cut there it carries an anchor bonus,
    usually wins, then fails the min-gap test and the loop skips forward
    without cutting at all. That silently converted per-beat cutting into
    per-bar cutting. Defaults to 0 so the legacy path is unchanged.
    """
    if beat_indices.size == 0:
        return None
    lo = max(0, min_pos, target_pos - radius)
    hi = min(beat_indices.size, target_pos + radius + 1)
    if lo >= hi:
        return None
    candidates = beat_indices[lo:hi]
    if candidates.size == 0:
        return None

    def candidate_score(idx: int) -> float:
        s = scores.get(int(idx), 0.0)
        # Strongly prefer bar/phrase anchors when nearby. This makes cuts feel
        # locked to the music instead of slightly mismatched.
        if bool(features["is_phrase_anchor"][idx]):
            s += 0.22 * anchor_pull
        elif bool(features["is_bar_anchor"][idx]):
            s += 0.12 * anchor_pull
        # Do not move too far from the intended rhythmic target.
        pos_penalty = abs(int(np.where(beat_indices == idx)[0][0]) - target_pos) * 0.06
        return s - pos_penalty

    best = max(candidates.tolist(), key=candidate_score)
    return int(best)


def compute_cut_scores(beat_indices: np.ndarray, features: BeatFeatures, section: Section,
                       cfg: AutoWaveConfig) -> np.ndarray:
    idx = beat_indices
    impact = features["impact_score"][idx]
    wave = features["wave"][idx]
    rhythm = features["rhythm_score"][idx]
    novelty = features["novelty"][idx]

    # Wave-13: SuperFlux onset strength joins the base score (novelty 0.10->0.06,
    # arc 0.07->0.06 rebalanced down to make room; superflux weighted 0.10).
    # Wave-14: when the demucs drum stem is available, its per-beat drum onset is
    # the same transient signal uncontaminated by melody/vocals, so it takes over
    # the 0.10 onset weight IN PLACE of SuperFlux. Absent (Windows/no-stem) =>
    # SuperFlux stays, byte-identical to wave 13.
    if "drum_onset" in features:
        onset_term = 0.10 * np.asarray(features["drum_onset"][idx], dtype=float)
    else:
        onset_term = 0.10 * features["onset_superflux"][idx]
    score = (0.36 * impact + 0.27 * rhythm + 0.20 * wave
             + 0.06 * novelty + 0.06 * features["arc"][idx]
             + onset_term)

    # Wave-14: when the vocal stem is available, discourage cutting mid-vocal-
    # phrase unless the bar structure justifies it. Deliberately conservative
    # (-0.06) and applied only to beats that are NEITHER phrase nor bar anchors.
    # Absent => no penalty, byte-identical to wave 13.
    if "vocal_presence" in features:
        off_anchor = ~(features["is_bar_anchor"][idx] | features["is_phrase_anchor"][idx])
        score = score - 0.06 * np.asarray(features["vocal_presence"][idx], dtype=float) * off_anchor

    score = score.copy()
    score[features["is_bar_anchor"][idx]] += cfg.anchor_bonus
    score[features["is_phrase_anchor"][idx]] += cfg.phrase_bonus

    section_type = section.get("type", "verse")
    pattern = section.get("dominant_pattern", "mixed")

    # Wave-13: harmonic-change (HCDF) bonus only where percussion is weak, i.e.
    # 'mixed' pattern in the quieter/melodic section types. Chord-change cuts
    # matter there; drops/choruses stay percussion-driven.
    if pattern == "mixed" and section_type in {"intro", "outro", "breakdown", "verse", "bridge"}:
        score = score + 0.12 * features["harmonic_change"][idx]

    if pattern == "kick_clap":
        score += 0.12 * features["is_strong_kick"][idx] + 0.10 * features["is_strong_clap"][idx]
    elif pattern == "kick":
        score += 0.16 * features["is_strong_kick"][idx]
    elif pattern == "bass":
        score += 0.16 * features["is_strong_bass"][idx]
    elif pattern == "clap":
        score += 0.14 * features["is_strong_clap"][idx]
    elif pattern == "hihat":
        # Hi-hat alone should not cause frantic switching.
        score += 0.05 * features["is_strong_hihat"][idx]

    return _normalize(score)


def min_interval_for_wave(wave: float, section_type: str, cfg: AutoWaveConfig,
                          beat_period: float | None = None,
                          intensity: float | None = None) -> float:
    density = _density(cfg)
    if density <= 0.0 or intensity is None:
        # Legacy floors, keyed on the section label.
        if section_type in {"intro", "outro", "breakdown"}:
            return cfg.low_energy_min_interval
        if wave >= 0.90 and section_type in {"drop", "finale"}:
            return cfg.peak_energy_min_interval
        if wave >= 0.68:
            return cfg.high_energy_min_interval
        if wave >= 0.38:
            return cfg.medium_energy_min_interval
        return cfg.low_energy_min_interval

    # Density path: pick the floor from measured signals, not the label. The
    # label branch above returns the 0.90s low-energy floor for any section
    # tagged intro/outro/breakdown, which is ~2 beats at 130 BPM -- on its own
    # enough to cap density hard, and it fires on every section whenever the
    # structure backend mislabels (see _section_intensity).
    level = max(float(wave), float(intensity))
    if level >= 0.80:
        base = cfg.peak_energy_min_interval
    elif level >= 0.62:
        base = cfg.high_energy_min_interval
    elif level >= 0.35:
        base = cfg.medium_energy_min_interval
    else:
        # Genuinely calm material keeps its full hold at every density -- the
        # dial sharpens the loud/quiet CONTRAST, it does not raise the floor
        # everywhere.
        return cfg.low_energy_min_interval

    # P3: relax the medium/high/peak floors with the dial, but never below a
    # half beat. At 130 BPM the legacy 0.30s peak floor made half-beat cutting
    # (0.23s) structurally impossible while per-beat (0.46s) was legal.
    relaxed = base * (1.0 - 0.55 * density)
    if beat_period and beat_period > 0.0:
        relaxed = max(relaxed, beat_period * 0.46)
    return max(_ABSOLUTE_MIN_GAP, relaxed)


def max_hold_for_section(section: Section, cfg: AutoWaveConfig,
                         sections: List[Section] | None = None) -> float:
    section_type = section.get("type", "verse")
    energy = float(section.get("energy", 0.5))

    # Density path: interpolate the hold ceiling on measured intensity for the
    # same reason the min-interval floor does -- a mislabelled section would
    # otherwise inherit the 3.80s low-energy hold and never force a cut.
    density = _density(cfg)
    if density > 0.0:
        intensity = _section_intensity(section, sections)
        legacy_span = cfg.low_energy_max_hold - cfg.peak_energy_max_hold
        return cfg.low_energy_max_hold - intensity * legacy_span

    if section_type in {"intro", "outro", "breakdown"}:
        return cfg.low_energy_max_hold
    if section_type in {"drop", "finale"} and energy >= 0.78:
        return cfg.peak_energy_max_hold
    if section_type in {"chorus", "drop", "finale"}:
        return cfg.high_energy_max_hold
    if energy >= 0.55:
        return cfg.medium_energy_max_hold
    return cfg.low_energy_max_hold


def add_rare_micro_cuts(selected: np.ndarray, beat_times: np.ndarray, features: BeatFeatures,
                        audio_duration: float, cfg: AutoWaveConfig,
                        sections: List[Section] | None = None) -> np.ndarray:
    """Add half-beat cuts on huge impacts.

    At density 0 these stay "almost disabled" by design (2.5% of cuts, gated at
    the 96.5th percentile). P3 opens the budget as the dial rises, but only
    inside high-intensity sections -- a micro-cut in a verse just reads as a
    mistake, whereas on a drop it is the whole point.
    """
    if not cfg.enable_rare_micro_cuts or len(beat_times) < 3 or selected.size == 0:
        return selected

    beat_diffs = np.diff(beat_times)
    median_beat = float(np.median(beat_diffs)) if beat_diffs.size else 0.5
    if median_beat < 0.22:
        return selected

    density = _density(cfg)
    micro_percentile = cfg.micro_percentile - density * 10.5   # 96.5 -> 86.0
    micro_ratio = cfg.max_micro_cut_ratio + density * 0.115    # 0.025 -> 0.14
    wave_gate = 0.88 - density * 0.18                          # 0.88 -> 0.70

    threshold = _safe_percentile(features["impact_score"], micro_percentile, 0.97)
    max_extra = int(max(0, round(len(selected) * micro_ratio)))
    if max_extra <= 0:
        return selected

    # Which sections may host a micro-cut. An absolute intensity threshold does
    # not travel: it depends on the structure backend's labels and on how the
    # track's energy happens to normalize. Gate on the track's OWN peak
    # instead, with a floor so a uniformly calm track gets none at all.
    # Intensity is already normalized to the track peak, so the hottest
    # section sits at 1.0 and this is simply "the top slice of this track".
    micro_floor = 0.75

    extras: List[float] = []
    selected_sorted = np.sort(selected)

    candidates = np.where((features["impact_score"] >= threshold)
                          & (features["wave"] >= wave_gate))[0]
    for idx in candidates:
        if len(extras) >= max_extra or idx >= len(beat_times) - 1:
            break
        t = float(beat_times[idx] + 0.5 * (beat_times[idx + 1] - beat_times[idx]))
        if t <= 0.0 or t >= audio_duration:
            continue
        if density > 0.0 and sections:
            host = _section_of(sections, t)
            if host is None or _section_intensity(host, sections) < micro_floor:
                continue
        # The 0.34s micro_min_gap is wider than a half beat below ~88 BPM, so
        # once P1 is cutting per beat it would reject every micro-cut on the
        # drops this is meant to serve. Relax it with the dial; the
        # half-beat-of-the-local-tempo term stays the real floor.
        micro_gap = max(cfg.micro_min_gap * (1.0 - 0.8 * density), median_beat * 0.45)
        nearest = np.min(np.abs(selected_sorted - t)) if selected_sorted.size else 999.0
        if nearest >= micro_gap:
            extras.append(t)

    if not extras:
        return selected
    return np.concatenate([selected, np.asarray(extras, dtype=float)])


def _nearest_beat_indices(beat_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``[int(np.argmin(np.abs(beat_times - t))) for t in times]``.

    Assumes beat_times is sorted ascending (the beat grid always is). The
    distance from a sorted array to any query point is minimized at one of
    the two neighbors straddling ``t`` found by np.searchsorted, so those two
    are the only VALUES that can attain the minimum. np.argmin returns the
    FIRST minimal index, so both tie cases must resolve to the earliest one:
    an exact midpoint tie between two distinct beats picks the earlier beat
    (``left_dist <= right_dist`` prefers left), and duplicate beat values
    must map to the first index holding that value -- a second searchsorted
    with side="left" on the winning value does exactly that.
    """
    beat_times = np.asarray(beat_times, dtype=float)
    n = beat_times.size
    times = np.asarray(times, dtype=float)
    if n <= 1:
        return np.zeros(times.shape, dtype=int)
    raw = np.searchsorted(beat_times, times, side="left")
    left = np.clip(raw - 1, 0, n - 1)
    right = np.clip(raw, 0, n - 1)
    left_dist = np.abs(beat_times[left] - times)
    right_dist = np.abs(beat_times[right] - times)
    nearest_val = np.where(left_dist <= right_dist,
                           beat_times[left], beat_times[right])
    # First occurrence of the winning value = argmin's first-minimum index.
    return np.searchsorted(beat_times, nearest_val, side="left")


def final_wave_cleanup(selected: np.ndarray, beat_times: np.ndarray, features: BeatFeatures,
                       audio_duration: float, cfg: AutoWaveConfig,
                       sections: List[Section] | None = None) -> np.ndarray:
    arr = np.asarray(selected, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr > 0.0) & (arr < audio_duration)]
    if arr.size == 0:
        return arr

    density = _density(cfg)
    if density > 0.0 and sections:
        return _final_wave_cleanup_sectioned(arr, beat_times, features, cfg, sections, density)
    return _final_wave_cleanup_global(arr, beat_times, features, cfg)


def _final_wave_cleanup_global(arr: np.ndarray, beat_times: np.ndarray,
                               features: BeatFeatures, cfg: AutoWaveConfig) -> np.ndarray:
    """Legacy (density 0) cleanup, unchanged."""
    # Main global no-flicker pass. The effective minimum changes with energy, but
    # this conservative base removes most too-fast switches.
    base_gap = cfg.peak_energy_min_interval
    arr = _unique_sorted(arr, base_gap)

    # Global ratio cap: if still too dense, keep strongest/most anchored cuts.
    max_allowed = int(max(1, round(len(beat_times) * cfg.target_cut_ratio_max)))
    min_allowed = int(max(1, round(len(beat_times) * cfg.target_cut_ratio_min)))
    if arr.size > max_allowed:
        keep_scores = []
        nearest_idx = _nearest_beat_indices(beat_times, arr)
        for idx in nearest_idx:
            idx = int(idx)
            s = float(features["impact_score"][idx])
            if bool(features["is_phrase_anchor"][idx]):
                s += 0.55
            elif bool(features["is_bar_anchor"][idx]):
                s += 0.32
            keep_scores.append(s)
        order = np.argsort(keep_scores)[::-1][:max_allowed]
        arr = np.sort(arr[order])

    # Make sure it does not become *too* sparse by adding clean phrase/bar anchors.
    if arr.size < min_allowed and len(beat_times) > 0:
        candidates = []
        for idx, t in enumerate(beat_times):
            if bool(features["is_phrase_anchor"][idx]) or bool(features["is_bar_anchor"][idx]):
                candidates.append((float(features["impact_score"][idx]), float(t)))
        for _, t in sorted(candidates, reverse=True):
            if arr.size >= min_allowed:
                break
            if np.min(np.abs(arr - t)) >= cfg.medium_energy_min_interval:
                arr = np.sort(np.append(arr, t))

    # Last pass with a slightly relaxed gap so high-energy drops can still breathe fast.
    arr = _unique_sorted(arr, cfg.peak_energy_min_interval)
    return arr


def _keep_scores(nearest_idx: np.ndarray, features: BeatFeatures,
                 phrase_bonus: float, bar_bonus: float) -> List[float]:
    scores: List[float] = []
    for idx in nearest_idx:
        idx = int(idx)
        s = float(features["impact_score"][idx])
        if bool(features["is_phrase_anchor"][idx]):
            s += phrase_bonus
        elif bool(features["is_bar_anchor"][idx]):
            s += bar_bonus
        scores.append(s)
    return scores


def _final_wave_cleanup_sectioned(arr: np.ndarray, beat_times: np.ndarray,
                                  features: BeatFeatures, cfg: AutoWaveConfig,
                                  sections: List[Section], density: float) -> np.ndarray:
    """P2: per-section cap, prune and min-gap, budgeted from section energy.

    The global version capped the whole track at one ratio and pruned by
    ``impact + 0.55*phrase + 0.32*bar``. Because those bonuses dwarf impact's
    0..1 range, the prune was effectively "keep anchors, drop everything else"
    applied uniformly -- so the denser a section got, the harder it was pulled
    back onto the bar grid, cancelling exactly the contrast P1 builds. Here
    each section gets its own budget, and the anchor bonuses shrink as the
    dial rises so impact decides which cuts survive.
    """
    beat_period = _median_beat_period(beat_times)
    phrase_bonus = 0.55 * (1.0 - 0.75 * density)
    bar_bonus = 0.32 * (1.0 - 0.75 * density)

    kept: List[np.ndarray] = []
    gaps: List[float] = []

    for section in sections:
        start = float(section["start"])
        end = float(section["end"])
        intensity = _section_intensity(section, sections)

        sec_beat_idx = np.where((beat_times >= start) & (beat_times < end))[0]
        sec_cuts = arr[(arr >= start) & (arr < end)]
        if sec_cuts.size == 0:
            continue
        n_beats = int(sec_beat_idx.size)
        if n_beats == 0:
            kept.append(sec_cuts)
            continue

        # P3: tempo-relative no-flicker floor, scaled by section intensity.
        gap = max(_ABSOLUTE_MIN_GAP, beat_period * _gap_beats(intensity, density))
        sec_cuts = _unique_sorted(sec_cuts, gap)
        gaps.append(gap)

        # Cap, budgeted from the section's own intensity.
        ratio_max = ((1.0 - density) * cfg.target_cut_ratio_max
                     + density * _cap_peak(intensity))
        max_allowed = int(max(1, round(n_beats * ratio_max)))
        if sec_cuts.size > max_allowed:
            nearest_idx = _nearest_beat_indices(beat_times, sec_cuts)
            scores = _keep_scores(nearest_idx, features, phrase_bonus, bar_bonus)
            order = np.argsort(scores)[::-1][:max_allowed]
            sec_cuts = np.sort(sec_cuts[order])

        # Floor, so a section never collapses to nothing.
        ratio_min = cfg.target_cut_ratio_min * ((1.0 - density)
                                                + density * _min_scale(intensity))
        min_allowed = int(max(1, round(n_beats * ratio_min)))
        if sec_cuts.size < min_allowed:
            candidates = []
            for idx in sec_beat_idx:
                idx = int(idx)
                if bool(features["is_phrase_anchor"][idx]) or bool(features["is_bar_anchor"][idx]):
                    candidates.append((float(features["impact_score"][idx]), float(beat_times[idx])))
            for _, t in sorted(candidates, reverse=True):
                if sec_cuts.size >= min_allowed:
                    break
                if sec_cuts.size == 0 or np.min(np.abs(sec_cuts - t)) >= gap:
                    sec_cuts = np.sort(np.append(sec_cuts, t))

        kept.append(sec_cuts)

    # Cuts outside every section (gaps in the section map) pass through
    # untouched rather than being silently dropped.
    if sections:
        covered = np.zeros(arr.shape, dtype=bool)
        for section in sections:
            covered |= (arr >= float(section["start"])) & (arr < float(section["end"]))
        if (~covered).any():
            kept.append(arr[~covered])

    if not kept:
        return arr
    out = np.sort(np.concatenate(kept))

    # One cross-boundary pass at the most permissive gap in play, so two
    # sections meeting cannot produce a flicker pair the per-section passes
    # never compared against each other.
    return _unique_sorted(out, max(_ABSOLUTE_MIN_GAP, min(gaps) if gaps else _ABSOLUTE_MIN_GAP))
