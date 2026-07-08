#!/usr/bin/env python3
"""Stage 4: deliberate rhythmic cut selection and final cleanup."""

from typing import Dict, List, Tuple
import numpy as np

from . import AutoWaveConfig
from . import _normalize, _safe_percentile, _unique_sorted

def select_wave_cuts(beat_times: np.ndarray, sections: List[Dict], features: Dict,
                     tempo: float, audio_duration: float,
                     cfg: AutoWaveConfig) -> Tuple[np.ndarray, List[Dict]]:
    selected: List[float] = []
    info: List[Dict] = []

    for section in sections:
        beat_indices = np.where((beat_times >= section["start"]) & (beat_times < section["end"]))[0]
        if beat_indices.size == 0:
            continue

        section_selected = select_section_wave_cuts(beat_indices, beat_times, features, section, cfg)
        selected.extend(section_selected)
        info.append({
            "section": section,
            "selected_count": len(section_selected),
            "beat_count": int(beat_indices.size),
            "density": len(section_selected) / max(1, int(beat_indices.size)),
        })

    # Rare micro-cuts only after the stable main grid is built.
    selected_arr = np.asarray(selected, dtype=float)
    selected_arr = add_rare_micro_cuts(selected_arr, beat_times, features, audio_duration, cfg)

    # Global cleanup: fewer cuts, exact rhythm, no jitter.
    selected_arr = final_wave_cleanup(selected_arr, beat_times, features, audio_duration, cfg)
    return selected_arr, info


def select_section_wave_cuts(beat_indices: np.ndarray, beat_times: np.ndarray,
                             features: Dict, section: Dict,
                             cfg: AutoWaveConfig) -> List[float]:
    section_type = section.get("type", "verse")
    pattern = section.get("dominant_pattern", "mixed")
    selected: List[float] = []

    scores = compute_cut_scores(beat_indices, features, section, cfg)
    score_map = {int(idx): float(score) for idx, score in zip(beat_indices, scores)}

    current_pos = 0
    last_cut_time = -999.0
    max_hold = max_hold_for_section(section, cfg)

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
        step = adaptive_beat_step(wave, impact, section_type, pattern)

        target_pos = min(beat_indices.size - 1, current_pos + step)
        target_idx = choose_best_nearby(
            beat_indices,
            target_pos,
            radius=1 if step <= 2 else 2,
            scores=score_map,
            features=features,
        )
        if target_idx is None:
            break

        target_time = float(beat_times[target_idx])
        min_gap = min_interval_for_wave(float(features["wave"][target_idx]), section_type, cfg)

        # If too close, move one more rhythmic step forward instead of cutting fast.
        if target_time - last_cut_time < min_gap:
            current_pos = min(beat_indices.size - 1, target_pos + 1)
            continue

        # If the target score is weak and we are not exceeding max hold, let the
        # shot breathe until the next cleaner beat.
        score = score_map.get(int(target_idx), 0.0)
        if score < 0.42 and target_time - last_cut_time < max_hold:
            current_pos = target_pos
            continue

        selected.append(target_time)
        last_cut_time = target_time
        current_pos = int(np.where(beat_indices == target_idx)[0][0])

        # Safety: if we somehow hold too long, force a clean bar/downbeat.
        if current_pos < beat_indices.size - 1:
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


def choose_best_nearby(beat_indices: np.ndarray, target_pos: int, radius: int,
                       scores: Dict[int, float], features: Dict) -> int | None:
    if beat_indices.size == 0:
        return None
    lo = max(0, target_pos - radius)
    hi = min(beat_indices.size, target_pos + radius + 1)
    candidates = beat_indices[lo:hi]
    if candidates.size == 0:
        return None

    def candidate_score(idx: int) -> float:
        s = scores.get(int(idx), 0.0)
        # Strongly prefer bar/phrase anchors when nearby. This makes cuts feel
        # locked to the music instead of slightly mismatched.
        if bool(features["is_phrase_anchor"][idx]):
            s += 0.22
        elif bool(features["is_bar_anchor"][idx]):
            s += 0.12
        # Do not move too far from the intended rhythmic target.
        pos_penalty = abs(int(np.where(beat_indices == idx)[0][0]) - target_pos) * 0.06
        return s - pos_penalty

    best = max(candidates.tolist(), key=candidate_score)
    return int(best)


def compute_cut_scores(beat_indices: np.ndarray, features: Dict, section: Dict,
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


def min_interval_for_wave(wave: float, section_type: str, cfg: AutoWaveConfig) -> float:
    if section_type in {"intro", "outro", "breakdown"}:
        return cfg.low_energy_min_interval
    if wave >= 0.90 and section_type in {"drop", "finale"}:
        return cfg.peak_energy_min_interval
    if wave >= 0.68:
        return cfg.high_energy_min_interval
    if wave >= 0.38:
        return cfg.medium_energy_min_interval
    return cfg.low_energy_min_interval


def max_hold_for_section(section: Dict, cfg: AutoWaveConfig) -> float:
    section_type = section.get("type", "verse")
    energy = float(section.get("energy", 0.5))
    if section_type in {"intro", "outro", "breakdown"}:
        return cfg.low_energy_max_hold
    if section_type in {"drop", "finale"} and energy >= 0.78:
        return cfg.peak_energy_max_hold
    if section_type in {"chorus", "drop", "finale"}:
        return cfg.high_energy_max_hold
    if energy >= 0.55:
        return cfg.medium_energy_max_hold
    return cfg.low_energy_max_hold


def add_rare_micro_cuts(selected: np.ndarray, beat_times: np.ndarray, features: Dict,
                        audio_duration: float, cfg: AutoWaveConfig) -> np.ndarray:
    """Add extremely rare half-beat cuts only for huge impacts, not normal density."""
    if not cfg.enable_rare_micro_cuts or len(beat_times) < 3 or selected.size == 0:
        return selected

    beat_diffs = np.diff(beat_times)
    median_beat = float(np.median(beat_diffs)) if beat_diffs.size else 0.5
    if median_beat < 0.22:
        return selected

    threshold = _safe_percentile(features["impact_score"], cfg.micro_percentile, 0.97)
    max_extra = int(max(0, round(len(selected) * cfg.max_micro_cut_ratio)))
    if max_extra <= 0:
        return selected

    extras: List[float] = []
    selected_sorted = np.sort(selected)

    candidates = np.where((features["impact_score"] >= threshold) & (features["wave"] >= 0.88))[0]
    for idx in candidates:
        if len(extras) >= max_extra or idx >= len(beat_times) - 1:
            break
        t = float(beat_times[idx] + 0.5 * (beat_times[idx + 1] - beat_times[idx]))
        if t <= 0.0 or t >= audio_duration:
            continue
        nearest = np.min(np.abs(selected_sorted - t)) if selected_sorted.size else 999.0
        if nearest >= max(cfg.micro_min_gap, median_beat * 0.45):
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


def final_wave_cleanup(selected: np.ndarray, beat_times: np.ndarray, features: Dict,
                       audio_duration: float, cfg: AutoWaveConfig) -> np.ndarray:
    arr = np.asarray(selected, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr > 0.0) & (arr < audio_duration)]
    if arr.size == 0:
        return arr

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
