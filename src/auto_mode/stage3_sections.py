#!/usr/bin/env python3
"""Stage 3: broad musical section detection and labeling."""

from typing import Dict, List, Optional
import librosa
import numpy as np

from . import AutoWaveConfig
from . import _safe_percentile

# Wave-14: map the structure backend's functional labels onto the internal
# section vocabulary the rest of the pipeline (stage4 stepping, stage6 planner,
# effects) already understands. Unknown labels default to the neutral body type.
#   intro/start -> intro   outro/end -> outro   break -> breakdown
#   chorus -> chorus       bridge -> bridge     verse/inst/solo -> verse
_STRUCTURE_LABEL_MAP = {
    "start": "intro",
    "intro": "intro",
    "end": "outro",
    "outro": "outro",
    "break": "breakdown",
    "bridge": "bridge",
    "chorus": "chorus",
    "verse": "verse",
    "inst": "verse",
    "solo": "verse",
}


def _map_structure_label(label: object) -> str:
    return _STRUCTURE_LABEL_MAP.get(str(label).strip().lower(), "verse")


def _label_for_interval(backend_sections: List[Dict], start: float, end: float) -> str:
    """Pick the backend label for an output interval by midpoint containment,
    falling back to the maximally-overlapping backend section (robust to
    merge_boundaries collapsing near-adjacent backend edges)."""
    mid = 0.5 * (start + end)
    best_label = "verse"
    best_overlap = -1.0
    for s in backend_sections:
        ss = float(s.get("start", 0.0))
        se = float(s.get("end", 0.0))
        if se <= ss:
            continue
        if ss <= mid < se:
            return _map_structure_label(s.get("label", ""))
        overlap = min(end, se) - max(start, ss)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = _map_structure_label(s.get("label", ""))
    return best_label


def analyze_sections(y: np.ndarray, y_harmonic: np.ndarray, y_percussive: np.ndarray,
                     sr: int, beat_times: np.ndarray, features: Dict,
                     cfg: AutoWaveConfig,
                     structure: Optional[Dict] = None) -> List[Dict]:
    duration = len(y) / sr
    if duration <= 0 or len(beat_times) == 0:
        return [{"index": 0, "start": 0.0, "end": duration, "duration": duration, "type": "body"}]

    # Wave-14: use the backend's functional boundaries + labels when available
    # with >=2 sections; otherwise the entire heuristic path below runs untouched
    # (structure=None => byte-identical to HEAD).
    backend_sections: Optional[List[Dict]] = None
    if isinstance(structure, dict) and structure.get("available"):
        raw = structure.get("sections") or []
        if len(raw) >= 2:
            backend_sections = [s for s in raw if isinstance(s, dict)]
            if len(backend_sections) < 2:
                backend_sections = None

    bass_energy = features.get("bass_energy")
    if bass_energy is not None:
        bass_energy = np.asarray(bass_energy, dtype=float)

    if backend_sections is not None:
        # Backend boundaries replace clustering + novelty-peak discovery.
        boundaries: List[float] = [0.0, duration]
        for s in backend_sections:
            for edge in (s.get("start"), s.get("end")):
                try:
                    t = float(edge)
                except (TypeError, ValueError):
                    continue
                if 0.0 < t < duration:
                    boundaries.append(t)
    else:
        boundaries = [0.0, duration]

        try:
            chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=cfg.hop_length, n_fft=cfg.n_fft)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, hop_length=cfg.hop_length)
            onset = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=cfg.hop_length)[np.newaxis, :]
            min_frames = min(chroma.shape[1], mfcc.shape[1], onset.shape[1])
            frame_features = np.vstack([
                librosa.util.normalize(chroma[:, :min_frames], axis=1),
                librosa.util.normalize(mfcc[:, :min_frames], axis=1),
                librosa.util.normalize(onset[:, :min_frames], axis=1),
            ])
            target_sections = int(np.clip(round(duration / 32.0), 3, 8))
            boundary_frames = librosa.segment.agglomerative(frame_features, k=target_sections)
            boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=cfg.hop_length)
            boundaries.extend([float(t) for t in boundary_times if 0.0 < t < duration])
        except Exception as e:
            print(f"      ⚠️ Structure clustering fallback: {e}")

        # Add only major transition peaks, not every novelty spike.
        novelty = np.asarray(features.get("novelty", []), dtype=float)
        wave = np.asarray(features.get("wave", []), dtype=float)
        if novelty.size == len(beat_times):
            threshold = _safe_percentile(novelty, 91, 0.90)
            wave_gate = _safe_percentile(wave, 45, 0.45)
            last_added = -999.0
            for idx, value in enumerate(novelty):
                if value >= threshold and wave[idx] >= wave_gate:
                    t = float(beat_times[idx])
                    if 0.0 < t < duration and t - last_added >= cfg.section_min_seconds:
                        boundaries.append(t)
                        last_added = t

    boundaries_arr = merge_boundaries(np.asarray(boundaries, dtype=float), duration, min_gap=cfg.section_min_seconds)
    if len(boundaries_arr) < 3:
        step = 32.0
        boundaries_arr = np.asarray([0.0] + list(np.arange(step, duration, step)) + [duration], dtype=float)

    sections: List[Dict] = []
    median_wave = _safe_percentile(features["wave"], 50, 0.5)
    prev_bass_mean: Optional[float] = None

    for i in range(len(boundaries_arr) - 1):
        start = float(boundaries_arr[i])
        end = float(boundaries_arr[i + 1])
        if end - start < 1.0:
            continue
        beat_idx = np.where((beat_times >= start) & (beat_times < end))[0]
        if beat_idx.size:
            section_wave = float(np.mean(features["wave"][beat_idx]))
            section_impact = float(np.mean(features["impact_score"][beat_idx]))
            section_brightness = float(np.mean(features["brightness"][beat_idx]))
        else:
            section_wave = median_wave
            section_impact = 0.5
            section_brightness = 0.5

        section_bass_mean: Optional[float] = None
        if bass_energy is not None and beat_idx.size and bass_energy.size == len(beat_times):
            section_bass_mean = float(np.mean(bass_energy[beat_idx]))

        rel_start = start / max(duration, 1e-6)
        rel_end = end / max(duration, 1e-6)
        if backend_sections is not None:
            section_type = _map_structure_label(_label_for_interval(backend_sections, start, end))
            # Upgrade a labeled chorus to drop/finale when the energy supports it.
            # Thresholds reused verbatim from classify_section: finale gate
            # (rel_end > 0.91 with wave > median_wave*1.18) is checked first to
            # mirror classify_section's positional-first ordering, then the drop
            # gate (wave >= max(0.72, median_wave*1.22) and impact >= 0.58). The
            # bass-step is an additional stem-only drop confirmation.
            if section_type == "chorus":
                bass_step = (
                    section_bass_mean is not None
                    and prev_bass_mean is not None
                    and (section_bass_mean - prev_bass_mean) >= 0.25
                )
                drop_energy = (
                    section_wave >= max(0.72, median_wave * 1.22)
                    and section_impact >= 0.58
                )
                if rel_end > 0.91 and section_wave > median_wave * 1.18:
                    section_type = "finale"
                elif drop_energy or bass_step:
                    section_type = "drop"
        else:
            section_type = classify_section(rel_start, rel_end, section_wave, section_impact, median_wave)
        dominant_pattern = detect_section_pattern(features, beat_idx)

        section = {
            "index": len(sections),
            "start": start,
            "end": end,
            "duration": end - start,
            "type": section_type,
            "energy": section_wave,
            "impact": section_impact,
            "brightness": section_brightness,
            "dominant_pattern": dominant_pattern,
        }
        sections.append(section)
        if section_bass_mean is not None:
            prev_bass_mean = section_bass_mean
        print(
            f"      • {section_type:<10} {start:6.1f}s → {end:6.1f}s "
            f"({end - start:5.1f}s), wave={section_wave:.2f}, pattern={dominant_pattern}"
        )

    return sections or [{"index": 0, "start": 0.0, "end": duration, "duration": duration, "type": "body"}]


def merge_boundaries(boundaries: np.ndarray, duration: float, min_gap: float) -> np.ndarray:
    arr = np.asarray(boundaries, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, duration)
    arr = np.unique(np.round(arr, 3))
    arr = np.sort(arr)

    merged: List[float] = []
    for t in arr:
        if not merged:
            merged.append(float(t))
            continue
        if t in (0.0, duration) or t - merged[-1] >= min_gap:
            merged.append(float(t))

    if not merged or abs(merged[0]) > 1e-6:
        merged.insert(0, 0.0)
    if abs(merged[-1] - duration) > 1e-6:
        merged.append(float(duration))
    return np.asarray(merged, dtype=float)


def classify_section(rel_start: float, rel_end: float, wave: float, impact: float, median_wave: float) -> str:
    if rel_start < 0.11:
        return "intro" if wave <= median_wave * 1.18 else "hook"
    if rel_end > 0.91:
        return "outro" if wave <= median_wave * 1.18 else "finale"
    if wave >= max(0.72, median_wave * 1.22) and impact >= 0.58:
        return "drop"
    if wave >= max(0.60, median_wave * 1.12):
        return "chorus"
    if impact >= 0.64 and wave < median_wave:
        return "bridge"
    if wave < median_wave * 0.82:
        return "breakdown"
    return "verse"


def detect_section_pattern(features: Dict, beat_indices: np.ndarray) -> str:
    if beat_indices.size == 0:
        return "mixed"
    kick_ratio = float(np.mean(features["is_strong_kick"][beat_indices]))
    clap_ratio = float(np.mean(features["is_strong_clap"][beat_indices]))
    bass_ratio = float(np.mean(features["is_strong_bass"][beat_indices]))
    hihat_ratio = float(np.mean(features["is_strong_hihat"][beat_indices]))

    if kick_ratio > 0.42 and clap_ratio > 0.34:
        return "kick_clap"
    if kick_ratio >= max(clap_ratio, bass_ratio, hihat_ratio) and kick_ratio > 0.34:
        return "kick"
    if clap_ratio >= max(kick_ratio, bass_ratio, hihat_ratio) and clap_ratio > 0.34:
        return "clap"
    if bass_ratio >= max(kick_ratio, clap_ratio, hihat_ratio) and bass_ratio > 0.34:
        return "bass"
    if hihat_ratio > 0.46:
        return "hihat"
    return "mixed"
