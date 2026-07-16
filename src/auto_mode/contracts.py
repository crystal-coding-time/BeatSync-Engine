#!/usr/bin/env python3
"""Typed contracts (W2-7) for the auto-mode dict shapes.

Annotation-only: these ``TypedDict``s codify the comment-enforced dict shapes
that already flow through stages 2-6. Nothing here changes runtime behavior —
producers keep returning plain ``dict`` literals and consumers keep reading
them with ``[]``/``.get()``; a ``TypedDict`` is purely a structural type hint
so a checker can see the shape. No key is renamed and no default is altered.

Length/normalization contracts below are transcribed verbatim from the
producing modules' comments (stage2_features.analyze_wave_features,
stage3_sections.analyze_sections, stage6_av_planner._build_segment_profiles /
_materialize_clip / _materialize_partner / _assign_boundary_transitions /
_assign_retime_specs).

Stage1Result (the 5-tuple ``detect_master_beat_grid`` returns —
``beat_times, tempo, beat_frames, onset_env, downbeat_times``) is deliberately
NOT modeled here as a NamedTuple: adopting one would require changing the
return site in ``stage1_audio.py`` to construct it, and that module is outside
this refactor's editable set. The two call sites unpack it as a plain 5-tuple
(``a, b, c, d, e = detect_master_beat_grid(...)`` and ``a, b, _, _, _ = ...``),
which a NamedTuple would support — but only if the producer actually builds
one. Left as a plain tuple; see the W2-7 report.
"""

from typing import List, Optional, TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Stage 2 — per-beat feature arrays (stage2_features.analyze_wave_features).
# ---------------------------------------------------------------------------
# Every array in the REQUIRED block below has length == len(beat_times) and is
# normalized to 0..1, EXCEPT:
#   * energy_levels: dtype=object array of the labels
#     "peak"/"high"/"medium"/"low".
#   * is_strong_kick / is_strong_clap / is_strong_hihat / is_strong_bass /
#     is_bar_anchor / is_phrase_anchor: dtype=bool masks.
# The *_curve arrays (rms_curve/centroid_curve/flux_curve) are the normalized
# per-FRAME curves (length == n_frames, NOT len(beat_times)) threaded into
# energy_profile; they are smoothed then _normalize'd 0..1.


class _BeatFeaturesRequired(TypedDict):
    # Beat-level band strengths (0..1, len == len(beat_times)).
    kick: np.ndarray
    bass: np.ndarray
    clap: np.ndarray
    hihat: np.ndarray
    # Beat-level spectral/energy features (0..1, len == len(beat_times)).
    rms: np.ndarray
    centroid: np.ndarray
    flux: np.ndarray
    onset: np.ndarray
    novelty: np.ndarray
    energy: np.ndarray          # energy_raw
    wave: np.ndarray            # slow density-controlling wave
    brightness: np.ndarray
    rhythm_score: np.ndarray
    impact_score: np.ndarray
    arc: np.ndarray
    # Per-beat categorical energy label (dtype=object).
    energy_levels: np.ndarray
    # Per-beat boolean masks (dtype=bool, len == len(beat_times)).
    is_strong_kick: np.ndarray
    is_strong_clap: np.ndarray
    is_strong_hihat: np.ndarray
    is_strong_bass: np.ndarray
    is_bar_anchor: np.ndarray
    is_phrase_anchor: np.ndarray
    # Normalized per-FRAME curves (len == n_frames) carried to energy_profile.
    rms_curve: np.ndarray
    centroid_curve: np.ndarray
    flux_curve: np.ndarray
    # Wave-13 per-beat features (0..1, len == len(beat_times); zero-filled on
    # extraction failure so the key is ALWAYS present).
    onset_superflux: np.ndarray
    harmonic_change: np.ndarray
    loudness: np.ndarray        # EBU R128 momentary loudness, 0..1


class BeatFeatures(_BeatFeaturesRequired, total=False):
    # Raw (pre-normalization) full-mix flux curve, per FRAME. Always set by
    # stage2 but consumed via .get() (stage3) because pre-wave-13 / fallback
    # beat_info may lack it — hence optional here.
    flux_curve_raw: np.ndarray
    # Wave-14 demucs-mlx stem signals, grafted onto the features dict only when
    # the optional structure/stem backend delivered them, each length-matched
    # to the beat grid (0..1). Absent on Windows / no-stem / disabled paths.
    drum_onset: np.ndarray
    vocal_presence: np.ndarray
    bass_energy: np.ndarray


# ---------------------------------------------------------------------------
# Stage 3 — musical sections (stage3_sections.analyze_sections).
# ---------------------------------------------------------------------------
# The degenerate/empty fallback sections carry only the REQUIRED five keys
# (index/start/end/duration/type); the normal path adds the four measured
# fields below (energy/impact/brightness/dominant_pattern).


class _SectionRequired(TypedDict):
    index: int
    start: float
    end: float
    duration: float
    type: str


class Section(_SectionRequired, total=False):
    energy: float           # section mean wave (0..1)
    impact: float           # section mean impact_score (0..1)
    brightness: float       # section mean brightness (0..1)
    dominant_pattern: str   # detect_section_pattern label


# ---------------------------------------------------------------------------
# Stage 6 — per-segment profile (stage6_av_planner._build_segment_profiles).
# ---------------------------------------------------------------------------
# All keys are always present (single construction site). Floats are 0..1
# except start/end/duration/mid which are seconds.


class SegmentProfile(TypedDict):
    index: int
    start: float
    end: float
    duration: float
    mid: float
    wave: float
    impact: float
    rhythm: float
    novelty: float
    arc: float
    loudness: float
    section: dict           # a Section, or {} when no section covers `mid`
    section_type: str
    target: str             # drop/soft/build/rhythm/flow
    is_downbeat: bool


# ---------------------------------------------------------------------------
# Stage 6 — split-transition and retime specs (additive, optional on a clip).
# ---------------------------------------------------------------------------


class TransitionSpec(TypedDict, total=False):
    # `type` is always present on a real spec; `direction` only for whip_pan.
    type: str
    direction: str          # "left"/"right" (whip_pan only)


class RetimeSpec(TypedDict, total=False):
    # `kind` is always present ("constant"/"ramp"/"freeze"); the rest depend on
    # kind (constant->speed[,interp]; ramp->speed_start/speed_end;
    # freeze->freeze_frames).
    kind: str
    speed: float
    interp: int
    speed_start: float
    speed_end: float
    freeze_frames: int


# ---------------------------------------------------------------------------
# Stage 6 — duo partner payload (stage6_av_planner._materialize_partner).
# ---------------------------------------------------------------------------


class PartnerClip(TypedDict):
    video_file: Optional[str]
    start_time: float
    source_duration: float
    candidate_id: object
    source_name: Optional[str]
    loudness: Optional[float]
    subject_anchor: Optional[dict]


# ---------------------------------------------------------------------------
# Stage 6 — planner output clip (stage6_av_planner._materialize_clip).
# ---------------------------------------------------------------------------
# The REQUIRED block is _materialize_clip's literal; the optional block is the
# additive keys attached later by _materialize_partner (partner),
# _assign_boundary_transitions (transition_in/out) and _assign_retime_specs
# (retime).


class _PlannedClipRequired(TypedDict):
    index: int
    video_file: Optional[str]
    source_name: Optional[str]
    start_time: float
    source_duration: float
    final_duration: float
    target: str
    score: float
    candidate_id: object
    tags: List[str]
    subject_anchor: Optional[dict]
    ai_analyzed: bool
    audio_start: Optional[float]
    audio_end: Optional[float]
    wave: Optional[float]
    impact: Optional[float]
    loudness: Optional[float]
    # Candidate content profile, forwarded verbatim for the semantic_fx
    # effect gate (effects._sem_field); None/absent degrades gracefully.
    kinetic: Optional[float]
    subject_motion: Optional[float]
    motion: Optional[float]
    action_score: Optional[float]
    beauty_score: Optional[float]
    semantic: Optional[dict]
    # v13 additions: analysis media type (still/gif/video) for the semantic_fx
    # still gates, and measured camera-drift direction for the whip-pan
    # direction match (native variants carry sub-15fps re-measurements).
    media_type: Optional[str]
    camera_dir_x: Optional[float]
    camera_dir_y: Optional[float]
    camera_dir_x_native: Optional[float]
    camera_dir_y_native: Optional[float]


class PlannedClip(_PlannedClipRequired, total=False):
    partner: PartnerClip
    transition_in: TransitionSpec
    transition_out: TransitionSpec
    retime: RetimeSpec
