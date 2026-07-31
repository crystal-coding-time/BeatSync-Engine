#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import (PartnerClip, PlannedClip, RetimeSpec, Section,
                        SegmentProfile, TransitionSpec)

# Proportional-fair usage pressure (variety>0): how many segments of history one
# appearance is worth. Larger = slower forgetting = the fair-share view reaches
# further back. 24 segments ≈ a few phrases at typical cut rates.
_PF_TAU = 24.0


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _stable_rng(*parts) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    return random.Random(seed)


def _segment_energy(profile: SegmentProfile) -> float:
    """Quantized cross-modal energy of a segment (wave 16), in [0, 1].

    Blends the three audio intensity signals a profile always carries
    (loudness dominant, wave for the sustained envelope, impact for the hit)
    into one number `_score_candidate` matches clip kinetics against. The 2dp
    round is the determinism firewall: this value enters `_ScoreCache` keys,
    the coverage-reservation column dedup, and score comparisons, so it must
    be an exact, platform-stable float. This helper is the single authority —
    every consumer calls it rather than re-deriving the blend.
    """
    return round(0.5 * _clamp(profile.get("loudness", 0.5), default=0.5)
                 + 0.3 * _clamp(profile.get("wave", 0.5), default=0.5)
                 + 0.2 * _clamp(profile.get("impact", 0.5), default=0.5), 2)


class _ScoreCache:
    """Per-run memo for ``_score_candidate(candidate, profile)``.

    ``_score_candidate`` depends on ``profile`` through exactly two derived
    values: ``profile.get("target", "flow")`` and (since wave 16) the
    2dp-quantized segment energy ``_segment_energy(profile)`` — everything
    else it uses comes from ``candidate``. So its result only varies along
    three axes: which candidate, which target, and which quantized energy.
    This class memoizes on those axes for a single
    ``build_planned_clip_sequence()`` run, so the O(candidates x segments)
    scoring the auction, coverage-reservation, duo-partner and materialize
    call sites all did independently collapses to O(candidates x distinct
    (target, energy) pairs) real evaluations — energy is on a 0.01 grid, so
    the pair count stays small. If ``_score_candidate`` ever grows another
    profile input, it must be quantized and added to this key.

    Candidates are plain dicts drawn, by reference, from one shared
    ``candidates`` list for the whole run -- no call site copies a candidate
    dict before scoring it -- so ``id(candidate)`` is a safe, collision-free
    key for the lifetime of this cache. The candidate dict's own ``"id"``
    field is deliberately NOT used as the key: sample/real candidate data can
    carry duplicate or ``None`` ids, which would corrupt the cache.
    """

    __slots__ = ("_index_of", "_scores")

    def __init__(self, candidates: Sequence[Dict]):
        self._index_of = {id(c): i for i, c in enumerate(candidates)}
        self._scores: Dict[tuple, float] = {}

    def score(self, candidate: Dict, profile: Dict) -> float:
        target = profile.get("target", "flow")
        # Same quantized energy _score_candidate computes — 2dp rounding is
        # what makes it a safe dict key (exact float, no drift).
        e_seg = _segment_energy(profile)
        idx = self._index_of.get(id(candidate))
        key = (idx, target, e_seg)
        if key not in self._scores:
            # Call the real scorer once per (candidate, target, energy) --
            # never reimplement its math here.
            self._scores[key] = _score_candidate(candidate, profile)
        return self._scores[key]


class _FairShareEWMA:
    """Lazy-decayed EWMA of per-source appearances, for proportional-fair usage.

    Replaces the linear ``usage[file] * file_rate`` reuse penalty when
    variety>0 (see ``build_planned_clip_sequence``). Each source stores
    ``(value, last_index)``; both reads and writes decay that value by
    ``(1 - 1/TAU) ** (index - last_index)`` at the moment they touch it, so no
    O(sources) sweep runs per segment — only the single ``total()`` call the
    scorer makes once per segment does, which is fine at dozens of sources.

    Determinism: pure float math over integer step counts (``index`` is the
    segment index, monotonically non-decreasing across appends), so identical
    inputs replay bit-identically. Reads always see ``index >= last_index``
    (an entry is written at the segment where it is picked and only read on
    that or a later segment), so the exponent is never negative.
    """

    __slots__ = ("_decay", "_state")

    def __init__(self, tau: float = _PF_TAU):
        self._decay = 1.0 - 1.0 / tau
        self._state: Dict[object, tuple] = {}

    def value(self, key, index: int) -> float:
        entry = self._state.get(key)
        if entry is None:
            return 0.0
        val, last = entry
        return val * self._decay ** (index - last)

    def total(self, index: int) -> float:
        return sum(val * self._decay ** (index - last)
                   for val, last in self._state.values())

    def add(self, key, index: int, amount: float = 1.0) -> None:
        # Decay the running value to `index`, then credit this appearance.
        self._state[key] = (self.value(key, index) + amount, index)


# --- Split-screen duo segments ----------------------------------------------
# A duo pairs the auction's primary clip with a partner clip from a different
# cross-orientation source (portrait sources on a landscape canvas, or the
# converse). The planner only ever ADDS a "partner" key: primary selection,
# reservations and transitions are bit-identical with split_screen off.

# Seeded rarity cap: ~1 in 4 eligible segments becomes a duo.
DUO_RATE = 0.25
# Duos live on hard cuts: drop/rhythm targets with at least this impact.
DUO_TARGETS = frozenset({"drop", "rhythm"})
DUO_MIN_IMPACT = 0.5
# Brightness-coherence gate: panes whose mean brightness differs by more than
# this clash side by side, so such partners are skipped outright.
DUO_BRIGHTNESS_MAX_DELTA = 0.30
# Partners whose subject anchor is below ANCHOR_MIN_CONFIDENCE still compete,
# but at a deficit — the pane crop should land on a subject.
DUO_LOW_ANCHOR_PENALTY = 0.12


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
    variety: float = 0.0,
    speed_ramps: bool = False,
    lossless: bool = False,
    fps: float = 30.0,
    split_screen: bool = False,
    target_size: tuple | None = None,
    semantic_variety: float = 0.0,
    media_aware: bool = False,
    semantic_fx: bool = False,
) -> List[PlannedClip]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to keep its old fallback sampling.

    variety=0 keeps the legacy quality-auction STRUCTURE (no coverage
    guarantee — weak sources can lose every pick), though wave-16 rebalanced
    the scores it ranks (see PLAN STABILITY below). Any variety>0 reserves one
    segment per source so every upload with usable candidates appears at least
    once, and applies proportional-fair usage pressure that pushes over-used
    sources toward an even spread at 1.0.

    PLAN STABILITY / SLIDER SEMANTICS
      * Wave-17 repetition fix (accepted, deliberate — same precedent as the
        wave-12, wave-15 and wave-16 redesigns below; ALL plans change,
        INCLUDING variety=0, and there is no kill switch). Three coupled
        changes attack "clips loop noticeably", whose root cause was that
        ``_plan_source_window`` had no memory: for a shared anchor and a fixed
        per-target ``align``, two windows of different durations are strictly
        NESTED around the same anchor frame, so every reuse of a candidate
        replayed essentially the same footage even though analysis windows run
        up to 5.2s against ~1.2s segments (roughly 4 disjoint segment-length
        windows per candidate that the formula could never reach).
        (a) A per-run SUB-WINDOW LEDGER (``spent``, candidate id -> spent
        source intervals) is threaded into ``_plan_source_window`` — still the
        single authority for window semantics, so the primary and duo-partner
        paths stay identical — which now offsets a reuse onto the first
        0.85*d-strided sub-window that overlaps nothing spent by more than
        0.35*d, falling back to the anchor window when the span is under 2.2*d
        or every sub-window is taken.
        (b) The saturating ``min(0.28, usage[cid] * 0.10)`` reuse penalty
        becomes ``_reuse_penalty``: an unbounded ``0.10 * log1p(usage) *
        (1 + 2*variety)`` term plus an exhaustion term (up to 0.25) in the
        fraction of the candidate's analysis span the ledger has served.
        Previously the ONLY unbounded pressure was ``pf_ewma``, which is keyed
        on ``video_file``, not candidate identity — so one candidate inside a
        favoured source could win without limit once its capped penalty
        saturated at the third use. Ledger-dependent, therefore applied
        OUTSIDE ``_ScoreCache`` like the recency/PF/semantic/media-aware terms.
        (c) ``recent_ids`` / ``recent_videos`` are sized from the MEDIAN
        segment duration to cover ~20s / ~10s of screen time instead of a
        fixed 10 / 5 segments, so fast passages keep a real recency memory.
        Determinism is unaffected — no new rng; the ledger is a pure function
        of the sequential pick order. Precise (lossless/ProRes) mode also gets the
        new windows — window CHOICE is not on that mode's veto list (effects,
        retimes, duos, crossfades, text); its extractions stay plain,
        static-framed source windows, just better spread across the footage.
      * Wave-16 scoring rebalance (accepted, deliberate — same precedent as
        the wave-12 and wave-15 redesigns below): ``_score_candidate`` now
        (a) scores untagged (non-``ai_analyzed``) candidates on drop segments
        from measured signals instead of the fabricated combat/chase/explosion
        placeholders, and (b) adds a continuous cross-modal energy-matching
        term (clip kinetics vs. the segment's quantized ``_segment_energy``)
        to EVERY score. ALL plans change vs. wave 15, INCLUDING variety=0 —
        the pre-wave-12 byte-for-byte claim for the legacy auction no longer
        holds, and there is no kill switch. Only the no-candidates path (no
        video analysis at all → empty list → renderer fallback) is untouched.
      * The mere PRESENCE of ``embedding`` / ``visual_cluster`` keys on
        candidates still changes nothing at semantic_variety=0.
      * variety>0 plans CHANGED vs. wave 11 (accepted, deliberate): the
        linear, capped ``file_rate`` reuse penalty is replaced by an EWMA
        proportional-fair pressure term (``_FairShareEWMA``). The old formula
        scaled ``file_rate`` linearly and uncapped it, which late in long
        videos swamped content scores and degenerated to score-blind
        round-robin; PF instead penalizes only sources running ABOVE their fair
        share (``share * source_count > 1``), so content ordering survives. The
        variety slider's meaning is redesigned; variety=0 keeps the legacy
        auction STRUCTURE (no reservations, linear file penalty) even though
        wave-16 rebalanced the scores it ranks.

    semantic_variety (0..1, needs candidate ``embedding``/``visual_cluster``
    keys from src/visual_embeddings.py — both OPTIONAL, missing → no penalty)
    discourages runs of visually similar shots: a cluster-run penalty plus a
    windowed max-cosine-similarity (MMR-style) penalty against recently picked
    embeddings. semantic_variety=0 short-circuits the whole feature (no deque
    bookkeeping, no embedding arithmetic) so it cannot perturb legacy plans.

    split_screen=False (the default) adds nothing on top of the plan the
    other settings produce (no partner keys, no extra usage accounting).
    When True, some eligible drop/rhythm segments gain an
    additive "partner" dict (a second, cross-orientation source for a 2-up
    pane composite); everything else about the plan is unchanged apart from
    the partner's usage accounting. target_size is the output canvas
    (width, height) used only to decide pairing orientation — the caller
    (video_processor, wave 7C) passes its resolved target resolution; None is
    treated as a landscape 16:9 canvas.

    media_aware=False (the default) adds NOTHING — plans stay byte-identical.
    When True, the auctions add signed adjustments computed OUTSIDE the score
    cache (see _media_aware_adjustment): an upscale penalty for sub-canvas
    sources (target_size supplies the canvas width), a loop-seam penalty for
    seamy GIFs on segments longer than one GIF pass, and a native-frame-pair
    kinetic correction for sub-15fps sources. All the fields it reads
    (media_type, source_width, loop_seam, kinetic_native) are optional —
    missing data means no adjustment.
    """
    beat_info = beat_info or {}
    video_analysis = beat_info.get("video_analysis") or {}
    candidates = list(video_analysis.get("candidates") or [])
    candidates = [c for c in candidates if c.get("video_file")]
    if not candidates:
        return []
    score_cache = _ScoreCache(candidates)

    cut_times_arr = np.asarray(cut_times, dtype=float)
    durations_arr = np.asarray(segment_durations, dtype=float)
    if cut_times_arr.size < 2 or durations_arr.size == 0:
        return []

    profiles = _build_segment_profiles(cut_times_arr, durations_arr, beat_info)
    variety = _clamp(variety)
    semantic_variety = _clamp(semantic_variety)

    # Legacy reuse penalty: 0.012/use capped at 0.18 (variety=0 only — kept
    # byte-identical). variety>0 no longer uses it: proportional-fair usage
    # pressure (below, `pf_ewma`) replaces it, so weak-but-unused sources win
    # without a leader's penalty growing without bound.
    file_rate, file_cap = 0.012, 0.18
    source_count = len({c.get("video_file") for c in candidates})
    reservations: Dict[int, Dict] = {}
    pf_ewma: "_FairShareEWMA | None" = None
    if variety > 0.0:
        pf_ewma = _FairShareEWMA()
        reservations = _plan_coverage_reservations(candidates, profiles,
                                                   score_cache)

    # Semantic diversity state (semantic_variety>0 only). Each candidate's
    # embedding is converted to a float64 array exactly ONCE here, keyed by
    # id(candidate) like _ScoreCache (candidates are shared by reference for
    # the whole run). Missing/malformed embeddings simply get no entry →
    # graceful zero penalty. recent_clusters / recent_embeds are the sliding
    # windows the MMR-style penalties compare against.
    embed_arrays: Dict[int, np.ndarray] = {}
    recent_clusters: deque = deque(maxlen=4)
    recent_embeds: deque = deque(maxlen=6)
    if semantic_variety > 0.0:
        for c in candidates:
            emb = c.get("embedding")
            if emb is None:
                continue
            try:
                arr = np.asarray(emb, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 1 and arr.size > 0:
                embed_arrays[id(c)] = arr

    # Duo pairing needs ≥2 distinct cross-orientation, non-still sources; the
    # context is None whenever duos are impossible (including lossless mode:
    # the ProRes branch must stay pristine), which keeps every plan bit-equal
    # to the split_screen=False path.
    duo_pair_files = None
    if split_screen and not lossless:
        duo_pair_files = _duo_pair_files(candidates, target_size)

    # Media-aware auction state (media_aware=True only): the canvas width the
    # upscale penalty measures against, plus fire counters for the one debug
    # line below. media_aware=False leaves canvas_w=0/stats=None and no call
    # sites touch them, so legacy plans cannot be perturbed.
    canvas_w = 0
    media_stats: Counter | None = None
    if media_aware:
        try:
            canvas_w = int(target_size[0]) if target_size else 0
        except (TypeError, ValueError, IndexError):
            canvas_w = 0
        media_stats = Counter()

    # Wave 17: recency windows measured in SECONDS, not in segments. The old
    # fixed maxlen=10/5 meant ~12s / ~6s of memory at typical 1.2s cuts, but
    # only ~4s / ~2s on a fast 0.4s-cut passage — exactly where a repeat is
    # most visible. Sizing from the median segment duration keeps the window
    # at a roughly constant ~20s (ids) / ~10s (sources) of screen time at any
    # cut rate. Deterministic: numpy median over the same durations array the
    # timeline is built from.
    median_dur = float(np.median(durations_arr))
    if not np.isfinite(median_dur) or median_dur <= 0.0:
        median_dur = 1.0
    recent_ids = deque(maxlen=max(8, int(round(20.0 / median_dur))))
    recent_videos = deque(maxlen=max(4, int(round(10.0 / median_dur))))
    usage = Counter()
    # Sub-window ledger: candidate id -> list of (start, end) source intervals
    # already spent. _plan_source_window offsets reuse into unspent footage,
    # and _reuse_penalty prices exhaustion. Purely a function of pick order.
    spent: Dict[object, List[tuple]] = {}
    planned: List[PlannedClip] = []

    for i, profile in enumerate(profiles):
        # Reservations are consumed inside the sequential loop so adjacency
        # deques and usage stay coherent for the auction picks around them.
        candidate = reservations.get(i)
        if candidate is None:
            candidate = _choose_candidate(
                candidates=candidates,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                file_rate=file_rate,
                file_cap=file_cap,
                score_cache=score_cache,
                variety=variety,
                source_count=source_count,
                pf_ewma=pf_ewma,
                semantic_variety=semantic_variety,
                recent_clusters=recent_clusters,
                recent_embeds=recent_embeds,
                embed_arrays=embed_arrays,
                media_aware=media_aware,
                canvas_w=canvas_w,
                media_stats=media_stats,
                spent=spent,
            )
        if not candidate:
            continue
        planned_clip = _materialize_clip(
            candidate=candidate,
            profile=profile,
            index=i,
            score_cache=score_cache,
            spent=spent,
        )
        partner_candidate = None
        if duo_pair_files is not None:
            partner_candidate = _maybe_choose_duo_partner(
                candidates=candidates,
                profile=profile,
                primary=candidate,
                planned=planned,
                index=i,
                pair_files=duo_pair_files,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                file_rate=file_rate,
                file_cap=file_cap,
                score_cache=score_cache,
                variety=variety,
                source_count=source_count,
                pf_ewma=pf_ewma,
                semantic_variety=semantic_variety,
                recent_clusters=recent_clusters,
                recent_embeds=recent_embeds,
                embed_arrays=embed_arrays,
                media_aware=media_aware,
                canvas_w=canvas_w,
                media_stats=media_stats,
                spent=spent,
            )
        if partner_candidate is not None:
            planned_clip["partner"] = _materialize_partner(
                partner_candidate, profile, spent)
        planned.append(planned_clip)
        recent_ids.append(candidate.get("id"))
        recent_videos.append(candidate.get("video_file"))
        usage[candidate.get("id")] += 1
        usage[candidate.get("video_file")] += 1
        _record_source_window(spent, candidate, planned_clip["start_time"],
                              planned_clip["source_duration"])
        if variety > 0.0:
            # Reservations flow through here too (they materialize above like
            # any pick), so a seated source counts toward its fair share.
            pf_ewma.add(candidate.get("video_file"), i)
        if semantic_variety > 0.0:
            _record_semantic(candidate, recent_clusters, recent_embeds,
                             embed_arrays)
        if partner_candidate is not None:
            # A pane appearance is an appearance: the partner pays the same
            # usage / file-recency / PF / semantic costs going forward and
            # counts for coverage. Its id deliberately does NOT enter
            # recent_ids — appending would evict primary ids from that
            # (wave-17 time-scaled) window faster; a partner only dodges the
            # -0.28 id-recency penalty, not the rest. Its source window IS
            # booked into the ledger: a pane replays frames just as visibly.
            recent_videos.append(partner_candidate.get("video_file"))
            usage[partner_candidate.get("id")] += 1
            usage[partner_candidate.get("video_file")] += 1
            partner_clip = planned_clip["partner"]
            _record_source_window(spent, partner_candidate,
                                  partner_clip["start_time"],
                                  partner_clip["source_duration"])
            if variety > 0.0:
                pf_ewma.add(partner_candidate.get("video_file"), i)
            if semantic_variety > 0.0:
                _record_semantic(partner_candidate, recent_clusters,
                                 recent_embeds, embed_arrays)

    if media_aware and media_stats is not None:
        print(f"   🧩 Media-aware auction (canvas {canvas_w}px): "
              f"upscale penalty ×{media_stats['upscale']}, "
              f"loop-seam penalty ×{media_stats['loop_seam']}, "
              f"native-kinetic correction ×{media_stats['kinetic_native']} "
              f"(candidate evaluations)")

    if len(planned) != len(durations_arr):
        return []
    _assign_boundary_transitions(planned, semantic_fx=semantic_fx)
    if speed_ramps and not lossless:
        # Retime specs never reach precise mode: the ProRes branch extracts
        # plain windows and must stay pristine for external editing.
        _assign_retime_specs(planned, candidates, fps)
        # Anticipation ramps claim drop-entry boundaries the per-segment pass
        # left untouched (same setting, dedicated rng stream).
        _assign_boundary_ramps(planned, candidates, fps)
    return planned


def _assign_retime_specs(planned: List[PlannedClip], candidates: Sequence[Dict],
                         fps: float) -> None:
    """Attach per-segment retime specs where the music and footage allow.

    Every gate here is deterministic and conservative: a ramp multiplies how
    much source a segment consumes, and a window that comes up short would
    break the frame-locked timeline (extraction guards against it, but a
    stripped ramp is a wasted plan — better to never attach one that can't
    run). Renderer probes (not the analysis metadata) decide runway.
    """
    from ffmpeg_processing import (
        get_cached_video_fps,
        is_image_source,
        seconds_to_frame_count,
    )

    by_id = {c.get("id"): c for c in candidates}
    for clip in planned:
        if "partner" in clip:
            # Duo segments never retime: a warped clock breaks both panes'
            # tracked crops and doubles the source-runway math.
            continue
        video_file = clip.get("video_file")
        final_duration = float(clip.get("final_duration", 0.0))
        if not video_file or is_image_source(video_file):
            continue
        if final_duration < 0.6:
            # Sub-0.6s cuts don't hold a readable speed change.
            continue

        source_fps = get_cached_video_fps(video_file)
        if source_fps < 24.0 or os.path.splitext(video_file)[1].lower() == ".gif":
            # Low-fps sources (GIFs run 10-15fps) are already frame-duplicated
            # by the fps= normalization; retiming them is pure judder. The
            # extension check closes the container-fps hole: a GIF whose
            # header claims >=24fps still runs on display durations and
            # retimes just as badly.
            continue

        target = str(clip.get("target", "flow"))
        impact = _clamp(clip.get("impact", 0.5), default=0.5)
        rng = _stable_rng("retime", clip.get("index"), video_file, target)
        roll = rng.random()

        retime: RetimeSpec | None = None
        if target == "soft" and roll < 0.45:
            # Slow-mo drift. On >=50fps footage the deep band (<0.6x) is real
            # source frames. On ordinary 24-50fps sources sub-0.6x used to be
            # pure duplicated-frame judder — now minterpolate synthesizes the
            # in-between frames (interp=2), so the deep band opens up there too.
            #
            # RNG DISCIPLINE: both branches draw exactly ONE uniform from the
            # shared 'retime' stream — the >=50 branch always did uniform(0.5,
            # 0.7); the <50 branch WAS uniform(0.6, 0.7) and is now uniform(0.5,
            # 0.7). Same draw count, so every other segment's plan is untouched;
            # only 24-50fps soft-slow-mo speeds shift (lower bound 0.6 -> 0.5),
            # and a drawn speed < 0.6 there gains the interp flag.
            speed = round(rng.uniform(0.5, 0.7), 3)
            retime = {"kind": "constant", "speed": speed}
            if source_fps < 50.0 and speed < 0.6:
                # 24 <= source_fps < 50 is guaranteed by the sub-24 skip above.
                retime["interp"] = 2
        elif target == "build" and roll < 0.40:
            retime = {"kind": "constant", "speed": round(rng.uniform(1.4, 2.0), 3)}
        elif target == "drop":
            if impact >= 0.75 and roll < 0.10 and final_duration >= 0.8:
                out_frames = max(1, seconds_to_frame_count(final_duration, fps))
                freeze = min(max(2, int(round(fps * 0.3))), out_frames // 2)
                retime = {"kind": "freeze", "freeze_frames": freeze}
            elif roll < 0.25:
                retime = {
                    "kind": "ramp",
                    "speed_start": round(rng.uniform(1.5, 2.0), 3),
                    "speed_end": round(rng.uniform(0.6, 0.8), 3),
                }
        if retime is None:
            continue
        start = _retime_fit_start(clip, retime, by_id, fps, source_fps)
        if start is None:
            continue
        _attach_retime(clip, retime, start)


def _retime_fit_start(clip: PlannedClip, retime: RetimeSpec, by_id: Dict,
                      fps: float, source_fps: float) -> Optional[float]:
    """Runway-gate a retime spec; returns the re-anchored start or None.

    Pure check — never mutates the clip. Shared by _assign_retime_specs and
    _assign_boundary_ramps so the two can never disagree on how much source a
    retimed segment needs. Renderer probes (not analysis metadata) decide.
    """
    from ffmpeg_processing import (
        get_cached_video_duration,
        retime_source_window,
    )

    final_duration = float(clip.get("final_duration", 0.0))
    # source_fps lets retime_source_window add the interp over-provision
    # (a no-op for non-interp specs); it must match extraction's window.
    window = retime_source_window(final_duration, retime, fps,
                                  source_fps=source_fps)

    # Per-window runway: the retimed window must fit inside the scene
    # window the candidate was chosen for (a ramp that spills across the
    # scene cut hides a hard cut mid-slow-mo), and inside the real file.
    candidate = by_id.get(clip.get("candidate_id")) or {}
    cand_start = float(candidate.get("start", 0.0))
    cand_end = float(candidate.get("end", cand_start))
    video_duration = get_cached_video_duration(clip.get("video_file"))
    scene_len = cand_end - cand_start
    if scene_len > 0.0 and window > scene_len:
        return None
    if window > video_duration - 0.05:
        return None

    # Re-anchor the start for the bigger window, inside both the scene
    # window and the file. (Mirrors _materialize_clip's anchor semantics.)
    start = float(clip.get("start_time", 0.0))
    hi = min(cand_end - window if scene_len > 0.0 else video_duration - window,
             video_duration - window)
    lo = max(0.0, cand_start if scene_len > 0.0 else 0.0)
    if hi < lo:
        return None
    return max(lo, min(start, hi))


def _attach_retime(clip: PlannedClip, retime: RetimeSpec, start: float) -> None:
    """Attach a runway-checked retime (start from _retime_fit_start)."""
    clip["start_time"] = start
    clip["retime"] = retime
    # Retiming warps the segment's local clock (setpts sits between the
    # source trim and the output fps=), so a source-time subject path no
    # longer lines up with the `t` the renderer's crop expressions see —
    # and this branch also re-anchors start_time, which the path was
    # rebased against. Drop the tracked path and let the static anchor
    # offset stand for retimed segments.
    anchor = clip.get("subject_anchor")
    if isinstance(anchor, dict) and "path_seg" in anchor:
        anchor = dict(anchor)
        anchor.pop("path_seg")
        clip["subject_anchor"] = anchor


# Boundary anticipation ramps (behind the same speed_ramps setting): how often
# an ELIGIBLE drop-entry boundary actually fires, and the density cap — at
# most one fired boundary per this many planned segments (floor of one).
_BOUNDARY_RAMP_PROBABILITY = 0.25
_BOUNDARY_RAMP_SEGMENTS_PER_FIRE = 8
# Minimum segment lengths: the outgoing tail accel needs room to read as a
# rush (and to keep the warped tail well under half the segment); the incoming
# slow-open needs room to ease back to 1.0x without looking like a stutter.
_BOUNDARY_RAMP_MIN_OUT_SECONDS = 1.2
_BOUNDARY_RAMP_MIN_IN_SECONDS = 1.0


def _assign_boundary_ramps(planned: List[PlannedClip],
                           candidates: Sequence[Dict], fps: float) -> None:
    """Anticipation ramps at drop entries: warp time across the boundary.

    A fired boundary accelerates the OUTGOING segment's final 0.4-0.7s
    (1.0x -> ~1.6x via the 'tail_ramp' retime kind) so the music's arrival
    feels rushed-into, and opens the INCOMING drop at 0.4-0.5x exactly on its
    first frame, easing back to 1.0x over the segment (the existing
    whole-segment 'ramp' kind). Both sides must clear runway or neither
    attaches — a half-fired boundary reads worse than a plain cut.

    Runs after _assign_retime_specs and only claims boundaries where neither
    side already carries a retime or partner. Crossfades stay mutually
    exclusive for free: _select_crossfade_boundaries (video_processor) runs
    later and skips any boundary whose clips carry a retime, so a fired
    anticipation ramp deterministically wins the boundary.

    RNG DISCIPLINE: every draw comes from a dedicated per-boundary
    'boundary_ramp' stream — the per-segment 'retime' streams above are
    seeded independently and keep their exact draw counts, so a run where no
    boundary fires plans byte-identically to the pre-feature speed_ramps
    engine.
    """
    from ffmpeg_processing import get_cached_video_fps, is_image_source

    by_id = {c.get("id"): c for c in candidates}
    cap = max(1, len(planned) // _BOUNDARY_RAMP_SEGMENTS_PER_FIRE)
    fired = 0
    for i in range(len(planned) - 1):
        if fired >= cap:
            break
        out_clip, in_clip = planned[i], planned[i + 1]
        if str(in_clip.get("target", "flow")) != "drop":
            continue
        if "retime" in out_clip or "retime" in in_clip:
            continue
        if "partner" in out_clip or "partner" in in_clip:
            continue
        out_file = out_clip.get("video_file")
        in_file = in_clip.get("video_file")
        if not out_file or not in_file:
            continue
        # Mirror _assign_retime_specs' media gates on BOTH sides: stills and
        # GIFs (any container fps — display durations, not real frames) never
        # retime, and sub-24fps sources are already frame-duplicated by the
        # fps= normalization, so warping them is pure judder.
        side_fps = []
        for f in (out_file, in_file):
            if is_image_source(f) or os.path.splitext(f)[1].lower() == ".gif":
                break
            probed = get_cached_video_fps(f)
            if probed < 24.0:
                break
            side_fps.append(probed)
        if len(side_fps) < 2:
            continue
        out_fps, in_fps = side_fps
        out_len = float(out_clip.get("final_duration", 0.0))
        in_len = float(in_clip.get("final_duration", 0.0))
        if (out_len < _BOUNDARY_RAMP_MIN_OUT_SECONDS
                or in_len < _BOUNDARY_RAMP_MIN_IN_SECONDS):
            continue

        rng = _stable_rng("boundary_ramp", i, out_file, in_file)
        if rng.random() >= _BOUNDARY_RAMP_PROBABILITY:
            continue
        # out_len >= 1.2 keeps min(0.7, out_len / 2.0) >= 0.6 > 0.4, so the
        # uniform's bounds never invert and the tail stays under half the
        # outgoing segment (the cut must read as a rush, not a whole-segment
        # speed-up).
        tail = round(rng.uniform(0.4, min(0.7, out_len / 2.0)), 3)
        out_speed = round(rng.uniform(1.45, 1.75), 3)
        in_speed = round(rng.uniform(0.4, 0.5), 3)
        out_retime: RetimeSpec = {"kind": "tail_ramp", "speed_end": out_speed,
                                  "tail_seconds": tail}
        in_retime: RetimeSpec = {"kind": "ramp", "speed_start": in_speed,
                                 "speed_end": 1.0}

        out_start = _retime_fit_start(out_clip, out_retime, by_id, fps,
                                      out_fps)
        if out_start is None:
            continue
        in_start = _retime_fit_start(in_clip, in_retime, by_id, fps,
                                     in_fps)
        if in_start is None:
            continue
        _attach_retime(out_clip, out_retime, out_start)
        _attach_retime(in_clip, in_retime, in_start)
        fired += 1
        print(f"   ⏱ Anticipation ramp @ boundary {i}: out tail "
              f"1.0→{out_speed}x ({tail}s), drop opens at {in_speed}x")


def _plan_coverage_reservations(candidates: Sequence[Dict],
                                profiles: Sequence[SegmentProfile],
                                score_cache: "_ScoreCache | None" = None) -> Dict[int, Dict]:
    """Reserve one segment per source: a globally optimal (source, segment) seating.

    Solves a single rectangular linear assignment problem
    (``scipy.optimize.linear_sum_assignment``) that maximizes the total
    seat score over all sources at once, instead of the pre-wave-15 greedy
    that seated sources one at a time in best-seat order. Greedy could hand a
    high-priority source a seat that a later source needed as its *only* good
    seat, forcing that later source onto a terrible seat (or off the video);
    the global solve avoids those dead ends, so it seats at least as many
    sources as greedy and at no worse total score, in every case.

    PLAN STABILITY (wave-15 change, deliberate — same precedent as the wave-12
    proportional-fair redesign): variety>0 plans CHANGE, because reservation
    seating is now globally optimal rather than greedy. variety=0 never calls
    this function (no reservations); note however that the wave-16 scoring
    rebalance in ``_score_candidate`` changes ALL plans, variety=0 included —
    see ``build_planned_clip_sequence``'s PLAN STABILITY notes.

    Preserved semantics vs. greedy:
      * The per-(source, segment) seat score is identical, including the
        short-candidate duration penalty (-0.18 when the candidate is shorter
        than 0.55x the segment) that steers short sources — GIFs and short
        scenes — onto short segments. It does NOT reach stills, despite what
        this docstring claimed before wave 17: video_analysis pins a still's
        planner-facing ``duration`` at 5.0 (one zero-width window, but the
        still can fill any segment via Ken Burns), so the penalty would need a
        segment longer than 5.0 / 0.55 = 9.09s, while the longest hold in
        ``auto_mode/__init__.py`` is ``low_energy_max_hold = 3.80``. The claim
        is corrected rather than the penalty widened: a still genuinely has no
        length limit to protect, so penalizing it on long segments would be a
        content regression, not a fix.
      * The drop-exemption tier is preserved: on a 'drop' segment a candidate
        scoring below ``best_raw[j] - 0.35`` is "exempt" — coverage should
        spend the flow/soft filler slots, not the money-shot drops. Exempt
        seats are shifted strictly below every primary (non-exempt) seat, so
        the solver only ever seats a source on an exempt drop when it has no
        primary seat available.
      * Per (source, segment) the reserved candidate is that source's
        best-scoring candidate for the segment under the exact greedy
        tie-break ``key = (-score, j, str(id))``. The LAP decides *which*
        segment each source gets; the candidate for a given seat is unchanged.

    WHY A COVERAGE FLOOR (and how both guarantees hold at once). Pure
    score-maximization would leave a marginal source unseated whenever seating
    it lowers the total — but greedy always seats a source that has a free
    seat, so pure score-max can DROP coverage below greedy. Pure
    coverage-maximization (seat as many as possible) can instead force a
    lower-scoring layout than greedy. Neither dominates greedy on both axes.
    The fix: run greedy's exact seating once to learn its coverage ``cov_g``,
    then solve ``maximize total seat score subject to seating at least cov_g
    sources``. Greedy's own matching is feasible for that program, so the LAP
    optimum is >= greedy on score AND >= cov_g on coverage — both required
    parities hold by construction, in every scenario.

    The cardinality floor is encoded structurally, not as a penalty: the cost
    matrix gets exactly ``n_src - cov_g`` shared "unseated" columns (cost 0,
    any source may take one). With only that many escape hatches, at most
    ``n_src - cov_g`` sources can sit out, so at least ``cov_g`` land on real
    segments. Sources whose real seats beat 0 still fill in freely, so
    coverage floats up to the score-max level when that exceeds ``cov_g``. A
    source with NO option at all (neither tier) has only forbidden real cells,
    so it takes one of the unseated columns — no reservation, exactly like
    greedy. With more sources than segments, ``cov_g`` == segment count and
    the LAP fills every segment with the globally best-fitting sources while
    the rest sit out — the same coverage target greedy reached seat-by-seat.

    TIER SHIFT. Primary seats keep their real score. Exempt seats are shifted
    down by ``SHIFT = (Pmax - Pmin) + (Emax - Emin) + 1`` — one unit past the
    combined span of both tiers' scores, so ``Emax - SHIFT < Pmin`` and every
    exempt seat is strictly below every primary seat no matter the data (no
    hardcoded magic number: the shift is derived from the actual ranges and
    can't be crossed). The shift is a single constant, so it preserves the
    real spacing *within* the exempt tier; seating a source on an exempt drop
    only happens when it has no primary seat, and the coverage floor still
    forces it in when greedy seated it there.

    DETERMINISM. The cost matrix is built in ``sorted(by_source)`` row order,
    and each (source, segment) seat aggregates its candidates with the exact
    ``(-score, j, str(id))`` tie-break — so the matrix is a pure function of
    the candidate *content*, independent of the input list's order. Seat
    values are quantized to a 1e-9 grid, and a per-cell epsilon derived from
    (row rank, segment index) and scaled strictly below that grid is added, so
    genuine >1e-9 score gaps can never be flipped by the tie-break while exact
    ties resolve deterministically toward the lower segment index (matching
    greedy's ``j`` tie-break) — identical across platforms and input orders.
    """
    from scipy.optimize import linear_sum_assignment

    by_source: Dict[str, List[Dict]] = {}
    rows_of: Dict[str, List[int]] = {}       # source -> candidate list indices
    for i, c in enumerate(candidates):
        src = str(c.get("video_file"))
        by_source.setdefault(src, []).append(c)
        rows_of.setdefault(src, []).append(i)

    _score = score_cache.score if score_cache is not None else _score_candidate

    sources = sorted(by_source)              # deterministic row order
    n_src = len(sources)
    n_seg = len(profiles)
    n_cand = len(candidates)

    # --- W2-8 vectorized seat scoring. ------------------------------------
    # The (candidate, segment) score matrix is assembled FROM the exact
    # floats `_score` returns -- `_score_candidate` reads the profile only
    # through its target and (wave 16) its 2dp-quantized `_segment_energy`,
    # so one real call per (candidate, distinct (target, energy) pair) covers
    # every (candidate, segment) pair, exactly like the memoized sweep did.
    # Deduping on target alone would smear one segment's energy score onto
    # every same-target segment -- the representative-column key MUST match
    # `_ScoreCache`'s profile axes. The only arithmetic applied on top is the
    # same short-candidate `- 0.18` and the same `best_raw - 0.35` exemption
    # threshold, as float64 ops bit-identical to the scalar originals. Never
    # re-derive score math here.
    targets = [p.get("target", "flow") for p in profiles]
    energies = [_segment_energy(p) for p in profiles]
    tcol: Dict[tuple, int] = {}
    rep_profiles: List[Dict] = []
    for p, t, e in zip(profiles, targets, energies):
        if (t, e) not in tcol:
            tcol[(t, e)] = len(rep_profiles)
            rep_profiles.append(p)
    S = np.empty((n_cand, len(rep_profiles)), dtype=np.float64)
    for k, p in enumerate(rep_profiles):
        for i, c in enumerate(candidates):
            S[i, k] = _score(c, p)
    col = np.array([tcol[(t, e)] for t, e in zip(targets, energies)],
                   dtype=np.intp)
    M = S[:, col]                            # (n_cand, n_seg) seat scores

    # Auction-best approximation per segment (raw scores, no duration
    # penalty), used only for the drop exemption threshold.
    best_raw = S.max(axis=0)[col]

    # Short-candidate penalty: candidates shorter than 0.55x the segment lose
    # 0.18. A missing "duration" defaulted to the segment's own requirement,
    # which can never test short -- +inf encodes exactly that.
    req = np.array([max(0.05, p["duration"]) for p in profiles],
                   dtype=np.float64)
    cand_dur = np.empty(n_cand, dtype=np.float64)
    for i, c in enumerate(candidates):
        if "duration" in c:
            cand_dur[i] = max(0.05, float(c["duration"]))
        else:
            cand_dur[i] = np.inf
    short = cand_dur[:, None] < (req * 0.55)[None, :]
    M = np.where(short, M - 0.18, M)

    is_drop = np.fromiter((t == "drop" for t in targets), dtype=bool,
                          count=n_seg)
    exempt_m = is_drop[None, :] & (M < (best_raw - 0.35)[None, :])

    # Per (row r, segment j) keep the single best candidate in each tier under
    # the exact greedy tie-break key (-score, j, str(id)); j is constant within
    # a cell, so this reduces to "highest score, then lowest str(id)", with
    # first-encounter (source-local candidate order) breaking full ties --
    # rank candidates once by (str(id), position) and take, per segment, the
    # lowest rank among the EXACTLY-equal maxima.
    prim_cell: Dict[tuple, tuple] = {}       # (r, j) -> (score, candidate)
    exempt_cell: Dict[tuple, tuple] = {}     # (r, j) -> (score, candidate)

    for r, src in enumerate(sources):
        cands_r = by_source[src]
        rows = np.array(rows_of[src], dtype=np.intp)
        k = len(cands_r)
        sub = M[rows, :]                     # (k, n_seg)
        ex = exempt_m[rows, :]
        sids = [str(c.get("id")) for c in cands_r]
        perm = sorted(range(k), key=lambda i: (sids[i], i))
        rank = np.empty(k, dtype=np.intp)
        for m, i_loc in enumerate(perm):
            rank[i_loc] = m
        for tier_mask, cell in ((~ex, prim_cell), (ex, exempt_cell)):
            has = tier_mask.any(axis=0)
            if not has.any():
                continue
            masked = np.where(tier_mask, sub, -np.inf)
            col_best = masked.max(axis=0)
            tie_rank = np.where((masked == col_best[None, :]) & tier_mask,
                                rank[:, None], k)
            win = tie_rank.min(axis=0)
            for j in np.nonzero(has)[0].tolist():
                i_loc = perm[int(win[j])]
                # Store the winner's own (possibly penalized) score as a
                # Python float -- the same value the scalar loop kept.
                cell[(r, int(j))] = (float(sub[i_loc, j]), cands_r[i_loc])

    if not prim_cell and not exempt_cell:
        return {}

    # --- Coverage floor: greedy's exact cardinality (pre-wave-15 seating). ---
    # Reproduces HEAD's greedy on the aggregated per-(source,segment) cells:
    # sources in descending best-seat order, each taking its highest-scoring
    # free primary seat, else its highest-scoring free exempt seat. We only
    # need the resulting COUNT to floor the LAP at, but computing it exactly
    # keeps the parity guarantee tight.
    prim_by_src: Dict[int, list] = {}
    exempt_by_src: Dict[int, list] = {}
    for (r, j), (score, _c) in prim_cell.items():
        prim_by_src.setdefault(r, []).append((score, j))
    for (r, j), (score, _c) in exempt_cell.items():
        exempt_by_src.setdefault(r, []).append((score, j))
    for d in (prim_by_src, exempt_by_src):
        for lst in d.values():
            lst.sort(key=lambda t: (-t[0], t[1]))   # (-score, j)
    rows_with_opts = sorted(set(prim_by_src) | set(exempt_by_src))
    g_order = sorted(
        rows_with_opts,
        key=lambda r: (-(prim_by_src[r][0][0] if prim_by_src.get(r)
                         else exempt_by_src[r][0][0]), sources[r]))
    g_taken: set = set()
    cov_g = 0
    for r in g_order:
        seat = next((j for _s, j in prim_by_src.get(r, []) if j not in g_taken), None)
        if seat is None:
            seat = next((j for _s, j in exempt_by_src.get(r, []) if j not in g_taken),
                        None)
        if seat is not None:
            g_taken.add(seat)
            cov_g += 1

    # --- Tier shift: exempt strictly below primary, from actual ranges. ---
    prim_scores = [v[0] for v in prim_cell.values()]
    exempt_scores = [v[0] for v in exempt_cell.values()]
    if prim_scores and exempt_scores:
        p_span = max(prim_scores) - min(prim_scores)
        e_span = max(exempt_scores) - min(exempt_scores)
        shift = p_span + e_span + 1.0        # Emax - shift < Pmin, guaranteed
    else:
        shift = 0.0

    # --- Cost matrix: rows = sources, cols = [segments | (n_src-cov_g) dummies].
    # Minimizing cost == maximizing value. Shared dummy cols (cost 0) cap the
    # number of unseated sources at n_src - cov_g -> at least cov_g get seated.
    BIG = 1.0e6
    n_dummy = n_src - cov_g
    n_cols = n_seg + n_dummy
    cost = np.full((n_src, n_cols), BIG, dtype=float)
    for col in range(n_seg, n_cols):
        cost[:, col] = 0.0                   # any source may sit out here

    GRID = 1.0e-9
    # Total epsilon across the matrix must stay strictly under one grid step so
    # it can only order exact ties, never flip a genuine >1e-9 score gap.
    eps_unit = GRID / (16.0 * (n_src * max(n_cols, 1) + 1))

    def _fill(cell: Dict[tuple, tuple], delta: float) -> None:
        for (r, j), (score, _c) in cell.items():
            v = score - delta
            qv = round(v / GRID) * GRID
            # +eps increasing in (r, j): lower segment index is cheaper, so
            # exact ties resolve toward the lower j (greedy's tie-break).
            cost[r, j] = -qv + (r * n_cols + j) * eps_unit

    # Primary first (real value); exempt only where no primary holds the cell,
    # so a drop seat above threshold counts as primary exactly as today.
    _fill(prim_cell, 0.0)
    _fill({k: v for k, v in exempt_cell.items() if k not in prim_cell}, shift)

    row_ind, col_ind = linear_sum_assignment(cost)

    reserved: Dict[int, Dict] = {}
    for r, col in zip(row_ind.tolist(), col_ind.tolist()):
        if col >= n_seg:
            continue                         # dummy column -> source unseated
        j = col
        seat = prim_cell.get((r, j)) or exempt_cell.get((r, j))
        if seat is not None:                 # skip forbidden cells (no option)
            reserved[j] = seat[1]
    return reserved


# Segments shorter than this get no transition window on either side: the
# out/in chains occupy ~0.2s and need normal footage around them to read.
_TRANSITION_MIN_SEG = 0.3

# semantic_fx only: minimum |camera_dir_x| before a whip pan's direction is
# overridden to continue the outgoing clip's measured horizontal camera drift.
# Below it the rng coin flip stands. Analysis v14 put camera_dir_x on a
# 1.0 screen-height/second full scale, so 0.10 = the camera drifting a tenth
# of the frame height per second — a visible pan, not measurement noise.
_WHIP_DIR_MATCH_MIN = 0.10


def _whip_camera_dir_x(clip: PlannedClip) -> Optional[float]:
    """Horizontal camera-drift component for the whip-pan direction match.

    Prefers the native-pair measurement (sub-15fps sources measure every
    consecutive frame in the window instead of only the 3-8 sample positions,
    which matters most exactly where per-frame displacement is largest); None
    when analysis produced neither."""
    for key in ("camera_dir_x_native", "camera_dir_x"):
        value = clip.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _assign_boundary_transitions(planned: List[PlannedClip],
                                 semantic_fx: bool = False) -> None:
    """Label consecutive clip pairs with split-transition specs.

    Each transition is rendered as two per-segment effect chains (an out-chain
    on clip i's tail and an in-chain on clip i+1's head), so assembly stays
    concat stream-copy. Kept occasional on purpose; deterministic per boundary.

    semantic_fx=False is the byte-identical historical path. semantic_fx=True
    adds two content-aware rules: (a) whip pans never land on a boundary
    where either side is a still image (a whip sells camera motion; a frozen
    frame can't), and (b) when the outgoing clip carries a meaningful
    measured horizontal camera drift, the whip's direction continues that
    drift across the cut instead of the coin flip. Both are safe for the rng
    invariants because every boundary owns its own _stable_rng stream and
    nothing draws from it after the direction flip.
    """
    for i in range(len(planned) - 1):
        a, b = planned[i], planned[i + 1]
        if (float(a.get("final_duration", 0.0)) < _TRANSITION_MIN_SEG
                or float(b.get("final_duration", 0.0)) < _TRANSITION_MIN_SEG):
            continue
        t_out = str(a.get("target", "flow"))
        t_in = str(b.get("target", "flow"))
        impact_in = _clamp(b.get("impact", 0.5), default=0.5)
        rng = _stable_rng("transition", i, t_out, t_in)
        roll = rng.random()

        kind = None
        if t_in == "drop":
            if t_out == "drop":
                if roll < 0.40:
                    kind = "whip_pan"
                elif roll < 0.55:
                    kind = "glitch_cut"
            else:
                if roll < 0.30:
                    kind = "whip_pan"
                elif roll < 0.40:
                    kind = "dip_flash"
                elif impact_in >= 0.6 and roll < 0.55:
                    kind = "glitch_cut"
        elif t_out == "soft" and t_in == "soft":
            if roll < 0.30:
                kind = "dip_black"
        elif impact_in >= 0.65:
            if roll < 0.25:
                kind = "glitch_cut"
        if kind is None:
            continue
        if (semantic_fx and kind == "whip_pan"
                and "still" in (_candidate_media_type(a), _candidate_media_type(b))):
            # A whip pan fakes a fast camera move; against a frozen frame it
            # reads as a rendering glitch. Skipping here is rng-safe: each
            # boundary has its own stream, and the direction draw below is
            # the stream's last consumer.
            print(f"   🎛 semantic_fx boundary {i}: whip_pan vetoed (still image on the cut)")
            continue

        spec: TransitionSpec = {"type": kind}
        if kind == "whip_pan":
            # Same direction on both sides = continuous camera motion across
            # the cut.
            direction = "right" if rng.random() < 0.5 else "left"
            if semantic_fx:
                dir_x = _whip_camera_dir_x(a)
                if dir_x is not None and abs(dir_x) >= _WHIP_DIR_MATCH_MIN:
                    # Continue the outgoing clip's on-screen drift. Measured
                    # +x = content moving right; direction 'left' sweeps the
                    # crop window left, which keeps content moving right —
                    # so the sign inverts. The coin flip above is still drawn
                    # (dead) to keep the stream's draw positions fixed.
                    direction = "left" if dir_x > 0 else "right"
                    print(f"   🎛 semantic_fx boundary {i}: whip_pan direction "
                          f"'{direction}' matched to camera drift (dir_x {dir_x:+.2f})")
            spec["direction"] = direction
        a["transition_out"] = dict(spec)
        b["transition_in"] = dict(spec)


def summarize_clip_plan(plan: Sequence[Dict],
                        video_files: Sequence[str] | None = None,
                        candidates: Sequence[Dict] | None = None) -> Dict:
    if not plan:
        return {"clip_count": 0, "targets": {}, "ai_tagged": 0}
    targets = Counter(str(item.get("target", "flow")) for item in plan)
    ai_tagged = sum(1 for item in plan if item.get("ai_analyzed"))
    # A source seen in a duo pane has been seen: partners count for coverage.
    seen_files = [item.get("video_file") for item in plan]
    seen_files += [item["partner"].get("video_file") for item in plan
                   if isinstance(item.get("partner"), dict)]
    source_count = len(set(seen_files))
    duo_count = sum(1 for item in plan if item.get("partner"))
    transitions = Counter(
        str((item.get("transition_out") or {}).get("type"))
        for item in plan if item.get("transition_out")
    )
    retimes = Counter(
        str((item.get("retime") or {}).get("kind"))
        for item in plan if item.get("retime")
    )
    summary = {
        "clip_count": len(plan),
        "targets": dict(targets),
        "ai_tagged": ai_tagged,
        "source_count": source_count,
        "transitions": dict(transitions),
        "retimes": dict(retimes),
    }
    if duo_count:
        summary["duos"] = duo_count
    if video_files is not None:
        import os
        usage = Counter(os.path.basename(str(f)) for f in seen_files)
        candidate_files = {os.path.basename(str(c.get("video_file")))
                           for c in (candidates or [])}
        all_files = [os.path.basename(str(p)) for p in video_files]
        summary["source_usage"] = dict(sorted(usage.items(),
                                              key=lambda kv: (-kv[1], kv[0])))
        summary["sources_never_selected"] = sorted(
            f for f in all_files if f not in usage and f in candidate_files)
        summary["sources_without_candidates"] = sorted(
            f for f in all_files if f not in candidate_files)
    return summary


def _build_segment_profiles(cut_times: np.ndarray, segment_durations: np.ndarray, beat_info: Dict) -> List[SegmentProfile]:
    beat_times = np.asarray(beat_info.get("times", []), dtype=float)
    downbeat_times = np.asarray(beat_info.get("downbeat_times", []), dtype=float)
    energy_profile = beat_info.get("energy_profile") or {}
    rhythm_data = beat_info.get("rhythm_data") or {}
    sections = beat_info.get("sections") or []

    wave = np.asarray(energy_profile.get("wave", []), dtype=float)
    arc = np.asarray(energy_profile.get("arc", []), dtype=float)
    # EBU momentary loudness per beat (0..1); a stage-2 feature threaded through
    # energy_profile alongside wave/arc. Always present going forward, but ABSENT
    # in beat_info cached before it landed — _interp_feature's default (0.5) keeps
    # those runs neutral, and effects (_loudness_gain) reads the resulting clip key.
    loudness = np.asarray(energy_profile.get("loudness", []), dtype=float)
    impact = np.asarray(rhythm_data.get("impact_strength", []), dtype=float)
    rhythm = np.asarray(rhythm_data.get("combined_strength", []), dtype=float)
    novelty = np.asarray(rhythm_data.get("novelty_strength", []), dtype=float)

    profiles: List[SegmentProfile] = []
    for i, duration in enumerate(segment_durations):
        start = float(cut_times[i])
        end = float(cut_times[i + 1])
        mid = (start + end) * 0.5
        local_wave = _interp_feature(mid, beat_times, wave, 0.5)
        local_arc = _interp_feature(mid, beat_times, arc, 0.5)
        local_loudness = _interp_feature(mid, beat_times, loudness, 0.5)
        local_impact = _interp_feature(start, beat_times, impact, 0.5)
        local_rhythm = _interp_feature(start, beat_times, rhythm, 0.5)
        local_novelty = _interp_feature(start, beat_times, novelty, 0.4)
        section = _section_at(sections, mid)
        target = _target_for_segment(section, local_wave, local_impact, local_rhythm, local_novelty, local_arc)
        profile: SegmentProfile = {
            "index": i,
            "start": start,
            "end": end,
            "duration": float(duration),
            "mid": mid,
            "wave": local_wave,
            "impact": local_impact,
            "rhythm": local_rhythm,
            "novelty": local_novelty,
            "arc": local_arc,
            "loudness": local_loudness,
            "section": section,
            "section_type": section.get("type", "body") if section else "body",
            "target": target,
            "is_downbeat": bool(
                downbeat_times.size and float(np.min(np.abs(downbeat_times - start))) <= 0.05
            ),
        }
        profiles.append(profile)
    return profiles


def _interp_feature(time_s: float, beat_times: np.ndarray, values: np.ndarray, default: float) -> float:
    if beat_times.size == 0 or values.size != beat_times.size:
        return default
    return _clamp(np.interp(time_s, beat_times, values, left=float(values[0]), right=float(values[-1])), default=default)


def _section_at(sections: Sequence[Section], time_s: float) -> Dict:
    for section in sections:
        if float(section.get("start", 0.0)) <= time_s < float(section.get("end", 0.0)):
            return section
    return sections[-1] if sections else {}


def _target_for_segment(section: Section, wave: float, impact: float, rhythm: float, novelty: float, arc: float) -> str:
    section_type = section.get("type", "body")
    if section_type in {"drop", "finale"} and (wave >= 0.58 or impact >= 0.55):
        return "drop"
    if impact >= 0.76 or (wave >= 0.78 and rhythm >= 0.62):
        return "drop"
    if section_type in {"breakdown", "intro", "outro"} and wave <= 0.54:
        return "soft"
    if wave <= 0.32 and impact <= 0.48:
        return "soft"
    if section_type in {"bridge", "hook"} or novelty >= 0.68 or (arc >= 0.62 and wave >= 0.48):
        return "build"
    if wave >= 0.58 and rhythm >= 0.54:
        return "rhythm"
    return "flow"


def _record_semantic(candidate: Dict, recent_clusters: deque,
                     recent_embeds: deque,
                     embed_arrays: Dict[int, np.ndarray]) -> None:
    """Push a just-picked candidate's cluster/embedding onto the MMR windows.

    Missing keys append nothing (graceful degradation); the embedding array is
    the one precomputed up front, never re-derived here.
    """
    cluster = candidate.get("visual_cluster")
    if cluster is not None:
        recent_clusters.append(cluster)
    emb = embed_arrays.get(id(candidate))
    if emb is not None:
        recent_embeds.append(emb)


def _semantic_penalty(candidate: Dict, semantic_variety: float,
                      recent_clusters: deque, recent_embeds: deque,
                      embed_arrays: Dict[int, np.ndarray]) -> float:
    """Diversity penalty for a candidate given the recent-pick windows.

    Two terms, both zero when the candidate lacks the relevant key (graceful
    degradation) or the window is empty:
      * cluster-run: -0.14 * sv when this candidate's visual_cluster is one of
        the last few picked clusters.
      * windowed MMR: -0.22 * sv * max(0, max_cos_sim - 0.55) against the last
        few picked embeddings. Each dot product is rounded to 4dp BEFORE the
        max — that quantization is the determinism firewall against float
        reassociation drift, so it must not be removed.
    Returns a POSITIVE amount to subtract from the score.
    """
    penalty = 0.0
    cluster = candidate.get("visual_cluster")
    if cluster is not None and cluster in recent_clusters:
        penalty += 0.14 * semantic_variety
    e_cand = embed_arrays.get(id(candidate))
    if e_cand is not None and recent_embeds:
        max_sim = max(round(float(np.dot(e_cand, e_recent)), 4)
                      for e_recent in recent_embeds)
        penalty += 0.22 * semantic_variety * max(0.0, max_sim - 0.55)
    return penalty


def _candidate_media_type(candidate: Dict) -> str:
    """'still' | 'gif' | 'video' for a candidate.

    Reads the analysis-time media_type field; candidates from sidecars that
    predate it (or external callers) fall back to the same extension-based
    derivation video_analysis uses, so the answer is identical either way.
    """
    media_type = candidate.get("media_type")
    if media_type:
        return str(media_type)
    path = str(candidate.get("video_file") or "")
    if os.path.splitext(path)[1].lower() == ".gif":
        return "gif"
    from ffmpeg_processing import is_image_source
    return "still" if is_image_source(path) else "video"


def _media_aware_adjustment(candidate: Dict, profile: SegmentProfile,
                            canvas_w: int,
                            stats: "Counter | None" = None) -> float:
    """Signed auction score delta for media-aware planning (media_aware only).

    Deliberately applied OUTSIDE ``_ScoreCache`` alongside the other auction
    penalties (recency/usage/PF/semantic), so the cache keeps its exact
    (candidate, target, quantized-energy) key semantics. Three terms, each
    zero when its data is missing (graceful degradation):

      * upscale: -0.12 * log2(canvas_w / source_w) for sources narrower than
        the output canvas — every halving of resolution costs another 0.12.
        Applies to stills too (wave 2): their probe width/height are real
        (and EXIF-oriented), so a low-res photo pays the same blow-up cost
        as a low-res clip.
      * loop-seam: -0.20 * loop_seam, scaled by how many EXTRA passes the
        segment forces (min(2, wraps-1)) — a GIF wrapping 1.1× pays almost
        nothing, 2× pays the full seam penalty, 3×+ pays double. Seamless
        GIFs (seam≈0) stay preferred fillers at any length.
      * native-kinetic: sub-15fps sources carry ``kinetic_native`` (optical
        flow over EVERY consecutive native frame pair in the window, not just
        the pairs at the 3-8 metric sample positions — a density refinement
        that matters most at 10-15fps, where a window holds few frames).
        Since analysis v14 both measurements share the same native dt, so the
        delta is small; before v14 it corrected an outright broken ``kinetic``.
        The cached score already paid the wave-16 energy term with the sampled
        ``kinetic``; this adds exactly the difference so the effective energy
        term uses the denser measurement. Wave 2 completes the swap for
        untagged DROP segments, whose cached formula also paid 0.10 * kinetic
        directly.

    ``stats`` (when provided) counts per-evaluation fires for the one debug
    line the planner prints.
    """
    delta = 0.0
    media_type = _candidate_media_type(candidate)

    if canvas_w > 0:
        try:
            source_w = float(candidate.get("source_width") or 0.0)
        except (TypeError, ValueError):
            source_w = 0.0
        if 0.0 < source_w < canvas_w:
            delta -= 0.12 * math.log2(canvas_w / source_w)
            if stats is not None:
                stats["upscale"] += 1

    if media_type == "gif":
        seam = _clamp(candidate.get("loop_seam", 0.0))
        try:
            loop_duration = float(candidate.get("video_duration") or 0.0)
            segment_duration = float(profile["duration"])
        except (TypeError, ValueError, KeyError):
            loop_duration = segment_duration = 0.0
        if seam > 0.0 and loop_duration > 0.0 and segment_duration > loop_duration:
            wrap_factor = min(2.0, segment_duration / loop_duration - 1.0)
            delta -= 0.20 * seam * wrap_factor
            if stats is not None:
                stats["loop_seam"] += 1

    kinetic_native = candidate.get("kinetic_native")
    if kinetic_native is not None:
        e_seg = _segment_energy(profile)
        # Mirror _score_candidate's e_clip expression exactly, then swap it
        # for the native measurement: delta = new energy term - old one.
        e_old = _clamp(candidate.get("kinetic", candidate.get("motion", 0.0)))
        e_new = _clamp(kinetic_native)
        if e_new != e_old:
            delta += 0.18 * (abs(e_old - e_seg) - abs(e_new - e_seg))
            if stats is not None:
                stats["kinetic_native"] += 1
            # Untagged drop segments also paid 0.10 * kinetic inside the
            # cached formula (_score_candidate's measured-signal branch);
            # swap that term too, mirroring its exact fallback chain.
            if (str(profile.get("target", "flow")) == "drop"
                    and not candidate.get("ai_analyzed")):
                delta += 0.10 * (e_new - e_old)

    return delta


def _choose_candidate(
    candidates: Sequence[Dict],
    profile: SegmentProfile,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    file_rate: float = 0.012,
    file_cap: float = 0.18,
    score_cache: "_ScoreCache | None" = None,
    variety: float = 0.0,
    source_count: int = 1,
    pf_ewma: "_FairShareEWMA | None" = None,
    semantic_variety: float = 0.0,
    recent_clusters: "deque | None" = None,
    recent_embeds: "deque | None" = None,
    embed_arrays: "Dict[int, np.ndarray] | None" = None,
    media_aware: bool = False,
    canvas_w: int = 0,
    media_stats: "Counter | None" = None,
    spent: "Dict[object, List[tuple]] | None" = None,
) -> Dict | None:
    best_candidate = None
    best_score = -999.0
    rng = _stable_rng(index, profile.get("target"), profile.get("start"))
    _score = score_cache.score if score_cache is not None else _score_candidate

    # Proportional-fair usage denominator, computed once per segment (O(sources)
    # once, not per candidate). variety=0 keeps the legacy linear file penalty.
    use_pf = variety > 0.0 and pf_ewma is not None
    pf_total = pf_ewma.total(index) if use_pf else 0.0

    for candidate in candidates:
        score = _score(candidate, profile)
        cid = candidate.get("id")
        video_file = candidate.get("video_file")

        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.10
        # Wave 17: unbounded candidate-identity reuse cost + footage
        # exhaustion. Ledger-dependent, so it lives OUTSIDE _ScoreCache.
        score -= _reuse_penalty(candidate, usage, variety, spent)
        if use_pf:
            share = pf_ewma.value(video_file, index) / max(1e-9, pf_total)
            pressure = share * source_count
            score -= variety * 0.30 * max(0.0, pressure - 1.0)
        else:
            score -= min(file_cap, usage[video_file] * file_rate)

        # Short-source penalty. Reachable for GIFs and short scenes (their
        # candidate "duration" is the real analysis-window length); NOT for
        # stills, whose duration video_analysis pins at 5.0 so they can fill
        # any segment — firing would need a segment > 9.09s, above every
        # *_max_hold ceiling in auto_mode/__init__.py. See the note in
        # _plan_coverage_reservations.
        required_source = max(0.05, profile["duration"])
        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        if semantic_variety > 0.0:
            score -= _semantic_penalty(candidate, semantic_variety,
                                       recent_clusters, recent_embeds,
                                       embed_arrays)

        if media_aware:
            score += _media_aware_adjustment(candidate, profile, canvas_w,
                                             media_stats)

        score += rng.random() * 0.015
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_candidate(candidate: Dict, profile: SegmentProfile) -> float:
    target = profile.get("target", "flow")
    semantic = candidate.get("semantic") or {}
    tags = {str(t).lower() for t in candidate.get("tags", [])}
    quality = _clamp(candidate.get("quality_score", semantic.get("visual_quality", 0.5)), default=0.5)
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    character = _clamp(semantic.get("character_focus", 0.0))
    combat = _clamp(semantic.get("combat", 0.0))
    chase = _clamp(semantic.get("chase", 0.0))
    explosion = _clamp(semantic.get("explosion", 0.0))

    tag_bonus = 0.0
    if target in tags:
        tag_bonus += 0.08
    if target == "drop" and tags.intersection({"action", "combat", "chase", "explosion", "hype"}):
        tag_bonus += 0.12
    if target == "soft" and tags.intersection({"soft", "beauty", "sad"}):
        tag_bonus += 0.10
    if target == "build" and tags.intersection({"tension", "transition"}):
        tag_bonus += 0.10

    if target == "drop":
        if candidate.get("ai_analyzed"):
            match = 0.46 * action + 0.16 * motion + 0.12 * combat + 0.10 * chase + 0.08 * explosion + 0.08 * quality
        else:
            # Wave 16: without Qwen, video_analysis FABRICATES the semantic
            # fields (chase=motion*0.6, combat=explosion=0.0), so the tagged
            # formula silently degenerates to a motion echo plus dead weight.
            # Untagged drops instead read real measured signals. The flow keys
            # are normally always present (ANALYSIS_VERSION v11 recomputes
            # stale sidecars); the plain-motion fallback engages only when the
            # flow measurement FAILED for a candidate (video_analysis omits
            # the keys on that path), so this branch works either way.
            subject = _clamp(candidate.get("subject_motion", motion))
            kinetic = _clamp(candidate.get("kinetic", motion))
            contrast = _clamp(candidate.get("contrast", 0.5), default=0.5)
            match = 0.46 * action + 0.16 * motion + 0.12 * subject + 0.10 * kinetic + 0.08 * quality + 0.08 * contrast
    elif target == "soft":
        match = 0.45 * beauty + 0.18 * soft + 0.13 * character + 0.14 * (1.0 - action) + 0.10 * quality
    elif target == "build":
        match = 0.40 * tension + 0.18 * motion + 0.15 * character + 0.14 * action + 0.13 * quality
    elif target == "rhythm":
        match = 0.30 * action + 0.24 * motion + 0.18 * quality + 0.16 * tension + 0.12 * beauty
    else:
        match = 0.28 * quality + 0.24 * beauty + 0.20 * action + 0.16 * tension + 0.12 * soft

    brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
    visibility_penalty = 0.0
    if brightness < 0.13:
        visibility_penalty += 0.18
    if quality < 0.24:
        visibility_penalty += 0.16

    # Wave 16: continuous cross-modal energy matching, every target. Calm
    # clips court quiet segments, kinetic clips court loud ones; the reward
    # peaks (+0.18) at a perfect match and fades linearly with the gap.
    # _segment_energy is 2dp-quantized — the determinism firewall that also
    # keys _ScoreCache and the reservation-matrix column dedup, so all three
    # must keep consuming the same helper.
    e_seg = _segment_energy(profile)
    e_clip = _clamp(candidate.get("kinetic", candidate.get("motion", 0.0)))
    match += 0.18 * (1.0 - abs(e_clip - e_seg))

    return _clamp(match + tag_bonus + 0.12 * quality - visibility_penalty, lo=-1.0, hi=2.0)


def _rebase_subject_anchor(candidate: Dict, start_time: float,
                           source_duration: float):
    """Copy the candidate's subject_anchor with a segment-local tracked path.

    The analysis path ("path", from video_analysis) is timestamped relative
    to the CANDIDATE's start in the source, but the extractor only knows the
    segment's start_time. The rebased samples go under the key "path_seg":
    a list of [t_seg, cx, cy] where t_seg = (candidate_start + t_rel) -
    start_time, i.e. seconds from the segment's first frame, cx/cy still
    normalized 0..1. Samples outside [-0.5, source_duration + 0.5] are
    dropped.

    "path_seg" (not "path") is deliberately the renderer's trigger for the
    tracked pan: an anchor that still carries only the raw candidate-relative
    "path" (older plans, external callers) simply keeps the static offset
    crop instead of panning on a wrong clock. The shared candidate dict is
    never mutated — the copy happens only when a rebase actually attaches.
    """
    anchor = candidate.get("subject_anchor")
    if not isinstance(anchor, dict):
        return anchor
    path = anchor.get("path")
    if not isinstance(path, (list, tuple)) or not path:
        return anchor
    try:
        cand_start = float(candidate.get("start", 0.0) or 0.0)
    except (TypeError, ValueError):
        cand_start = 0.0
    rebased = []
    for sample in path:
        try:
            t_rel = float(sample[0])
            cx = float(sample[1])
            cy = float(sample[2])
        except (TypeError, ValueError, IndexError):
            continue
        t_seg = (cand_start + t_rel) - float(start_time)
        if -0.5 <= t_seg <= float(source_duration) + 0.5:
            rebased.append([round(t_seg, 4), cx, cy])
    if not rebased:
        return anchor
    out = dict(anchor)
    out["path_seg"] = rebased
    return out


def _plan_source_window(candidate: Dict, profile: SegmentProfile,
                        spent: "Dict[object, List[tuple]] | None" = None) -> tuple:
    """(start_time, source_duration) for a candidate serving a segment.

    This is the single authority for source-window semantics — the primary
    clip (_materialize_clip) and a duo partner (_materialize_partner) must
    place their windows identically, so the math lives here once.

    ANCHOR WINDOW (unchanged, and still the fallback). The window is placed so
    the candidate's salient instant (``peak_time`` on drop/build, ``center``
    otherwise) lands at a fixed fraction ``align`` into the segment.

    SUB-WINDOW LEDGER (wave 17). The anchor formula alone has no memory, so
    every reuse of one candidate replays the same frames: for a shared anchor
    ``a`` and durations d1 < d2 the two windows are strictly NESTED
    (``start1 - start2 = (d2-d1)*align > 0``, ``end1 - end2 = (d1-d2)*(1-align)
    < 0``), always straddling the same frame — the direct cause of "clips loop
    noticeably". Analysis windows run up to 5.2s (``_make_candidate_windows``)
    while segments average ~1.2s, so a candidate typically holds ~4 disjoint
    segment-length windows the anchor formula structurally cannot reach.

    When ``spent`` is supplied (the per-run ledger of already-used source
    intervals, keyed by candidate id) and the candidate's analysis span is at
    least 2.2x the needed duration, this scans that span on a ``0.85 * d``
    stride and returns the FIRST offset whose overlap with every spent
    interval is under ``0.35 * d`` — a deterministic left-to-right scan, no
    rng. A span too short to vary, or one whose sub-windows are all spent,
    falls back to the anchor window, so behaviour degrades to the legacy
    formula instead of failing. The returned start is clamped exactly as
    before (video_processor re-clamps against the probed duration).
    """
    source_duration = max(0.05, float(profile["duration"]))
    video_duration = max(source_duration, float(candidate.get("video_duration", source_duration)))
    target = profile.get("target", "flow")

    if target == "drop":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.36
    elif target == "soft":
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50
    elif target == "build":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.48
    else:
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.44

    start_time = anchor - source_duration * align

    if spent is not None:
        used = spent.get(candidate.get("id"))
        span = _candidate_span(candidate)
        if used and span >= source_duration * 2.2:
            try:
                span_start = float(candidate.get("start", 0.0) or 0.0)
            except (TypeError, ValueError):
                span_start = 0.0
            stride = source_duration * 0.85
            tolerance = source_duration * 0.35
            steps = int((span - source_duration) / stride) + 1
            for k in range(steps):
                t = span_start + k * stride
                if all(min(t + source_duration, e) - max(t, s) < tolerance
                       for s, e in used):
                    start_time = t
                    break

    start_time = max(0.0, min(start_time, max(0.0, video_duration - source_duration)))
    return start_time, source_duration


def _candidate_span(candidate: Dict) -> float:
    """Extent of the candidate's analysis window in source seconds (>= 0.0).

    Stills carry start == end == 0.0 (video_analysis gives them one zero-width
    window), so their span is 0.0 and every span-gated term below is inert for
    them — exactly right: a single frame has no unspent footage to find.
    """
    try:
        span = float(candidate.get("end", 0.0) or 0.0) - float(candidate.get("start", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(span) or span <= 0.0:
        return 0.0
    return span


def _record_source_window(spent: "Dict[object, List[tuple]] | None",
                          candidate: Dict, start_time: float,
                          source_duration: float) -> None:
    """Book one materialized window into the sub-window ledger.

    Called once per pick (and once more for a duo partner) from
    ``build_planned_clip_sequence``'s sequential loop, alongside the usage /
    recency / PF bookkeeping. Ledger state is therefore a pure function of the
    pick order — no rng, deterministic replay.
    """
    if spent is None:
        return
    spent.setdefault(candidate.get("id"), []).append(
        (float(start_time), float(start_time) + float(source_duration)))


def _reuse_penalty(candidate: Dict, usage: Counter, variety: float,
                   spent: "Dict[object, List[tuple]] | None") -> float:
    """Per-candidate reuse cost (wave 17): log-growth + footage exhaustion.

    Replaces the pre-wave-17 ``min(0.28, usage[cid] * 0.10)``, which saturated
    at the third use — after that the only unbounded pressure was ``pf_ewma``,
    and that is keyed on ``video_file``, not on candidate identity, so a single
    candidate inside a favoured source could be picked without limit. The log
    term never saturates and scales with the variety slider.

    The exhaustion term prefers candidates that still hold unspent footage: it
    grows with the fraction of the candidate's analysis span the ledger has
    already served, capped at 0.25. Because it reads ledger state it MUST be
    applied OUTSIDE ``_ScoreCache`` (whose key is only
    (candidate, target, quantized energy)) — same rule as the recency, PF,
    semantic and media-aware terms.
    """
    penalty = 0.10 * math.log1p(usage[candidate.get("id")]) * (1.0 + 2.0 * variety)
    if spent:
        used = spent.get(candidate.get("id"))
        if used:
            span = _candidate_span(candidate)
            if span > 0.0:
                covered = sum(e - s for s, e in used)
                penalty += 0.25 * min(1.0, covered / max(0.1, span))
    return penalty


def _materialize_clip(candidate: Dict, profile: SegmentProfile, index: int,
                      score_cache: "_ScoreCache | None" = None,
                      spent: "Dict[object, List[tuple]] | None" = None) -> PlannedClip:
    final_duration = max(0.05, float(profile["duration"]))
    start_time, source_duration = _plan_source_window(candidate, profile, spent)
    target = profile.get("target", "flow")
    _score = score_cache.score if score_cache is not None else _score_candidate

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "target": target,
        "score": _score(candidate, profile),
        "candidate_id": candidate.get("id"),
        "tags": list(candidate.get("tags", [])),
        "subject_anchor": _rebase_subject_anchor(candidate, start_time,
                                                 source_duration),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
        "loudness": profile.get("loudness"),
        # Content profile for the semantic_fx effect gate: measured motion and
        # stage5 semantic scores travel with the plan so effects.py can veto
        # primitives that clash with the footage. Extra keys are inert to the
        # renderer; missing candidate fields forward as None (graceful).
        "kinetic": candidate.get("kinetic"),
        "subject_motion": candidate.get("subject_motion"),
        # Native-pair variants (sub-15fps sources measure every consecutive
        # frame in the window, not just the sample positions). Forwarded so
        # effects._sem_needs_motion / _sem_calm_subject can prefer them, the
        # same native-first rule _whip_camera_dir_x already applies.
        "kinetic_native": candidate.get("kinetic_native"),
        "subject_motion_native": candidate.get("subject_motion_native"),
        "motion": candidate.get("motion"),
        "action_score": candidate.get("action_score"),
        "beauty_score": candidate.get("beauty_score"),
        "semantic": candidate.get("semantic"),
        # v13 additions (semantic_fx consumers): analysis media type for the
        # still-image gates, and measured camera-drift direction for the
        # whip-pan direction match (native variants win on sub-15fps media).
        "media_type": candidate.get("media_type"),
        "camera_dir_x": candidate.get("camera_dir_x"),
        "camera_dir_y": candidate.get("camera_dir_y"),
        "camera_dir_x_native": candidate.get("camera_dir_x_native"),
        "camera_dir_y_native": candidate.get("camera_dir_y_native"),
    }


def _duo_pair_files(candidates: Sequence[Dict], target_size) -> set | None:
    """Sources eligible to occupy a duo pane, or None when duos are impossible.

    A pane source must be cross-orientation relative to the canvas (portrait
    display AR < 1.0 on a landscape canvas; the converse stacks on a portrait
    canvas), a real video (still images are a v1 exclusion — no Ken Burns
    interplay inside panes), and probeable. Display AR comes from the shared
    ffprobe cache so orientation matches what the renderer's fit ladder sees
    (SAR folded in). Fewer than two such sources means no segment can ever
    pair, so the whole feature short-circuits to None.
    """
    from ffmpeg_processing import get_cached_display_info, is_image_source

    tw, th = (1920.0, 1080.0)
    if target_size:
        try:
            tw, th = float(target_size[0]), float(target_size[1])
        except (TypeError, ValueError, IndexError):
            tw, th = (1920.0, 1080.0)
    want_portrait = tw >= th  # landscape (or square) canvas pairs portrait sources

    pair_files: set = set()
    for video_file in sorted({str(c.get("video_file")) for c in candidates
                              if c.get("video_file")}):
        if is_image_source(video_file):
            continue
        info = get_cached_display_info(video_file)
        if not info or info[1] <= 0:
            continue
        aspect = info[0] / info[1]
        if (aspect < 1.0) if want_portrait else (aspect > 1.0):
            pair_files.add(video_file)
    if len(pair_files) < 2:
        return None
    return pair_files


def _maybe_choose_duo_partner(
    candidates: Sequence[Dict],
    profile: SegmentProfile,
    primary: Dict,
    planned: List[PlannedClip],
    index: int,
    pair_files: set,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    file_rate: float,
    file_cap: float,
    score_cache: "_ScoreCache | None" = None,
    variety: float = 0.0,
    source_count: int = 1,
    pf_ewma: "_FairShareEWMA | None" = None,
    semantic_variety: float = 0.0,
    recent_clusters: "deque | None" = None,
    recent_embeds: "deque | None" = None,
    embed_arrays: "Dict[int, np.ndarray] | None" = None,
    media_aware: bool = False,
    canvas_w: int = 0,
    media_stats: "Counter | None" = None,
    spent: "Dict[object, List[tuple]] | None" = None,
) -> Dict | None:
    """Partner candidate for a duo segment, or None to render the primary solo.

    Eligibility gates (all deterministic), then a seeded rarity roll, then a
    mini-auction over candidates from OTHER pane-eligible sources. The primary
    pick is never touched — this only decides whether it gets a pane-mate.
    """
    if str(profile.get("target", "flow")) not in DUO_TARGETS:
        return None
    if _clamp(profile.get("impact", 0.0), default=0.0) < DUO_MIN_IMPACT:
        return None
    # Never two duo segments adjacent: a run of split screens reads as a
    # wall, not an accent. (planned[-1] is the previous output segment.)
    if planned and "partner" in planned[-1]:
        return None
    primary_file = primary.get("video_file")
    if primary_file not in pair_files:
        # The idiom pairs two cross-orientation clips; a landscape primary on
        # a landscape canvas keeps its normal solo fit.
        return None

    rng = _stable_rng("duo", index, profile.get("start"), profile.get("target"))
    if rng.random() >= DUO_RATE:
        return None

    from ffmpeg_processing import ANCHOR_MIN_CONFIDENCE

    primary_brightness = _clamp(primary.get("brightness", 0.5), default=0.5)
    best_candidate = None
    best_score = -999.0
    _score = score_cache.score if score_cache is not None else _score_candidate
    use_pf = variety > 0.0 and pf_ewma is not None
    pf_total = pf_ewma.total(index) if use_pf else 0.0
    for candidate in candidates:
        video_file = candidate.get("video_file")
        if video_file == primary_file or video_file not in pair_files:
            continue
        brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
        if abs(brightness - primary_brightness) > DUO_BRIGHTNESS_MAX_DELTA:
            continue

        # Same shape as the primary auction: fit score minus recent-use and
        # usage penalties (plus the short-source penalty — a looping pane on
        # a drop is worse than no pane at all). A pane appearance is an
        # appearance, so it carries the same PF pressure / semantic diversity
        # terms as a solo pick.
        score = _score(candidate, profile)
        cid = candidate.get("id")
        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.10
        # Same wave-17 reuse cost as the primary auction (shared helper — the
        # two blocks must never drift apart).
        score -= _reuse_penalty(candidate, usage, variety, spent)
        if use_pf:
            share = pf_ewma.value(video_file, index) / max(1e-9, pf_total)
            pressure = share * source_count
            score -= variety * 0.30 * max(0.0, pressure - 1.0)
        else:
            score -= min(file_cap, usage[video_file] * file_rate)
        required_source = max(0.05, profile["duration"])
        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        if semantic_variety > 0.0:
            score -= _semantic_penalty(candidate, semantic_variety,
                                       recent_clusters, recent_embeds,
                                       embed_arrays)

        if media_aware:
            score += _media_aware_adjustment(candidate, profile, canvas_w,
                                             media_stats)

        anchor = candidate.get("subject_anchor")
        try:
            confidence = float((anchor or {}).get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < ANCHOR_MIN_CONFIDENCE:
            score -= DUO_LOW_ANCHOR_PENALTY

        score += rng.random() * 0.015
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _materialize_partner(candidate: Dict, profile: SegmentProfile,
                         spent: "Dict[object, List[tuple]] | None" = None) -> PartnerClip:
    """The additive "partner" payload carried by a duo's planned clip.

    Window semantics are identical to the primary's (_plan_source_window is
    shared, ledger included), and the subject anchor is rebased onto the
    segment clock exactly as for the primary so the pane crop can track the
    subject.
    """
    start_time, source_duration = _plan_source_window(candidate, profile, spent)
    return {
        "video_file": candidate.get("video_file"),
        "start_time": start_time,
        "source_duration": source_duration,
        "candidate_id": candidate.get("id"),
        "source_name": candidate.get("source_name"),
        "loudness": profile.get("loudness"),
        "subject_anchor": _rebase_subject_anchor(candidate, start_time,
                                                 source_duration),
    }
