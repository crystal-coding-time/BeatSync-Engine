#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from typing import Dict, List, Sequence

import numpy as np


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


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
    variety: float = 0.0,
    speed_ramps: bool = False,
    lossless: bool = False,
    fps: float = 30.0,
) -> List[Dict]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to keep its old fallback sampling.

    variety=0 is the exact legacy quality auction (no coverage guarantee —
    weak sources can lose every pick). Any variety>0 reserves one segment per
    source so everything the user uploaded appears at least once, and scales
    the per-source reuse penalty toward an even spread at 1.0.
    """
    beat_info = beat_info or {}
    video_analysis = beat_info.get("video_analysis") or {}
    candidates = list(video_analysis.get("candidates") or [])
    candidates = [c for c in candidates if c.get("video_file")]
    if not candidates:
        return []

    cut_times_arr = np.asarray(cut_times, dtype=float)
    durations_arr = np.asarray(segment_durations, dtype=float)
    if cut_times_arr.size < 2 or durations_arr.size == 0:
        return []

    profiles = _build_segment_profiles(cut_times_arr, durations_arr, beat_info)
    variety = _clamp(variety)

    # Legacy reuse penalty: 0.012/use capped at 0.18 — too weak for coverage
    # (the cap means a source ~0.5 below the leaders can never win). With
    # variety on, the rate scales so a leader at its fair share of segments
    # yields to unused sources, and the cap goes away.
    file_rate, file_cap = 0.012, 0.18
    reservations: Dict[int, Dict] = {}
    if variety > 0.0:
        source_count = len({c.get("video_file") for c in candidates})
        fair_share = len(profiles) / max(1, source_count)
        file_rate = 0.012 + 1.3 * variety / max(1.0, fair_share)
        file_cap = float("inf")
        reservations = _plan_coverage_reservations(candidates, profiles)

    recent_ids = deque(maxlen=10)
    recent_videos = deque(maxlen=5)
    usage = Counter()
    planned: List[Dict] = []

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
            )
        if not candidate:
            continue
        planned_clip = _materialize_clip(
            candidate=candidate,
            profile=profile,
            index=i,
        )
        planned.append(planned_clip)
        recent_ids.append(candidate.get("id"))
        recent_videos.append(candidate.get("video_file"))
        usage[candidate.get("id")] += 1
        usage[candidate.get("video_file")] += 1

    if len(planned) != len(durations_arr):
        return []
    _assign_boundary_transitions(planned)
    if speed_ramps and not lossless:
        # Retime specs never reach precise mode: the ProRes branch extracts
        # plain windows and must stay pristine for external editing.
        _assign_retime_specs(planned, candidates, fps)
    return planned


def _assign_retime_specs(planned: List[Dict], candidates: Sequence[Dict],
                         fps: float) -> None:
    """Attach per-segment retime specs where the music and footage allow.

    Every gate here is deterministic and conservative: a ramp multiplies how
    much source a segment consumes, and a window that comes up short would
    break the frame-locked timeline (extraction guards against it, but a
    stripped ramp is a wasted plan — better to never attach one that can't
    run). Renderer probes (not the analysis metadata) decide runway.
    """
    from ffmpeg_processing import (
        get_cached_video_duration,
        get_cached_video_fps,
        is_image_source,
        retime_source_window,
        seconds_to_frame_count,
    )

    by_id = {c.get("id"): c for c in candidates}
    for clip in planned:
        video_file = clip.get("video_file")
        final_duration = float(clip.get("final_duration", 0.0))
        if not video_file or is_image_source(video_file):
            continue
        if final_duration < 0.6:
            # Sub-0.6s cuts don't hold a readable speed change.
            continue

        source_fps = get_cached_video_fps(video_file)
        if source_fps < 24.0:
            # Low-fps sources (GIFs run 10-15fps) are already frame-duplicated
            # by the fps= normalization; retiming them is pure judder.
            continue

        target = str(clip.get("target", "flow"))
        impact = _clamp(clip.get("impact", 0.5), default=0.5)
        rng = _stable_rng("retime", clip.get("index"), video_file, target)
        roll = rng.random()

        retime = None
        if target == "soft" and roll < 0.45:
            # Slow-mo drift. Depths below 0.6x are duplicated-frame judder on
            # ordinary 24-30fps sources, so they need high-fps footage.
            lo = 0.5 if source_fps >= 50.0 else 0.6
            retime = {"kind": "constant", "speed": round(rng.uniform(lo, 0.7), 3)}
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

        window = retime_source_window(final_duration, retime, fps)

        # Per-window runway: the retimed window must fit inside the scene
        # window the candidate was chosen for (a ramp that spills across the
        # scene cut hides a hard cut mid-slow-mo), and inside the real file.
        candidate = by_id.get(clip.get("candidate_id")) or {}
        cand_start = float(candidate.get("start", 0.0))
        cand_end = float(candidate.get("end", cand_start))
        video_duration = get_cached_video_duration(video_file)
        scene_len = cand_end - cand_start
        if scene_len > 0.0 and window > scene_len:
            continue
        if window > video_duration - 0.05:
            continue

        # Re-anchor the start for the bigger window, inside both the scene
        # window and the file. (Mirrors _materialize_clip's anchor semantics.)
        start = float(clip.get("start_time", 0.0))
        hi = min(cand_end - window if scene_len > 0.0 else video_duration - window,
                 video_duration - window)
        lo = max(0.0, cand_start if scene_len > 0.0 else 0.0)
        if hi < lo:
            continue
        clip["start_time"] = max(lo, min(start, hi))
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


def _plan_coverage_reservations(candidates: Sequence[Dict],
                                profiles: Sequence[Dict]) -> Dict[int, Dict]:
    """Reserve one segment per source: its best (candidate, segment) pairing.

    Deterministic: sources are seated in descending best-seat-score order,
    each taking its highest-scoring free segment. Drop segments are exempt
    when the seated pick would fall clearly below what the auction would put
    there — coverage should cost the flow/soft filler slots, not the money
    shots. The short-candidate duration penalty stays in force, which
    naturally steers short sources (GIFs, stills) onto short segments.
    """
    by_source: Dict[str, List[Dict]] = {}
    for c in candidates:
        by_source.setdefault(str(c.get("video_file")), []).append(c)

    # Auction-best approximation per segment (raw scores, no deque state),
    # used only for the drop exemption threshold.
    best_raw = [max(_score_candidate(c, p) for c in candidates) for p in profiles]

    options_by_source: Dict[str, List] = {}
    fallback_by_source: Dict[str, List] = {}
    for src in sorted(by_source):
        opts = []
        exempted = []
        for c in by_source[src]:
            for j, p in enumerate(profiles):
                score = _score_candidate(c, p)
                required = max(0.05, p["duration"])
                cand_duration = max(0.05, float(c.get("duration", required)))
                if cand_duration < required * 0.55:
                    score -= 0.18
                if p.get("target") == "drop" and score < best_raw[j] - 0.35:
                    exempted.append((score, j, c))
                    continue
                opts.append((score, j, c))
        key = lambda t: (-t[0], t[1], str(t[2].get("id")))
        opts.sort(key=key)
        exempted.sort(key=key)
        if opts or exempted:
            options_by_source[src] = opts
            fallback_by_source[src] = exempted

    order = sorted(options_by_source.items(),
                   key=lambda kv: (-(kv[1][0][0] if kv[1]
                                     else fallback_by_source[kv[0]][0][0]), kv[0]))
    reserved: Dict[int, Dict] = {}
    taken: set = set()
    for src, opts in order:
        seat = next(((j, c) for score, j, c in opts if j not in taken), None)
        if seat is None:
            # Every non-drop-worthy seat is taken (or this source only fits
            # drops): coverage still wins — fall back onto an exempted drop
            # segment rather than dropping the source from the video.
            seat = next(((j, c) for score, j, c in fallback_by_source[src]
                         if j not in taken), None)
        if seat is not None:
            j, cand = seat
            reserved[j] = cand
            taken.add(j)
    return reserved


# Segments shorter than this get no transition window on either side: the
# out/in chains occupy ~0.2s and need normal footage around them to read.
_TRANSITION_MIN_SEG = 0.3


def _assign_boundary_transitions(planned: List[Dict]) -> None:
    """Label consecutive clip pairs with split-transition specs.

    Each transition is rendered as two per-segment effect chains (an out-chain
    on clip i's tail and an in-chain on clip i+1's head), so assembly stays
    concat stream-copy. Kept occasional on purpose; deterministic per boundary.
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

        spec: Dict = {"type": kind}
        if kind == "whip_pan":
            # Same direction on both sides = continuous camera motion across
            # the cut.
            spec["direction"] = "right" if rng.random() < 0.5 else "left"
        a["transition_out"] = dict(spec)
        b["transition_in"] = dict(spec)


def summarize_clip_plan(plan: Sequence[Dict],
                        video_files: Sequence[str] | None = None,
                        candidates: Sequence[Dict] | None = None) -> Dict:
    if not plan:
        return {"clip_count": 0, "targets": {}, "ai_tagged": 0}
    targets = Counter(str(item.get("target", "flow")) for item in plan)
    ai_tagged = sum(1 for item in plan if item.get("ai_analyzed"))
    source_count = len(set(item.get("video_file") for item in plan))
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
    if video_files is not None:
        import os
        usage = Counter(os.path.basename(str(item.get("video_file")))
                        for item in plan)
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


def _build_segment_profiles(cut_times: np.ndarray, segment_durations: np.ndarray, beat_info: Dict) -> List[Dict]:
    beat_times = np.asarray(beat_info.get("times", []), dtype=float)
    downbeat_times = np.asarray(beat_info.get("downbeat_times", []), dtype=float)
    energy_profile = beat_info.get("energy_profile") or {}
    rhythm_data = beat_info.get("rhythm_data") or {}
    sections = beat_info.get("sections") or []

    wave = np.asarray(energy_profile.get("wave", []), dtype=float)
    arc = np.asarray(energy_profile.get("arc", []), dtype=float)
    impact = np.asarray(rhythm_data.get("impact_strength", []), dtype=float)
    rhythm = np.asarray(rhythm_data.get("combined_strength", []), dtype=float)
    novelty = np.asarray(rhythm_data.get("novelty_strength", []), dtype=float)

    profiles: List[Dict] = []
    for i, duration in enumerate(segment_durations):
        start = float(cut_times[i])
        end = float(cut_times[i + 1])
        mid = (start + end) * 0.5
        local_wave = _interp_feature(mid, beat_times, wave, 0.5)
        local_arc = _interp_feature(mid, beat_times, arc, 0.5)
        local_impact = _interp_feature(start, beat_times, impact, 0.5)
        local_rhythm = _interp_feature(start, beat_times, rhythm, 0.5)
        local_novelty = _interp_feature(start, beat_times, novelty, 0.4)
        section = _section_at(sections, mid)
        target = _target_for_segment(section, local_wave, local_impact, local_rhythm, local_novelty, local_arc)
        profiles.append({
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
            "section": section,
            "section_type": section.get("type", "body") if section else "body",
            "target": target,
            "is_downbeat": bool(
                downbeat_times.size and float(np.min(np.abs(downbeat_times - start))) <= 0.05
            ),
        })
    return profiles


def _interp_feature(time_s: float, beat_times: np.ndarray, values: np.ndarray, default: float) -> float:
    if beat_times.size == 0 or values.size != beat_times.size:
        return default
    return _clamp(np.interp(time_s, beat_times, values, left=float(values[0]), right=float(values[-1])), default=default)


def _section_at(sections: Sequence[Dict], time_s: float) -> Dict:
    for section in sections:
        if float(section.get("start", 0.0)) <= time_s < float(section.get("end", 0.0)):
            return section
    return sections[-1] if sections else {}


def _target_for_segment(section: Dict, wave: float, impact: float, rhythm: float, novelty: float, arc: float) -> str:
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


def _choose_candidate(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    file_rate: float = 0.012,
    file_cap: float = 0.18,
) -> Dict | None:
    best_candidate = None
    best_score = -999.0
    rng = _stable_rng(index, profile.get("target"), profile.get("start"))

    for candidate in candidates:
        score = _score_candidate(candidate, profile)
        cid = candidate.get("id")
        video_file = candidate.get("video_file")

        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.10
        score -= min(0.28, usage[cid] * 0.10)
        score -= min(file_cap, usage[video_file] * file_rate)

        required_source = max(0.05, profile["duration"])
        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        score += rng.random() * 0.015
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_candidate(candidate: Dict, profile: Dict) -> float:
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
        match = 0.46 * action + 0.16 * motion + 0.12 * combat + 0.10 * chase + 0.08 * explosion + 0.08 * quality
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


def _materialize_clip(candidate: Dict, profile: Dict, index: int) -> Dict:
    final_duration = max(0.05, float(profile["duration"]))
    source_duration = final_duration
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
    start_time = max(0.0, min(start_time, max(0.0, video_duration - source_duration)))

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "target": target,
        "score": _score_candidate(candidate, profile),
        "candidate_id": candidate.get("id"),
        "tags": list(candidate.get("tags", [])),
        "subject_anchor": _rebase_subject_anchor(candidate, start_time,
                                                 source_duration),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
    }
