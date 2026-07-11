#!/usr/bin/env python3
"""Multi-song analysis + sample-exact audio concatenation for Auto Mode.

Multiple songs go in; ONE audio file plus ONE merged analysis comes out, with
exactly the shape the single-song flow gets from::

    selected_beats, beat_info = analyze_beats_auto(audio_file, ...)

so callers swap in the three returned values and change nothing else.

TIMING AUTHORITY = THE MUXED AUDIO
    The returned audio path is not just analyzed — ``assemble_final_video``
    (ffmpeg_processing) muxes it into the rendered video as the final
    soundtrack. So each song is decoded ONCE at mux quality (MUX_SR = 44100
    Hz, stereo; mono sources upmixed, >2ch downmixed) and the raw sample
    blocks are written back-to-back into one PCM_24 WAV. Per-song offsets are
    cumulative frame counts of THIS hi-fi decode / MUX_SR — the single source
    of truth for every merged timestamp, so the merged timeline and the audio
    people actually hear can never disagree. Per-song analysis still decodes
    internally at its own rate (CONFIG.sr mono, inside analyze_beats_auto);
    its beat times are song-relative and are simply offset by the hi-fi
    cumulative starts (a per-song duration cross-check catches decoder
    disagreement between the two decodes). Because the WAV holds exactly the
    decoded samples, its ffprobe duration equals ``total_frames / MUX_SR``
    exactly. This dodges the classic mp3 trap: an mp3's ffprobe (container)
    duration includes encoder delay/padding and routinely differs from its
    decoded sample count, so summing container durations would desynchronize
    the merged timeline from the frame-locked timeline ``video_processor``
    later builds by probing the returned audio path.

MERGE POLICY (key -> consumer -> policy) — every key a downstream consumer
reads from beat_info, traced not guessed:

    times            stage6 _build_segment_profiles (feature interp grid),
                     video_processor segment_beats derivation,
                     text_overlay plan_text_windows snap points
                     -> per-song beat arrays offset by song start, concatenated
                        (strictly increasing: per-song beats live in
                        [0, duration_i) so they never cross the next offset).
    downbeat_times   stage6 profiles ("is_downbeat")
                     -> offset + concatenate.
    selected_times   mirror of the returned cut array (kept equal to it)
                     -> merged cut array (see RETURNED CUTS below).
    tempo            effects render_opts['tempo'], gui success message,
                     merged audio_visual_profile
                     -> duration-weighted mean (scalar consumers want one
                        number); exact per-song tempos live in the added
                        ``songs`` list.
    sections         stage6 _section_at (start/end/type), selection_info
                     -> start/end offset, ``index`` renumbered sequentially
                        across songs, concatenated (sorted, non-overlapping
                        because each song's sections tile [0, duration_i]).
    energy_profile   stage6 reads wave/arc/loudness per-beat; the rest ride
                     along for cache/debug parity
                     -> per-beat arrays (beat_energy, energy_levels, wave,
                        arc, loudness, optional bass_energy) concatenate in
                        beat order; per-FRAME curves (rms, spectral_centroid,
                        zcr — stage2's normalized frame curves) concatenate in
                        song order (constant hop, so frame index stays
                        time-ordered). Per-song 0..1 normalization is kept:
                        each song's dynamics are measured against itself,
                        exactly like a single-song run of that song.
    rhythm_data      stage6 reads impact_strength/combined_strength/
                     novelty_strength per-beat
                     -> every per-beat array (floats and bool anchors)
                        concatenates in beat order. Optional stem keys
                        (drum_onset, vocal_presence — and bass_energy in
                        energy_profile) present on only SOME songs: songs
                        missing the key contribute zeros of their beat count,
                        so one song's missing backend never drops another
                        song's data.
    rhythm_patterns  rebuilt from the merged, renumbered sections
                     ({index: dominant_pattern}).
    selection_info   gui success message (_format_auto_section_summary)
                     -> lists concatenated; each item's embedded ``section``
                        dict replaced by its merged (offset, renumbered) twin.
    audio_visual_profile  fed to video analysis at analysis time (per song 1)
                     -> recomputed over the MERGED data with the exact
                        single-song logic (auto_mode._build_audio_visual_profile),
                        so smart_preset/averages describe the whole timeline.
    video_analysis   stage6 planner candidates, gui _stage5_summary
                     -> property of the visual library, not the song: video
                        analysis runs ONCE (first song's call), and that
                        result is attached to the merged beat_info.
    audio_duration   gui success message
                     -> total hi-fi frames / MUX_SR (identical to the concat
                        WAV's ffprobe duration by construction).
    structure        no downstream consumer outside analyze_beats_auto (it is
                     consumed per-song inside stage 3); merged informationally
                     from the songs that HAVE it (offset label sections +
                     downbeats, duration-weighted bpm) — missing on some
                     songs never drops the others' data. Omitted if no song
                     has it, exactly like single-song.
    mode/auto_style  logging strings -> first song's values, unchanged.
    render_info/clip_plan_summary are written LATER by video_processor; not
    produced here.

Additional key ``songs``: [{file, offset, duration, tempo, beat_count,
cut_count}, ...] for any consumer that wants per-song detail. Purely
additive; every existing consumer uses .get() and ignores it.

RETURNED CUTS
    Per-song selected-cut arrays are offset and concatenated, and each song's
    offset (the JOIN) is inserted exactly once as a locked boundary — a
    multi-song edit must cut ON the song change. Any per-song cut landing
    within JOIN_EPS (0.02 s — below one frame even at 48 fps; the output fps
    is unknown at analysis time) of a join is dropped in favor of the join
    itself, so frame quantization can never produce a duplicate/1-frame
    segment at the seam.

DETERMINISM
    Pure function of its inputs: analyze_beats_auto is deterministic, the
    merge adds no randomness, and key iteration follows explicit first-seen
    order over song order (never set()/dict-race order). The concat WAV is
    byte-identical across calls (same samples, fixed header, no timestamps).
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from auto_mode import analyze_beats_auto, _build_audio_visual_profile

# Cuts within this many seconds of a join collapse onto the join. 0.02 s is
# under one frame at any realistic output fps (<= 1 frame even at 48 fps),
# and far smaller than the tightest cut interval Auto Mode ever emits
# (peak_energy_min_interval = 0.30 s), so it can only ever swallow a cut that
# genuinely sits on the seam.
JOIN_EPS = 0.02

# The concat file's name inside work_dir. Fixed so a re-run of the same
# session overwrites rather than accumulates.
CONCAT_BASENAME = "multisong_concat.wav"

# Mux-quality decode format for the concat WAV (the final soundtrack of the
# rendered video): CD-rate stereo. The final mux transcodes to 48 kHz PCM_24
# stereo, so 44100/stereo/PCM_24 loses nothing audible vs muxing the original
# files, unlike the 22050 mono analysis rate.
MUX_SR = 44100
MUX_CHANNELS = 2

# Analysis (CONFIG.sr mono) and hi-fi (MUX_SR stereo) decodes of the same
# song must agree on its length to well under a video frame; both go through
# librosa/soundfile off the same source stream, differing only by resampler
# edge rounding (sub-sample scale). A bigger gap means the two decoders
# genuinely disagree about the song's extent — merged timestamps would be
# audibly wrong, so refuse to continue.
_DECODE_AGREEMENT_TOL = 0.005

# energy_profile keys that are per-FRAME curves (stage2's normalized
# rms_curve/centroid_curve/flux_curve), not per-beat: they concatenate in
# song order but are NOT length-checked against the beat count.
_FRAME_CURVE_KEYS = frozenset({"rms", "spectral_centroid", "zcr"})


def analyze_and_concat(audio_files: List[str], work_dir: str, *,
                       video_files: Optional[List[str]] = None,
                       use_gpu: bool = False,
                       enable_video_analysis: bool = True,
                       enable_qwen_semantics: bool = True,
                       qwen_model_path: Optional[str] = None,
                       progress_callback: Optional[Callable[[str], None]] = None,
                       console_callback: Optional[Callable[[int, str], None]] = None,
                       ) -> Tuple[str, np.ndarray, Dict]:
    """Analyze one or more songs and return (audio_path, cut_beats, beat_info).

    Exactly what the single-song flow gets from
    ``(audio_file, *analyze_beats_auto(audio_file, ...))`` — callers swap in
    these three values and change nothing else.

    * One file: byte-identical passthrough to analyze_beats_auto (no concat
      file is written; the original path is returned).
    * Several files: per-song analysis (video analysis only on the first
      call — it describes the visual library, not the song), a sample-exact
      concatenated WAV written into ``work_dir``, and a merged beat_info per
      the module-level policy table.
    """
    if not audio_files:
        raise ValueError("analyze_and_concat needs at least one audio file.")

    if len(audio_files) == 1:
        # Single-song passthrough: the exact call the single-song flow makes.
        return (audio_files[0], *analyze_beats_auto(
            audio_files[0],
            use_gpu=use_gpu,
            video_files=video_files,
            enable_video_analysis=enable_video_analysis,
            enable_qwen_semantics=enable_qwen_semantics,
            qwen_model_path=qwen_model_path,
            progress_callback=progress_callback,
            console_callback=console_callback,
        ))

    import librosa  # deferred: matches analyze_beats_auto's own import scope
    import soundfile as sf

    # ------------------------------------------------------------------
    # 1) Per-song analysis. Video analysis runs ONCE, on the first call:
    #    it is a property of the visual library, not the song. Per-song
    #    caches (structure/stems/Qwen) key on the audio file, so each song
    #    keeps its own cache entries.
    # ------------------------------------------------------------------
    per_song: List[Tuple[np.ndarray, Dict]] = []
    for i, audio_file in enumerate(audio_files):
        first = i == 0
        cuts_i, info_i = analyze_beats_auto(
            audio_file,
            use_gpu=use_gpu,
            video_files=video_files if first else None,
            enable_video_analysis=enable_video_analysis if first else False,
            enable_qwen_semantics=enable_qwen_semantics,
            qwen_model_path=qwen_model_path,
            progress_callback=progress_callback,
            console_callback=console_callback,
        )
        per_song.append((cuts_i, info_i))

    # ------------------------------------------------------------------
    # 2) Sample-exact hi-fi concat — the timing authority AND the final
    #    soundtrack (assemble_final_video muxes the returned path into the
    #    rendered video). Each song is decoded at mux quality (MUX_SR
    #    stereo); offsets are cumulative frame counts of this decode /
    #    MUX_SR, so the merged timeline and the audible audio can never
    #    disagree. Samples are written UN-normalized (the single-song flow
    #    muxes the original file, so original levels are preserved;
    #    analyze_beats_auto's librosa.util.normalize is analysis-internal).
    # ------------------------------------------------------------------
    sample_blocks: List[np.ndarray] = []
    frame_counts: List[int] = []
    for i, audio_file in enumerate(audio_files):
        data, _sr = librosa.load(audio_file, sr=MUX_SR, offset=0.0,
                                 duration=None, mono=False)
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            raise ValueError(f"Audio file decoded to zero samples: {audio_file}")
        if data.ndim == 1:
            # Mono source: upmix by duplication (identical L/R).
            data = np.stack([data, data])
        elif data.shape[0] > MUX_CHANNELS:
            # >2 channels: deterministic mean downmix, then duplicate.
            mono = np.mean(data, axis=0, dtype=np.float64).astype(np.float32)
            data = np.stack([mono, mono])
        n_frames = data.shape[1]
        analyzed = float(per_song[i][1].get("audio_duration", 0.0))
        decoded = n_frames / MUX_SR
        if abs(decoded - analyzed) > _DECODE_AGREEMENT_TOL:
            # The analysis decode (CONFIG.sr mono) and this hi-fi decode must
            # agree on the song's extent; anything else would silently
            # desynchronize every merged timestamp from the muxed audio.
            raise RuntimeError(
                f"Decode mismatch for {os.path.basename(audio_file)}: "
                f"analysis saw {analyzed:.6f}s, mux decode saw {decoded:.6f}s"
            )
        sample_blocks.append(data)
        frame_counts.append(n_frames)

    boundaries = np.concatenate(([0], np.cumsum(frame_counts)))
    offsets = [int(b) / MUX_SR for b in boundaries[:-1]]  # per-song start times
    durations = [int(c) / MUX_SR for c in frame_counts]   # per-song lengths
    total_duration = int(boundaries[-1]) / MUX_SR         # single division: exact

    os.makedirs(work_dir, exist_ok=True)
    concat_path = os.path.join(work_dir, CONCAT_BASENAME)
    # PCM_24, not FLOAT: libsndfile stamps float WAVs with a PEAK chunk that
    # embeds a WALL-CLOCK timestamp, breaking byte-determinism. PCM_24 has no
    # PEAK chunk, keeps ~full float32 mantissa precision, and its duration is
    # sample-exact either way. The clip is defined behavior for the float->int
    # conversion (libsndfile wraps, not clips, by default); decoded audio is
    # normally within [-1, 1] so it is almost always a no-op. soundfile wants
    # (frames, channels), hence the transpose.
    sf.write(concat_path,
             np.clip(np.concatenate(sample_blocks, axis=1), -1.0, 1.0).T,
             MUX_SR, subtype="PCM_24")

    # Verify the returned path really carries the duration the merged timeline
    # assumes: video_processor later ffprobes it to build the frame-locked cut
    # timeline. WAV is sample-exact (unlike mp3 container durations), so this
    # must agree to within one sample.
    from ffmpeg_processing import get_video_duration, invalidate_media_info

    # concat_path is a FIXED name in a per-session dir and we just overwrote
    # it; the process-global probe cache would otherwise serve the PREVIOUS
    # render's duration here (and to video_processor's timeline build) after
    # any mid-session song-list change.
    invalidate_media_info(concat_path)
    probed = float(get_video_duration(concat_path))
    if abs(probed - total_duration) > 1.5 / MUX_SR:
        raise RuntimeError(
            f"Concat WAV duration drift: ffprobe {probed:.6f}s vs "
            f"sample-count {total_duration:.6f}s — timing authority broken"
        )

    # ------------------------------------------------------------------
    # 3) Merge.
    # ------------------------------------------------------------------
    cut_beats = _merge_cut_arrays([c for c, _ in per_song], offsets)
    beat_info = _merge_beat_info(
        per_song=per_song,
        audio_files=audio_files,
        offsets=offsets,
        durations=durations,
        total_duration=total_duration,
        cut_beats=cut_beats,
    )

    parts = ", ".join(
        f"{offsets[i]:.2f}s {os.path.basename(audio_files[i])}"
        for i in range(len(audio_files))
    )
    print(
        f"   ℹ️ 🎶 Multi-song merge: {len(audio_files)} songs, "
        f"{total_duration:.2f}s total, {len(cut_beats)} cuts | boundaries: {parts}"
    )

    return concat_path, cut_beats, beat_info


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _merge_cut_arrays(cut_arrays: Sequence[np.ndarray],
                      offsets: Sequence[float]) -> np.ndarray:
    """Offset + concatenate per-song cuts; each join present exactly once.

    Joins (every song's offset except the first song's 0.0) are locked
    boundaries. Per-song cuts within JOIN_EPS of a join are dropped in favor
    of the join itself, so the seam can never carry a near-duplicate cut.
    """
    joins = [float(o) for o in offsets[1:]]

    merged: List[float] = []
    for song_idx, cuts in enumerate(cut_arrays):
        offset = float(offsets[song_idx])
        for t in np.asarray(cuts, dtype=float).reshape(-1):
            t_abs = float(t) + offset
            if any(abs(t_abs - j) <= JOIN_EPS for j in joins):
                continue
            merged.append(t_abs)
    merged.extend(joins)
    merged.sort()

    # Strictly increasing by construction (per-song cuts are strictly
    # increasing and live in [0, duration); joins sit between songs and their
    # EPS neighborhoods were vacated above) — enforce anyway for safety.
    out: List[float] = []
    for t in merged:
        if not out or t > out[-1]:
            out.append(t)
    return np.asarray(out, dtype=float)


def _ordered_key_union(dicts: Sequence[Dict]) -> List:
    """Union of keys in first-seen order over song order (deterministic)."""
    seen: List = []
    for d in dicts:
        for k in d.keys():
            if k not in seen:
                seen.append(k)
    return seen


def _concat_per_beat(dicts: Sequence[Dict], key,
                     beat_counts: Sequence[int]) -> np.ndarray:
    """Concatenate one per-beat array across songs, zero-filling songs that
    lack the key (optional stem features) so no song's data is ever dropped.

    energy_levels is a dtype=object string array; zero-fill uses "medium"
    (the neutral class) there instead of 0.0.
    """
    pieces: List[np.ndarray] = []
    for d, n_beats in zip(dicts, beat_counts):
        value = d.get(key)
        if value is None:
            if key == "energy_levels":
                pieces.append(np.asarray(["medium"] * n_beats, dtype=object))
            else:
                pieces.append(np.zeros(n_beats, dtype=float))
            continue
        arr = np.asarray(value)
        if arr.shape[0] != n_beats:
            raise RuntimeError(
                f"Per-beat key {key!r} has length {arr.shape[0]} but the song "
                f"has {n_beats} beats — merge would corrupt beat alignment"
            )
        pieces.append(arr)
    return np.concatenate(pieces)


def _concat_frame_curves(dicts: Sequence[Dict], key) -> np.ndarray:
    """Concatenate per-frame curves (constant hop -> time order preserved)."""
    pieces = [np.asarray(d.get(key, []), dtype=float) for d in dicts]
    pieces = [p for p in pieces if p.size]
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def _offset_sections(per_song_sections: Sequence[List[Dict]],
                     offsets: Sequence[float]) -> Tuple[List[Dict], List[Dict]]:
    """Offset per-song sections and renumber ``index`` sequentially.

    Returns (merged_sections, per_song_index_maps) where index_maps[i] maps a
    song's original section index -> the merged section dict (used to rebind
    selection_info's embedded section references).
    """
    merged: List[Dict] = []
    index_maps: List[Dict] = []
    for song_idx, sections in enumerate(per_song_sections):
        offset = float(offsets[song_idx])
        index_map: Dict = {}
        for section in sections:
            out = dict(section)
            out["start"] = float(section.get("start", 0.0)) + offset
            out["end"] = float(section.get("end", 0.0)) + offset
            out["index"] = len(merged)
            index_map[section.get("index")] = out
            merged.append(out)
        index_maps.append(index_map)
    return merged, index_maps


def _merge_structure(infos: Sequence[Dict], offsets: Sequence[float],
                     durations: Sequence[float]) -> Optional[Dict]:
    """Informational merge of the optional wave-14 structure payloads.

    Only songs that HAVE structure contribute; if none do, returns None and
    the key stays absent, exactly like single-song. bpm is duration-weighted
    over contributing songs.
    """
    contributing = [
        (i, info["structure"]) for i, info in enumerate(infos)
        if isinstance(info.get("structure"), dict) and info["structure"].get("available")
    ]
    if not contributing:
        return None

    sections: List[Dict] = []
    downbeats: List[float] = []
    bpm_weighted = 0.0
    bpm_weight = 0.0
    for i, struct in contributing:
        offset = float(offsets[i])
        for s in struct.get("sections", []):
            out = dict(s)
            out["start"] = float(s.get("start", 0.0)) + offset
            out["end"] = float(s.get("end", 0.0)) + offset
            sections.append(out)
        for t in struct.get("downbeats", []):
            downbeats.append(float(t) + offset)
        try:
            bpm = float(struct.get("bpm", 0.0))
        except (TypeError, ValueError):
            bpm = 0.0
        if bpm > 0.0:
            bpm_weighted += bpm * float(durations[i])
            bpm_weight += float(durations[i])

    return {
        "available": True,
        "multisong": True,
        "backend": contributing[0][1].get("backend"),
        "sections": sections,
        "downbeats": downbeats,
        "bpm": round(bpm_weighted / bpm_weight, 3) if bpm_weight > 0 else 0.0,
    }


def _merge_beat_info(per_song: Sequence[Tuple[np.ndarray, Dict]],
                     audio_files: Sequence[str],
                     offsets: Sequence[float],
                     durations: Sequence[float],
                     total_duration: float,
                     cut_beats: np.ndarray) -> Dict:
    infos = [info for _, info in per_song]
    n_songs = len(infos)

    beat_counts = [int(np.asarray(info.get("times", []), dtype=float).size)
                   for info in infos]

    # --- time-like arrays: offset + concat -------------------------------
    times = np.concatenate([
        np.asarray(infos[i]["times"], dtype=float) + offsets[i]
        for i in range(n_songs)
    ])
    downbeat_times = np.concatenate([
        np.asarray(infos[i].get("downbeat_times", []), dtype=float) + offsets[i]
        for i in range(n_songs)
    ]) if any(np.asarray(info.get("downbeat_times", [])).size for info in infos) \
        else np.asarray([], dtype=float)

    # --- sections + everything keyed off them ----------------------------
    merged_sections, index_maps = _offset_sections(
        [list(info.get("sections", [])) for info in infos], offsets)

    rhythm_patterns = {
        s["index"]: s.get("dominant_pattern", "mixed") for s in merged_sections
    }

    selection_info: List[Dict] = []
    for song_idx, info in enumerate(infos):
        for item in info.get("selection_info", []):
            out = dict(item)
            orig_section = item.get("section")
            if isinstance(orig_section, dict):
                mapped = index_maps[song_idx].get(orig_section.get("index"))
                if mapped is not None:
                    out["section"] = mapped
            selection_info.append(out)

    # --- per-beat carrier dicts ------------------------------------------
    energy_dicts = [info.get("energy_profile") or {} for info in infos]
    rhythm_dicts = [info.get("rhythm_data") or {} for info in infos]

    energy_profile: Dict = {}
    for key in _ordered_key_union(energy_dicts):
        if key in _FRAME_CURVE_KEYS:
            energy_profile[key] = _concat_frame_curves(energy_dicts, key)
        else:
            energy_profile[key] = _concat_per_beat(energy_dicts, key, beat_counts)

    rhythm_data: Dict = {}
    for key in _ordered_key_union(rhythm_dicts):
        rhythm_data[key] = _concat_per_beat(rhythm_dicts, key, beat_counts)

    # --- scalars ----------------------------------------------------------
    tempos = [float(info.get("tempo", 0.0)) for info in infos]
    tempo = float(
        sum(t * d for t, d in zip(tempos, durations)) / max(1e-9, sum(durations))
    )

    # Merged profile recomputed with the exact single-song logic over the
    # merged data (wave/impact/rhythm arrays, merged sections, merged cuts).
    audio_visual_profile = _build_audio_visual_profile(
        tempo=tempo,
        sections=merged_sections,
        features={
            "wave": energy_profile.get("wave", np.asarray([], dtype=float)),
            "impact_score": rhythm_data.get("impact_strength", np.asarray([], dtype=float)),
            "rhythm_score": rhythm_data.get("combined_strength", np.asarray([], dtype=float)),
        },
        selected_beats=cut_beats,
        beat_times=times,
    )

    songs = [
        {
            "file": str(audio_files[i]),
            "offset": float(offsets[i]),
            "duration": float(durations[i]),
            "tempo": tempos[i],
            "beat_count": beat_counts[i],
            "cut_count": int(np.asarray(per_song[i][0]).size),
        }
        for i in range(n_songs)
    ]

    beat_info: Dict = {
        "times": times,
        "downbeat_times": downbeat_times,
        "selected_times": cut_beats,
        "tempo": tempo,
        "sections": merged_sections,
        "energy_profile": energy_profile,
        "rhythm_data": rhythm_data,
        "rhythm_patterns": rhythm_patterns,
        "selection_info": selection_info,
        "audio_visual_profile": audio_visual_profile,
        # Video analysis ran once, on the first song's call (property of the
        # visual library, not the song).
        "video_analysis": infos[0].get("video_analysis"),
        "audio_duration": float(total_duration),
        "mode": infos[0].get("mode", "auto_v4_audio_visual_rhythmic_planner"),
        "auto_style": infos[0].get("auto_style", "audio_visual_rhythmic_gmv_amv"),
        "songs": songs,
    }

    structure = _merge_structure(infos, offsets, durations)
    if structure is not None:
        beat_info["structure"] = structure

    return beat_info


__all__ = ["analyze_and_concat", "JOIN_EPS", "CONCAT_BASENAME"]
