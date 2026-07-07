# Design: split-screen vertical pairing ("duo segments")

Status: **implemented** (wave 7, 2026-07-07). Decision outcomes: pane crop policy =
Option A (anchored ≤40%); checkbox **default ON** (owner direction, changed from the
draft's off recommendation); butted panes; drop/rhythm only. Triptych and "hot pane"
beat accent remain v2 candidates.

## Goal

When the batch contains multiple portrait sources and the canvas is landscape,
some high-energy segments render **two portrait clips side by side** instead of
forcing one clip through the fit ladder. This is the concert-multicam idiom:
verticals become a deliberate look instead of a compromise. (Generalizes to the
converse — landscape sources stacked on a 9:16 canvas — same math, other axis.)

## Inherited constraints (non-negotiable)

- Determinism: seeded decisions only (`_stable_rng`), double-run framemd5 equality.
- Frame-exactness: `-vframes` stays the frame-count authority; the per-segment
  frame guard applies to the composed output unchanged.
- The 15%-crop lesson: never *silently* discard most of a frame. Pane crops
  exceed 15% by design (see geometry below), which is acceptable **only**
  because it is a deliberate, subject-anchored style choice — see the crop-cap
  decision point.
- ProRes precise mode stays pristine: **no duos in lossless mode**.
- Concat assembly is untouched: a duo segment is still one normal H.264 segment file.

## Geometry (the numbers that shape everything)

On a 1920×1080 canvas, each 2-up pane is **960×1080 (8:9 ≈ 0.889 AR)**.
A 9:16 portrait source (0.5625 AR) filled into a pane loses
`1 − 0.5625/0.889 ≈ 36.7%` of its height — above `MAX_CROP_PER_AXIS` (15%),
below `SCAN_CROP_LOSS` (40%). So per-pane fitting cannot reuse the default
ladder unmodified; it needs a pane-specific policy:

| Option | Per-pane look | Notes |
|---|---|---|
| **A. Anchored pane crop (recommended)** | fill + crop ≤ ~40%, window centered/tracked on the subject | 8:9 crops of vertical footage are standard broadcast practice; the wave-5/6 anchor + tracked pan is exactly the machinery that makes this safe. New constant `PANE_MAX_CROP = 0.40`. |
| B. Per-pane echo hybrid | ≤15% crop + blurred margins inside each pane | Honors the 15% cap literally but panes look small and busy — blur inside blur when both panes need it. |
| C. Triptych (3-up) | panes 640×1080 (0.593 AR) → only ~5% crop of 9:16 — under the 15% cap natively | Best geometric fit, but needs 3 distinct portrait sources per segment and reads busier. Proposed as **v2**, not v1. |

Pane widths: `pane_w = target_w // 2` rounded to even; the second pane takes
`target_w − pane_w` (may differ by 2 px — `hstack` only requires equal heights).
No divider bar in v1 (butted panes are the music-video norm); a 2–4 px gap is a
one-line follow-up if wanted.

## Planner design (stage6_av_planner.py)

**Eligibility** — a segment can become a duo when ALL hold:
- Canvas is landscape and ≥2 distinct portrait sources have usable candidates
  (or the converse for a portrait canvas, stacking with `vstack`).
- Segment target is `drop` or `rhythm` (the idiom lives on hard cuts), and
  `impact` ≥ ~0.5.
- Not lossless mode; segment not selected for a retime (and duos are excluded
  from `_assign_retime_specs` — a warped clock breaks the tracked pan and
  doubles the runway math).
- Neither member is a still image (v1 exclusion: Ken Burns interplay deferred).

**Frequency & spacing** — seeded roll (`_stable_rng('duo', index, song)`), cap
≈ **1 in 4 eligible segments**, and never two duo segments adjacent (a run of
split screens reads as a wall, not an accent).

**Partner selection** — the primary candidate comes from the existing auction
*unchanged*. The partner is a mini-auction over candidates from a **different
portrait source**: existing `_score_candidate` score, minus the recent-use and
usage penalties, plus a coherence gate (|Δbrightness| below a threshold so the
two panes don't clash), preferring anchor confidence ≥ `ANCHOR_MIN_CONFIDENCE`
(the pane crop should land on a subject). Partner usage increments the same
`usage` counters and `recent_videos` deque, and a partner appearance **counts
for source coverage** (a source seen in a pane has been seen).

**Data shape** — additive, backward compatible. The planned clip keeps all its
flat fields (every existing consumer reads those untouched) and gains:

```python
"partner": {
    "video_file": ..., "start_time": ..., "source_duration": ...,
    "candidate_id": ..., "subject_anchor": {..., "path_seg": [...]},  # rebased
    "source_name": ...,
}
```

`_rebase_subject_anchor` runs for the partner exactly as for the primary.
`_assign_boundary_transitions` needs no change (transitions act on the composed
frame). `_assign_retime_specs` skips any clip with `"partner"`.

## Renderer design

**video_processor.create_clip_parallel** — if `planned_clip.get('partner')`:
clamp the partner's `start_time` against its probed duration (same logic as the
primary), and pass a `partner=` spec into `extract_kwargs`. If the partner
fails any probe/runway check at render time, **drop it and render the primary
solo through the normal fit ladder** (deterministic planner gates make this
rare; the fallback keeps a bad probe from killing the segment).

**ffmpeg_processing.extract_clip_segment_ffmpeg(..., partner: dict = None)** —
the function already builds multi-input `filter_complex` graphs (the text PNG
is input 1 today), so the structure extends rather than forks:

```
[0:v] pre_filters_L, pane_fit_L [L];
[1:v] pre_filters_R, pane_fit_R [R];
[L][R] hstack, setsar=1 [comp];
[comp] post_filters (effects, look) [basev];   # then text overlay as today
```

- Each input gets its own loop/seek decision (`get_loop_input_args`, the
  trim=start= gotcha) and its own `build_segment_pre_filters` — the per-input
  logic is already factored; it runs twice.
- `pane_fit` = SAR fix + anchored/tracked crop to pane size (`PANE_MAX_CROP`),
  reusing `_anchored_crop`/`_tracked_crop` with pane headroom. No scan, no blur
  inside panes (v1, per Option A).
- **Text input index**: `build_text_overlay_graph` hardcodes `[1:v]` for the
  PNG; it gains a `text_input_index` parameter (1 solo, 2 duo). This is the one
  pre-existing function whose signature changes (default preserves behavior).
- `-vframes` caps the composed stream; each branch over-provides frames exactly
  as solo segments do; the existing frame guard verifies the output unchanged.
- Effects apply to the composed frame — punch zooms/tilts on a split screen
  are correct and look intentional. The one-zoompan guard is unaffected.
- v2 polish (not v1): "hot pane" accent — per-pane `eq` brightness/saturation
  boost alternating with `local_beats` enable windows before the hstack.

## GUI & defaults

Checkbox in the Style group: **"Pair vertical clips (split screen)"** —
**opt-in, default off for v1** (same rollout pattern as speed ramps: owner
tests on real footage, then we consider defaulting on). Tooltip notes it needs
two or more vertical sources and fires on hard cuts only.

## Testing plan

- Unit-ish: pane fit math (crop ≤ `PANE_MAX_CROP`, anchor offset/tracked pan in
  pane headroom), partner auction determinism, adjacency/rarity caps.
- Render: synthetic portrait pair (moving boxes at different positions) → duo
  segment renders at canvas size, exact frame count, both subjects visible;
  double-run framemd5 identical; text overlay on a duo segment (input-index
  shift); partner-drop fallback renders solo; portrait canvas variant (vstack).
- Full pipeline smoke with ≥2 portrait sources and the checkbox on; frame
  guards + determinism as always.

## Implementation wave (wave 7, on approval)

- **7A** `stage6_av_planner.py`: eligibility, rarity/spacing, partner auction,
  data shape, retime exclusion (~150 lines).
- **7B** `ffmpeg_processing.py`: `partner=` path, pane fit, text index
  parameter (~200 lines).
- **7C** `video_processor.py` + `gui.py`: partner clamping + fallback,
  checkbox plumbing (~60 lines).
- Integration/docs/smoke by the coordinator, owner tests, then commit.

7A and 7B can run in parallel (disjoint files, `partner` dict shape above is
the contract); 7C follows or runs parallel too (its files are disjoint).

## Decision points for the owner

1. **Pane crop policy** — Option A (anchored ≤40% pane crop, recommended) vs
   B (blur inside panes) vs C-first (triptych)?
2. **Default off (recommended) or on** for the checkbox at v1?
3. **Divider**: butted panes (recommended) or a thin gap/line?
4. Is `drop`/`rhythm`-only right, or should calm `flow` sections also pair
   (lounge-y side-by-side vibe)? Recommended: drop/rhythm only at v1.
