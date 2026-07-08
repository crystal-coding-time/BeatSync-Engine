# Work tracker — waves 11+

Working to-do document for the 2026-07-07 review/design cycle. Status key:
⬜ pending · 🤖 agent dispatched · 🔧 integrating · ✅ landed (uncommitted) · 💾 committed · ⏸ awaiting owner decision

Workflow reminder: agents work disjoint files → integration + docs sync (ROADMAP/README) →
headless smoke test → restart `./run.sh` → **owner tests → commit only on his OK** → next wave.

## Wave 11 — verified review fixes 💾 (committed 2026-07-08)

From the 2026-07-07 adversarially-verified pipeline review (13 confirmed/plausible findings + 1 latent hardening).

| # | Task | File(s) | Sev | Status |
|---|------|---------|-----|--------|
| 11.1 | Qwen failure cached as `ai_enabled=True` — set from real success flag; also guard batch per-video merge (lines ~776–807) | `src/video_analysis.py` | High | 💾 |
| 11.2 | Serial-retry in analysis loop has no try/except — one corrupt file discards the whole batch pre-cache | `src/video_analysis.py` | Med | 💾 |
| 11.3 | 3 ffprobe spawns per source → single combined probe (duration+fps+resolution) local to video_analysis | `src/video_analysis.py` | Low | 💾 |
| 11.4 | `get_cached_video_fps` caches probe failures (pins 30.0 for the run) — mirror duration-cache no-cache-on-failure | `src/ffmpeg_processing.py` | Med | 💾 |
| 11.5 | Unknown duration → fictitious 10.0s (seek past EOF → frame-guard abort; early-content bias) — safe unknown-duration path | `src/ffmpeg_processing.py` | Low | 💾 |
| 11.6 | Latent decel-ramp `sqrt(negative)` → NaN PTS: `sqrt(max(0,…))` hardening in `_retime_filters` (verified unreachable today; cheap landmine removal) | `src/ffmpeg_processing.py` | Hardening | 💾 |
| 11.7 | ProRes mode transcodes ALL sources — convert only plan-referenced files (+ fallback pool from that subset) | `src/video_processor.py` | Med | 💾 |
| 11.8 | `_fx_push_pull_zoom` missing one-zoompan guard (stacks with punch_zoom on rhythm segments in Custom/Shuffle) | `src/effects.py` | Med | 💾 |
| 11.9 | Unconditional trailing `scale` on effect segments — only restore after dimension-shrinking filters (shake/whip_pan) | `src/effects.py` | Low | 💾 |
| 11.10 | `_score_candidate` recomputed ~900k× in stage6 — memoize by (candidate id, target), byte-identical values | `src/auto_mode/stage6_av_planner.py` | Med | 💾 |
| 11.11 | Loop-invariant `_safe_percentile(wave,45,…)` recomputed in section scan — hoist | `src/auto_mode/stage3_sections.py` | Med | 💾 |
| 11.12 | Rhythm-band exception zero-fills silently — add ⚠️ log line | `src/auto_mode/stage2_features.py` | Med | 💾 |
| 11.13 | Per-cut `np.argmin` beat re-mapping in ratio cap — `np.searchsorted` (identical picks) | `src/auto_mode/stage4_select.py` | Low | 💾 |
| 11.14 | HPSS failure swallowed with no log line — add ⚠️ log line | `src/auto_mode/__init__.py` | Low | 💾 |
| 11.15 | Integration: cross-file check, ROADMAP.md sync, headless smoke (determinism + frame guards), restart run.sh | (lead) | — | 💾 |
| 11.16 | Owner-reported render failure (clip 133, 75/105 frames): GIF trailing display-duration gap — `tpad=stop_mode=clone` before `fps=` in the shared segment chain; pre-existing bug, reproduced + fixed against the real render | `src/ffmpeg_processing.py` | High | 💾 |

## Wave 12 — semantic diversity + fair-share variety 💾 (committed 2026-07-08)

Approach A (DINOv2 embeddings + windowed-MMR/cluster penalties) + Approach B (proportional-fair
usage). Approach C (min-cost-flow seating) stays backlog. Contract: candidates gain optional
`embedding` (384-d unit list, 4dp) + `visual_cluster` (int) keys; sidecar cache keeps existing
analysis/Qwen caches valid (no ANALYSIS_VERSION bump).

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 12.1 | Embedding infra: DINOv2 ViT-S/14 ONNX (CPU, deterministic), sidecar per-video cache, online cosine clustering, fetch script, kill switches | `src/visual_embeddings.py`, `scripts/fetch_dinov2.py` (new) | 💾 |
| 12.2 | Auction: windowed-MMR + cluster-run penalties (`semantic_variety`, 0 = byte-identical) + proportional-fair EWMA usage term for variety>0 (variety=0 stays exact legacy) | `src/auto_mode/stage6_av_planner.py` | 💾 |
| 12.3 | Plumbing: Visual variety slider (Advanced), settings threading, embed annotation call before planning (graceful skip) | `src/gui.py`, `src/video_processor.py` | 💾 |
| 12.4 | Integration: fetch model, cross-check, docs sync (ROADMAP + README), smoke (with + without model), restart | (lead) | 💾 |

## Wave 13 — pacing + kinetic (in flight, owner approved)

Contract: features dict gains `onset_superflux` / `harmonic_change` / `loudness` (per-beat,
normalized, zero-filled + ⚠️ on failure); planned clips gain optional `loudness` float;
retime specs gain optional `interp` factor (deep slow-mo on 24–50fps sources only).

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 13.1 | SuperFlux onset + tonnetz harmonic-change + ebur128 momentary-loudness per-beat features; conservative cut-score integration (onset bonus everywhere, HCDF bonus in low-percussive sections) | `src/auto_mode/stage2_features.py`, `stage4_select.py`, `__init__.py` | 🤖 Agent I |
| 13.2 | Loudness → segment profiles → planned clips; `interp` retime spec (lifts ≥50fps gate for sub-0.6x) + inline `minterpolate` in the retime chain (+4-frame over-provision, tpad synergy) | `src/auto_mode/stage6_av_planner.py`, `src/ffmpeg_processing.py` | 🤖 Agent J |
| 13.3 | Loudness-scaled punch/flash amplitudes (byte-identical when key absent) | `src/effects.py` | 🤖 Agent K |
| 13.4 | Integration: cross-check, docs sync, smoke (determinism + frame guards + interp render cost), restart | (lead) | ⬜ |

## Backlog / opt-in follow-ups

- `all-in-one-mlx` structure labels + `demucs-mlx` stems behind env flag (verified installable on py3.13)
- RIFE 4.6 via rife-ncnn-vulkan pre-pass (cached, framemd5-verified intermediates; models >4.6 broken on Apple Silicon)
- Flow motion-blur effect primitive (minterpolate/tmix); FFglitch datamosh profile (later)
- xfade crossfades on low-energy boundaries (pre-existing roadmap item)
- Backburnered by owner: Real-ESRGAN. Rejected: preset picker.
