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

## Wave 12 — source variety + semantic diversity ⏸ (owner picks direction after wave 11)

- Approach A: DINOv2 ViT-S/14 (ONNX, CPU) embeddings in the existing analysis frame pass → deterministic
  online cosine clustering → windowed-MMR + cluster-run penalties in the stage6 auction (0 = byte-identical)
- Approach B: proportional-fair EWMA usage term replacing the linear uncapped file penalty (reservations stay)
- Approach C (optional): min-cost-flow reservation seating (OR-Tools), auction untouched
- Prereq: 11.10 (score memoization) — done in wave 11

## Wave 13 — pacing + kinetic ⏸ (owner picks direction)

- Tier 0 audio (no deps): SuperFlux onsets + backtracking, strict-margin HPSS pseudo-stems,
  tonnetz harmonic-change cut candidates for flow/soft, ffmpeg ebur128 momentary LUFS as impact scalar
- Inline `minterpolate` slow-mo (measured deterministic, ~12s CPU per output second; +4-frame
  over-provision; lifts the ≥50fps gate for sub-0.6x)

## Backlog / opt-in follow-ups

- `all-in-one-mlx` structure labels + `demucs-mlx` stems behind env flag (verified installable on py3.13)
- RIFE 4.6 via rife-ncnn-vulkan pre-pass (cached, framemd5-verified intermediates; models >4.6 broken on Apple Silicon)
- Flow motion-blur effect primitive (minterpolate/tmix); FFglitch datamosh profile (later)
- xfade crossfades on low-energy boundaries (pre-existing roadmap item)
- Backburnered by owner: Real-ESRGAN. Rejected: preset picker.
