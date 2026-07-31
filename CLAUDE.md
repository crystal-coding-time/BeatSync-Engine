# BeatSync Engine (mac-port branch)

macOS port of a Windows-only beat-synced music video generator. Owner is learning self-hosting; explain non-obvious decisions briefly.

## Ground rules
- **The only durable docs are this file and `README.mac.md` (user-facing usage). Every feature change keeps both accurate in the same commit — need-to-know only, factual and current-state. No changelogs, backlogs, or planning files.**
- Work on the `mac-port` branch. `origin` is upstream (`Merserk/BeatSync-Engine`, read-only); push to the `fork` remote (`crystal-coding-time/BeatSync-Engine`). Keep Windows behavior intact: platform-specific code branches on `os.name == 'nt'` or falls back from bundled `bin/` paths to `PATH` lookups.
- Python env: `.venv` (Homebrew python@3.13), no CuPy on Mac. Launch with `./run.sh` (Gradio UI on 7860).
- Working rhythm: land a change uncommitted → restart the service → owner tests → commit only on his OK.

## Pipeline

`src/gui.py` (Gradio UI) → `src/orchestrator.py` (per-render `RenderSettings` model + headless `_process_video_impl`/`process_video` pipeline driver, re-exported through gui; console/render logging in `src/render_log.py`) → for multiple songs `src/multisong.py` (per-song analysis, sample-exact 44.1kHz concat = the timing authority, offset-merged beat_info) → the 6-stage auto pipeline in `src/auto_mode/` → `src/video_processor.py` (parallel segment extraction; a failed extraction is rescued at the same time slot and frame count from seeded fallback sources, and only an unrescuable segment aborts the render) → `src/ffmpeg_processing.py` (per-segment ffmpeg commands; the per-segment `-vf` chain is the hook point for effects/retimes/text) → concat stream-copy assembly, with opt-in xfade boundary chunks re-encoded in place.

### Stages (`src/auto_mode/`)
1. **Beats + downbeats** — beat-this is the Mac default via run.sh; librosa fallback estimates bar phase.
2. **Per-beat features** — SuperFlux onsets, harmonic-change, EBU loudness.
3. **Sections** — optional all-in-one-mlx structure labels via `src/structure_stems.py`, which also supplies demucs-mlx stem signals.
4. **Cut selection** on the beat grid (see below).
5. **Optional Qwen3-VL tagging** via llama.cpp.
6. **Planner** — one global auction per segment with coverage reservations (LAP-seated), proportional-fair source variety, DINOv2 semantic diversity via `src/visual_embeddings.py` (3-frame pooled), and cross-modal energy matching against optical-flow clip motion.

### Stage 4 — cut selection
The `cut_density` setting (0..1, default 0.55) rides in `AutoWaveConfig` via `dataclasses.replace`, so every stage-4 helper sees it without a signature change and the module `CONFIG` singleton is never mutated. `0.0` selects the legacy pacing on every branch and is byte-identical to the pre-dial engine.

Above 0:
- The beat-step map (`adaptive_beat_step_density`) reads a **section-local percentile** of `0.6·wave + 0.4·impact`.
- The cut cap and prune run per-section on a section-intensity budget, with the anchor prune bonuses shrinking as the dial rises.
- Min-interval and cleanup floors are expressed in **beat periods**, so half-beat cutting is reachable at any tempo at the top of the range. `_ABSOLUTE_MIN_GAP` = 0.20s is the hard floor.
- The weak-score gate is a section-local percentile. This matters because `compute_cut_scores` adds the anchor bonuses *before* `_normalize`, so anchors saturate near 1.0 and non-anchor beats squash toward 0; a fixed threshold rejects nearly every non-anchor beat and cuts then only happen where `max_hold` forces one, landing on the bar grid.
- `choose_best_nearby` takes `min_pos` so it cannot re-select the beat just cut (which would fail the min-gap test and skip forward without cutting, making per-beat cutting unreachable), and the max-hold safety block is skipped at `step <= 2` so it cannot leapfrog the step map.

**Section handling keys on `_section_intensity`**: the label prior blended with measured `energy`, then normalized against the track's own peak. The structure backend's labels are not reliable — it can tag every section of a contrast-heavy track `intro` while `energy` tracks the sections exactly — so anything keyed on the label alone goes inert. `min_interval_for_wave` and `max_hold_for_section` both have label-keyed legacy branches for the `0.0` path.

### Stage 6 — planner details
- **Clip motion** (`src/video_analysis.py`): optical flow runs on adjacent native frame pairs (dt = 1/fps) taken at the metric-sample positions, never sample-to-sample — Farneback correspondence is exact only to ~14px of displacement. Full scale is 1.0 screen-height/sec. Fields are `kinetic`, `subject`, and camera direction. Note `semantic["camera_motion"]` is a different, Qwen-schema quantity.
- **Anti-repetition** rests on a per-run sub-window ledger of spent source intervals. `_plan_source_window` is the single window authority (shared by primary and duo partner) and offsets a candidate's reuse onto the first unspent 0.85×duration-strided sub-window of its analysis span. `_reuse_penalty` adds an unbounded log-growth reuse cost plus a footage-exhaustion term; being ledger-dependent, it is applied OUTSIDE `_ScoreCache`. The id/source recency deques are sized from the median segment duration to cover ~20s/~10s of screen time.
- **`media_aware`** (default ON) adds mixed-media auction penalties (upscale incl. stills, wrap-scaled GIF loop-seam, native-fps kinetics incl. the untagged-drop term), applied OUTSIDE the memoized `_ScoreCache` alongside the recency/variety penalties.
- **Speed ramps** never touch GIFs regardless of container fps. Under `speed_ramps`, a low-probability boundary pass (`_assign_boundary_ramps`, 'boundary_ramp' rng stream) attaches drop-entry anticipation ramps: the outgoing tail gets the piecewise `tail_ramp` retime kind and the incoming drop opens 0.4–0.5× via the existing ramp kind. Anticipation claims a boundary before crossfade selection can.

### Effects and text
`src/effects.py` is the effect-primitive registry. The `semantic_fx` content gate (default ON) applies `SEMANTIC_FX_MATRIX` vetoes plus impact-weighted firing; stills are hard-vetoed from motion effects, and whip pans follow the analysis camera-direction fields and never cross a still boundary. The plan dict forwards candidate motion/semantic/media_type fields for it. The matrix covers every primitive except `vignette_grain`, which is deliberately uncovered because it is the style's constant finishing look and a per-segment veto would make the vignette blink between cuts. `fisheye` and `vignette_grain`'s grain roll additionally take an inline `ctx.semantic_fx`-only target gate — they are the two primitives that fire on a bare probability roll with no music condition.

`still_motion` (default ON) gives stills energy/anchor-conditioned moves (`build_still_motion_filter`, dedicated rng stream) instead of the generic Ken Burns builder, which remains for the off path.

Text timing lives in `src/text_overlay.py`. The opt-in Motion text style swaps the static Pillow PNG for per-frame SVG sequences from `src/styled_text.py` — same planner, same overlay/fade graph.

## Hard invariants (never break these)
- **Determinism**: all randomness through seeded `_stable_rng` streams; two identical runs must produce byte-identical video (test = double-run framemd5 equality). New effects/features get dedicated rng streams so existing plans stay byte-identical; MLX/ONNX backends achieve determinism by memoization (4dp-rounded sidecar caches).
- **Frame-exactness**: `-vframes` is the frame-count authority; per-segment and assembly frame guards must pass; the frame-locked timeline (`segment_frames`) is the duration authority. `tpad=stop_mode=clone` before `fps=` guards demuxer underrun (GIF trailing display durations). Gotcha: `tmix` toggled with `enable=` silently drops frames — use split/trim/concat instead.
- **Byte-identical off-paths**: every optional feature's disabled state (checkbox off, slider 0, model absent) must reproduce the pre-feature engine exactly. `cut_density`'s off path is `0.0`. `semantic_fx`/`media_aware`/`still_motion` default ON, so their off path is the explicitly-unticked state. The `RenderSettings` defaults (orchestrator) and the `gr.Slider`/`gr.Checkbox(value=...)` defaults (gui) must always agree or the UI and headless paths diverge.
- **Settings plumbing**: `SETTINGS_KEYS` derives from `RenderSettings` field order; `gui.py` asserts it matches `settings_components` and `process_video` zips them `strict=True`, so a new field must land together with its GUI component and its slot in the positional tuple.
- **Analysis-cache keying**: the orchestrator's per-session stage 1-5 cache (`analysis_key`) holds `selected_beats`, so any setting that changes stage-4 output must be part of that key — `cut_density` is, rounded to 4dp. Miss this and the setting looks inert on a re-render because the previous cut list is served.
- **ProRes precise mode stays pristine**: no effects, retimes, duos, crossfades, or text; static centered framing on proxies.
- **Probe-cache staleness**: `_MEDIA_INFO_CACHE` and `_SOURCE_NORMALIZE_CACHE` (ffmpeg_processing; the latter holds per-still/GIF EXIF orientation + alpha flags feeding `source_normalize_filters`) are path-keyed and process-lifetime; any file REWRITTEN at a fixed path mid-session (e.g. multisong's concat wav) must call `invalidate_media_info(path)` right after writing, or later probes serve the previous file's duration.
- **Graceful degradation**: optional backends (YuNet faces, beat-this, Qwen, DINOv2 embeddings, all-in-one-mlx/demucs-mlx, cairosvg for Motion text) fall back with one log line when missing; kill switches: `BEATSYNC_DISABLE_QWEN/EMBED/STRUCTURE/STEMS/STYLEDTEXT`, `BEATSYNC_YUNET_MODEL`, `BEATSYNC_BEAT_BACKEND` (=librosa reverts the Mac beat-this default), `BEATSYNC_DOWNBEAT_PHASE=off` (beat-0 anchor grid).
- **15%-crop rule**: `MAX_CROP_PER_AXIS` caps content loss; subject anchors change WHERE we crop, never HOW MUCH.

## Testing
No test suite. Smoke test = run the pipeline headless on synthetic media:
`ffmpeg -f lavfi -i "sine=frequency=440:beep_factor=8:duration=20" beat.wav`, testsrc clips, then call `gui._process_video_impl(audio_files=..., video_files=..., ...)` with `PYTHONPATH=src` (`audio_files` accepts one path or a list). `BEATSYNC_DISABLE_QWEN=1` skips the slow vision stage. Isolate runs by monkeypatching `paths.get_processing_dir` AND `get_output_dir` to a scratch dir so nothing writes to `output/`. Verify: double-run framemd5 identical + "✓ Frame guard" lines. Renders log to `output/render_<timestamp>.log`.

Pacing work additionally needs a cuts-per-bar density curve over the timeline, annotated with section labels — determinism only proves nothing broke. The click track must have real loud/quiet section contrast or section-relative behavior cannot be observed.

**Comparing against an older tree**: `ROOT_DIR` is derived from the module's own location, so a `git archive HEAD src` checkout elsewhere resolves `models/`, `input/`, `bin/`, `looks/` relative to THAT directory — it silently runs with DINOv2/YuNet absent and produces a legitimately different plan. Symlink those four dirs into the comparison tree or the off-path test compares two different feature sets, not two code versions.
