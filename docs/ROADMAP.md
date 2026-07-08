# Roadmap — automated local music video generation

Goal: drop in audio + source footage, get a finished, styled music video with zero manual editing. Everything runs locally.

Status key: ✅ done · 🚧 in progress · ⬜ planned

## Phase 0 — macOS port ✅ (2026-07-06)
- ✅ Cross-platform ffmpeg/ffprobe/llama.cpp resolution (bundled `bin/` first, then `PATH`)
- ✅ CPU fallback for CuPy audio analysis
- ✅ `run.sh` launcher + `.venv` workflow
- ✅ Qwen3-VL scene tagging via llama.cpp Metal

## Phase 1 — broader inputs + Mac hardware encoding ✅ (2026-07-06)
- ✅ Source formats: `.mov`, `.webm`, `.m4v`, `.avi`, `.gif` (was mp4/mkv only)
- ✅ Short sources (GIFs) loop via `-stream_loop` instead of drifting the timeline
- ✅ Apple VideoToolbox H.264/HEVC encoding (`get_gpu_quality_args` dispatch in `ffmpeg_processing.py`; GUI auto-offers it when NVENC is absent)
- ✅ Still images (2026-07-07): `.jpg/.jpeg/.png/.webp/.bmp` as sources. Images get a
  synthetic duration in the probe layer, one analysis candidate from the single frame, and
  deterministic Ken Burns motion (seeded `zoompan` pan+zoom) per segment at render time
  (`is_image_source`/`build_ken_burns_filter` in `ffmpeg_processing.py`). In ProRes precise
  mode stills become fixed-length static proxies (no Ken Burns — proxies stay pristine).
  HEIC is **not** supported: Homebrew ffmpeg ships without a HEIF demuxer/decoder

## Phase 1.5 — aspect-ratio frame fit ✅ (2026-07-06)
- Mixed resolutions/aspect ratios no longer stretch-distort. GUI "Frame fit" option:
  smart crop-to-fill (default) · blurred-background fill · letterbox · stretch (legacy)
- `get_fit_filters` / `build_blur_fit_graph` in `ffmpeg_processing.py`; `setsar=1` normalizes
  sample aspect across heterogeneous sources
- ~~Target resolution/aspect = highest-resolution source by pixel area (`get_max_resolution`); a dominant vertical source makes a vertical video~~ superseded 2026-07-07 — fixed 16:9 canvas by default, see Phase 4.5 (legacy behavior lives on as the "Match best source" canvas option)
- Smart crop is a **limited-crop hybrid** (2026-07-06, owner feedback: pure fill-crop
  over-cropped): mild mismatches keep the classic fill-and-center-crop, but at most
  `MAX_CROP_PER_AXIS` (15%) of the source may be cropped away — beyond that the source is
  composited over the blurred-background fill instead (a 9:16 source in a 16:9 target keeps
  ≥85% of its content instead of 31%). `plan_source_fit`/`plan_smart_crop` in
  `ffmpeg_processing.py`; ProRes proxies use the same chain (`build_source_fit_chain`)
- Anamorphic sources (SAR ≠ 1) are resampled to square pixels before fitting —
  `scale=force_original_aspect_ratio` compares storage dimensions, so they previously
  rendered distorted

## Phase 2 — beat-aware effects engine ✅ (2026-07-06)
- `src/effects.py`: per-segment filter chains driven by stage 6 planner metadata
  (`target` drop/rhythm/build/soft/flow, `impact` energy). Effects land where the music does:
  punch-in zoom (zoompan) on drop/rhythm cuts, white flash on drops, saturation pulse at the
  beat frequency, chromatic aberration occasionally on hard cuts; hype adds shake, vignette, grain
- GUI: Effect style Clean (default) / AMV / Hype + intensity slider
- Deterministic (seeded per segment); ffmpeg gotcha encoded in code comments: `crop` can't animate w/h → zoom uses `zoompan`
- Effects recipe pack (2026-07-06): pixelize bursts, negative-flash strobes and posterize
  flashes on Hype drop cuts; directional zoom-blur smears on drops; `tmix`/`lagfun` motion
  trails on calm segments; hue sweeps through build-ups; rare subtle fisheye (Hype).
  Core filters only, every recipe verified frame-count-safe, capped at 2 (AMV) / 3 (Hype)
  pack effects per segment (`shuffleframes`/`elbg` deliberately excluded — see effects.py)
- Effects registry + picker (2026-07-07): `effects.py` is now a registry of named primitives
  (`EFFECT_REGISTRY`). GUI "Effect mode": **Curated** (the classic Clean/AMV/Hype presets,
  chains proven character-identical to the pre-registry code) · **Custom** (checkbox palette —
  pick any primitive subset; empty palette means *no* effects) · **Shuffle** (seeded random
  palette; seed 0 derives from the song filename, same seed → identical video). The applied
  recipe is printed into the render log. New primitives: half/quad mirrors, beat-flip,
  sustained push-in/pull-out zooms
- Punch-to-full-bleed (2026-07-07): `punch_fill` primitive — drop cuts on hybrid-tier fits
  punch in to exactly full-bleed and decay back to the framed view; amplitude computed from
  the clip's real fill geometry (capped at true full-bleed +10%), ~1 in 4 eligible drops in
  AMV/Hype via a dedicated rng stream (existing curated plans stay byte-identical); no-op
  on duo segments and whenever another zoompan is present (`punch_zoom` gained the same
  one-zoompan guard)
- Dutch tilt (2026-07-07): `dutch_tilt` primitive — a few degrees of rotation scaled to
  cover (no black corners on any canvas), sign alternating per inter-beat interval when the
  beat grid is present. Curated: AMV/Hype, drop/rhythm segments only, ~1 in 6, seeded from a
  dedicated rng stream so pre-existing render plans stay byte-identical
- Effects don't apply in ProRes precise mode (kept pristine for external editing); the GUI
  now says so next to the mode picker
- ~~Not included by design: speed ramps~~ superseded — see Phase 4 speed ramps below: the
  original exclusion assumed whole-timeline `setpts`; per-segment retiming inside the
  `-vframes` anchor is safe

## Phase 3 — text overlay system ✅ (2026-07-06)
- General-purpose text over the video at planned moments (quotes, captions, titles — any text), entered one-per-line in the GUI
- `src/text_overlay.py`: planning happens on the **global audio timeline**, not segment indexes — every entry gets its own disjoint, beat-snapped time window (guaranteed to appear), which is then projected onto whatever segments it intersects with fades translated to each segment's local clock
- `@<time>` prefix pins an entry (`@15 Finish strong`, `@1:23 Halfway`); pinned windows take priority, auto-placed entries flow around them
- Rendering: **Pillow → transparent PNG → ffmpeg `overlay`** (core filter), NOT drawtext — Homebrew ffmpeg ships without libfreetype/libass. Bonus: real word wrapping, stroke + shadow, any TTF (auto-detects Arial Bold on macOS; `BEATSYNC_FONT` overrides)
- GUI: text entries box, position (lower third/center/top), size slider
- Gotcha encoded in code: ffmpeg `fade` rejects negative `st` — continuation segments omit the fade-in filter instead
- Later ideas: per-entry timing control, color/font options in GUI, text file import

## Hardening pass ✅ (2026-07-06)
Code-review fixes across the render pipeline; verified with repeat-run `framemd5` comparisons and targeted repros:
- Frame-exact looping when a source window spills past EOF: `-ss` before `-i` combined with `-stream_loop` re-seeks on **every** loop iteration (repro: 45 frames instead of 60), so looped seeks moved into the filter chain (`trim=start=`) — `extract_clip_segment_ffmpeg`
- Deterministic renders on every path: all fallback sampling (no visual plan, ProRes source/start picks) now uses the seeded `_stable_rng` pattern; two identical runs produce byte-identical video streams
- Text entries are now truly guaranteed: two windows landing in the same segment used to silently overwrite each other in `seg_map` — planning treats "shares a segment" as a clash and relocates; continuation segments no longer restart the fade (alpha pop at cuts)
- The planner's beat grid actually reaches text planning (`beat_info['times']` key mismatch meant snapping always fell back to cut boundaries)
- ProRes precise mode normalizes mixed-resolution sources to the target resolution during proxy conversion (concat stream-copy requires identical dimensions), and its GUI preview is fixed (encoder options were placed before `-i`, so preview generation always failed silently)
- Pipeline diagnostics (including ffmpeg stderr) are captured to `output/render_<timestamp>.log` instead of being discarded; error statuses point at the log
- Output FPS follows the highest-resolution source (matching how target resolution is picked) instead of the arbitrary first upload; ffprobe failures are no longer cached for the whole run
- Simplification: `ClipJob` dataclass replaces the 10-element args tuple; shared pre-filter/audio-mux helpers in `ffmpeg_processing.py`

## Phase 3.5 — beat-sync tightening + audio backends ✅ (2026-07-07)
- Interior-beat effect gating: each segment's real beat offsets (from the stage-1 grid) are
  threaded into `build_effect_filters` (`segment_beats` in `video_processor.py`), so pulses,
  strobes and beat-flips fire on actual beats via `enable=` windows instead of a
  sine-at-tempo approximation
- Cut-lead bias: interior cut boundaries land `BEATSYNC_CUT_LEAD_FRAMES` (default **1**)
  frames *before* the beat — the editor's trick of having the new shot onscreen when the
  transient hits. First/last boundaries stay locked; total frame count unchanged
- Optional `beat-this` backend (MIT, CPJKU): `BEATSYNC_BEAT_BACKEND=beat_this` swaps the
  librosa tracker for the SOTA transformer (beats **and downbeats**; downbeats drive the
  bar/phrase anchor bonuses in stage 4). CPU by default (faster than MPS for short tracks;
  `BEATSYNC_BEAT_THIS_DEVICE=mps` overrides), librosa fallback on any failure, off by
  default — install with `.venv/bin/pip install beat-this torch torchaudio`
- Frame guards (from the speed-ramps audit): every extracted segment is verified against its
  planned frame count (demux-only ffprobe), and final assembly is asserted equal to
  `render_info['timeline_frames']` — the zero-drift constraint is now machine-checked on
  every render, both standard and ProRes paths

## Phase 4 — transitions & polish 🚧
- ✅ Split transitions (2026-07-07): whip pans, glitch cuts and dip-to-black/white on
  planner-labeled boundaries (drop boundaries → whip/glitch, soft → dips; occasional and
  seeded). Implemented as *per-segment* out/in effect chains — the tail of segment A and the
  head of segment B each carry half the transition — so concat stream-copy assembly is
  untouched. Active for AMV/Hype styles
- ✅ LUT color looks (2026-07-07): six self-baked looks (Vintage, Cross Process, Cool, Warm,
  High Contrast, Day For Night) generated by `scripts/bake_looks.py` (`haldclutsrc` +
  curves/colorbalance chains → HaldCLUT PNG → `.cube`), applied per segment via the core
  `lut3d` filter (~4 ms/frame). GUI "Look" dropdown; AGPL-clean since the LUTs are baked
  from ffmpeg expressions, not third-party packs. PNGs are committed; `.cube` files are
  regenerated locally (`looks/*.cube` gitignored, rebuilt on demand by `src/looks.py`)
- ✅ Source coverage + variety (2026-07-07, owner-reported: many uploads never appeared —
  23/29 sources in one render): stage-6 planning now seats every source with ≥1 usable
  candidate into its best-matching segment (reservations consumed inside the sequential
  auction; weak sources kept off drop segments when possible) and scales the reuse penalty.
  GUI "Source variety" slider: 0 = pure quality auction (**exact legacy behavior**),
  1 = even spread; **default 0.4** — note this changes default plans vs. renders made
  before 2026-07-07. The render log prints a per-source usage histogram plus unused /
  zero-candidate sources
- ✅ Speed ramps (2026-07-07, **experimental, opt-in GUI checkbox**): per-segment retiming —
  slow-mo drift on soft segments, rushes through builds, decelerate-into-the-cut on drops,
  rare freeze-hits. Safe within the frame-locked timeline because the retime `setpts` sits
  between `setpts=PTS-STARTPTS` and the `fps` filter with `-vframes` still the frame-count
  authority; source windows are over-provisioned by 2 frames and every retimed segment is
  frame-count-verified after extraction (an independent audit showed naive retiming silently
  produced 59/60- and 23/60-frame segments). Ramps skip images, sub-24fps sources
  (0.5x floor needs ≥50fps), segments that would loop or spill their scene window, and
  ProRes precise mode
- ⬜ Opt-in `xfade` crossfades/wipes on low-energy boundaries (requires boundary-chunk
  re-encode assembly; hard cuts stay the fast default)

## Phase 4.5 — 16:9 canvas + subject-aware reframe ✅ (2026-07-07)
Redesign of aspect-ratio handling (owner feedback: one high-res portrait upload flipped the
whole video to portrait). The output canvas is now fixed and the fit engine follows the subject:
- **Fixed output canvas, default 16:9 1080p**: GUI "Output canvas" dropdown — 16:9 1080p
  (default) · 16:9 4K · 9:16 portrait · Match best source (exact legacy `get_max_resolution`
  behavior). `resolve_target_resolution(output_format, ...)` in `video_processor.py`; unknown
  keys warn and fall back to the default. Output FPS still follows the highest-resolution source
- **Subject anchor** (`video_analysis.py`): every analysis candidate carries
  `subject_anchor {cx, cy, confidence, source, path}` — YuNet face detection when available
  (model at `models/face_detection_yunet_2023mar.onnx`, re-fetch with `scripts/fetch_yunet.py`,
  `BEATSYNC_YUNET_MODEL` env overrides/kill-switches), else motion-diff centroid, else
  Laplacian-detail centroid; deterministic, no extra frame decodes, CPU-only (schema identical
  on Windows/GPU). `ANALYSIS_VERSION` bump invalidates stale caches
- **Anchor-aware smart crop** (`ffmpeg_processing.py`): the crop window centers on the subject
  (clamped) instead of the frame center — anchors change *where* we crop, never *how much*;
  `MAX_CROP_PER_AXIS` (15%) still governs. Anchors below `ANCHOR_MIN_CONFIDENCE` (0.2) are
  ignored (centered, byte-identical to legacy). Threaded stage6 → `ClipJob.planned_clip`
  → `extract_clip_segment_ffmpeg(anchor=...)`
- **Tracked pan** (2026-07-07, v2 of the reframe): the crop window *follows* the subject
  through the shot. Stage6 rebases `subject_anchor.path` onto the segment clock
  (`path_seg`; retimed segments drop it — a warped clock would desync the pan), and the fit
  engine emits speed-clamped piecewise-linear crop expressions (`PAN_*` constants:
  ≤6 knots, ≤25% of crop headroom/second, <3% travel → static). Applies to the offset-crop
  and hybrid tiers; scan-fit keeps its own sweep (motions never compound). Anchors without
  `path_seg` (or any fallback condition) reproduce the wave-5 static offset exactly
- **"Echo" blur fill**: the blurred background behind hybrid/blur fits is now graded
  (110% overscan, darker, desaturated, vignette) with a slow ~3% drift over the segment
  (direction seeded from the source path — deterministic)
- **Kinetic fill upgrades** (2026-07-07, wave 8): the hybrid/blur foreground gets a slow
  continuous push-in (`HYBRID_FG_ZOOM = 0.035`; 0 = kill switch; zoompan after the tracked
  pan with `s=` locked so the frame rectangle never moves; skips retimes/stills/duo panes),
  and the echo margins pulse with the music — a windowed saturation/brightness lift on each
  beat (`ECHO_PULSE_*` constants; `local_beats` threaded into
  `extract_clip_segment_ffmpeg`, only for non-Minimal styles, ≤8 beats/segment)
- **Scan-fit for extreme mismatches** (`SCAN_CROP_LOSS = 0.40`): instead of a blur-fit
  postage stamp, the frame fills the short axis and the crop window sweeps the long axis with
  smoothstep easing, ending on the subject anchor; sweep speed is capped
  (`SCAN_MAX_SPEED_FRAC`) so short segments shrink the travel rather than whip. Requires a
  known segment duration — ProRes proxies (whole-file, duration-less) keep static centered
  framing per the precise-mode-stays-pristine rule

## Phase 4.6 — split-screen vertical pairing ✅ (2026-07-07)
Design doc: `docs/DESIGN_split_screen.md`. Two portrait clips share one landscape frame
(concert-multicam idiom) instead of each fighting the fit ladder — **on by default**
(owner call; GUI "Pair vertical clips" checkbox disables):
- Planner (`stage6_av_planner.py`): duos fire on drop/rhythm segments (impact ≥ 0.5),
  seeded ~1 in 4 eligible, never adjacent; the partner is a mini-auction over a *different*
  portrait source (score minus the standard penalties, brightness-coherence gate, anchor
  preferred). Partner counts toward usage + source coverage. `split_screen`/`target_size`
  kwargs on `build_planned_clip_sequence`; off = byte-identical legacy plans. Duos never
  combine with retimes or stills; generalizes to landscape pairs stacked on a 9:16 canvas
- Renderer (`ffmpeg_processing.py`): two-input filter_complex — per-input loop/seek +
  pre-filters, pane fit via the anchored/tracked crop at `PANE_MAX_CROP` (0.40; a pane crop
  is a deliberate style, and the subject anchor is what makes it safe), hstack/vstack,
  effects/look/text on the composed frame (`build_text_overlay_graph` gained
  `text_input_index`). Every failure path degrades to a solo render — a bad partner never
  kills a segment. ProRes precise mode never sees duos
- Plumbing (`gui.py`/`video_processor.py`): checkbox → planner; partner start-clamp +
  drop-to-solo fallbacks in `create_clip_parallel`; `resolve_target_resolution` hoisted
  above planning so the planner knows the canvas

## Phase 4.7 — intent-based UI ✅ (2026-07-07)
Design doc: `docs/DESIGN_ui_redesign.md` (owner: no preset picker; Stretch removed from the
UI, engine value kept). Pure layout reorg of `create_ui()` — zero engine/signature changes,
the process inputs list order untouched by construction:
- Tier 1: files, Style (relabeled Minimal / Music video / Hype — values frozen), Look,
  Output canvas, Text entries
- Tier 2: one collapsed ⚙️ Advanced accordion (effect mode/palette/seed/intensity, variety,
  speed ramps, split-screen, frame fit, processing mode + ProRes note, custom FPS,
  filename, text position/size)
- Frame fit is now Auto (the smart ladder) / Blurred background / Letterbox; `'stretch'`
  survives for headless/settings callers only

## Wave 12 — semantic diversity + fair-share variety ✅ (2026-07-08)
Two upgrades to stage-6 planning (tracker `docs/TODO.md`; designs from the 2026-07-07 research):
- **Visual variety** (new GUI slider, Advanced, default 0.4): avoids runs of visually similar
  shots even across different files. `src/visual_embeddings.py` embeds one frame per analysis
  candidate with DINOv2 ViT-S/14 via ONNX Runtime (CPU-only single-threaded → deterministic;
  vectors unit-normalized and 4dp-quantized; per-video sidecar cache in the analysis cache dir
  keyed by file+model signature — existing analysis/Qwen caches stay valid) and clusters them
  online by cosine threshold. The auction subtracts a cluster-run penalty (−0.14·sv) and a
  windowed MMR penalty (−0.22·sv·max(0, max_cos_sim−0.55) vs the last 6 picks; dots rounded to
  4dp before max — the determinism firewall). Fetch the model with `scripts/fetch_dinov2.py`
  (~87 MB, gitignored); missing model/onnxruntime or `BEATSYNC_DISABLE_EMBED=1` degrades to
  zero penalty. sv=0 (and mere key presence) is byte-identical to legacy plans. Duo partners
  pay and record the same penalties
- **Proportional-fair source variety** (variety>0 redesigned; variety=0 exact legacy): the
  linear uncapped per-file reuse penalty — which late in long videos swamped content scores and
  degenerated to score-blind round-robin — is replaced by a lazily-decayed EWMA fair-share
  pressure term (`_FairShareEWMA`, τ=24 segments): only sources above fair share pay,
  proportionally (−variety·0.30·max(0, share·sources−1)). Verified: a dominant high-quality
  source keeps a bounded, quality-justified lead instead of flattening; coverage reservations
  unchanged and still hard. **variety>0 plans change vs wave 11 (deliberate)**
- Plumbing: `semantic_variety` threaded gui → settings → `create_music_video` →
  `build_planned_clip_sequence`; annotation runs just before planning, never in ProRes mode.
  `onnxruntime==1.27.0` added to requirements (feature degrades gracefully without it)

## Wave 11 — review-driven hardening + perf ✅ (2026-07-08)
Fixes from a multi-agent, adversarially-verified pipeline review (13 confirmed findings +
1 latent hardening; tracker in `docs/TODO.md`). No new features; behavior changes only on
failure paths and one effects edge case:
- **Qwen results are only cached as AI-complete on real success** (≥1 merged semantic tag) in
  both the deferred single-video and batch merge paths — a transient llama-server failure now
  retries on the next run instead of being pinned in the AI cache (`video_analysis.py`)
- **One corrupt source can no longer discard a whole analysis batch**: the serial retry stores
  an empty result and the siblings still reach the cache
- **Probe-failure hygiene**: fps probe failures are no longer cached (they pinned fps=30 for the
  run and skewed retime gating); an unknown duration forces `-stream_loop -1` from t=0 instead of
  a fictitious 10s (which could seek past EOF and abort on the frame guard); analysis metadata now
  uses one combined ffprobe call per source instead of three
- **Effects**: `push_in`/`pull_out` gained the one-zoompan-per-segment guard (Custom/Shuffle
  rhythm segments could stack two zoompans); the trailing restore `scale` is emitted only after
  dimension-shrinking filters (shake, whip pan) — chains are otherwise unchanged, verified over a
  73k-case before/after matrix
- **Perf**: stage-6 candidate scoring memoized by (candidate, target) — 8–12× faster planning on
  large libraries, plans byte-identical at every variety level; ProRes precise mode converts only
  the sources the plan references instead of the whole library; loop-invariant percentile hoisted
  (stage 3) and vectorized nearest-beat lookup (stage 4), both value-identical
- **Silent failures now log**: HPSS fallback and rhythm-band zero-fills print ⚠️ warnings
- Latent hardening: decel-ramp `setpts` sqrt radicand clamped with `max(0,…)` (unreachable via
  current planner gates; guards future callers — negative radicand stalls ffmpeg's fps stage)
- **GIF/VFR trailing-frame fix** (owner-reported failed render): a GIF whose *final* frame
  carries a long display duration (container says 3.75s, last packet at 2.5s) came up short
  through `fps=` when a segment window ended inside the gap — the frame guard then correctly
  refused the drifted timeline. `build_segment_pre_filters` now inserts
  `tpad=stop_mode=clone:stop=-1` before `fps=`: the last frame is held through the window
  (the correct rendering for display-duration sources), a no-op when the input covers the
  window, and always bounded by `-vframes`. Pre-existing bug, not a wave-11 regression —
  reproduced and verified against the failing render (149/149 clips, frame guards pass)

## Stage-5 hardening ✅ (2026-07-07)
Incident: an owner render froze for ~35 min inside llama-server (Homebrew llama.cpp 9870,
Qwen3VL-2B + mmproj) — a slot wedged mid-prompt while `/health` stayed ok; the worker had
no effective per-request timeout and the parent backstop scaled to ~23 h. Not reproducible
deterministically (65 requests / 8 fresh servers, 0 hangs); fingerprint matches open
upstream bugs (ggml-org/llama.cpp #24265 prompt-cache/ctx-checkpoint stall, #17297
flash-attn, #20921 wedged slot). Fixes:
- Server flags now disable the implicated build-9870 default-on subsystems:
  `--cache-ram 0 --ctx-checkpoints 0 --flash-attn off` (~15% gen cost, throughput
  unchanged), plus `--log-file` (piped stdout is block-buffered — killed servers used to
  leave 0-byte logs)
- Per-request watchdog: 180 s warmup allowance until a server's first success
  (`BEATSYNC_QWEN_WARMUP_TIMEOUT`), then 60 s steady-state (`BEATSYNC_QWEN_REQUEST_TIMEOUT`),
  1 retry, failed items keep deterministic tags
- Circuit breaker: 6 consecutive timeouts (`BEATSYNC_QWEN_TIMEOUT_BREAKER`) → one server
  restart at halved slots → trips: partial results written (`"wedged": true`), remaining
  jobs skipped. Worst case ≈ 16 min, was ~23 h
- Orphan prevention: worker signal handlers + atexit stop llama-server
  (SIGTERM→SIGKILL escalation); the parent runs the worker in its own process group and
  kills the whole group on timeout (POSIX; Windows keeps prior behavior)
- Parent backstop: `min(7200, 300 + 15·candidates)` s (`BEATSYNC_QWEN_BATCH_TIMEOUT`) —
  7200 cap because a healthy 1000+-candidate batch legitimately needs >1 h at ~3.3 s/frame

## Phase 5 — full automation ⬜
- Watch-folder mode built on the existing `video_processor.py` CLI: drop audio + clips, video appears in `output/`
- launchd job on macOS; candidate for running on the-all-thing server later

## Documentation policy
Every feature change updates: this file (status + any new hook points), `README.mac.md` (user-facing usage), and `CLAUDE.md` if conventions change.
