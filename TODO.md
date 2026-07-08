# Refactor tracker (from the 2026-07-08 architecture audit)
Temporary tracking file — delete when all waves land (per CLAUDE.md docs policy).

## Verification protocol (every item, no exceptions)
1. **Baseline before any edit**: headless smoke render on synthetic media
   (`ffmpeg -f lavfi -i "sine=frequency=440:beep_factor=8:duration=20" beat.wav`,
   testsrc clips, `gui._process_video_impl` with `PYTHONPATH=src`,
   `BEATSYNC_DISABLE_QWEN=1`, `paths.get_processing_dir` pointed at a scratch dir).
   Save the video framemd5.
2. Apply the change; run the smoke **twice**.
3. Require: baseline framemd5 == run1 == run2, plus "✓ Frame guard" lines.
   Any mismatch → revert that change and record why.
4. No commits until the owner tests and OKs (working rhythm in CLAUDE.md).

## Wave 1 — complete (owner-tested 2026-07-08)

| ID | Item | Files | Status |
|----|------|-------|--------|
| A1 | Consolidate the 4 per-property ffprobe helpers into one `_probe_media_info` + one cache (fallback semantics preserved exactly) | ffmpeg_processing.py | done (owner-tested, committed) |
| A2 | Extract shared `_segment_encode_tail` + `_run_and_verify_segment`, use at all 4 encode sites; catch `TimeoutExpired` so hung encoders hit existing fallbacks | ffmpeg_processing.py | done (owner-tested, committed) |
| B1 | Parallelize the lossless pipeline: ProRes proxy conversion loop + segment extraction (RNG decisions planned serially, ffmpeg jobs pooled) | video_processor.py | done (owner-tested, committed) |
| B2 | Gate the `time.sleep(1.0)` lossless-cleanup on `os.name == 'nt'`; delete the dead `frame_count` local | video_processor.py | done (owner-tested, committed) |
| C1 | Share spectral computations across stages 1–3 (librosa `S=` params; reuse the raw flux curve in stage 3). Riskiest item: strict framemd5 gate, revert if not bit-identical | auto_mode/stage1_audio.py, stage2_features.py, stage3_sections.py, auto_mode/__init__.py | done (owner-tested, committed) |
| C2 | Delete dead `_select_vulkan_device` (stage5, zero callers) | auto_mode/stage5_qwen_scene_worker.py | done (owner-tested, committed) |
| C3 | Drop the top-level `import librosa` in logger.py (banner uses `importlib.metadata.version`, same printed string) | logger.py | done (owner-tested, committed) |

## Wave 2 — backlog (start after Wave 1 is owner-approved)

| ID | Item | Depends on |
|----|------|-----------|
| W2-1 | Split `extract_clip_segment_ffmpeg` into plan/execute halves | A2 |
| W2-2 | Split `create_music_video` into `_render_lossless` / `_render_standard` + RenderContext | B1 |
| W2-3 | `RenderSettings` dataclass; kill the 4× settings listing and the 22-arg positional chain in gui.py | — |
| W2-4 | Frame guard from the encode's own stderr `frame=` counter (drop the per-segment ffprobe spawn) | A1 |
| W2-5 | Cold-run audio decode dedup: shared `_decoded_wav` cache for structure+stems; overlap the ebur128 pass via Popen | — |
| W2-6 | Extract `analyze_beats_auto` phase helpers (incl. the buried downbeat-anchor override) | C1 |
| W2-7 | TypedDict contracts (BeatFeatures / SegmentProfile / PlannedClip) + `EffectContext` dataclass | — |
| W2-8 | Vectorize stage-6 reservation/auction sweeps (exact tie-breaks preserved) | — |
| W2-9 | `OptionalBackend` protocol + shared `SidecarCache` for Qwen/YuNet/DINOv2 | — |
| W2-10 | Stage-5: persistent executor with bounded in-flight window (no per-wave pool churn) | — |
| W2-11 | Merge Qwen single/batch orchestration; extract the shared CPU/GPU metric-scoring formula | — |
| W2-12 | gui.py module split; explicit ui_content imports; delete the 18 dead symbols | W2-3 |
| W2-13 | `concatenate_videos_ffmpeg` strategy split | A2 |
| W2-14 | Low-severity sweep: bisect in text_overlay, naming/annotation fixes, `_decoded_wav` temp-dir leak | — |

## Results log
- **Combined port (all of Wave 1 in the working tree)**: PASS. Standard (cpu)
  config: original-code baseline (via stash) == patched run1 == patched run2
  (framemd5 `130608ac…`). ProRes precise mode: patched double-run identical
  (`a3e50024…`; vs-original identity proven 4-way in the B worktree). Frame
  guards green in every run. Service restarted on 7860 with the patched code.
  Owner manual test passed (full 61-source render, frame guard green); committed.
- **C1/C2/C3 (auto_mode/*, logger.py)**: PASS, all three C1 sub-steps landed,
  each gated individually. framemd5 identical across baseline×2, each sub-step,
  and final×2; ~30 analysis arrays bit-equal (`np.array_equal`) on BOTH stage-3
  paths (heuristic and installed all-in-one-mlx/demucs-mlx backends). The
  `rms(S=)` share was correctly skipped: librosa's `rms(y=)` is time-domain
  frame RMS, provably different from spectral RMS — no redundancy existed there.
  Saved work ≈ 3× mel+power_to_db + 1× full STFT per song (~0.4 s on a 4-min
  track; load+HPSS dominate short clips). Dead `_select_vulkan_device` deleted
  (zero callers confirmed). logger.py no longer imports librosa at module top
  (banner string byte-identical; note: librosa 0.11 lazy-loads, so the startup
  win is ~40 ms, smaller than the audit estimated).
- **A1/A2 (ffmpeg_processing.py, +225/−231, net −6 lines)**: PASS. framemd5
  identical baseline vs after×2 on defaults AND crossfades-forced AND the
  crossfades-off-path config; frame guards green. Beyond the render tests:
  direct argv capture at all 4 encode sites byte-identical pre/post; probe
  values identical across 8 inputs (incl. anamorphic SAR, corrupt, missing)
  on both cold and cached calls. `MediaInfo` now carries duration/fps/w/h/SAR
  from ONE ffprobe (was 3–4 spawns per source); only successful probes cached.
  `_run_media_command` synthesizes returncode 124 on `TimeoutExpired`, so hung
  encoders now degrade into existing fallbacks instead of crashing assembly.
  Log-only deltas: consolidated probe warning; solo-path error print shows the
  400-char stderr tail.
- **B1/B2 (video_processor.py, +70/−19)**: PASS. framemd5 4-way identical on the
  ProRes config (baseline == after×2 == original-code re-run via stash) and 3-way
  on standard; frame guards green in every run. Lossless video stage ~2s faster
  even on the tiny synthetic set (gains scale with library size). Bonus find:
  `convert_to_prores_proxy` derives the proxy path from the source *basename*, so
  two sources sharing a basename collide on one output file — fixed by grouping
  same-stem conversions serially (input order, reproducing serial last-one-wins)
  while distinct stems parallelize; filenames unchanged (they seed
  `_stable_rng('prores_start', i, prores_file)`). All RNG draws stay serial.
