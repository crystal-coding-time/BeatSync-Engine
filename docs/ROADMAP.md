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
- ⬜ Still images (JPG/PNG) as sources via `zoompan` (Ken Burns) — deferred, needs duration synthesis in the analysis stages

## Phase 1.5 — aspect-ratio frame fit ✅ (2026-07-06)
- Mixed resolutions/aspect ratios no longer stretch-distort. GUI "Frame fit" option:
  smart crop-to-fill (default) · blurred-background fill · letterbox · stretch (legacy)
- `get_fit_filters` / `build_blur_fit_graph` in `ffmpeg_processing.py`; `setsar=1` normalizes
  sample aspect across heterogeneous sources
- Target resolution/aspect = highest-resolution source by pixel area (`get_max_resolution`); a dominant vertical source makes a vertical video
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
- Not included by design: speed ramps (`setpts` would break the zero-drift frame-locked timeline); effects don't apply in ProRes precise mode (kept pristine for external editing)

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

## Phase 4 — transitions & polish ⬜
- Opt-in `xfade` crossfades/wipes on low-energy boundaries (requires re-encode assembly path; hard cuts stay the fast default)
- LUT-based color grading for a consistent look across mismatched sources

## Phase 5 — full automation ⬜
- Watch-folder mode built on the existing `video_processor.py` CLI: drop audio + clips, video appears in `output/`
- launchd job on macOS; candidate for running on the-all-thing server later

## Documentation policy
Every feature change updates: this file (status + any new hook points), `README.mac.md` (user-facing usage), and `CLAUDE.md` if conventions change.
