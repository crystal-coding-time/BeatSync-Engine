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

## Phase 2 — beat-aware effects engine ✅ (2026-07-06)
- `src/effects.py`: per-segment filter chains driven by stage 6 planner metadata
  (`target` drop/rhythm/build/soft/flow, `impact` energy). Effects land where the music does:
  punch-in zoom (zoompan) on drop/rhythm cuts, white flash on drops, saturation pulse at the
  beat frequency, chromatic aberration occasionally on hard cuts; hype adds shake, vignette, grain
- GUI: Effect style Clean (default) / AMV / Hype + intensity slider
- Deterministic (seeded per segment); ffmpeg gotcha encoded in code comments: `crop` can't animate w/h → zoom uses `zoompan`
- Not included by design: speed ramps (`setpts` would break the zero-drift frame-locked timeline); effects don't apply in ProRes precise mode (kept pristine for external editing)

## Phase 3 — text overlay system ⬜
- General-purpose text over the video at planned moments (motivational quotes are one use case; captions, lyrics snippets, watermarks, titles are others)
- Text pool: user-editable file (e.g. `input/text.txt`, one entry per line) and/or direct GUI input
- Placement driven by the section planner: e.g. one entry per section, fade in/out on beat boundaries, avoid drops where fast cuts fight legibility
- `drawtext` with macOS system fonts (configurable font/size/position/color), burned per-segment so fast concat assembly is preserved
- GUI: enable toggle, text source, style options

## Phase 4 — transitions & polish ⬜
- Opt-in `xfade` crossfades/wipes on low-energy boundaries (requires re-encode assembly path; hard cuts stay the fast default)
- LUT-based color grading for a consistent look across mismatched sources

## Phase 5 — full automation ⬜
- Watch-folder mode built on the existing `video_processor.py` CLI: drop audio + clips, video appears in `output/`
- launchd job on macOS; candidate for running on the-all-thing server later

## Documentation policy
Every feature change updates: this file (status + any new hook points), `README.mac.md` (user-facing usage), and `CLAUDE.md` if conventions change.
