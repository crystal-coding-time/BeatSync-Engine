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

## Phase 2 — beat-aware effects engine ⬜
- Composable ffmpeg `-vf` snippet library: punch-in zoom on beat, white-flash on drop cuts, RGB split, shake, speed ramps, saturation pulse, grain/vignette
- Driven by stage 6 planner profiles (energy / section / `drop`·`build`·`rhythm` targets) — effects hit harder on drops, ease off on intros
- Hook point: per-segment filter chain in `extract_clip_segment_ffmpeg` (`ffmpeg_processing.py`)
- Deterministic via existing `_stable_rng`
- GUI: style preset (Clean / AMV / Hype) + intensity slider

## Phase 3 — text overlays: quotes throughout the video ⬜
- Motivational quotes (and similar text) rendered over the video at planned moments — not title cards
- Quote pool: user-editable file (e.g. `input/quotes.txt`), plus curated built-in default set
- Placement driven by the section planner: e.g. one quote per section, fade in/out on beat boundaries, avoid drops where fast cuts fight legibility
- `drawtext` with macOS system fonts (configurable font/size/position), burned per-segment so fast concat assembly is preserved
- GUI: enable toggle, quote file picker, style options

## Phase 4 — transitions & polish ⬜
- Opt-in `xfade` crossfades/wipes on low-energy boundaries (requires re-encode assembly path; hard cuts stay the fast default)
- LUT-based color grading for a consistent look across mismatched sources

## Phase 5 — full automation ⬜
- Watch-folder mode built on the existing `video_processor.py` CLI: drop audio + clips, video appears in `output/`
- launchd job on macOS; candidate for running on the-all-thing server later

## Documentation policy
Every feature change updates: this file (status + any new hook points), `README.mac.md` (user-facing usage), and `CLAUDE.md` if conventions change.
