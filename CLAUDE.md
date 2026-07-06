# BeatSync Engine (mac-port branch)

macOS port of a Windows-only beat-synced music video generator. Owner is learning self-hosting; explain non-obvious decisions briefly.

## Ground rules
- **Keep docs in sync with code — every feature change updates `docs/ROADMAP.md` (status) and `README.mac.md` (usage) in the same commit.**
- Work on the `mac-port` branch; upstream is `main` (`Merserk/BeatSync-Engine`). Keep Windows behavior intact: platform-specific code branches on `os.name == 'nt'` or falls back from bundled `bin/` paths to `PATH` lookups.
- Python env: `.venv` (Homebrew python@3.13), no CuPy on Mac. Launch with `./run.sh` (Gradio UI on 7860).

## Architecture in one paragraph
`src/gui.py` (Gradio) → 6-stage auto pipeline in `src/auto_mode/` (stage1 librosa beats → stage2 features → stage3 sections → stage4 cut selection → stage5 optional Qwen3-VL tagging via llama.cpp → stage6 planner assigns each segment a source clip + profile) → `src/video_processor.py` extracts segments in parallel → `src/ffmpeg_processing.py` builds per-segment ffmpeg commands (per-segment `-vf` chain = the hook point for effects/text) → concat stream-copy assembly.

## Testing
No test suite. Smoke test = run the pipeline headless on synthetic media:
`ffmpeg -f lavfi -i "sine=frequency=440:beep_factor=8:duration=20" beat.wav`, testsrc clips, then call `gui._process_video_impl(...)` with `PYTHONPATH=src`. `BEATSYNC_DISABLE_QWEN=1` skips the slow vision stage.
