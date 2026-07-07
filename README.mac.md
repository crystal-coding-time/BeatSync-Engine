# BeatSync Engine — macOS Port

This fork runs [BeatSync-Engine](https://github.com/Merserk/BeatSync-Engine) (originally Windows-only) on macOS, including Apple Silicon. All Mac-specific changes live on the `mac-port` branch.

## Setup

```bash
brew install ffmpeg llama.cpp python@3.13
python3.13 -m venv .venv
grep -v cupy requirements.txt | .venv/bin/pip install -r /dev/stdin
```

The Qwen3-VL vision models (for AI scene tagging) go in `bin/models/`:

```bash
mkdir -p bin/models && cd bin/models
curl -LO 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf'
curl -LO 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf'
```

## Run

```bash
./run.sh
```

Opens the web UI at http://127.0.0.1:7860.

## What's different from the Windows build

| Windows | macOS |
|---|---|
| Bundled portable Python/FFmpeg/CUDA | Homebrew `python@3.13`, `ffmpeg`, `llama.cpp` |
| CuPy CUDA audio analysis | CPU (librosa/numpy) fallback — automatic |
| NVIDIA NVENC encoding | Apple VideoToolbox (`h264_videotoolbox` / `hevc_videotoolbox`) |
| llama.cpp Vulkan (`bin/llama-bin-win-vulkan-x64/*.exe`) | llama.cpp Metal from Homebrew, found via `PATH` |
| `run.bat` | `run.sh` |

Binary resolution order: the bundled `bin/` locations are tried first (so a portable layout still works), then the system `PATH`. Overridable with `BEATSYNC_QWEN_LLAMA_DIR`, `BEATSYNC_QWEN_LLAMA_MODEL`, `BEATSYNC_QWEN_LLAMA_MMPROJ`.

Useful env vars:
- `BEATSYNC_DISABLE_QWEN=1` — skip AI scene tagging (works without the model downloads)
- `GRADIO_SERVER_PORT=<port>` — pin the UI port

## Input formats

Audio: `.mp3`, `.wav`, `.flac`. Video sources: `.mp4`, `.mkv`, `.mov`, `.webm`, `.m4v`, `.avi`, `.gif` (upper- or lowercase). Sources shorter than a cut segment (e.g. GIFs) are looped automatically to keep the timeline frame-accurate.

The video dropzone stays empty and always accepts drops; loaded files accumulate in the "Loaded videos" list below it, where they can be removed individually or cleared. (This sidesteps gradio#10325 — the stock File component ignores drops once it holds files.) After the app is restarted with code changes, refresh the browser tab: the accepted-file-type filter is baked in at page load.

## Style controls

- **Frame fit** — how sources with a different aspect ratio fill the frame: Smart crop (fill and center-crop, default), Blurred background (undistorted over a blurred fill), Letterbox, or Stretch (legacy). Output resolution/aspect follows the highest-resolution source (by pixel area), and the output FPS follows that same source — set Custom FPS to override (so a low-fps GIF in the list can't drag the whole render down).
- **Effect style** — Clean (no effects), AMV, or Hype. Effects are beat-aware: zoom punches and flashes on drop cuts, saturation pulsing at the song's tempo, plus shake/vignette/grain in Hype. Same inputs always render the same video.
- **Effect intensity** — scales all effect strengths.

Effects apply to the H.264/HEVC modes only; ProRes precise mode stays untouched for external editing. Precise mode handles mixed-resolution sources: everything is normalized to the target resolution during the ProRes proxy conversion so the lossless assembly stays valid.

Every run writes its full pipeline output (including ffmpeg errors) to `output/render_<timestamp>.log` — if a render fails, the error message points at that log.

## Text overlays

Enter lines in the "Text entries" box (one entry per line — quotes, captions, titles, anything). Every line is guaranteed its own time window: entries are spaced evenly across the video, window starts snap to beats (preferring non-drop moments), each stays on screen ~3 seconds spanning cuts, fading in/out. Pin an entry to a moment with `@`: `@15 Finish strong` or `@1:23 Halfway there` — pins win, auto-placed entries move around them. The render log prints the exact schedule. Position (lower third/center/top) and size are configurable. Text renders via Pillow with Arial Bold by default; set `BEATSYNC_FONT=/path/to/font.ttf` to change the font. Like effects, text applies to H.264/HEVC modes only.

## Docs

- `docs/ROADMAP.md` — feature roadmap and status. **Keep it and this README updated when features land.**
