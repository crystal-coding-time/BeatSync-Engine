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

UI quirk: Gradio's file input ignores drag-drops once it already holds files (gradio#10325), so use the "➕ Drop here to add more videos" zone beneath it to append — it merges into the main list and clears itself. After the app is restarted with code changes, refresh the browser tab: the accepted-file-type filter is baked in at page load.

## Docs

- `docs/ROADMAP.md` — feature roadmap and status. **Keep it and this README updated when features land.**
