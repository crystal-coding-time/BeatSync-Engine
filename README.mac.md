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

The vision stage is watchdogged: a hung llama-server request times out (60s steady-state, 180s during model warmup; `BEATSYNC_QWEN_REQUEST_TIMEOUT`/`BEATSYNC_QWEN_WARMUP_TIMEOUT`), and repeated timeouts trip a circuit breaker that restarts the server once, then abandons AI tagging for the run and continues with deterministic visual tags — a wedged server costs minutes, never a frozen render (`BEATSYNC_QWEN_TIMEOUT_BREAKER`, `BEATSYNC_QWEN_BATCH_TIMEOUT`). Known-buggy llama.cpp subsystems (prompt cache, context checkpoints, flash-attn — default-on in recent Homebrew builds) are explicitly disabled, and the server writes a real log file next to the batch files in `input/video_analysis_cache/`. `BEATSYNC_DISABLE_QWEN=1` still skips the stage entirely.

Useful env vars:
- `BEATSYNC_DISABLE_QWEN=1` — skip AI scene tagging (works without the model downloads)
- `GRADIO_SERVER_PORT=<port>` — pin the UI port
- `BEATSYNC_BEAT_BACKEND` — beat-tracking backend. **`run.sh` now defaults this to `beat_this`** — the [Beat This!](https://github.com/CPJKU/beat_this) transformer for beat **and downbeat** tracking (cuts prefer real bar lines). Requires `.venv/bin/pip install beat-this torch torchaudio` (~8 MB model auto-downloads on first use); runs on CPU by default, `BEATSYNC_BEAT_THIS_DEVICE=mps` forces the GPU. Falls back to librosa on any failure. Export `BEATSYNC_BEAT_BACKEND=librosa` before `./run.sh` to use the classic librosa tracker instead.
- `BEATSYNC_DOWNBEAT_PHASE=off` — on the librosa backend, bar/phrase anchors (the beats that attract the biggest cut bonuses) are phased onto the estimated downbeat (strongest kick/bass/chord-change grid position) instead of assuming beat 0 starts a bar. `off` (or `0`) restores the old beat-0 grid exactly. Irrelevant when beat-this downbeats are available — those override the anchors either way.
- `BEATSYNC_CUT_LEAD_FRAMES=<0-2>` — how many frames before each beat the cut lands (default 1, the classic editor's trick; 0 restores exact on-beat cuts)

## Input formats

Audio: `.mp3`, `.wav`, `.flac`. Video sources: `.mp4`, `.mkv`, `.mov`, `.webm`, `.m4v`, `.avi`, `.gif` (upper- or lowercase). Sources shorter than a cut segment (e.g. GIFs) are looped automatically to keep the timeline frame-accurate.

Still images work too: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Each segment cut from a still gets its own gentle Ken Burns pan/zoom so photos read as footage. **HEIC is not supported** (Homebrew ffmpeg has no HEIF decoder) — export iPhone photos to JPEG first. In ProRes precise mode stills render as static frames (proxies stay pristine).

The video dropzone stays empty and always accepts drops; loaded files accumulate in the "Loaded videos" list below it, where they can be removed individually or cleared. (This sidesteps gradio#10325 — the stock File component ignores drops once it holds files.) After the app is restarted with code changes, refresh the browser tab: the accepted-file-type filter is baked in at page load.

## Multiple songs

Drop as many songs as you like into the audio box — they play in the order shown in the list
(remove/reorder before rendering). Each song gets its own full musical analysis (beat grid,
sections, structure), the join lands as a clean cut on the next song's first boundary, and the
footage plan spans the whole extended video so your clips spread across all songs instead of
repeating per track. The final soundtrack is assembled sample-exactly at full quality. One song
behaves exactly as before. (Shuffle-mode's filename-derived seed uses the first song.)

## Style controls

The interface is organized around the workflow — **1 · Media** (drop audio + clips),
**2 · Direct the edit**, **3 · Render & result** — because the engine makes the mechanical
decisions itself, so the defaults are the real UI. Zone 2 groups every control into five
tabs by what you're deciding: **Vibe** (Style, Look, effect intensity, and an optional
Customize-effects panel), **The Cut** (source & visual variety, split-screen, crossfades,
speed ramps), **Framing** (output canvas + frame fit), **Text** (entries, position, size),
and **Export** (encoder/quality, FPS, filename).

Cut placement listens to more than the beat grid: sharp transients (SuperFlux onsets) pull
cuts toward real percussive hits, and in quiet low-percussion passages the engine cuts on
chord changes instead of an empty grid. Same inputs still always render the same video.

With the optional structure backend installed (`.venv/bin/pip install "all-in-one-mlx"
"demucs-mlx[convert]"`, Apple Silicon only), the engine goes further: a local model labels
the song's real sections (verse/chorus/bridge/intro/outro, with choruses upgraded to drops
when the energy or the bass stem confirms it), drum-stem onsets drive cut snapping without
vocal/pad contamination, and cuts avoid landing mid-vocal-phrase off the bar grid. The first
analysis of a new track takes ~2 minutes (stem separation); after that it's cached and free.
Without the packages everything falls back to the built-in heuristics.
`BEATSYNC_DISABLE_STRUCTURE=1` / `BEATSYNC_DISABLE_STEMS=1` switch it off.

- **Style** — Minimal (clean cuts, no effects), Music video (AMV), or Hype. See "Effect
  style" details below.
- **Output canvas** — the frame every render targets: 16:9 1080p (default), 16:9 4K, 9:16 Portrait, or Match best source (the old behavior, where the highest-resolution upload decides the resolution *and* aspect — one portrait phone clip could flip the whole video vertical). A fixed canvas makes the output predictable for YouTube/TV; pick Portrait deliberately for Shorts/Reels. Output FPS still follows the highest-resolution source — set Custom FPS (Export tab) to override (so a low-fps GIF in the list can't drag the whole render down).
- **Frame fit** (Framing tab) — how sources with a different aspect ratio fill the canvas: Auto (default), Blurred background (undistorted over a blurred fill), or Letterbox. (The legacy Stretch distortion is no longer offered in the UI; headless callers can still pass `fit_mode='stretch'`.) Auto is subject-aware, in three tiers by how badly a source mismatches:
  - *Mild mismatch* — fill-and-crop, but the crop window follows the detected subject *through the shot* — a gentle tracked pan that keeps a moving subject in frame (face detection via a small local YuNet model in `models/`, falling back to motion/detail tracking) instead of blindly taking the frame center. Never crops away more than ~15% of a source; barely-moving subjects get a static subject-centered crop.
  - *Bigger mismatch* — the full shot composites over a graded "echo" fill: the clip's own image blurred, darkened and desaturated behind it, drifting slowly so the margins never look static. The foreground gets a gentle continuous push-in for kinetic energy, and in Music video/Hype styles the margins pulse subtly on each beat. On drop cuts these clips occasionally punch in to full-bleed and settle back into the framed view.
  - *Extreme mismatch* (e.g. a vertical phone video in a landscape edit) — a slow scanning pan: the frame fills the canvas width and sweeps along the clip, easing to a stop on the subject. No content is lost and it reads as camerawork, not a compromise.
  Anamorphic sources are resampled to square pixels first, so they no longer render distorted. If the face model goes missing, re-fetch it with `.venv/bin/python scripts/fetch_yunet.py`; set `BEATSYNC_YUNET_MODEL=/nonexistent` to disable face detection entirely.
- **Effect style details** — Minimal renders clean cuts with no effects. Effects in the other styles are beat-aware: zoom punches and flashes on drop cuts, saturation pulsing on the actual beat grid, plus shake/vignette/grain in Hype. AMV also mixes in occasional zoom-blur smears and motion trails; Hype adds pixelize bursts, strobes, posterize flashes, hue sweeps, mirrors, punchy motion-smear hits and a rare fisheye — a few per segment at most, all beat-placed. Both get an occasional dutch tilt on drop cuts (a few degrees of rotation, flipping sides on the beat, never showing black corners). AMV/Hype also get occasional split transitions at cut boundaries: whip pans and glitch cuts into drops, dips to black/white between calm sections. Same inputs always render the same video.
- **Effect mode** (Vibe tab › Customize effects) — *Curated* (the presets above, default) · *Custom* (a checkbox palette appears — tick exactly the effects you want; the music-aware planner still decides where they land; nothing ticked = no effects) · *Shuffle* (a seeded random palette; the seed box makes it reproducible — same seed, same video; leave 0 to derive it from the song's filename; bump it to re-roll the mix).
- **Effect intensity** (Vibe tab) — scales all effect strengths. Impact effects (zoom punches, flashes, shake) additionally scale with the music's actual momentary loudness at each cut, so a huge drop hits harder than a mild one.
- **Look** — optional color grade applied per segment (`lut3d`): Vintage, Cross Process, Cool, Warm, High Contrast, Day For Night. The LUTs are baked locally by `scripts/bake_looks.py`; ~free at render time.
- **Source variety** (The Cut tab) — how hard the planner spreads cuts across your uploads. 0 = pure quality auction (the best-scoring moments win, some files may never appear), 1 = near-even spread. Default 0.4 guarantees every usable source appears at least once while keeping the best footage on the big drops; above 0 the spread uses proportional-fair pressure, so a genuinely stronger source keeps a bounded lead instead of being ground down to round-robin. The render log prints a per-source usage histogram.
- **Visual variety** (The Cut tab) — avoids runs of *visually similar* shots even across different files (e.g. five wide landscapes in a row). A small local DINOv2 model embeds one frame per candidate moment and clusters the footage (CPU-only, deterministic, cached per source); the planner then penalizes picking from a just-used cluster or anything too cosine-similar to the last few picks. Fetch the model once with `.venv/bin/python scripts/fetch_dinov2.py` (~87 MB into `models/`); without it the slider silently does nothing. 0 = off (exact legacy plans). Default 0.4. `BEATSYNC_DISABLE_EMBED=1` is the kill switch.
- **Crossfade calm cuts** (The Cut tab) — opt-in, off by default: roughly one in three boundaries between two calm segments dissolves (mostly fades, occasional wipes) instead of hard-cutting; energetic cuts always stay hard. Costs one extra short re-encode per crossfade; everything else keeps the fast assembly path. Not available in ProRes precise mode.
- **Speed ramps (experimental, The Cut tab)** — opt-in: slow-motion drifts through calm segments, speed-ups through builds, decelerating hits into drops, and rare freeze-frames. Deep slow-mo (below 0.6×) on ordinary 24–30fps footage is motion-interpolated (ffmpeg `minterpolate`) so it stays smooth instead of juddering — it costs a few extra seconds of render per affected segment, so the planner keeps it rare. Frame-count-safe by construction (every retimed segment is verified after extraction); skips images, sub-24fps sources, and ProRes mode.
- **Pair vertical clips (split screen)** (The Cut tab) — on by default: when two or more vertical sources are loaded, some hard-cut segments render two of them side by side, filling the widescreen frame completely (the concert-multicam look). Occasional and never back-to-back; each pane gets its own subject-tracked framing. Untick to route every vertical clip through the frame-fit ladder instead. H.264/HEVC modes only.

Effects, text, looks and speed ramps apply to the H.264/HEVC modes only; ProRes precise mode stays untouched for external editing (the UI notes this next to the mode picker). Precise mode handles mixed-resolution sources: everything is normalized to the target resolution during the ProRes proxy conversion so the lossless assembly stays valid.

Every run writes its full pipeline output (including ffmpeg errors) to `output/render_<timestamp>.log` — if a render fails, the error message points at that log.

## Text overlays

Enter lines in the "Text entries" box (one entry per line — quotes, captions, titles, anything). Every line is guaranteed its own time window: entries are spaced evenly across the video, window starts snap to beats (preferring non-drop moments), each stays on screen ~3 seconds spanning cuts, fading in/out. Pin an entry to a moment with `@`: `@15 Finish strong` or `@1:23 Halfway there` — pins win, auto-placed entries move around them. The render log prints the exact schedule. Position (lower third/center/top) and size are configurable. The **Font** dropdown lists every `.ttf`/`.otf` found on the machine (user, library, and system-supplemental font folders); the custom-path box beside it takes any font file — no install needed (Motion registers it with CoreText at render time) — and overrides the dropdown. *Auto* keeps the historic default: `$BEATSYNC_FONT` if set, else Arial Bold. Prefer a Bold cut; the caption styling is built around one. Like effects, text applies to H.264/HEVC modes only.

**Text style.** *Classic* (default) is the original static white caption. *Motion* renders the same schedule as beat-reactive motion graphics: text pulses subtly with a glow halo that spreads on musical onsets, and two inline tags unlock extras per line — `[style:glitch]` (accent-colored split copies that burst on strong hits, e.g. `@45 [style:glitch] Drop!`) and `[widget:progress duration:4s]` (a progress bar under the text; `duration:4` without the `s` also works; omit it to span the whole window; capped at 60s). A widget's duration widens that entry's screen time beyond the usual ~3s so the bar really runs its full length, and widget entries get placement priority so crowded scripts don't drop them first. Other bracketed text (`[note: whatever]`) is left in the caption untouched. The accent color picker tints the glitch and the bar fill. Quick-add buttons under the entries box append ready-made template lines — the Loading bar and Glitch templates also switch the style to Motion, since their tags only render there. Motion needs the `cairosvg` package (installed in the project venv); if it's missing the render falls back to Classic with a log line (tags are stripped, never burned into the video). `BEATSYNC_DISABLE_STYLEDTEXT=1` forces Classic.

**Knowing what will render.** The **📋 Preview schedule** button on the Text tab shows every line's planned time window — or why it won't render (`SKIPPED — no room on the timeline`, unreadable `@`-stamp, tags in Classic style) — before you spend minutes on a render. It's beat-exact once the session has analyzed the track (after your first render), and a labeled estimate before that. The render status also reports `Text: placed X/Y entries, N skipped`. Pins are forgiving: `@1:23:45` (h:mm:ss) works, and a malformed stamp warns and auto-places the line instead of printing the raw `@…` on screen. When lines are skipped for room, pin the ones that matter — pinned entries claim their windows first.
