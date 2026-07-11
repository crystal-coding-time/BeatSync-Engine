#!/usr/bin/env python3
"""Video source analysis for Auto Mode.

This module builds a reusable library of source moments. FFmpeg/OpenCV provide
fast deterministic signals for every candidate; Qwen3-VL optionally adds
semantic editing tags for the best candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np

from gpu_cpu_utils import GPU_AVAILABLE, cp
from ffmpeg_processing import (
    FFPROBE_PATH,
    detect_video_scene_changes,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    is_image_source,
)
from logger import ROOT_DIR, setup_environment
from visual_embeddings import OptionalBackend, SidecarCache


setup_environment()

# v12: candidates additionally carry media_type/source_width/source_height/
# source_fps, GIF loop_seam, and *_native flow for sub-15fps sources (the
# media-aware planner's inputs). The bump recomputes stale sidecars; every
# pre-existing field is derived exactly as in v11, so plans that ignore the
# new fields stay byte-identical.
ANALYSIS_VERSION = "auto_av_analysis_v12_media"
DEFAULT_QWEN_MODEL_DIR = os.path.join(ROOT_DIR, "bin", "models")
DEFAULT_QWEN_GGUF_MODEL = os.path.join(DEFAULT_QWEN_MODEL_DIR, "Qwen3VL-2B-Instruct-Q8_0.gguf")
DEFAULT_QWEN_MMPROJ_MODEL = os.path.join(DEFAULT_QWEN_MODEL_DIR, "mmproj-Qwen3VL-2B-Instruct-F16.gguf")
_BUNDLED_LLAMA_CPP_DIR = os.path.join(ROOT_DIR, "bin", "llama-bin-win-vulkan-x64")
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def _find_default_llama_dir() -> str:
    if os.path.isdir(_BUNDLED_LLAMA_CPP_DIR):
        return _BUNDLED_LLAMA_CPP_DIR
    found = shutil.which("llama-server")
    return os.path.dirname(os.path.realpath(found)) if found else _BUNDLED_LLAMA_CPP_DIR


DEFAULT_LLAMA_CPP_DIR = _find_default_llama_dir()
VIDEO_ANALYSIS_CACHE_DIR = os.path.join(ROOT_DIR, "input", "video_analysis_cache")
_LLAMA_VERSION_TOKENS: Dict[str, str] = {}


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _safe_name(path: str) -> str:
    return os.path.basename(path) or path


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _resolve_qwen_backend_paths(qwen_model_path: str | None) -> Dict[str, str]:
    llama_dir = os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", DEFAULT_LLAMA_CPP_DIR)
    model_override = os.environ.get("BEATSYNC_QWEN_LLAMA_MODEL")
    mmproj_override = os.environ.get("BEATSYNC_QWEN_LLAMA_MMPROJ")

    requested = os.path.abspath(qwen_model_path) if qwen_model_path else ""
    if model_override:
        model_path = os.path.abspath(model_override)
    elif requested and os.path.isfile(requested) and requested.lower().endswith(".gguf"):
        model_path = requested
    else:
        candidates = []
        if requested:
            candidates.extend([
                os.path.join(requested, os.path.basename(DEFAULT_QWEN_GGUF_MODEL)),
                os.path.join(os.path.dirname(requested), os.path.basename(DEFAULT_QWEN_GGUF_MODEL)),
            ])
        candidates.append(DEFAULT_QWEN_GGUF_MODEL)
        model_path = next((path for path in candidates if os.path.exists(path)), DEFAULT_QWEN_GGUF_MODEL)

    if mmproj_override:
        mmproj_path = os.path.abspath(mmproj_override)
    else:
        candidates = [
            os.path.join(os.path.dirname(model_path), os.path.basename(DEFAULT_QWEN_MMPROJ_MODEL)),
            DEFAULT_QWEN_MMPROJ_MODEL,
        ]
        mmproj_path = next((path for path in candidates if os.path.exists(path)), DEFAULT_QWEN_MMPROJ_MODEL)

    return {
        "llama_dir": os.path.abspath(llama_dir),
        "server": os.path.abspath(os.path.join(llama_dir, f"llama-server{EXE_SUFFIX}")),
        "mtmd": os.path.abspath(os.path.join(llama_dir, f"llama-mtmd-cli{EXE_SUFFIX}")),
        "model": os.path.abspath(model_path),
        "mmproj": os.path.abspath(mmproj_path),
    }


def _qwen_backend_available(qwen_model_path: str | None) -> bool:
    paths = _resolve_qwen_backend_paths(qwen_model_path)
    return all(os.path.exists(paths[key]) for key in ["server", "mtmd", "model", "mmproj"])


def _qwen_backend_model_path(qwen_model_path: str | None) -> str:
    return _resolve_qwen_backend_paths(qwen_model_path)["model"]


def _path_signature_token(path: str) -> str:
    try:
        stat = os.stat(path)
        return f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return f"{os.path.basename(path)}:missing"


def _llama_version_token(llama_dir: str) -> str:
    llama_dir = os.path.abspath(llama_dir)
    cached = _LLAMA_VERSION_TOKENS.get(llama_dir)
    if cached:
        return cached

    mtmd = os.path.join(llama_dir, f"llama-mtmd-cli{EXE_SUFFIX}")
    token = _path_signature_token(mtmd)
    if os.path.exists(mtmd):
        env = os.environ.copy()
        env["PATH"] = llama_dir + os.pathsep + env.get("PATH", "")
        try:
            result = subprocess.run(
                [mtmd, "--version"],
                cwd=llama_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            version = ((result.stdout or "") + (result.stderr or "")).strip()
            if version:
                token = version.splitlines()[0][:120]
        except Exception:
            pass
    _LLAMA_VERSION_TOKENS[llama_dir] = token
    return token


def _qwen_backend_signature_token(qwen_model_path: str | None) -> str:
    paths = _resolve_qwen_backend_paths(qwen_model_path)
    required = ["server", "mtmd", "model", "mmproj"]
    if not all(os.path.exists(paths[key]) for key in required):
        return "ai_missing"
    raw = "|".join([
        "llama_vulkan",
        _path_signature_token(paths["model"]),
        _path_signature_token(paths["mmproj"]),
        _path_signature_token(paths["server"]),
        _path_signature_token(paths["mtmd"]),
        _llama_version_token(paths["llama_dir"]),
    ])
    return "ai_" + _hash_text(raw, length=20)


def _video_signature(video_file: str, enable_ai: bool, qwen_model_path: str | None) -> str:
    stat = os.stat(video_file)
    model_token = "no_ai"
    if enable_ai:
        model_token = _qwen_backend_signature_token(qwen_model_path)
    raw = "|".join([
        ANALYSIS_VERSION,
        os.path.abspath(video_file),
        str(stat.st_size),
        str(int(stat.st_mtime)),
        model_token,
    ])
    return _hash_text(raw, length=24)


# Shared sidecar cache for the full per-video analysis result. Same on-disk
# layout as before ({sha1(stem)[:8]}_{signature}.json), pretty-printed
# (indent=2), with logged read/write failures. make_dir_on_path=True mirrors the
# old _cache_path, which created the cache directory up front.
_VIDEO_ANALYSIS_CACHE = SidecarCache(
    VIDEO_ANALYSIS_CACHE_DIR, "", indent=2,
    log_label="video analysis cache", make_dir_on_path=True,
)


def _cache_path(video_file: str, enable_ai: bool, qwen_model_path: str | None) -> str:
    return _VIDEO_ANALYSIS_CACHE.path(
        video_file, _video_signature(video_file, enable_ai, qwen_model_path)
    )


def _load_cache(path: str, require_ai: bool = False) -> Dict | None:
    data = _VIDEO_ANALYSIS_CACHE.load(path)
    if not isinstance(data, dict):
        return None
    if data.get("analysis_version") == ANALYSIS_VERSION:
        if require_ai and not data.get("ai_enabled"):
            return None
        return data
    return None


def _save_cache(path: str, data: Dict) -> None:
    _VIDEO_ANALYSIS_CACHE.save(path, data)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if hi is None:
        hi = max(lo, value)
    return max(lo, min(hi, value))


def _video_analysis_workers(video_count: int) -> int:
    if video_count <= 1:
        return 1
    cpu = os.cpu_count() or 4
    # Decode + OpenCV scoring are native workloads, so multiple source videos can
    # safely use otherwise idle CPU cores. Keep the default conservative enough
    # to avoid crushing slow disks, and expose an env override for Ryzen-class CPUs.
    default_workers = min(video_count, max(1, cpu // 4))
    return _env_int("BEATSYNC_VIDEO_ANALYSIS_WORKERS", default_workers, lo=1, hi=video_count)


def _candidate_metric_workers(window_count: int, use_gpu: bool) -> int:
    if use_gpu or window_count < 80:
        return 1
    cpu = os.cpu_count() or 4
    default_workers = min(4, max(1, cpu // 4), window_count)
    return _env_int("BEATSYNC_CANDIDATE_METRIC_WORKERS", default_workers, lo=1, hi=max(1, window_count))


def _opencv_decode_threads() -> int:
    cpu = os.cpu_count() or 4
    default_threads = min(8, max(2, cpu // 2))
    return _env_int("BEATSYNC_OPENCV_FFMPEG_THREADS", default_threads, lo=1, hi=max(1, cpu))


def _open_video_capture(video_file: str) -> cv2.VideoCapture:
    """Open video with FFmpeg decoder threads configured for OpenCV.

    This keeps the same sampled frames and scoring formulas, but lets FFmpeg use
    more CPU cores while OpenCV decodes/seeks source frames.
    """
    threads = _opencv_decode_threads()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"threads;{threads}")
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    return cv2.VideoCapture(video_file)


def analyze_video_sources(
    video_files: Sequence[str],
    audio_profile: Dict | None = None,
    use_gpu: bool = False,
    enable_ai: bool = True,
    qwen_model_path: str | None = None,
) -> Dict:
    """Analyze all source videos and return candidate moments for Auto Mode."""
    total_started = time.perf_counter()
    requested_qwen_model_path = qwen_model_path or DEFAULT_QWEN_MODEL_DIR
    qwen_model_path = _qwen_backend_model_path(requested_qwen_model_path)
    existing = [os.path.abspath(p) for p in video_files if p and os.path.exists(p)]
    if not existing:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "videos": [],
            "candidates": [],
            "ai_enabled": False,
            "summary": "No source videos available for analysis.",
        }

    ai_available = bool(enable_ai and _qwen_backend_available(requested_qwen_model_path))
    if ai_available:
        qwen_model_path = _qwen_backend_model_path(requested_qwen_model_path)
    print("\n   VIDEO ANALYSIS - Auto Mode visual library")
    print(f"   Source videos: {len(existing)}")
    print(f"   Qwen semantic tags: {'enabled' if ai_available else 'disabled/fallback'}")

    results_by_index: Dict[int, Dict] = {}
    cache_paths: Dict[int, str] = {}
    jobs: List[Dict] = []
    cache_hits = 0

    for idx, video_file in enumerate(existing, 1):
        cache_file = _cache_path(video_file, ai_available, qwen_model_path)
        cache_paths[idx] = cache_file
        cached = _load_cache(cache_file, require_ai=ai_available)
        if cached:
            cache_hits += 1
            print(f"   Reusing cached visual analysis {idx}/{len(existing)}: {_safe_name(video_file)}")
            results_by_index[idx] = cached
        else:
            jobs.append({"index": idx, "video_file": video_file, "cache_file": cache_file})

    workers = _video_analysis_workers(len(jobs))
    if jobs:
        if workers > 1:
            print(
                f"   CPU visual analysis workers: {workers} "
                f"(parallel videos; Qwen stays sequential to protect VRAM)"
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _analyze_single_video,
                        job["video_file"],
                        use_gpu,
                        ai_available,
                        qwen_model_path,
                        audio_profile or {},
                        True,
                        job["index"],
                        len(existing),
                    ): job
                    for job in jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        results_by_index[job["index"]] = future.result()
                    except Exception as exc:
                        print(
                            f"   ⚠️  Parallel analysis failed for "
                            f"{_safe_name(job['video_file'])}: {exc}"
                        )
                        print("   ↪ Retrying that video safely in serial mode...")
                        try:
                            results_by_index[job["index"]] = _analyze_single_video(
                                job["video_file"],
                                use_gpu,
                                ai_available,
                                qwen_model_path,
                                audio_profile or {},
                                True,
                                job["index"],
                                len(existing),
                            )
                        except Exception as retry_exc:
                            print(
                                f"   ⚠️  Serial retry also failed for "
                                f"{_safe_name(job['video_file'])}: {retry_exc}"
                            )
                            print(
                                "   ↪ Storing an empty result for it so the other "
                                "videos still get analyzed and cached."
                            )
                            results_by_index[job["index"]] = _empty_video_result(
                                job["video_file"]
                            )
        else:
            print("   CPU visual analysis workers: 1 (serial)")
            for job in jobs:
                results_by_index[job["index"]] = _analyze_single_video(
                    job["video_file"],
                    use_gpu,
                    ai_available,
                    qwen_model_path,
                    audio_profile or {},
                    False,
                    job["index"],
                    len(existing),
                )

    # When the deterministic CPU-heavy pass ran in parallel, run Qwen after it in
    # original video order. A single multi-video Qwen worker is used by default so
    # the llama.cpp model loads once, not once per source video.
    deferred_jobs = [
        job for job in jobs
        if results_by_index.get(job["index"], {}).get("ai_deferred") and ai_available
    ]
    batch_qwen = os.environ.get("BEATSYNC_QWEN_BATCH_VIDEOS", "1") != "0"
    if len(deferred_jobs) > 1 and batch_qwen:
        _complete_deferred_qwen_batch(
            video_items=[(job, results_by_index[job["index"]]) for job in deferred_jobs],
            use_gpu=use_gpu,
            qwen_model_path=qwen_model_path,
            audio_profile=audio_profile or {},
            total_video_count=len(existing),
        )
    else:
        for job in deferred_jobs:
            idx = job["index"]
            video_data = results_by_index.get(idx)
            if not video_data:
                continue
            _complete_deferred_qwen(
                video_data=video_data,
                use_gpu=use_gpu,
                qwen_model_path=qwen_model_path,
                audio_profile=audio_profile or {},
                label=f"{idx}/{len(existing)}",
            )

    for job in jobs:
        video_data = results_by_index.get(job["index"])
        if video_data:
            _save_cache(job["cache_file"], video_data)

    videos: List[Dict] = [results_by_index[i] for i in range(1, len(existing) + 1) if i in results_by_index]
    all_candidates: List[Dict] = []
    for video_data in videos:
        all_candidates.extend(video_data.get("candidates", []))

    if not all_candidates:
        summary = "No usable candidate moments were found; renderer will use fallback sampling."
    else:
        action_avg = float(np.mean([c.get("action_score", 0.0) for c in all_candidates]))
        beauty_avg = float(np.mean([c.get("beauty_score", 0.0) for c in all_candidates]))
        quality_avg = float(np.mean([c.get("quality_score", 0.0) for c in all_candidates]))
        summary = (
            f"{len(all_candidates)} visual moments, "
            f"action={action_avg:.2f}, beauty={beauty_avg:.2f}, quality={quality_avg:.2f}"
        )

    total_elapsed = time.perf_counter() - total_started
    qwen_tag_count = sum(int((v.get("timings") or {}).get("qwen_tag_count", 0)) for v in videos)
    qwen_frame_count = sum(int((v.get("timings") or {}).get("qwen_frame_count", 0)) for v in videos)
    qwen_seconds = sum(float((v.get("timings") or {}).get("qwen_seconds", 0.0)) for v in videos)
    qwen_inference_seconds = sum(float((v.get("timings") or {}).get("qwen_inference_seconds", 0.0)) for v in videos)
    qwen_model_id = next(
        (
            str((v.get("timings") or {}).get("qwen_model_id"))
            for v in videos
            if (v.get("timings") or {}).get("qwen_model_id")
        ),
        "",
    )
    qwen_concurrency = next(
        (
            int((v.get("timings") or {}).get("qwen_concurrency"))
            for v in videos
            if (v.get("timings") or {}).get("qwen_concurrency")
        ),
        0,
    )
    qwen_peak_vram_gb = max(
        [float((v.get("timings") or {}).get("qwen_peak_vram_gb") or 0.0) for v in videos] or [0.0]
    )
    print(
        f"   Visual library ready: {summary} "
        f"[total {_fmt_seconds(total_elapsed)}, cache hits {cache_hits}/{len(existing)}]"
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "videos": videos,
        "candidates": all_candidates,
        "ai_enabled": ai_available,
        "qwen_model_path": qwen_model_path if ai_available else None,
        "summary": summary,
        "analysis_seconds": total_elapsed,
        "cache_hits": cache_hits,
        "source_count": len(existing),
        "worker_count": workers,
        "qwen_tag_count": qwen_tag_count,
        "qwen_frame_count": qwen_frame_count,
        "qwen_seconds": qwen_seconds,
        "qwen_inference_seconds": qwen_inference_seconds,
        "qwen_model_id": qwen_model_id,
        "qwen_concurrency": qwen_concurrency,
        "qwen_peak_vram_gb": qwen_peak_vram_gb,
    }


def _parse_fps_fraction(fps_str) -> float | None:
    """Evaluate an ffprobe frame-rate string ('30000/1001', '30/1', '25') the
    same way get_video_fps does; None when unparseable so the caller can fall
    back to the standalone getter."""
    if not fps_str:
        return None
    fps_str = str(fps_str).strip()
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return None


def _probe_media_fields(video_file: str) -> tuple[float, float, int, int]:
    """One ffprobe call for (duration, fps, width, height).

    Collapses the three separate ffprobe spawns (get_video_duration,
    get_video_fps, get_video_resolution) into a single combined invocation.
    Every field falls back to its standalone getter when the combined probe
    fails or a field is missing/unparseable, so the returned values are exactly
    what the three getters would have produced — including the image-source
    synthetic values: get_video_duration / get_video_fps short-circuit on stills
    (SYNTHETIC_IMAGE_DURATION and 30.0), so for an image we deliberately skip the
    combined probe's real ~1-frame duration and image2 demuxer default fps and
    take those from the getters, while width/height still come from the probe
    (get_video_resolution probes stills anyway).

    Uses the same ffprobe binary resolution as the rest of the codebase
    (FFPROBE_PATH from ffmpeg_processing → bundled bin/ffprobe, else PATH).
    """
    is_image = is_image_source(video_file)
    duration = None
    fps = None
    width = None
    height = None
    try:
        probe_cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            video_file,
        ]
        result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams") or []
            stream = streams[0] if streams else {}
            fmt = data.get("format") or {}
            try:
                width = int(stream["width"])
                height = int(stream["height"])
            except (KeyError, TypeError, ValueError):
                width = height = None
            if not is_image:
                # Match get_video_fps: evaluate r_frame_rate exactly.
                fps = _parse_fps_fraction(stream.get("r_frame_rate"))
                try:
                    duration = float(fmt["duration"])
                except (KeyError, TypeError, ValueError):
                    duration = None
    except Exception:
        # Any probe/parse failure → each field falls back to its getter below.
        pass

    if duration is None:
        duration = get_video_duration(video_file)
    if fps is None:
        fps = get_video_fps(video_file)
    if width is None or height is None:
        width, height = get_video_resolution(video_file)
    return float(duration), float(fps), int(width), int(height)


def _empty_video_result(video_file: str) -> Dict:
    """Minimal valid analysis result for a source that failed even the serial
    retry. Mirrors the exact key shape _analyze_single_video returns (empty
    candidates, safe metadata) so the caller can still cache every sibling and
    finish the run instead of one bad file crashing the whole batch.

    ai_deferred=False keeps the deferred-Qwen pass from touching it;
    ai_enabled=False keeps a later _load_cache(require_ai=True) from trusting the
    cached empty result as an AI hit, so the file is re-analyzed on the next run.
    """
    return {
        "analysis_version": ANALYSIS_VERSION,
        "video_file": os.path.abspath(video_file),
        "source_name": _safe_name(video_file),
        "duration": 0.0,
        "fps": 24.0,
        "width": 0,
        "height": 0,
        "scene_changes": [],
        "candidate_count": 0,
        "candidates": [],
        "analysis_seconds": 0.0,
        "timings": {},
        "ai_enabled": False,
        "ai_deferred": False,
    }


def _analyze_single_video(
    video_file: str,
    use_gpu: bool,
    enable_ai: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    defer_ai: bool = False,
    index: int | None = None,
    total: int | None = None,
) -> Dict:
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    name = _safe_name(video_file)
    prefix = f"{index}/{total} " if index and total else ""
    print(f"   Analyzing video {prefix}{name}")

    step_started = time.perf_counter()
    probe_duration, probe_fps, probe_width, probe_height = _probe_media_fields(video_file)
    duration = max(0.0, float(probe_duration))
    fps = max(1.0, float(probe_fps))
    width, height = probe_width, probe_height
    timings["metadata_seconds"] = time.perf_counter() - step_started
    print(
        f"      Metadata: {duration:.1f}s, {fps:.3g} fps, "
        f"{int(width)}x{int(height)} [{_fmt_seconds(timings['metadata_seconds'])}]"
    )

    candidates: List[Dict] = []
    if is_image_source(video_file):
        # Stills: no scenes or motion to analyze — one candidate scored from
        # the single frame. Its start/end are equal so the Qwen worker's
        # mid-frame seek lands on frame 0 (the only frame); the planner-facing
        # duration lets the still fill any segment, and the renderer gives
        # each segment deterministic Ken Burns motion.
        scene_changes: List[float] = []
        timings["scene_detection_seconds"] = 0.0
        step_started = time.perf_counter()
        frame = cv2.imread(video_file)
        if frame is None:
            print(f"      Warning: OpenCV could not read image {name}; candidate analysis skipped.")
        else:
            frame = _resize_for_analysis(frame, max_width=360)
            window = {"start": 0.0, "end": 0.0, "kind": "image"}
            metrics = _measure_frame_samples([frame], np.asarray([0.0]), 0.0, 0.01, False)
            if metrics:
                candidate = _build_candidate(video_file, name, duration, 0, window, metrics)
                candidate["duration"] = 5.0
                candidates.append(candidate)
        timings["candidate_scoring_seconds"] = time.perf_counter() - step_started
        print(
            f"      Still image: {len(candidates)} candidate "
            f"[{_fmt_seconds(timings['candidate_scoring_seconds'])}]"
        )
    else:
        scene_use_gpu = _use_gpu_scene_detection(use_gpu)
        if use_gpu and not scene_use_gpu:
            print("      Scene detection mode: CPU FFmpeg scene filter (CUDA path is optional; default off)")

        step_started = time.perf_counter()
        scene_changes = detect_video_scene_changes(
            video_file,
            threshold=0.27,
            use_gpu=scene_use_gpu,
            analysis_fps=6.0,
            analysis_width=384,
        )
        timings["scene_detection_seconds"] = time.perf_counter() - step_started
        print(
            f"      ⏱ Scene detection total: {_fmt_seconds(timings['scene_detection_seconds'])} "
            f"({len(scene_changes)} scene cuts)"
        )

        step_started = time.perf_counter()
        boundaries = _build_boundaries(scene_changes, duration)
        windows = _make_candidate_windows(boundaries, duration)
        timings["window_build_seconds"] = time.perf_counter() - step_started
        print(
            f"      Candidate windows: {len(windows)} from {len(boundaries)} boundaries "
            f"[{_fmt_seconds(timings['window_build_seconds'])}]"
        )

        cap = _open_video_capture(video_file)
        if cap.isOpened():
            gpu_candidate_metrics = _use_gpu_candidate_metrics(use_gpu)
            if gpu_candidate_metrics:
                print("      Candidate scoring: GPU CuPy metrics + CPU frame decode")
            else:
                print("      Candidate scoring: CPU metrics")
            step_started = time.perf_counter()
            window_metrics = _measure_windows(cap, fps, windows, use_gpu=gpu_candidate_metrics)
            timings["candidate_scoring_seconds"] = time.perf_counter() - step_started
            valid_metrics = 0
            for i, (window, metrics) in enumerate(zip(windows, window_metrics)):
                if not metrics:
                    continue
                valid_metrics += 1
                candidate = _build_candidate(video_file, name, duration, i, window, metrics)
                candidates.append(candidate)
            cap.release()
            print(
                f"      ⏱ Candidate scoring total: {_fmt_seconds(timings['candidate_scoring_seconds'])} "
                f"({valid_metrics}/{len(windows)} windows usable)"
            )
        else:
            print(f"      Warning: OpenCV could not open {name}; candidate analysis skipped.")

    # Media-intelligence fields (always computed — they are labels/metadata the
    # planner only ACTS on when media_aware is enabled, so adding them never
    # changes a legacy plan). media_type is derived purely from the extension,
    # matching ffmpeg_processing.is_image_source's still/video split with GIFs
    # called out separately.
    if candidates:
        ext = os.path.splitext(video_file)[1].lower()
        media_type = ("still" if is_image_source(video_file)
                      else "gif" if ext == ".gif" else "video")
        for candidate in candidates:
            candidate["media_type"] = media_type
            candidate["source_width"] = int(width)
            candidate["source_height"] = int(height)
            candidate["source_fps"] = round(float(fps), 4)
        if media_type != "still":
            _annotate_lowfps_media(video_file, fps, media_type == "gif", candidates)

    qwen_seconds = 0.0
    if enable_ai and candidates and not defer_ai:
        try:
            step_started = time.perf_counter()
            qwen_info = _annotate_candidates_with_qwen(
                video_file=video_file,
                fps=fps,
                candidates=candidates,
                qwen_model_path=qwen_model_path,
                use_gpu=use_gpu,
                audio_profile=audio_profile,
            )
            qwen_seconds = time.perf_counter() - step_started
            timings.update(qwen_info or {})
            timings["qwen_seconds"] = qwen_seconds
            print(f"      ⏱ Qwen semantic analysis total: {_fmt_seconds(qwen_seconds)}")
        except Exception as e:
            print(f"      Warning: Qwen semantic analysis failed for {name}: {e}")
    elif enable_ai and candidates and defer_ai:
        timings["qwen_seconds"] = 0.0
        print("      Qwen semantic analysis: deferred until parallel CPU pass completes")

    sort_started = time.perf_counter()
    if not defer_ai:
        candidates = sorted(candidates, key=lambda c: c.get("editorial_score", 0.0), reverse=True)
    timings["sort_seconds"] = time.perf_counter() - sort_started

    elapsed = time.perf_counter() - started
    timings["total_seconds"] = elapsed
    print(
        f"   Found {len(candidates)} usable visual moments in {_fmt_seconds(elapsed)} "
        f"(scene {_fmt_seconds(timings.get('scene_detection_seconds', 0))}, "
        f"scoring {_fmt_seconds(timings.get('candidate_scoring_seconds', 0))}, "
        f"qwen {_fmt_seconds(timings.get('qwen_seconds', 0))})"
    )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "video_file": os.path.abspath(video_file),
        "source_name": name,
        "duration": duration,
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "scene_changes": scene_changes,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "analysis_seconds": elapsed,
        "timings": timings,
        "ai_enabled": bool(enable_ai and not defer_ai),
        "ai_deferred": bool(enable_ai and defer_ai),
    }


def _complete_deferred_qwen(
    video_data: Dict,
    use_gpu: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    label: str = "",
) -> None:
    candidates = video_data.get("candidates") or []
    if not candidates:
        video_data["ai_deferred"] = False
        video_data["ai_enabled"] = False
        return

    name = _safe_name(video_data.get("video_file", video_data.get("source_name", "video")))
    print(f"   Running deferred Qwen semantic analysis {label}: {name}")
    started = time.perf_counter()
    qwen_info: Dict = {}
    # The single-video worker call and merge; the ai_enabled rule (only a real
    # tag merge marks the video ai_enabled) is enforced by _apply_qwen_result,
    # the same site the batch path uses. On any failure the flags stay off so a
    # transient llama-server error is never cached as an AI hit.
    applied = False
    try:
        max_windows = _qwen_max_windows()
        if max_windows == 0:
            print("   Qwen semantic analysis skipped (BEATSYNC_QWEN_MAX_WINDOWS=0).")
        else:
            ai_candidates = _select_ai_candidates(candidates, max_windows)
            if not ai_candidates:
                qwen_info = {"qwen_frame_count": 0, "qwen_tag_count": 0}
            else:
                print(f"   Qwen semantic analysis: {len(ai_candidates)} candidate moments")
                response = _run_qwen_worker(
                    _qwen_request(
                        qwen_model_path, use_gpu, audio_profile,
                        video_file=video_data["video_file"],
                        fps=float(video_data.get("fps") or 24.0),
                        candidates=ai_candidates,
                    ),
                    batch=False,
                )
                semantics = response.get("semantics") if isinstance(response, dict) else {}
                merged_count = _apply_qwen_result(video_data, ai_candidates, semantics)
                applied = True
                if not semantics:
                    print("      Qwen returned no semantic tags; deterministic visual tags remain active.")
                    qwen_info = {
                        "qwen_frame_count": len(ai_candidates),
                        "qwen_tag_count": 0,
                        "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
                        "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
                        "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
                    }
                else:
                    print(f"      Qwen semantic tags merged: {merged_count}/{len(ai_candidates)}")
                    timing = response.get("timings_by_job", {}).get("single", {}) if isinstance(response, dict) else {}
                    qwen_info = {
                        "qwen_frame_count": int(timing.get("frame_count") or len(ai_candidates)),
                        "qwen_tag_count": int(timing.get("tag_count") or merged_count),
                        "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
                        "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
                        "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
                    }
    except Exception as e:
        print(f"      Warning: Qwen semantic analysis failed for {name}: {e}")

    if not applied:
        # Skip / no candidates / exception: reproduce the original finalization
        # (sort + flags off). The success paths already sorted and set the flags
        # inside _apply_qwen_result.
        candidates.sort(key=lambda c: c.get("editorial_score", 0.0), reverse=True)
        video_data["ai_deferred"] = False
        video_data["ai_enabled"] = False
        video_data["candidate_count"] = len(candidates)

    qwen_seconds = time.perf_counter() - started
    timings = video_data.setdefault("timings", {})
    timings.update(qwen_info or {})
    timings["qwen_seconds"] = qwen_seconds
    timings["total_seconds"] = float(timings.get("total_seconds", video_data.get("analysis_seconds", 0.0))) + qwen_seconds
    video_data["analysis_seconds"] = timings["total_seconds"]
    if not video_data["ai_enabled"]:
        print(
            "      Deferred Qwen produced no semantic tags; leaving this video "
            "ai_enabled=False so it retries on the next run."
        )
    print(
        f"      ⏱ Deferred Qwen total: {_fmt_seconds(qwen_seconds)}; "
        f"video total now {_fmt_seconds(video_data['analysis_seconds'])}"
    )




def _qwen_max_windows() -> int:
    try:
        value = int(os.environ.get("BEATSYNC_QWEN_MAX_WINDOWS", "120"))
    except ValueError:
        value = 120
    return max(0, value)


def _complete_deferred_qwen_batch(
    video_items: Sequence[tuple[Dict, Dict]],
    use_gpu: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    total_video_count: int,
) -> None:
    max_windows = _qwen_max_windows()
    if max_windows == 0:
        for _, video_data in video_items:
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
        print("   Qwen semantic analysis skipped (BEATSYNC_QWEN_MAX_WINDOWS=0).")
        return

    request_jobs: List[Dict] = []
    job_to_video: Dict[str, Dict] = {}
    selected_by_job: Dict[str, List[Dict]] = {}
    for job, video_data in video_items:
        candidates = video_data.get("candidates") or []
        ai_candidates = _select_ai_candidates(candidates, max_windows)
        idx = int(job.get("index") or 0)
        job_id = str(idx or len(request_jobs) + 1)
        name = _safe_name(video_data.get("video_file", video_data.get("source_name", "video")))
        print(
            f"   Qwen semantic analysis batch input {idx}/{total_video_count}: "
            f"{name} - {len(ai_candidates)} candidate moments"
        )
        if not ai_candidates:
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
            continue
        selected_by_job[job_id] = ai_candidates
        job_to_video[job_id] = video_data
        request_jobs.append({
            "job_id": job_id,
            "video_file": video_data["video_file"],
            "fps": float(video_data.get("fps") or 24.0),
            "candidates": [
                {
                    "id": c.get("id"),
                    "start": c.get("start"),
                    "end": c.get("end"),
                }
                for c in ai_candidates
            ],
        })

    if not request_jobs:
        return

    print(f"   Running one shared Qwen worker for {len(request_jobs)} video(s) (model loads once)")
    batch_started = time.perf_counter()
    response = _run_qwen_worker(
        _qwen_request(qwen_model_path, use_gpu, audio_profile, jobs=request_jobs),
        batch=True,
    )
    batch_seconds = time.perf_counter() - batch_started
    semantics_by_job = response.get("semantics_by_job") or {}
    timings_by_job = response.get("timings_by_job") or {}
    if not semantics_by_job:
        for video_data in job_to_video.values():
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
        print("      Qwen llama.cpp returned no semantic response; AI analysis will retry on the next run.")
        print(f"      ⏱ Shared Qwen batch total: {_fmt_seconds(batch_seconds)}")
        return
    model_load_seconds = float(response.get("model_load_seconds") or 0.0)
    amortized_model = model_load_seconds / max(1, len(request_jobs))
    qwen_model_id = str(response.get("model_id") or "")
    qwen_concurrency = int(response.get("batch_size") or 0)
    qwen_peak_vram_gb = float(response.get("peak_vram_gb") or 0.0)

    for job_id, video_data in job_to_video.items():
        ai_candidates = selected_by_job.get(job_id, [])
        semantics = semantics_by_job.get(str(job_id), {})
        # Merge + sort + the ai_enabled rule all live in _apply_qwen_result (the
        # single enforcement site shared with the single-video deferred path).
        merged_count = _apply_qwen_result(video_data, ai_candidates, semantics)
        timing = timings_by_job.get(str(job_id), {}) if isinstance(timings_by_job, dict) else {}
        qwen_seconds = (
            float(timing.get("prefetch_seconds") or 0.0)
            + float(timing.get("inference_seconds") or 0.0)
            + amortized_model
        )
        timings = video_data.setdefault("timings", {})
        timings["qwen_seconds"] = qwen_seconds
        timings["qwen_model_load_seconds_amortized"] = amortized_model
        timings["qwen_prefetch_seconds"] = float(timing.get("prefetch_seconds") or 0.0)
        timings["qwen_inference_seconds"] = float(timing.get("inference_seconds") or 0.0)
        timings["qwen_frame_count"] = int(timing.get("frame_count") or len(ai_candidates))
        timings["qwen_tag_count"] = int(timing.get("tag_count") or merged_count)
        timings["qwen_model_id"] = qwen_model_id
        timings["qwen_concurrency"] = qwen_concurrency
        timings["qwen_peak_vram_gb"] = qwen_peak_vram_gb
        timings["total_seconds"] = float(timings.get("total_seconds", video_data.get("analysis_seconds", 0.0))) + qwen_seconds
        video_data["analysis_seconds"] = timings["total_seconds"]
        print(
            f"      Qwen semantic tags merged: {merged_count}/{len(ai_candidates)} "
            f"for {_safe_name(video_data.get('video_file', 'video'))}"
        )
        if merged_count == 0:
            print(
                "      No semantic tags merged for this video; leaving it "
                "ai_enabled=False so Qwen retries on the next run."
            )
        print(
            f"      ⏱ Shared-Qwen video cost: {_fmt_seconds(qwen_seconds)} "
            f"(prefetch {_fmt_seconds(timings['qwen_prefetch_seconds'])}, "
            f"inference {_fmt_seconds(timings['qwen_inference_seconds'])}, "
            f"model share {_fmt_seconds(amortized_model)})"
        )

    print(f"      ⏱ Shared Qwen batch total: {_fmt_seconds(batch_seconds)}")


def _qwen_worker_timeout(total_candidates: int) -> int:
    """Parent-level backstop timeout for the stage-5 Qwen worker.

    The worker now fail-fasts on its own (per-request watchdog + circuit
    breaker: a fully wedged llama-server ends the worker in minutes), so this
    is a last-resort backstop, not the primary defense. The old formula
    (max(1800, n*75)) allowed ~23h for 1099 candidates and froze a render for
    a day when llama-server wedged on its first request.

    Cap is 7200s, not lower: measured healthy throughput is ~3.3s/candidate
    (13s median request wall at 4 slots), so a healthy 1099-candidate batch
    plus frame prefetch legitimately needs over an hour; a tighter backstop
    would kill it and discard every tag. Override with
    BEATSYNC_QWEN_BATCH_TIMEOUT (seconds) in either direction.
    """
    override = os.environ.get("BEATSYNC_QWEN_BATCH_TIMEOUT")
    if override:
        try:
            return max(60, int(override))
        except ValueError:
            pass
    return min(7200, 300 + int(total_candidates) * 15)


def _kill_qwen_worker_group(proc: "subprocess.Popen") -> None:
    """POSIX-only: kill the worker's whole process group (SIGTERM, then SIGKILL).

    The worker is started with start_new_session=True, so its pgid == its pid
    and llama-server (its child) is in the same group. SIGTERM first gives the
    worker's signal handler a chance to stop llama-server cleanly.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Always follow with a group SIGKILL: even if the worker itself exited on
    # SIGTERM, a group member (llama-server) may have ignored it. killpg works
    # as long as any member of the group is still alive.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:
        pass


def _run_qwen_worker_process(command: List[str], env: Dict[str, str], timeout: int) -> "subprocess.CompletedProcess":
    """Run the stage-5 worker; a timeout must never orphan llama-server.

    POSIX: the worker runs in its own session/process group so that on
    TimeoutExpired we can os.killpg the whole tree. Plain subprocess.run only
    kills the direct child, which left llama-server orphaned (and the GPU
    busy) in the 2026-07-07 incident. Windows: killpg/setsid do not exist and
    process groups work differently (Job Objects would be needed), so Windows
    keeps the previous subprocess.run behavior unchanged.
    """
    if os.name == "nt":
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_qwen_worker_group(proc)
        raise
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _qwen_request(
    qwen_model_path: str,
    use_gpu: bool,
    audio_profile: Dict,
    *,
    jobs: Sequence[Dict] | None = None,
    video_file: str | None = None,
    fps: float | None = None,
    candidates: Sequence[Dict] | None = None,
) -> Dict:
    """Build the stage-5 worker request. This is the ONLY place the wire format
    branches: a multi-video request carries top-level ``jobs``; a single-video
    request carries top-level ``video_file``/``fps``/``candidates``. The worker
    process accepts both and its protocol is unchanged."""
    request: Dict = {
        "qwen_model_path": qwen_model_path,
        "use_gpu": bool(use_gpu),
        "audio_profile": audio_profile,
    }
    if jobs is not None:
        request["jobs"] = list(jobs)
    else:
        request["video_file"] = video_file
        request["fps"] = fps
        request["candidates"] = [
            {"id": c.get("id"), "start": c.get("start"), "end": c.get("end")}
            for c in (candidates or [])
        ]
    return request


def _run_qwen_worker(request: Dict, batch: bool) -> Dict:
    """Run the stage-5 Qwen worker for one request and return its parsed
    response dict (``{}`` on any failure). Single-video is just a 1-request
    batch; ``batch`` only selects the temp-file naming, backstop-timeout input,
    the stderr tail length and the log label — all preserved verbatim from the
    two former (single / batch) implementations."""
    os.makedirs(VIDEO_ANALYSIS_CACHE_DIR, exist_ok=True)
    if batch:
        jobs = request.get("jobs") or []
        token = _hash_text(f"batch|{time.time()}|{len(jobs)}", 12)
        prefix, label, error_tail = "qwen_batch", "Qwen batch worker", 2400
        total_candidates = sum(len(job.get("candidates") or []) for job in jobs)
    else:
        token = _hash_text(f"{request.get('video_file')}|{time.time()}", 12)
        prefix, label, error_tail = "qwen", "Qwen worker", 1800
        total_candidates = len(request.get("candidates") or [])
    request_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"{prefix}_request_{token}.json")
    response_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"{prefix}_response_{token}.json")
    worker_path = os.path.join(ROOT_DIR, "src", "auto_mode", "stage5_qwen_scene_worker.py")
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f)

    env = _qwen_worker_environment()
    timeout = _qwen_worker_timeout(total_candidates)
    try:
        result = _run_qwen_worker_process(
            [sys.executable, worker_path, "--request", request_path, "--response", response_path],
            env,
            timeout,
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"      {line}")
        if result.returncode != 0:
            error_tail_text = (result.stderr or "").strip()[-error_tail:]
            print(f"      {label} failed: {error_tail_text}")
            return {}
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except subprocess.TimeoutExpired:
        print(f"      {label} timed out; deterministic visual tags remain active.")
        return {}
    except Exception as e:
        print(f"      {label} error: {e}")
        return {}
    finally:
        # Keep Qwen worker request/response files in video_analysis_cache for
        # reproducibility and debugging. The user explicitly wants this cache
        # folder to be preserved.
        pass


def _merge_semantics_by_id(ai_candidates: Sequence[Dict], semantics) -> int:
    """Merge worker-returned semantics into ``ai_candidates`` by id and return
    the number of candidates that actually received a merge. Shared merge loop
    for the serial and deferred paths (a merge is not idempotent, so each
    candidate is merged at most once)."""
    semantic_by_id = {str(k): v for k, v in semantics.items()} if isinstance(semantics, dict) else {}
    merged_count = 0
    for candidate in ai_candidates:
        semantic = semantic_by_id.get(str(candidate.get("id")))
        if semantic:
            _merge_semantic(candidate, semantic)
            merged_count += 1
    return merged_count


def _apply_qwen_result(video_data: Dict, ai_candidates: Sequence[Dict], semantics) -> int:
    """Apply a deferred Qwen result to one video and return the merge count.

    This is the SINGLE place the deferred-path rule lives: only a real tag merge
    (>= 1 candidate actually annotated) may set ``ai_enabled``. A transient
    llama-server failure therefore never gets cached as an AI hit, so
    ``_load_cache(require_ai=True)`` re-runs Qwen for that video next time.
    Both deferred orchestrators (single and batch) funnel through here."""
    merged_count = _merge_semantics_by_id(ai_candidates, semantics)
    candidates = video_data.get("candidates") or []
    candidates.sort(key=lambda c: c.get("editorial_score", 0.0), reverse=True)
    video_data["ai_deferred"] = False
    video_data["ai_enabled"] = merged_count > 0
    video_data["candidate_count"] = len(candidates)
    return merged_count

def _build_boundaries(scene_changes: Iterable[float], duration: float) -> List[float]:
    if duration <= 0:
        return [0.0]
    raw = [0.0] + [float(t) for t in scene_changes if 0.0 < float(t) < duration] + [duration]
    raw = sorted(set(round(t, 3) for t in raw))
    cleaned: List[float] = []
    for t in raw:
        if not cleaned or t - cleaned[-1] >= 0.35 or t in (0.0, duration):
            cleaned.append(t)
    if cleaned[-1] != duration:
        cleaned.append(duration)
    return cleaned


def _make_candidate_windows(boundaries: Sequence[float], duration: float) -> List[Dict]:
    windows: List[Dict] = []
    if duration <= 0:
        return windows

    max_window = 5.2
    min_window = 0.55
    fallback_step = 3.5

    if len(boundaries) < 2:
        starts = np.arange(0.0, max(duration - min_window, 0.0), fallback_step)
        return [
            {"start": float(s), "end": float(min(duration, s + max_window)), "kind": "fallback"}
            for s in starts
        ]

    for scene_idx in range(len(boundaries) - 1):
        start = float(boundaries[scene_idx])
        end = float(boundaries[scene_idx + 1])
        scene_duration = end - start
        if scene_duration < min_window:
            continue

        if scene_duration <= max_window:
            windows.append({"start": start, "end": end, "kind": "scene", "scene_index": scene_idx})
            continue

        chunks = max(1, int(math.ceil(scene_duration / max_window)))
        step = scene_duration / chunks
        for chunk_idx in range(chunks):
            chunk_start = start + chunk_idx * step
            chunk_end = min(end, chunk_start + max_window)
            if chunk_end - chunk_start >= min_window:
                windows.append({
                    "start": float(chunk_start),
                    "end": float(chunk_end),
                    "kind": "scene_chunk",
                    "scene_index": scene_idx,
                })

    return windows


def _measure_windows(cap: cv2.VideoCapture, fps: float, windows: Sequence[Dict],
                     use_gpu: bool = False) -> List[Dict]:
    plans = [_window_sample_plan(fps, float(w["start"]), float(w["end"])) for w in windows]
    targets = sorted({idx for plan in plans for idx in plan["frame_indices"]})
    if not targets:
        return [{} for _ in windows]

    frame_span = max(1, targets[-1] - targets[0] + 1)
    seek_ratio = frame_span / max(1, len(targets))
    sequential_limit = float(os.environ.get("BEATSYNC_SEQUENTIAL_SAMPLE_RATIO", "80"))
    use_sequential = (
        os.environ.get("BEATSYNC_SEQUENTIAL_WINDOW_SAMPLING", "1") != "0"
        and seek_ratio <= sequential_limit
    )

    def build_metric(plan: Dict, frame_map: Dict[int, np.ndarray]) -> Dict:
        frames = []
        sample_times = []
        for frame_idx, sample_time in zip(plan["frame_indices"], plan["sample_times"]):
            frame = frame_map.get(frame_idx)
            if frame is not None:
                frames.append(frame)
                sample_times.append(sample_time)
        return _measure_frame_samples(
            frames,
            np.asarray(sample_times, dtype=float),
            plan["start"],
            plan["duration"],
            use_gpu,
        )

    if use_sequential:
        print(
            f"         Ordered frame sampling: {len(targets)} target frames "
            f"(span/target={seek_ratio:.1f}, limit={sequential_limit:g})"
        )
        read_started = time.perf_counter()
        frame_map = _read_ordered_frames(cap, targets)
        read_elapsed = time.perf_counter() - read_started
        approx_ram_mb = sum(getattr(frame, "nbytes", 0) for frame in frame_map.values()) / (1024 * 1024)
        print(
            f"         Frame decode/cache: {len(frame_map)}/{len(targets)} frames "
            f"in RAM, ~{approx_ram_mb:.0f} MB [{_fmt_seconds(read_elapsed)}]"
        )

        metric_workers = _candidate_metric_workers(len(plans), use_gpu)
        metrics_started = time.perf_counter()
        if metric_workers > 1:
            print(f"         Metric CPU workers: {metric_workers}")
            with ThreadPoolExecutor(max_workers=metric_workers) as executor:
                metrics = list(executor.map(lambda plan: build_metric(plan, frame_map), plans))
        else:
            metrics = [build_metric(plan, frame_map) for plan in plans]
        metric_elapsed = time.perf_counter() - metrics_started
        usable = sum(1 for metric in metrics if metric)
        print(
            f"         Metric math: {usable}/{len(plans)} windows "
            f"[{_fmt_seconds(metric_elapsed)}, {usable / max(metric_elapsed, 1e-6):.1f} windows/s]"
        )
        return metrics

    print(
        f"         Random-seek sampling fallback: {len(targets)} target frames "
        f"(span/target={seek_ratio:.1f} > limit {sequential_limit:g})"
    )
    fallback_started = time.perf_counter()
    metrics = [
        _measure_window(cap, fps, float(w["start"]), float(w["end"]), use_gpu=use_gpu)
        for w in windows
    ]
    fallback_elapsed = time.perf_counter() - fallback_started
    usable = sum(1 for metric in metrics if metric)
    print(
        f"         Random-seek scoring: {usable}/{len(windows)} windows "
        f"[{_fmt_seconds(fallback_elapsed)}]"
    )
    return metrics


def _window_sample_plan(fps: float, start: float, end: float) -> Dict:
    duration = max(0.0, end - start)
    if duration <= 0.05:
        return {"start": start, "duration": duration, "sample_times": [], "frame_indices": []}

    sample_count = int(np.clip(math.ceil(duration * 1.4), 3, 8))
    sample_times = np.linspace(start + duration * 0.12, end - duration * 0.12, sample_count)
    frame_indices = [max(0, int(round(float(t) * fps))) for t in sample_times]
    return {
        "start": start,
        "duration": duration,
        "sample_times": sample_times,
        "frame_indices": frame_indices,
    }


def _read_ordered_frames(cap: cv2.VideoCapture, frame_indices: Sequence[int]) -> Dict[int, np.ndarray]:
    frames: Dict[int, np.ndarray] = {}
    if not frame_indices:
        return frames

    first = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    current = first

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)
        if frame_idx < current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current = frame_idx
        while current < frame_idx:
            if not cap.grab():
                break
            current += 1
        if current != frame_idx:
            continue
        ok, frame = cap.read()
        current += 1
        if not ok or frame is None:
            continue
        frames[frame_idx] = _resize_for_analysis(frame, max_width=360)
    return frames


def _measure_window(cap: cv2.VideoCapture, fps: float, start: float, end: float,
                    use_gpu: bool = False) -> Dict:
    plan = _window_sample_plan(fps, start, end)
    if plan["duration"] <= 0.05:
        return {}

    frames: List[np.ndarray] = []
    used_times: List[float] = []
    for frame_idx, sample_time in zip(plan["frame_indices"], plan["sample_times"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _resize_for_analysis(frame, max_width=360)
        frames.append(frame)
        used_times.append(float(sample_time))

    return _measure_frame_samples(frames, np.asarray(used_times, dtype=float), start, plan["duration"], use_gpu)


def _score_from_primitives(brightness: float, contrast: float, saturation: float,
                           sharpness: float, motion: float, colorfulness: float,
                           peak_offset: float, duration: float) -> Dict:
    """Shared metric scorer: turn per-window primitives into the candidate
    metric dict. The weights, penalties and returned dict are identical to the
    inline copy that previously lived in both _measure_frame_samples (CPU) and
    _measure_frames_gpu (GPU); each caller still computes the primitives its own
    way (the CPU/GPU saturation etc. difference is pre-existing and untouched).
    subject_anchor is NOT added here — the CPU caller attaches it afterward."""
    darkness_penalty = _clamp((0.25 - brightness) / 0.25)
    blown_penalty = _clamp((brightness - 0.86) / 0.14)
    quality = _clamp(
        0.36 * sharpness
        + 0.22 * _clamp(contrast)
        + 0.18 * _clamp(saturation)
        + 0.14 * (1.0 - darkness_penalty)
        + 0.10 * (1.0 - blown_penalty)
    )
    return {
        "duration": duration,
        "brightness": _clamp(brightness),
        "contrast": _clamp(contrast),
        "saturation": _clamp(saturation),
        "sharpness": _clamp(sharpness),
        "motion": _clamp(motion),
        "colorfulness": _clamp(colorfulness),
        "quality_score": quality,
        "peak_offset": _clamp(peak_offset, 0.0, duration, default=duration * 0.5),
    }


def _mean_sample_dt(sample_times: np.ndarray) -> float:
    """Mean spacing in seconds between sampled frames; 0.0 when fewer than 2."""
    if sample_times is None or len(sample_times) < 2:
        return 0.0
    return float(np.mean(np.diff(np.asarray(sample_times, dtype=float))))


def _measure_flow(gray_frames: Sequence[np.ndarray], sample_dt: float) -> Dict[str, float]:
    """Dense Farneback optical flow over the already-decoded gray samples.

    Measurement only — nothing consumes these keys here; the stage-6 planner
    reads them separately. Returns kinetic / subject_motion / camera_motion,
    each 0..1: per-pair means are normalized by frame height and sample_dt
    (screen-heights/sec), then averaged and scaled. Values are rounded to 4dp
    as the determinism firewall for the JSON sidecar cache. Single frame
    (still image) or unusable spacing -> all zeros (genuinely motionless).
    A measurement FAILURE returns {} — keys omitted, NOT zeros — so the
    planner's `candidate.get("kinetic", motion)` fallback engages instead of
    scoring an errored clip as perfectly calm footage."""
    zeros = {"kinetic": 0.0, "subject_motion": 0.0, "camera_motion": 0.0}
    if len(gray_frames) < 2 or not (sample_dt > 1e-6):
        return zeros
    try:
        kin_vals: List[float] = []
        subj_vals: List[float] = []
        cam_vals: List[float] = []
        for i in range(1, len(gray_frames)):
            prev = gray_frames[i - 1]
            curr = gray_frames[i]
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )  # (H, W, 2) pixels of displacement over one sample interval
            norm = float(max(1, prev.shape[0])) * sample_dt  # px -> screen-heights/sec
            mag = np.linalg.norm(flow, axis=2)
            camera = np.median(flow.reshape(-1, 2), axis=0)  # global (camera) motion
            residual = np.linalg.norm(flow - camera, axis=2)  # subject motion
            kin_vals.append(float(np.mean(mag)) / norm)
            subj_vals.append(float(np.mean(residual)) / norm)
            cam_vals.append(float(np.linalg.norm(camera)) / norm)
        return {
            "kinetic": round(min(1.0, float(np.mean(kin_vals)) / 0.35), 4),
            "subject_motion": round(min(1.0, float(np.mean(subj_vals)) / 0.25), 4),
            "camera_motion": round(min(1.0, float(np.mean(cam_vals)) / 0.35), 4),
        }
    except Exception as exc:
        print(f"   ⚠️  Optical-flow motion metrics failed ({exc}); keys omitted "
              f"(scoring falls back to plain frame-diff motion).")
        return {}


# Sources below this fps get an additional native-frame-pair flow measurement:
# the standard window sampling spaces samples ~0.3-0.7s apart, and Farneback
# magnitudes over gaps that wide (several source frames of displacement) are
# garbage. GIFs typically run 10-15fps.
_NATIVE_FLOW_MAX_FPS = 15.0
# Full-decode cap for the extras pass (low-fps sources only, frames already
# resized to <=360px — at 15fps this covers 80s of source).
_LOWFPS_DECODE_MAX_FRAMES = 1200


def _annotate_lowfps_media(video_file: str, fps: float, is_gif: bool,
                           candidates: List[Dict]) -> None:
    """GIF loop-seam + native-pair optical flow extras for low-fps sources.

    Adds ONLY new candidate fields — ``loop_seam`` (grayscale absdiff mean of
    last vs. first frame, 0..1, 0 = seamless loop; GIFs only) and
    ``kinetic_native``/``subject_motion_native``/``camera_motion_native``
    (``_measure_flow`` on consecutive native frames with dt = 1/source_fps;
    sub-15fps sources only). The existing kinetic/motion keys are never
    touched, so plans that ignore these fields stay byte-identical. Values are
    4dp-rounded like the rest of the sidecar (determinism firewall). Any
    failure leaves the fields absent after one log line — the media-aware
    planner treats missing data as "no adjustment".
    """
    measure_native = float(fps) < _NATIVE_FLOW_MAX_FPS
    if not candidates or not (is_gif or measure_native):
        return
    try:
        cap = _open_video_capture(video_file)
        if not cap.isOpened():
            return
        gray_frames: List[np.ndarray] = []
        try:
            while len(gray_frames) < _LOWFPS_DECODE_MAX_FRAMES:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                gray_frames.append(cv2.cvtColor(
                    _resize_for_analysis(frame, max_width=360),
                    cv2.COLOR_BGR2GRAY))
        finally:
            cap.release()
        if len(gray_frames) < 2:
            return

        note_parts = [f"{len(gray_frames)} native frames"]
        if is_gif:
            seam = round(float(np.mean(cv2.absdiff(gray_frames[-1],
                                                   gray_frames[0]))) / 255.0, 4)
            for candidate in candidates:
                candidate["loop_seam"] = seam
            note_parts.append(f"loop seam {seam:.3f}")

        if measure_native:
            dt = 1.0 / max(1e-6, float(fps))
            whole_flow: Dict[str, float] | None = None
            annotated = 0
            for candidate in candidates:
                i0 = max(0, int(round(float(candidate.get("start", 0.0)) * fps)))
                i1 = min(len(gray_frames),
                         int(round(float(candidate.get("end", 0.0)) * fps)) + 1)
                window = gray_frames[i0:i1]
                if len(window) < 2:
                    # Degenerate window (still-like or probe drift): fall back
                    # to one whole-source measurement, computed at most once.
                    if whole_flow is None:
                        whole_flow = _measure_flow(gray_frames, dt)
                    flow = whole_flow
                else:
                    flow = _measure_flow(window, dt)
                for key in ("kinetic", "subject_motion", "camera_motion"):
                    if key in flow:
                        candidate[f"{key}_native"] = float(flow[key])
                        annotated += 1
            note_parts.append(f"native flow on {annotated // 3} candidates "
                              f"(dt={dt:.3f}s)")
        print(f"      Low-fps media extras: {', '.join(note_parts)}")
    except Exception as exc:
        print(f"      ⚠️  Low-fps media extras skipped for "
              f"{_safe_name(video_file)}: {exc}")


def _measure_frame_samples(frames: Sequence[np.ndarray], sample_times: np.ndarray,
                           start: float, duration: float, use_gpu: bool = False) -> Dict:
    if not frames:
        return {}

    if use_gpu and GPU_AVAILABLE and cp is not None:
        try:
            metrics = _measure_frames_gpu(frames, sample_times, start, duration)
            # Anchor stays on CPU (numpy/cv2) even on the GPU metrics path so
            # both paths emit an identical schema.
            metrics["subject_anchor"] = _compute_subject_anchor(frames, sample_times, start)
            # Flow too: computed on CPU from the same frames for schema parity.
            metrics.update(_measure_flow(
                [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames],
                _mean_sample_dt(sample_times),
            ))
            return metrics
        except Exception as exc:
            # Graceful degradation: one log line, then fall through to the CPU
            # scorer below. (Dead on Mac — no CuPy — so it never fires here.)
            print(f"      GPU candidate metrics failed ({exc}); using CPU scorer.")

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    hsv_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]

    brightness = float(np.mean([np.mean(g) / 255.0 for g in gray_frames]))
    contrast = float(np.mean([np.std(g) / 80.0 for g in gray_frames]))
    saturation = float(np.mean([np.mean(h[:, :, 1]) / 255.0 for h in hsv_frames]))
    blur_values = [cv2.Laplacian(g, cv2.CV_64F).var() for g in gray_frames]
    sharpness = _clamp(np.mean(blur_values) / 520.0)

    motion_values = []
    diff_maps: List[np.ndarray] = []
    peak_offset = duration * 0.5
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        diff_maps.append(diff)
        motion = float(np.mean(diff) / 42.0)
        motion_values.append(motion)
    if motion_values:
        motion = _clamp(np.mean(motion_values))
        peak_i = int(np.argmax(motion_values)) + 1
        peak_offset = float(sample_times[min(peak_i, len(sample_times) - 1)] - start)
    else:
        motion = 0.0

    colorfulness = _colorfulness(frames)
    metrics = _score_from_primitives(
        brightness, contrast, saturation, sharpness, motion, colorfulness,
        peak_offset, duration,
    )
    metrics["subject_anchor"] = _compute_subject_anchor(
        frames, sample_times, start, gray_frames=gray_frames, diff_maps=diff_maps
    )
    metrics.update(_measure_flow(gray_frames, _mean_sample_dt(sample_times)))
    return metrics


def _use_gpu_candidate_metrics(use_gpu: bool) -> bool:
    if not (use_gpu and GPU_AVAILABLE and cp is not None):
        return False
    return os.environ.get("BEATSYNC_GPU_CANDIDATE_METRICS", "0") == "1"


def _use_gpu_scene_detection(use_gpu: bool) -> bool:
    if not use_gpu:
        return False
    return os.environ.get("BEATSYNC_GPU_SCENE_DETECTION", "0") == "1"


def _measure_frames_gpu(frames: Sequence[np.ndarray], sample_times: np.ndarray,
                        start: float, duration: float) -> Dict:
    frame_stack = cp.asarray(np.stack(frames, axis=0), dtype=cp.float32)
    b = frame_stack[:, :, :, 0]
    g = frame_stack[:, :, :, 1]
    r = frame_stack[:, :, :, 2]

    gray = 0.114 * b + 0.587 * g + 0.299 * r
    brightness = float(cp.asnumpy(cp.mean(gray) / 255.0))
    contrast = float(cp.asnumpy(cp.mean(cp.std(gray, axis=(1, 2))) / 80.0))

    max_rgb = cp.max(frame_stack, axis=3)
    min_rgb = cp.min(frame_stack, axis=3)
    saturation_map = cp.where(max_rgb > 1e-6, (max_rgb - min_rgb) / max_rgb, 0.0)
    saturation = float(cp.asnumpy(cp.mean(saturation_map)))

    lap = _gpu_laplacian(gray)
    sharpness = _clamp(float(cp.asnumpy(cp.mean(cp.var(lap, axis=(1, 2)))) / 520.0))

    if gray.shape[0] > 1:
        diff = cp.abs(gray[1:] - gray[:-1])
        motion_values = cp.mean(diff, axis=(1, 2)) / 42.0
        motion = _clamp(float(cp.asnumpy(cp.mean(motion_values))))
        peak_i = int(cp.asnumpy(cp.argmax(motion_values))) + 1
        peak_offset = float(sample_times[min(peak_i, len(sample_times) - 1)] - start)
    else:
        motion = 0.0
        peak_offset = duration * 0.5

    rg = cp.abs(r - g)
    yb = cp.abs(0.5 * (r + g) - b)
    colorfulness = _clamp(float(cp.asnumpy(cp.mean(cp.std(rg, axis=(1, 2)) + cp.std(yb, axis=(1, 2)))) / 95.0))

    return _score_from_primitives(
        brightness, contrast, saturation, sharpness, motion, colorfulness,
        peak_offset, duration,
    )


def _gpu_laplacian(gray_stack):
    padded = cp.pad(gray_stack, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    center = padded[:, 1:-1, 1:-1]
    return (
        padded[:, :-2, 1:-1]
        + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
        - 4.0 * center
    )


def _resize_for_analysis(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Subject anchor: per-candidate estimate of where the subject sits in frame,
# so crops can follow the subject instead of always centering.
#
# Coordinate space: cx/cy are normalized 0..1 in the *decoded* frame
# (cv2 storage pixels, pre-fit, SAR-unaware — cv2 does not apply the sample
# aspect ratio, but normalized coordinates survive any later horizontal
# scaling, so consumers can apply them after SAR correction safely).
#
# Priority: YuNet face detection (if the model is available) > motion
# centroid (absdiff intensity) > detail centroid (gradient energy) for
# static shots. All paths are deterministic and CPU/numpy-only.
# ---------------------------------------------------------------------------

YUNET_MODEL_ENV = "BEATSYNC_YUNET_MODEL"
DEFAULT_YUNET_MODEL = os.path.join(ROOT_DIR, "models", "face_detection_yunet_2023mar.onnx")
_YUNET_SCORE_THRESHOLD = 0.65

# YuNet has no disable env of its own — the kill switch is BEATSYNC_YUNET_MODEL
# pointing at a missing/blank path (resolve returns "" → disabled). Its detector
# is not thread-safe, so concurrency="thread_local" keeps the original one-per-
# worker-thread discipline plus the process-wide latch.
_YUNET_BACKEND = OptionalBackend(
    "YuNet face detector",
    default_model=DEFAULT_YUNET_MODEL,
    model_env=YUNET_MODEL_ENV,
    concurrency="thread_local",
)

_NEUTRAL_ANCHOR = {"cx": 0.5, "cy": 0.5, "confidence": 0.0, "source": "detail", "path": []}


def _resolve_yunet_model_path() -> str:
    # Explicit override is authoritative: if it is missing/broken the face path
    # is disabled (also serves as a kill switch) rather than silently falling
    # back to the bundled model.
    return _YUNET_BACKEND.resolve_model_path()


def _get_yunet_detector():
    def build():
        if not hasattr(cv2, "FaceDetectorYN"):  # needs opencv>=4.5.4
            return None
        model_path = _YUNET_BACKEND.resolve_model_path()
        if not model_path:
            return None
        try:
            return cv2.FaceDetectorYN.create(model_path, "", (320, 320), _YUNET_SCORE_THRESHOLD)
        except Exception:
            return None

    return _YUNET_BACKEND.get_thread_local(build)


def _detect_face_center(frame: np.ndarray):
    """Weighted face center for one BGR frame: (cx, cy, weight) or None."""
    detector = _get_yunet_detector()
    if detector is None:
        return None
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return None
    try:
        detector.setInputSize((int(w), int(h)))
        _, faces = detector.detect(frame)
    except Exception:
        return None
    if faces is None or len(faces) == 0:
        return None
    frame_area = float(w * h)
    sum_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    best_score = 0.0
    for face in faces:
        # YuNet row: x, y, w, h, 10 landmark coords, score (index 14).
        score = float(face[14]) if len(face) > 14 else 1.0
        if score < _YUNET_SCORE_THRESHOLD:
            continue
        bw = max(float(face[2]), 1.0)
        bh = max(float(face[3]), 1.0)
        cx = (float(face[0]) + bw * 0.5) / w
        cy = (float(face[1]) + bh * 0.5) / h
        weight = score * math.sqrt((bw * bh) / frame_area)  # larger faces dominate
        sum_w += weight
        sum_x += cx * weight
        sum_y += cy * weight
        best_score = max(best_score, score)
    if sum_w <= 1e-9:
        return None
    return _clamp(sum_x / sum_w), _clamp(sum_y / sum_w), best_score


def _map_centroid(magnitude: np.ndarray):
    """Intensity centroid of a nonnegative 2D map: (cx, cy, concentration) or None.

    concentration is 0 for a uniform map and approaches 1 for a point source
    (spatial std of a uniform 2D distribution is ~0.408 in normalized units).
    """
    mag = magnitude.astype(np.float64, copy=False)
    total = float(mag.sum())
    if total <= 1e-6:
        return None
    h, w = mag.shape[:2]
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    col = mag.sum(axis=0) / total
    row = mag.sum(axis=1) / total
    cx = float(np.dot(col, xs))
    cy = float(np.dot(row, ys))
    variance = float(np.dot(col, (xs - cx) ** 2) + np.dot(row, (ys - cy) ** 2))
    concentration = _clamp(1.0 - math.sqrt(max(variance, 0.0)) / 0.408)
    return _clamp(cx), _clamp(cy), concentration


def _robust_center(points: List[List[float]]):
    """Outlier-rejecting weighted mean of [t, cx, cy, weight] samples.

    Returns (cx, cy, agreement, kept_points). A single weird sample far from
    the median is dropped before averaging.
    """
    kept = points
    if len(points) >= 3:
        med_x = float(np.median([p[1] for p in points]))
        med_y = float(np.median([p[2] for p in points]))
        filtered = [p for p in points if math.hypot(p[1] - med_x, p[2] - med_y) <= 0.28]
        if filtered:
            kept = filtered
    total_w = sum(p[3] for p in kept)
    if total_w <= 1e-9:
        return 0.5, 0.5, 0.0, kept
    cx = sum(p[1] * p[3] for p in kept) / total_w
    cy = sum(p[2] * p[3] for p in kept) / total_w
    if len(kept) > 1:
        residual = sum(math.hypot(p[1] - cx, p[2] - cy) * p[3] for p in kept) / total_w
        agreement = _clamp(1.0 - residual / 0.25)
    else:
        agreement = 0.6  # single sample: no cross-sample evidence either way
    return _clamp(cx), _clamp(cy), agreement, kept


def _compute_subject_anchor(frames: Sequence[np.ndarray], sample_times: np.ndarray,
                            start: float, gray_frames: Sequence[np.ndarray] = None,
                            diff_maps: Sequence[np.ndarray] = None) -> Dict:
    """Estimate the subject position across the sampled frames.

    Reuses the frames already decoded for candidate metrics (no extra
    decoding); frames arrive downsampled to <=360px wide. Never raises.
    """
    try:
        return _compute_subject_anchor_impl(frames, sample_times, start, gray_frames, diff_maps)
    except Exception:
        return dict(_NEUTRAL_ANCHOR, path=[])


def _compute_subject_anchor_impl(frames, sample_times, start, gray_frames, diff_maps) -> Dict:
    if not frames:
        return dict(_NEUTRAL_ANCHOR, path=[])
    if gray_frames is None:
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    if diff_maps is None:
        diff_maps = [cv2.absdiff(gray_frames[i], gray_frames[i - 1])
                     for i in range(1, len(gray_frames))]

    def t_rel(i: int) -> float:
        if len(sample_times) > i:
            return round(max(0.0, float(sample_times[i]) - start), 4)
        return 0.0

    # --- Face override -----------------------------------------------------
    face_points: List[List[float]] = []
    face_best_score = 0.0
    for i, frame in enumerate(frames):
        hit = _detect_face_center(frame)
        if hit is not None:
            cx, cy, score = hit
            face_points.append([t_rel(i), cx, cy, score])
            face_best_score = max(face_best_score, score)
    face_fraction = len(face_points) / float(len(frames))
    if face_points and (face_fraction >= 0.3 or len(face_points) >= 2):
        cx, cy, agreement, kept = _robust_center(face_points)
        confidence = _clamp((0.6 + 0.3 * agreement) * (0.55 + 0.45 * face_fraction)
                            * (0.7 + 0.3 * face_best_score))
        return {
            "cx": cx,
            "cy": cy,
            "confidence": confidence,
            "source": "face",
            "path": [[p[0], round(p[1], 4), round(p[2], 4)] for p in kept],
        }

    # --- Motion centroid ---------------------------------------------------
    motion_points: List[List[float]] = []
    motion_strengths: List[float] = []
    concentrations: List[float] = []
    for i, diff in enumerate(diff_maps):
        strength = float(np.mean(diff)) / 42.0
        if strength < 0.02:  # near-static pair: absdiff is mostly sensor noise
            continue
        moving = np.maximum(diff.astype(np.float32) - 6.0, 0.0)  # floor out noise
        moving *= moving  # square to concentrate on the strongest mover
        centroid = _map_centroid(moving)
        if centroid is None:
            continue
        cx, cy, concentration = centroid
        motion_points.append([t_rel(i + 1), cx, cy, strength * (0.3 + 0.7 * concentration)])
        motion_strengths.append(strength)
        concentrations.append(concentration)
    if motion_points and float(np.mean(motion_strengths)) >= 0.05:
        cx, cy, agreement, kept = _robust_center(motion_points)
        strength_factor = _clamp(float(np.mean(motion_strengths)) / 0.3)
        confidence = _clamp(0.2 + 0.45 * strength_factor * float(np.mean(concentrations))
                            + 0.15 * agreement, hi=0.75)
        return {
            "cx": cx,
            "cy": cy,
            "confidence": confidence,
            "source": "motion",
            "path": [[p[0], round(p[1], 4), round(p[2], 4)] for p in kept],
        }

    # --- Detail centroid fallback (static shots) ---------------------------
    detail_points: List[List[float]] = []
    energies: List[float] = []
    concentrations = []
    for i, gray in enumerate(gray_frames):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = gx * gx + gy * gy
        centroid = _map_centroid(grad)
        if centroid is None:
            continue
        cx, cy, concentration = centroid
        energy = _clamp(float(np.mean(grad)) / 4000.0)
        detail_points.append([t_rel(i), cx, cy, 0.2 + 0.8 * energy])
        energies.append(energy)
        concentrations.append(concentration)
    if detail_points:
        cx, cy, agreement, kept = _robust_center(detail_points)
        # Low ceiling; near-zero for uniform frames so consumers ignore it.
        confidence = _clamp(0.35 * float(np.mean(energies) ** 0.5)
                            * (0.4 + 0.6 * float(np.mean(concentrations)))
                            * (0.5 + 0.5 * agreement), hi=0.35)
        return {
            "cx": cx,
            "cy": cy,
            "confidence": confidence,
            "source": "detail",
            "path": [[p[0], round(p[1], 4), round(p[2], 4)] for p in kept],
        }

    return dict(_NEUTRAL_ANCHOR, path=[])


def _colorfulness(frames: Sequence[np.ndarray]) -> float:
    values = []
    for frame in frames:
        b, g, r = cv2.split(frame.astype("float"))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        values.append((np.std(rg) + np.std(yb)) / 95.0)
    return _clamp(float(np.mean(values)) if values else 0.0)


def _build_candidate(
    video_file: str,
    source_name: str,
    video_duration: float,
    index: int,
    window: Dict,
    metrics: Dict,
) -> Dict:
    start = float(window["start"])
    end = float(window["end"])
    duration = max(0.01, end - start)
    motion = metrics["motion"]
    quality = metrics["quality_score"]
    brightness = metrics["brightness"]
    saturation = metrics["saturation"]
    contrast = metrics["contrast"]
    sharpness = metrics["sharpness"]
    colorfulness = metrics["colorfulness"]

    balanced_light = 1.0 - min(abs(brightness - 0.52) / 0.52, 1.0)
    action = _clamp(0.58 * motion + 0.17 * contrast + 0.12 * saturation + 0.13 * quality)
    beauty = _clamp(0.34 * quality + 0.21 * colorfulness + 0.18 * saturation + 0.17 * balanced_light + 0.10 * sharpness)
    tension = _clamp(0.42 * motion + 0.22 * contrast + 0.18 * (1.0 - balanced_light) + 0.18 * saturation)
    soft = _clamp(0.55 * beauty + 0.25 * balanced_light + 0.20 * (1.0 - motion))
    editorial = _clamp(0.35 * max(action, beauty, tension) + 0.35 * quality + 0.16 * saturation + 0.14 * contrast)

    tags = _fallback_tags(action, beauty, tension, soft, quality)
    semantic = {
        "action_intensity": action,
        "beauty_score": beauty,
        "emotion": "hype" if action > 0.68 else "soft" if soft > 0.64 else "tension" if tension > 0.62 else "neutral",
        "combat": 0.0,
        "chase": _clamp(motion * 0.6),
        "explosion": 0.0,
        "character_focus": _clamp(0.35 * quality + 0.20 * sharpness + 0.10 * balanced_light),
        "camera_motion": motion,
        "visual_quality": quality,
        "recommended_use": "drop" if action > 0.68 else "soft" if soft > 0.64 else "build" if tension > 0.62 else "flow",
        "description": "",
    }

    candidate_id = f"{_hash_text(os.path.abspath(video_file), 10)}_{index:05d}_{int(start * 1000):08d}"
    candidate = {
        "id": candidate_id,
        "video_file": os.path.abspath(video_file),
        "source_name": source_name,
        "video_duration": video_duration,
        "start": start,
        "end": end,
        "duration": duration,
        "center": start + duration * 0.5,
        "peak_time": start + float(metrics.get("peak_offset", duration * 0.5)),
        "scene_index": int(window.get("scene_index", index)),
        "kind": window.get("kind", "scene"),
        "motion": motion,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": sharpness,
        "colorfulness": colorfulness,
        "quality_score": quality,
        "subject_anchor": metrics.get("subject_anchor", dict(_NEUTRAL_ANCHOR, path=[])),
        "action_score": action,
        "beauty_score": beauty,
        "tension_score": tension,
        "soft_score": soft,
        "editorial_score": editorial,
        "tags": tags,
        "semantic": semantic,
        "ai_analyzed": False,
    }
    # Optical-flow keys are copied ONLY when the measurement produced them —
    # a failed flow pass omits them so stage-6 scoring falls back to the plain
    # frame-diff `motion` instead of reading an errored clip as perfectly calm.
    for flow_key in ("kinetic", "subject_motion", "camera_motion"):
        if flow_key in metrics:
            candidate[flow_key] = float(metrics[flow_key])
    return candidate


def _fallback_tags(action: float, beauty: float, tension: float, soft: float, quality: float) -> List[str]:
    tags: List[str] = []
    if action >= 0.68:
        tags.append("action")
    if beauty >= 0.62:
        tags.append("beauty")
    if tension >= 0.62:
        tags.append("tension")
    if soft >= 0.64:
        tags.append("soft")
    if quality >= 0.68:
        tags.append("clean")
    if not tags:
        tags.append("flow")
    return tags


def _select_ai_candidates(candidates: List[Dict], max_windows: int) -> List[Dict]:
    if len(candidates) <= max_windows:
        return candidates

    selected: Dict[str, Dict] = {}

    def add_many(items: Sequence[Dict], count: int) -> None:
        for item in items[:count]:
            selected[item["id"]] = item

    ranked_action = sorted(candidates, key=lambda c: c.get("action_score", 0.0), reverse=True)
    ranked_beauty = sorted(candidates, key=lambda c: c.get("beauty_score", 0.0), reverse=True)
    ranked_quality = sorted(candidates, key=lambda c: c.get("quality_score", 0.0), reverse=True)

    add_many(ranked_action, max(1, max_windows // 3))
    add_many(ranked_beauty, max(1, max_windows // 4))
    add_many(ranked_quality, max(1, max_windows // 5))

    if len(selected) < max_windows:
        step = max(1, len(candidates) // max(1, max_windows - len(selected)))
        for item in candidates[::step]:
            selected[item["id"]] = item
            if len(selected) >= max_windows:
                break

    return list(selected.values())[:max_windows]


def _annotate_candidates_with_qwen(
    video_file: str,
    fps: float,
    candidates: List[Dict],
    qwen_model_path: str,
    use_gpu: bool,
    audio_profile: Dict,
) -> Dict | None:
    max_windows = int(os.environ.get("BEATSYNC_QWEN_MAX_WINDOWS", "120"))
    max_windows = max(0, max_windows)
    if max_windows == 0:
        print("   Qwen semantic analysis skipped (BEATSYNC_QWEN_MAX_WINDOWS=0).")
        return

    ai_candidates = _select_ai_candidates(candidates, max_windows)
    if not ai_candidates:
        return {"qwen_frame_count": 0, "qwen_tag_count": 0}

    print(f"   Qwen semantic analysis: {len(ai_candidates)} candidate moments")
    response = _run_qwen_worker(
        _qwen_request(
            qwen_model_path, use_gpu, audio_profile,
            video_file=video_file, fps=fps, candidates=ai_candidates,
        ),
        batch=False,
    )
    semantics = response.get("semantics") if isinstance(response, dict) else {}
    if not semantics:
        print("      Qwen returned no semantic tags; deterministic visual tags remain active.")
        return {
            "qwen_frame_count": len(ai_candidates),
            "qwen_tag_count": 0,
            "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
            "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
            "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
        }

    merged_count = _merge_semantics_by_id(ai_candidates, semantics)
    print(f"      Qwen semantic tags merged: {merged_count}/{len(ai_candidates)}")
    timing = response.get("timings_by_job", {}).get("single", {}) if isinstance(response, dict) else {}
    return {
        "qwen_frame_count": int(timing.get("frame_count") or len(ai_candidates)),
        "qwen_tag_count": int(timing.get("tag_count") or merged_count),
        "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
        "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
        "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
    }


def _qwen_worker_environment() -> Dict[str, str]:
    env = os.environ.copy()
    llama_dir = os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", DEFAULT_LLAMA_CPP_DIR)
    llama_root = os.path.normcase(os.path.abspath(llama_dir))
    path_parts = []
    for part in env.get("PATH", "").split(os.pathsep):
        if not part:
            continue
        norm = os.path.normcase(os.path.abspath(part))
        if norm == llama_root:
            continue
        path_parts.append(part)
    env["PATH"] = os.pathsep.join([llama_dir] + path_parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _merge_semantic(candidate: Dict, semantic: Dict) -> None:
    if not semantic:
        return
    current = candidate.get("semantic", {})
    merged = dict(current)
    numeric_keys = [
        "action_intensity",
        "beauty_score",
        "combat",
        "chase",
        "explosion",
        "character_focus",
        "camera_motion",
        "visual_quality",
    ]
    for key in numeric_keys:
        if key in semantic:
            merged[key] = _clamp(semantic[key])
    for key in ["emotion", "recommended_use", "description"]:
        if semantic.get(key):
            merged[key] = str(semantic[key])[:160]

    candidate["semantic"] = merged
    candidate["ai_analyzed"] = True

    deterministic_action = _clamp(candidate.get("action_score", 0.0))
    deterministic_motion = _clamp(candidate.get("motion", 0.0))
    semantic_action = _clamp(merged.get("action_intensity", 0.0))
    motion_gate = _clamp(0.35 + 0.65 * deterministic_motion)

    # Qwen is used as semantic guidance, not as the sole truth. Motion/quality
    # metrics stay dominant so a beautiful static frame is not hallucinated into
    # an action scene.
    action = _clamp(0.72 * deterministic_action + 0.28 * semantic_action * motion_gate)
    beauty = _clamp(0.65 * candidate.get("beauty_score", 0.0) + 0.35 * merged.get("beauty_score", 0.0))
    quality = _clamp(0.70 * candidate.get("quality_score", 0.0) + 0.30 * merged.get("visual_quality", 0.0))
    camera_motion = _clamp(0.75 * deterministic_motion + 0.25 * merged.get("camera_motion", deterministic_motion))
    tension = _clamp(0.48 * candidate.get("tension_score", 0.0) + 0.22 * camera_motion + 0.18 * action + 0.12 * merged.get("character_focus", 0.0))
    soft = _clamp(0.48 * beauty + 0.24 * merged.get("character_focus", 0.0) + 0.18 * (1.0 - action) + 0.10 * quality)

    candidate["action_score"] = action
    candidate["beauty_score"] = beauty
    candidate["quality_score"] = quality
    candidate["tension_score"] = tension
    candidate["soft_score"] = soft
    candidate["editorial_score"] = _clamp(0.30 * max(action, beauty, tension) + 0.42 * quality + 0.16 * candidate.get("saturation", 0.0) + 0.12 * candidate.get("contrast", 0.0))

    tags = set(_fallback_tags(action, beauty, tension, soft, quality))
    rec = str(merged.get("recommended_use", "")).lower()
    emotion = str(merged.get("emotion", "")).lower()
    if rec and (rec != "drop" or action >= 0.45):
        tags.add(rec)
    if emotion and (emotion != "hype" or action >= 0.45):
        tags.add(emotion)
    if merged.get("combat", 0.0) > 0.50 and action >= 0.38:
        tags.add("combat")
    if merged.get("chase", 0.0) > 0.50 and action >= 0.38:
        tags.add("chase")
    if merged.get("explosion", 0.0) > 0.55 and action >= 0.38:
        tags.add("explosion")
    candidate["tags"] = sorted(tags)
