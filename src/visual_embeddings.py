#!/usr/bin/env python3
"""Local visual embeddings for stage-6 semantic diversity.

Every analysis candidate (a dict carrying ``video_file``, ``start``, ``end``,
``center`` and ``id`` — see ``_build_candidate`` in ``src/video_analysis.py``)
can be annotated with a 384-d DINOv2 embedding and a deterministic
visual-cluster id. The planner uses the cluster id to avoid stacking visually
near-identical clips back to back.

Model: DINOv2 ViT-S/14 exported to ONNX (input ``input`` [1,3,224,224] float32,
output ``output`` [1,384] float32 — the pooled CLS embedding). Resolution mirrors
the YuNet integration in ``video_analysis.py``:

    1. $BEATSYNC_EMBED_MODEL   (explicit override / kill switch)
    2. models/dinov2_vits14.onnx   (fetched by scripts/fetch_dinov2.py)
    3. otherwise the feature is disabled — the planner treats missing keys as
       zero penalty, so absence is silent-but-logged.

$BEATSYNC_DISABLE_EMBED=1 turns the feature off entirely.

Everything here is CPU-only and deterministic across runs/machines: ONNX Runtime
runs single-threaded on the CPU provider, preprocessing is pure numpy/cv2, and
the stored vectors are L2-unit-normalized then rounded to 4 decimals. Embeddings
are cached in a sidecar JSON per source video that is completely independent of
the video_analysis cache / ANALYSIS_VERSION, so existing Qwen results stay valid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from typing import Dict, List, Sequence

import cv2
import numpy as np

from logger import ROOT_DIR


# --- Configuration ---------------------------------------------------------

EMBED_MODEL_ENV = "BEATSYNC_EMBED_MODEL"
EMBED_DISABLE_ENV = "BEATSYNC_DISABLE_EMBED"
DEFAULT_EMBED_MODEL = os.path.join(ROOT_DIR, "models", "dinov2_vits14.onnx")

EMBED_DIM = 384
QUANT_DECIMALS = 4  # L2-normalize, then round each component to this many dp

# Bump when preprocessing changes so stale sidecar caches recompute.
PREPROC_VERSION = "dino_preproc_v1"

# Same cache directory video_analysis uses (input/video_analysis_cache), but the
# sidecar filenames carry a distinct ``_dino`` suffix and their own signature, so
# they never collide with or invalidate the Qwen analysis cache.
_CACHE_DIR = os.path.join(ROOT_DIR, "input", "video_analysis_cache")

# ImageNet normalization (RGB), the standard DINOv2 preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_INPUT_SIZE = 224

_session_lock = threading.Lock()
_session_cache: Dict[str, object] = {}   # model_path -> InferenceSession (or None on failure)
_sha_cache: Dict[str, str] = {}          # (path:size:mtime) -> sha256 prefix
_ort_import_failed = False


# --- Model / runtime resolution -------------------------------------------

def _embed_disabled() -> bool:
    return os.environ.get(EMBED_DISABLE_ENV, "").strip() not in ("", "0", "false", "False")


def _resolve_model_path() -> str:
    """Explicit env override is authoritative (missing → disabled / kill switch);
    otherwise fall back to the bundled default, else disabled."""
    env_path = os.environ.get(EMBED_MODEL_ENV, "").strip()
    if env_path:
        return env_path if os.path.isfile(env_path) else ""
    return DEFAULT_EMBED_MODEL if os.path.isfile(DEFAULT_EMBED_MODEL) else ""


def _sha256_prefix(path: str, length: int = 12) -> str:
    try:
        stat = os.stat(path)
        key = f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return "missing"
    cached = _sha_cache.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return "missing"
    prefix = digest.hexdigest()[:length]
    _sha_cache[key] = prefix
    return prefix


def _get_session(model_path: str):
    """Return a single-threaded CPU ONNX Runtime session for the model, or None
    if onnxruntime is unavailable or the model cannot be loaded. onnxruntime is
    imported lazily so the rest of the pipeline runs even when it is absent."""
    global _ort_import_failed
    with _session_lock:
        if model_path in _session_cache:
            return _session_cache[model_path]
        if _ort_import_failed:
            return None
        try:
            import onnxruntime as ort  # lazy: optional dependency
        except Exception:
            _ort_import_failed = True
            return None
        try:
            opts = ort.SessionOptions()
            # Single-threaded CPU execution → deterministic across runs/machines.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model_path, opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            session = None
        _session_cache[model_path] = session
        return session


# --- Preprocessing / inference --------------------------------------------

def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → NCHW float32 tensor: RGB, resize shorter side to 224 with
    INTER_AREA, center-crop 224x224, scale to [0,1], ImageNet-normalize."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short <= 0:
        raise ValueError("empty frame")
    scale = _INPUT_SIZE / float(short)
    new_w = max(_INPUT_SIZE, int(round(w * scale)))
    new_h = max(_INPUT_SIZE, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Center crop 224x224.
    top = (new_h - _INPUT_SIZE) // 2
    left = (new_w - _INPUT_SIZE) // 2
    crop = resized[top:top + _INPUT_SIZE, left:left + _INPUT_SIZE]
    chw = np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.expand_dims(chw, 0).astype(np.float32)


def _embed_frame(session, input_name: str, frame_bgr: np.ndarray) -> List[float] | None:
    """Return the L2-normalized, 4dp-rounded 384-d embedding for one frame."""
    try:
        tensor = _preprocess(frame_bgr)
        out = session.run(None, {input_name: tensor})[0]
    except Exception:
        return None
    vec = np.asarray(out, dtype=np.float64).reshape(-1)
    if vec.shape[0] != EMBED_DIM or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    unit = vec / norm
    return [round(float(x), QUANT_DECIMALS) for x in unit]


# --- Sidecar cache ---------------------------------------------------------

def _video_signature(video_file: str, model_sha: str) -> str:
    try:
        stat = os.stat(video_file)
        size, mtime = stat.st_size, int(stat.st_mtime)
    except OSError:
        size, mtime = -1, -1
    raw = "|".join([
        os.path.abspath(video_file),
        str(size),
        str(mtime),
        model_sha,
        PREPROC_VERSION,
    ])
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _cache_path(video_file: str, signature: str) -> str:
    stem = os.path.splitext(os.path.basename(video_file))[0] or "video"
    short = hashlib.sha1(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"{short}_{signature}_dino.json")


def _window_key(start: float, end: float) -> str:
    return f"{round(float(start), 3)}_{round(float(end), 3)}"


def _load_sidecar(path: str, signature: str) -> Dict[str, List[float]]:
    """Return the cached {window_key: vector} map, or empty on miss/corrupt/stale."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if data.get("signature") != signature or data.get("preproc_version") != PREPROC_VERSION:
            return {}
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, dict):
            return {}
        clean: Dict[str, List[float]] = {}
        for key, vec in embeddings.items():
            if isinstance(vec, list) and len(vec) == EMBED_DIM:
                clean[key] = [float(x) for x in vec]
        return clean
    except Exception:
        # Corrupt / unreadable cache → recompute silently.
        return {}


def _save_sidecar(path: str, signature: str, embeddings: Dict[str, List[float]]) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        payload = {
            "signature": signature,
            "preproc_version": PREPROC_VERSION,
            "embeddings": embeddings,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        # Cache is an optimization; failure to persist must never break analysis.
        pass


# --- Deterministic online clustering --------------------------------------

def _cluster(items: Sequence[dict], sim_threshold: float) -> int:
    """Assign ``visual_cluster`` to each embedded candidate via deterministic
    online cosine-threshold clustering. ``items`` are the candidates that carry
    an ``embedding``; they are visited in a fully deterministic order. Returns
    the number of clusters created."""
    ordered = sorted(
        items,
        key=lambda c: (str(c.get("video_file", "")), round(float(c.get("start", 0.0)), 3), str(c.get("id", ""))),
    )
    centroids: List[np.ndarray] = []   # unit-norm running-mean centroids
    sums: List[np.ndarray] = []        # running vector sums
    counts: List[int] = []
    for cand in ordered:
        vec = np.asarray(cand["embedding"], dtype=np.float64)  # already 4dp, ~unit
        assigned = None
        for cid, centroid in enumerate(centroids):
            if float(np.dot(vec, centroid)) > sim_threshold:
                assigned = cid
                break
        if assigned is None:
            assigned = len(centroids)
            centroids.append(vec / (np.linalg.norm(vec) or 1.0))
            sums.append(vec.copy())
            counts.append(1)
        else:
            counts[assigned] += 1
            sums[assigned] += vec
            mean = sums[assigned] / counts[assigned]
            norm = np.linalg.norm(mean) or 1.0
            centroids[assigned] = mean / norm
        cand["visual_cluster"] = int(assigned)
    return len(centroids)


# --- Public API ------------------------------------------------------------

def annotate_candidates_with_embeddings(
    candidates: List[dict],
    sim_threshold: float = 0.82,
) -> dict:
    """Annotate candidates in place with ``embedding`` (384 floats) and
    ``visual_cluster`` (int >= 0).

    Candidates that cannot be embedded (decode failure, model absent) get
    NEITHER key. If the model/runtime is unavailable this returns immediately
    with ``available=False`` and touches nothing.

    Returns a stats dict:
        {'available': bool, 'embedded': int, 'skipped': int,
         'clusters': int, 'model': str, 'seconds': float}
    """
    started = time.perf_counter()
    stats = {
        "available": False,
        "embedded": 0,
        "skipped": 0,
        "clusters": 0,
        "model": "",
        "seconds": 0.0,
    }

    if not candidates:
        stats["seconds"] = time.perf_counter() - started
        return stats

    if _embed_disabled():
        print("   ℹ️  Visual embeddings disabled via BEATSYNC_DISABLE_EMBED; skipping semantic diversity.")
        stats["seconds"] = time.perf_counter() - started
        return stats

    model_path = _resolve_model_path()
    if not model_path:
        print("   ℹ️  Visual embedding model not found (run scripts/fetch_dinov2.py); semantic diversity disabled.")
        stats["seconds"] = time.perf_counter() - started
        return stats

    session = _get_session(model_path)
    if session is None:
        print("   ⚠️  onnxruntime/DINOv2 unavailable; visual embeddings skipped (planner treats this as zero penalty).")
        stats["seconds"] = time.perf_counter() - started
        return stats

    input_name = session.get_inputs()[0].name
    model_sha = _sha256_prefix(model_path)
    stats["available"] = True
    stats["model"] = os.path.basename(model_path)

    # Group candidates by source video so each file is opened at most once.
    by_video: Dict[str, List[dict]] = {}
    for cand in candidates:
        video_file = cand.get("video_file")
        if not video_file:
            stats["skipped"] += 1
            continue
        by_video.setdefault(str(video_file), []).append(cand)

    embedded: List[dict] = []

    for video_file, group in by_video.items():
        signature = _video_signature(video_file, model_sha)
        cache_path = _cache_path(video_file, signature)
        cache = _load_sidecar(cache_path, signature)
        cache_dirty = False

        # Attach the per-candidate window key and center time up front.
        planned = []
        for cand in group:
            start = float(cand.get("start", 0.0))
            end = float(cand.get("end", start))
            center = cand.get("center")
            if center is None:
                center = start + max(0.0, end - start) * 0.5
            planned.append((cand, _window_key(start, end), float(center)))

        # Anything not in cache needs a real decode; only open the capture then.
        need_decode = [p for p in planned if p[1] not in cache]
        cap = None
        if need_decode:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                cap.release()
                cap = None

        # Decode in ascending center time so a single forward pass over the file
        # keeps seeks cheap.
        for cand, wkey, center in sorted(need_decode, key=lambda p: p[2]):
            if cap is None:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, center) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            vec = _embed_frame(session, input_name, frame)
            if vec is None:
                continue
            cache[wkey] = vec
            cache_dirty = True

        if cap is not None:
            cap.release()

        # Assign embeddings (from cache or freshly decoded) to candidates.
        for cand, wkey, _center in planned:
            vec = cache.get(wkey)
            if vec is None:
                stats["skipped"] += 1
                continue
            cand["embedding"] = list(vec)
            embedded.append(cand)

        if cache_dirty:
            _save_sidecar(cache_path, signature, cache)

    stats["embedded"] = len(embedded)

    if embedded:
        stats["clusters"] = _cluster(embedded, sim_threshold)

    stats["seconds"] = time.perf_counter() - started
    print(
        f"   Visual embeddings: {stats['embedded']} embedded, {stats['skipped']} skipped, "
        f"{stats['clusters']} clusters [{stats['seconds']:.1f}s, model {stats['model']}]"
    )
    return stats
