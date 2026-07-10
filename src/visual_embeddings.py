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
# v2: 3 frames per window (25%/50%/75% of the window) mean-pooled, so a single
# transition/black center frame can no longer poison the embedding.
PREPROC_VERSION = "dino_preproc_v2_3frame"

# Fractions of the window duration at which frames are sampled and pooled.
_FRAME_FRACTIONS = (0.25, 0.50, 0.75)

# Same cache directory video_analysis uses (input/video_analysis_cache), but the
# sidecar filenames carry a distinct ``_dino`` suffix and their own signature, so
# they never collide with or invalidate the Qwen analysis cache.
_CACHE_DIR = os.path.join(ROOT_DIR, "input", "video_analysis_cache")

# ImageNet normalization (RGB), the standard DINOv2 preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_INPUT_SIZE = 224

_sha_cache: Dict[str, str] = {}          # (path:size:mtime) -> sha256 prefix
_ort_import_failed = False


# --- Shared optional-backend + sidecar abstractions ------------------------
# These two small helpers live here (the lighter of the two modules) and are
# imported by video_analysis.py, so the YuNet / DINOv2 / video-analysis sidecar
# logic exists in exactly one place. Qwen deliberately does NOT use
# OptionalBackend: it resolves a *set* of executables/models (not one model
# file), keeps no cached session (it shells out per call), and its disable lives
# upstream (auto_mode reads BEATSYNC_DISABLE_QWEN) — forcing it in would distort
# behavior, so it keeps its bespoke resolver/availability/signature helpers.


class OptionalBackend:
    """A single optional model backend: env-based disable + kill switch, the
    exact ``env override → bundled default → disabled`` path-resolution order,
    and a lazily-built instance whose *concurrency discipline is preserved, not
    homogenized* (``concurrency='thread_local'`` for YuNet's per-thread
    non-thread-safe detector; ``'shared'`` for DINOv2's process-wide
    lock+dict session cache)."""

    def __init__(self, name: str, *, default_model: str, model_env: str | None = None,
                 disable_env: str | None = None, concurrency: str = "shared"):
        self.name = name
        self.default_model = default_model
        self.model_env = model_env
        self.disable_env = disable_env
        self.concurrency = concurrency
        self._lock = threading.Lock()
        self._shared_cache: Dict[str, object] = {}   # key -> instance (shared strategy)
        self._thread_local = threading.local()       # per-thread instance (thread_local)
        self.disabled_latch = False                  # thread_local: latch off once impossible

    def disabled(self) -> bool:
        """True when the disable/kill-switch env var is set to a truthy value.
        Backends without a disable env (YuNet) are never disabled this way."""
        if not self.disable_env:
            return False
        return os.environ.get(self.disable_env, "").strip() not in ("", "0", "false", "False")

    def resolve_model_path(self) -> str:
        """Explicit env override is authoritative (missing → disabled / kill
        switch); otherwise fall back to the bundled default, else disabled.
        This order is preserved verbatim from both former resolvers."""
        env_path = os.environ.get(self.model_env, "").strip() if self.model_env else ""
        if env_path:
            return env_path if os.path.isfile(env_path) else ""
        return self.default_model if os.path.isfile(self.default_model) else ""

    def get_thread_local(self, build):
        """YuNet discipline: one instance per worker thread (FaceDetectorYN is
        not thread-safe), with a process-wide latch that short-circuits once
        construction is known impossible. ``build()`` returns the instance or
        None; a None result latches the backend off."""
        if self.disabled_latch:
            return None
        instance = getattr(self._thread_local, "instance", None)
        if instance is not None:
            return instance
        instance = build()
        if instance is None:
            self.disabled_latch = True
            return None
        self._thread_local.instance = instance
        return instance

    def get_shared(self, key: str, build):
        """DINOv2 discipline: a process-wide {key: instance-or-None} cache behind
        a lock; ``build()`` runs at most once per key and a None result is cached
        so a failed load is not retried."""
        with self._lock:
            if key in self._shared_cache:
                return self._shared_cache[key]
            instance = build()
            self._shared_cache[key] = instance
            return instance


class SidecarCache:
    """Shared per-source-video JSON sidecar cache. Path layout is exactly what
    both former copies produced, so existing on-disk caches keep hitting:

        {cache_dir}/{sha1(stem)[:8]}_{signature}{suffix}.json

    Serialization and failure behavior are parameterized to match each site:
    the video-analysis cache uses ``suffix='' , indent=2`` and *logs* read/write
    failures; the DINOv2 cache uses ``suffix='_dino'``, compact JSON and *silent*
    failures."""

    def __init__(self, cache_dir: str, suffix: str, *, indent: int | None = None,
                 log_label: str | None = None, make_dir_on_path: bool = True):
        self.cache_dir = cache_dir
        self.suffix = suffix
        self.indent = indent
        self.log_label = log_label
        self.make_dir_on_path = make_dir_on_path

    def path(self, source_file: str, signature: str) -> str:
        if self.make_dir_on_path:
            os.makedirs(self.cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source_file))[0] or "video"
        prefix = hashlib.sha1(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{prefix}_{signature}{self.suffix}.json")

    def load(self, path: str):
        """Return the parsed JSON object, or None on miss / unreadable / corrupt.
        Validation of the payload is the caller's job."""
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            if self.log_label:
                print(f"   Warning: could not read {self.log_label}: {e}")
            return None

    def save(self, path: str, data) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=self.indent)
            os.replace(tmp_path, path)
        except Exception as e:
            if self.log_label:
                print(f"   Warning: could not write {self.log_label}: {e}")


# --- Model / runtime resolution -------------------------------------------

_EMBED_BACKEND = OptionalBackend(
    "DINOv2 embeddings",
    default_model=DEFAULT_EMBED_MODEL,
    model_env=EMBED_MODEL_ENV,
    disable_env=EMBED_DISABLE_ENV,
    concurrency="shared",
)

# Same file layout / format as before (compact JSON, silent failures, `_dino`
# suffix); make_dir_on_path=False mirrors the old _cache_path which never
# created the directory (only _save_sidecar did).
_EMBED_SIDECAR = SidecarCache(
    _CACHE_DIR, "_dino", indent=None, log_label=None, make_dir_on_path=False,
)


def _embed_disabled() -> bool:
    return _EMBED_BACKEND.disabled()


def _resolve_model_path() -> str:
    return _EMBED_BACKEND.resolve_model_path()


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
    imported lazily so the rest of the pipeline runs even when it is absent.
    Caching keeps DINOv2's original lock+dict discipline (one session per model
    path, behind the backend lock)."""
    def build():
        global _ort_import_failed
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
            return ort.InferenceSession(
                model_path, opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            return None

    return _EMBED_BACKEND.get_shared(model_path, build)


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


def _embed_frame_raw(session, input_name: str, frame_bgr: np.ndarray) -> np.ndarray | None:
    """Return the L2-unit-normalized float64 384-d embedding for one frame
    (unquantized — quantization happens once, after pooling)."""
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
    return vec / norm


def _pool_embeddings(vecs: List[np.ndarray]) -> List[float] | None:
    """Mean-pool unit vectors in float64, L2-renormalize, round to 4dp.
    The 4dp rounding is the determinism firewall: only the quantized pooled
    vector is ever stored or compared, so ONNX float jitter cannot leak out."""
    if not vecs:
        return None
    mean = np.mean(np.stack(vecs), axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(mean))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    unit = mean / norm
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
    return _EMBED_SIDECAR.path(video_file, signature)


def _window_key(start: float, end: float) -> str:
    return f"{round(float(start), 3)}_{round(float(end), 3)}"


def _load_sidecar(path: str, signature: str) -> Dict[str, List[float]]:
    """Return the cached {window_key: vector} map, or empty on miss/corrupt/stale."""
    data = _EMBED_SIDECAR.load(path)
    try:
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
        # Corrupt cache → recompute silently.
        return {}


def _save_sidecar(path: str, signature: str, embeddings: Dict[str, List[float]]) -> None:
    payload = {
        "signature": signature,
        "preproc_version": PREPROC_VERSION,
        "embeddings": embeddings,
    }
    _EMBED_SIDECAR.save(path, payload)


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

        # Attach the per-candidate window key and sampled frame times up front.
        # Three frames per window (25/50/75% of the duration) — for zero-length
        # windows / stills all three collapse to the same time, so the sorted
        # de-duplicated tuple holds a single entry (one decode).
        planned = []
        for cand in group:
            start = float(cand.get("start", 0.0))
            end = float(cand.get("end", start))
            duration = max(0.0, end - start)
            frame_times = tuple(sorted({start + duration * f for f in _FRAME_FRACTIONS}))
            planned.append((cand, _window_key(start, end), frame_times))

        # Anything not in cache needs a real decode; only open the capture then.
        need_decode = [p for p in planned if p[1] not in cache]
        cap = None
        if need_decode:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                cap.release()
                cap = None

        # Decode every needed frame time in one ascending pass over the file so
        # seeks stay cheap; identical times (overlapping windows, stills) are
        # decoded and embedded exactly once.
        frame_vecs: Dict[float, np.ndarray] = {}
        if cap is not None:
            all_times = sorted({t for _cand, _wkey, times in need_decode for t in times})
            for t in all_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                vec = _embed_frame_raw(session, input_name, frame)
                if vec is not None:
                    frame_vecs[t] = vec

        if cap is not None:
            cap.release()

        # Pool the successful per-frame embeddings for each window; at least one
        # good frame → an embedding, zero → no keys (exactly like a failed
        # single-frame decode before).
        for cand, wkey, times in need_decode:
            pooled = _pool_embeddings([frame_vecs[t] for t in times if t in frame_vecs])
            if pooled is None:
                continue
            cache[wkey] = pooled
            cache_dirty = True

        # Assign embeddings (from cache or freshly decoded) to candidates.
        for cand, wkey, _times in planned:
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
