#!/usr/bin/env python3
"""Local music-structure labels + stem-derived beat signals (Apple Silicon / MLX).

Two independent, self-contained features for Auto Mode, both backed by the MLX
ports of the mir-aidj *All-In-One* structure analyzer and *Demucs*:

    analyze_structure(audio_file)          -> section labels + downbeats + bpm
    get_stem_signals(audio_file, beats)    -> per-beat drum/vocal/bass signals

Design mirrors the wave-12 ``visual_embeddings.py`` and wave-13 RIFE modules:

* **Lazy, optional imports.** ``allin1_mlx`` / ``demucs_mlx`` (and their ``mlx``
  backend) are imported only when a feature actually runs. On Intel/Windows the
  ``mlx`` wheel cannot import at all — that ImportError *is* the fallback path:
  the function logs one line and returns ``{'available': False, ...}``.
* **Kill switches.** ``BEATSYNC_DISABLE_STRUCTURE=1`` disables structure;
  ``BEATSYNC_DISABLE_STEMS=1`` disables stem signals.
* **Never raises.** Any failure (missing weights, decode error, backend crash)
  is caught and reported as ``available=False`` with exactly one ``ℹ️``/``⚠️``
  line, so the caller can treat both features as pure enhancements.
* **Determinism by memoization.** MLX offers no bit-exactness contract across
  runs/machines, so — like the RIFE integration — correctness of "same input →
  same output" comes from a sidecar JSON cache, not from the model. Cold calls
  compute, round every float to 4 dp, and persist; warm calls return the stored
  rounded values verbatim, so a cold call and a later warm call are identical.

Caches live in the same directory ``video_analysis.py`` uses
(``input/video_analysis_cache``) with distinct ``_structure`` / ``_stems``
suffixes and their own signatures, so they never collide with or invalidate the
Qwen/DINO caches. The signature keys on (abs path, size, mtime, backend package
versions, ``PREPROC_VERSION``); the stems signature *also* keys on a hash of the
rounded beat grid, so re-running with a different beat plan recomputes.

WAV-decode decision: All-In-One recommends WAV input because MP3 decoder
priming/offset shifts the beat/downbeat grid. Demucs likewise wants a clean
stereo waveform. So both entry points decode the source once with librosa (the
same decoder the rest of the pipeline uses in ``auto_mode``) to a temporary
stereo WAV before handing a *path* to the MLX backends. This removes MP3 offset
nondeterminism and keeps stem onsets aligned with the pipeline's beat grid.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from typing import Dict, List, Sequence

import numpy as np

from logger import ROOT_DIR


# --- Configuration ---------------------------------------------------------

STRUCTURE_DISABLE_ENV = "BEATSYNC_DISABLE_STRUCTURE"
STEMS_DISABLE_ENV = "BEATSYNC_DISABLE_STEMS"

# Explicit override for the All-In-One MLX structure weights directory.
ALLIN1_WEIGHTS_ENV = "BEATSYNC_ALLIN1_WEIGHTS_DIR"

STRUCTURE_BACKEND = "all-in-one-mlx"
_ALLIN1_MODEL = "harmonix-all"          # 8-fold Harmonix ensemble
_ALLIN1_FOLDS = 8
_DEMUCS_MODEL = "htdemucs"              # 4-stem: drums / bass / other / vocals

# Bump when the compute path (signal formulas, decode, model choice) changes so
# stale sidecar caches recompute rather than returning old-format values.
PREPROC_VERSION = "structstems_v1"

QUANT_DECIMALS = 4  # round every stored/returned float to this many dp

# Same cache dir video_analysis uses; distinct suffixes keep us isolated.
_CACHE_DIR = os.path.join(ROOT_DIR, "input", "video_analysis_cache")

# Auto-download location for the (small, ~10 MB) All-In-One MLX weights. The
# large Demucs checkpoint (~160 MB) is fetched+converted by demucs-mlx itself
# into ~/.cache/demucs-mlx on first use, so we do not manage that here.
_DEFAULT_ALLIN1_WEIGHTS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "beatsync", "allin1-mlx-weights"
)
_REPO_ALLIN1_WEIGHTS_DIR = os.path.join(ROOT_DIR, "models", "allin1-mlx-weights")
_ALLIN1_WEIGHTS_BASE_URL = (
    "https://raw.githubusercontent.com/ssmall256/all-in-one-mlx/main/mlx-weights"
)

# Stem-signal DSP constants (operate on the 44.1 kHz Demucs stems).
_STEM_SR = 44100
_STEM_HOP = 512
_STEM_FRAME = 2048
_VOCAL_SMOOTH_SECONDS = 0.4

_BACKENDS_IMPORT_FAILED = False  # set once if the mlx stack cannot import at all


# --- Small numerical helpers (inlined, matching auto_mode idioms) ----------

def _robust_normalize(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Percentile (2..98) min-max to [0, 1]; constant/zero/NaN input → ``default``.

    Same robust idiom as ``auto_mode._normalize`` so a single spike does not
    flatten everything and degenerate input never produces NaN/inf.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi - lo < 1e-8:
        return np.zeros_like(arr) + default
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Odd-window moving average with edge padding (mirrors auto_mode._smooth)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or width <= 1:
        return arr
    width = int(max(1, min(width, max(1, arr.size))))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _interp_to_beats(curve: np.ndarray, beat_times: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Sample a frame-rate ``curve`` at each beat time (edge-hold outside range)."""
    curve = np.asarray(curve, dtype=float).reshape(-1)
    beat_times = np.asarray(beat_times, dtype=float).reshape(-1)
    if curve.size == 0 or beat_times.size == 0:
        return np.zeros(beat_times.size, dtype=float)
    # librosa.frames_to_time without importing librosa at module scope: the frame
    # centres are n * hop / sr seconds.
    curve_times = np.arange(curve.size, dtype=float) * (float(hop) / float(sr))
    return np.interp(beat_times, curve_times, curve, left=float(curve[0]), right=float(curve[-1]))


def _round_list(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not np.isfinite(fv):
            fv = 0.0
        out.append(round(fv, QUANT_DECIMALS))
    return out


# --- Disable / availability -----------------------------------------------

def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0", "false", "False")


def _mlx_importable() -> bool:
    """True only if the ``mlx`` backend imports (i.e. we are on Apple Silicon).

    Intel Macs / Windows / Linux have no mlx wheel; that ImportError is the
    documented fallback path for both features.
    """
    global _BACKENDS_IMPORT_FAILED
    if _BACKENDS_IMPORT_FAILED:
        return False
    try:
        import mlx.core  # noqa: F401  (lazy: optional Apple-Silicon-only dependency)
        return True
    except Exception:
        _BACKENDS_IMPORT_FAILED = True
        return False


def _backend_versions() -> str:
    """Stable version string of the backends, folded into the cache signature."""
    import importlib.metadata as md

    parts = []
    for pkg in ("all-in-one-mlx", "demucs-mlx", "mlx", "natten-mlx", "librosa"):
        try:
            parts.append(f"{pkg}={md.version(pkg)}")
        except Exception:
            parts.append(f"{pkg}=?")
    return ";".join(parts)


# --- Sidecar cache ---------------------------------------------------------

def _file_stat_tuple(audio_file: str) -> tuple[int, int]:
    try:
        st = os.stat(audio_file)
        return int(st.st_size), int(st.st_mtime)
    except OSError:
        return -1, -1


def _signature(audio_file: str, extra: str = "") -> str:
    size, mtime = _file_stat_tuple(audio_file)
    raw = "|".join([
        os.path.abspath(audio_file),
        str(size),
        str(mtime),
        _backend_versions(),
        PREPROC_VERSION,
        extra,
    ])
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _cache_path(audio_file: str, signature: str, kind: str) -> str:
    stem = os.path.splitext(os.path.basename(audio_file))[0] or "audio"
    short = hashlib.sha1(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"{short}_{signature}_{kind}.json")


def _beats_hash(beat_times: np.ndarray) -> str:
    rounded = _round_list(np.asarray(beat_times, dtype=float).reshape(-1).tolist())
    raw = ",".join(f"{v:.4f}" for v in rounded)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_cache(path: str, signature: str) -> Dict | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("signature") != signature or data.get("preproc_version") != PREPROC_VERSION:
            return None  # stale / from another input → recompute
        return data
    except Exception:
        return None  # corrupt / unreadable → recompute silently


def _save_cache(path: str, signature: str, payload: Dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        record = {"signature": signature, "preproc_version": PREPROC_VERSION}
        record.update(payload)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(tmp, path)
    except Exception:
        # The cache is an optimization; failing to persist must never break the run.
        pass


# --- Shared decode ---------------------------------------------------------

@contextlib.contextmanager
def _decoded_wav(audio_file: str):
    """Yield a path to a temporary stereo WAV decoded from ``audio_file``.

    Decoding via librosa (the pipeline's decoder) to WAV removes MP3
    decoder-offset nondeterminism before the audio reaches the MLX backends, and
    keeps stem onsets aligned with the librosa-derived beat grid.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(audio_file, sr=None, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)          # mono → duplicated stereo
    data = np.ascontiguousarray(y.T)          # soundfile wants (frames, channels)

    tmp_dir = tempfile.mkdtemp(prefix="beatsync_ss_")
    wav_path = os.path.join(tmp_dir, "decoded.wav")
    try:
        sf.write(wav_path, data, int(sr))
        yield wav_path
    finally:
        for p in (wav_path, tmp_dir):
            try:
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    os.rmdir(p)
            except OSError:
                pass


@contextlib.contextmanager
def _quiet():
    """Silence the backends' progress/tqdm chatter so logs stay one-line-clean."""
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


# --- All-In-One structure weights -----------------------------------------

def _allin1_weight_files() -> List[str]:
    files: List[str] = []
    for i in range(_ALLIN1_FOLDS):
        files.append(f"harmonix-fold{i}_mlx.npz")
        files.append(f"harmonix-fold{i}_mlx.yaml")
    return files


def _weights_complete(weights_dir: str) -> bool:
    return bool(weights_dir) and all(
        os.path.isfile(os.path.join(weights_dir, name)) for name in _allin1_weight_files()
    )


def _resolve_weights_dir() -> str | None:
    """Locate an All-In-One MLX weights dir, downloading the small ensemble once
    if none is present. Returns a ready directory, or None on failure."""
    env_dir = os.environ.get(ALLIN1_WEIGHTS_ENV, "").strip()
    if env_dir:
        return env_dir if _weights_complete(env_dir) else None
    for candidate in (_REPO_ALLIN1_WEIGHTS_DIR, _DEFAULT_ALLIN1_WEIGHTS_DIR):
        if _weights_complete(candidate):
            return candidate
    # Not found anywhere → one-time download of the ~10 MB ensemble.
    if _download_allin1_weights(_DEFAULT_ALLIN1_WEIGHTS_DIR):
        return _DEFAULT_ALLIN1_WEIGHTS_DIR
    return None


def _download_allin1_weights(dest_dir: str) -> bool:
    """Best-effort one-time fetch of the 8-fold MLX weights + configs."""
    import urllib.request

    print("   ℹ️  Downloading All-In-One MLX structure weights (~10 MB, one-time)...")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        for name in _allin1_weight_files():
            out = os.path.join(dest_dir, name)
            if os.path.isfile(out):
                continue
            url = f"{_ALLIN1_WEIGHTS_BASE_URL}/{name}"
            tmp = out + ".tmp"
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                fh.write(resp.read())
            os.replace(tmp, out)
        return _weights_complete(dest_dir)
    except Exception:
        return False


# --- Public API: structure -------------------------------------------------

def analyze_structure(audio_file: str) -> dict:
    """Local music-structure labels for ``audio_file`` via All-In-One (MLX).

    Returns::

        {'available': bool, 'backend': 'all-in-one-mlx', 'seconds': float,
         'sections': [{'start': float, 'end': float, 'label': str}, ...],
         'downbeats': [float, ...], 'bpm': float | None}

    Labels are lowercase from the Harmonix set
    (start/end/intro/outro/break/bridge/inst/solo/verse/chorus). On any failure
    or when disabled/unavailable this returns ``available=False`` with empty
    lists and a single log line, and never raises.
    """
    started = time.perf_counter()

    def _unavailable() -> dict:
        return {
            "available": False,
            "backend": STRUCTURE_BACKEND,
            "seconds": round(time.perf_counter() - started, 4),
            "sections": [],
            "downbeats": [],
            "bpm": None,
        }

    if _env_on(STRUCTURE_DISABLE_ENV):
        print("   ℹ️  Music-structure analysis disabled via BEATSYNC_DISABLE_STRUCTURE; skipping.")
        return _unavailable()

    if not audio_file or not os.path.isfile(audio_file):
        print("   ℹ️  Music-structure analysis skipped: audio file not found.")
        return _unavailable()

    signature = _signature(audio_file)
    cache_path = _cache_path(audio_file, signature, "structure")
    cached = _load_cache(cache_path, signature)
    if cached is not None:
        return {
            "available": bool(cached.get("available", False)),
            "backend": STRUCTURE_BACKEND,
            "seconds": round(float(cached.get("seconds", 0.0)), 4),
            "sections": list(cached.get("sections", [])),
            "downbeats": list(cached.get("downbeats", [])),
            "bpm": cached.get("bpm"),
        }

    if not _mlx_importable():
        print("   ℹ️  All-In-One MLX backend unavailable (needs Apple Silicon / mlx); structure labels skipped.")
        return _unavailable()

    weights_dir = _resolve_weights_dir()
    if not weights_dir:
        print("   ⚠️  All-In-One MLX structure weights unavailable (download failed?); structure labels skipped.")
        return _unavailable()

    try:
        import allin1_mlx

        with _decoded_wav(audio_file) as wav_path, tempfile.TemporaryDirectory(prefix="beatsync_allin1_") as work:
            with _quiet():
                result = allin1_mlx.analyze(
                    wav_path,
                    demix_dir=os.path.join(work, "demix"),
                    spec_dir=os.path.join(work, "spec"),
                    keep_byproducts=False,
                    multiprocess=False,          # determinism / constrained env
                    mlx_weights_dir=weights_dir,
                )
        if isinstance(result, list):
            result = result[0]

        sections = []
        for seg in getattr(result, "segments", []) or []:
            sections.append({
                "start": round(float(seg.start), QUANT_DECIMALS),
                "end": round(float(seg.end), QUANT_DECIMALS),
                "label": str(seg.label).strip().lower(),
            })
        downbeats = _round_list(list(getattr(result, "downbeats", []) or []))
        bpm_raw = getattr(result, "bpm", None)
        bpm = round(float(bpm_raw), QUANT_DECIMALS) if bpm_raw is not None else None

    except Exception as e:
        print(f"   ⚠️  All-In-One MLX structure analysis failed ({type(e).__name__}: {e}); structure labels skipped.")
        return _unavailable()

    seconds = round(time.perf_counter() - started, 4)
    payload = {
        "available": True,
        "seconds": seconds,
        "sections": sections,
        "downbeats": downbeats,
        "bpm": bpm,
    }
    _save_cache(cache_path, signature, payload)
    print(
        f"   Music structure: {len(sections)} sections, {len(downbeats)} downbeats, "
        f"bpm {bpm} [{seconds:.1f}s, {STRUCTURE_BACKEND}]"
    )
    return {
        "available": True,
        "backend": STRUCTURE_BACKEND,
        "seconds": seconds,
        "sections": sections,
        "downbeats": downbeats,
        "bpm": bpm,
    }


# --- Public API: stem signals ----------------------------------------------

def get_stem_signals(audio_file: str, beat_times) -> dict:
    """Per-beat drum/vocal/bass signals from Demucs (MLX) stems.

    Returns::

        {'available': bool, 'seconds': float,
         'drum_onset':     [float] * len(beat_times),   # drum onset strength, 0..1
         'vocal_presence': [float] * len(beat_times),   # smoothed vocal RMS, 0..1
         'bass_energy':    [float] * len(beat_times)}   # bass RMS, 0..1

    On any failure or when disabled/unavailable this returns ``available=False``
    with empty lists and a single log line, and never raises.
    """
    started = time.perf_counter()
    beats = np.asarray(beat_times, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]

    def _unavailable() -> dict:
        return {
            "available": False,
            "seconds": round(time.perf_counter() - started, 4),
            "drum_onset": [],
            "vocal_presence": [],
            "bass_energy": [],
        }

    if _env_on(STEMS_DISABLE_ENV):
        print("   ℹ️  Stem signals disabled via BEATSYNC_DISABLE_STEMS; skipping.")
        return _unavailable()

    if not audio_file or not os.path.isfile(audio_file):
        print("   ℹ️  Stem signals skipped: audio file not found.")
        return _unavailable()

    if beats.size == 0:
        # No beats to annotate — nothing to compute, but this is not a failure.
        return {
            "available": True,
            "seconds": round(time.perf_counter() - started, 4),
            "drum_onset": [],
            "vocal_presence": [],
            "bass_energy": [],
        }

    signature = _signature(audio_file, extra=f"beats={_beats_hash(beats)}")
    cache_path = _cache_path(audio_file, signature, "stems")
    cached = _load_cache(cache_path, signature)
    if cached is not None:
        return {
            "available": bool(cached.get("available", False)),
            "seconds": round(float(cached.get("seconds", 0.0)), 4),
            "drum_onset": list(cached.get("drum_onset", [])),
            "vocal_presence": list(cached.get("vocal_presence", [])),
            "bass_energy": list(cached.get("bass_energy", [])),
        }

    if not _mlx_importable():
        print("   ℹ️  Demucs MLX backend unavailable (needs Apple Silicon / mlx); stem signals skipped.")
        return _unavailable()

    try:
        import librosa
        from demucs_mlx.api import Separator

        with _decoded_wav(audio_file) as wav_path:
            with _quiet():
                # shifts=0 + fixed seed → no randomized shift augmentation, so the
                # separation itself is as reproducible as MLX allows.
                separator = Separator(model=_DEMUCS_MODEL, shifts=0, seed=0, progress=False)
                _origin, stems = separator.separate_audio_file(wav_path)

        drum = _stem_mono(stems.get("drums"))
        vocal = _stem_mono(stems.get("vocals"))
        bass = _stem_mono(stems.get("bass"))
        if drum is None or vocal is None or bass is None:
            print("   ⚠️  Demucs MLX returned unexpected stems; stem signals skipped.")
            return _unavailable()

        # Drums: onset-strength envelope → sampled at each beat, robustly 0..1.
        drum_env = librosa.onset.onset_strength(y=drum, sr=_STEM_SR, hop_length=_STEM_HOP)
        drum_onset = _robust_normalize(_interp_to_beats(drum_env, beats, _STEM_SR, _STEM_HOP))

        # Vocals: frame RMS, smoothed ~0.4 s, sampled at beats, robustly 0..1.
        vocal_rms = librosa.feature.rms(
            y=vocal, frame_length=_STEM_FRAME, hop_length=_STEM_HOP
        )[0]
        smooth_frames = max(1, int(round(_VOCAL_SMOOTH_SECONDS * _STEM_SR / _STEM_HOP)))
        vocal_rms = _smooth(vocal_rms, smooth_frames)
        vocal_presence = _robust_normalize(_interp_to_beats(vocal_rms, beats, _STEM_SR, _STEM_HOP))

        # Bass: frame RMS, sampled at beats, robustly 0..1.
        bass_rms = librosa.feature.rms(
            y=bass, frame_length=_STEM_FRAME, hop_length=_STEM_HOP
        )[0]
        bass_energy = _robust_normalize(_interp_to_beats(bass_rms, beats, _STEM_SR, _STEM_HOP))

    except Exception as e:
        print(f"   ⚠️  Demucs MLX stem analysis failed ({type(e).__name__}: {e}); stem signals skipped.")
        return _unavailable()

    seconds = round(time.perf_counter() - started, 4)
    payload = {
        "available": True,
        "seconds": seconds,
        "drum_onset": _round_list(drum_onset.tolist()),
        "vocal_presence": _round_list(vocal_presence.tolist()),
        "bass_energy": _round_list(bass_energy.tolist()),
    }
    _save_cache(cache_path, signature, payload)
    print(
        f"   Stem signals: {len(beats)} beats annotated (drum/vocal/bass) "
        f"[{seconds:.1f}s, {_DEMUCS_MODEL} mlx]"
    )
    return {
        "available": True,
        "seconds": seconds,
        "drum_onset": payload["drum_onset"],
        "vocal_presence": payload["vocal_presence"],
        "bass_energy": payload["bass_energy"],
    }


def _stem_mono(stem) -> np.ndarray | None:
    """Demucs stem (channels, N) → mono float array, or None if unusable."""
    if stem is None:
        return None
    arr = np.asarray(stem, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = arr.mean(axis=0)          # (channels, N) → mono
    return np.ascontiguousarray(arr.reshape(-1))


__all__ = ["analyze_structure", "get_stem_signals"]
