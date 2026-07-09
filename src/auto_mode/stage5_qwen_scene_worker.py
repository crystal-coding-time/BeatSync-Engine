#!/usr/bin/env python3
"""Standalone Qwen3-VL semantic tagging worker via llama.cpp Vulkan.

The main app keeps Auto Mode's candidate selection and merge behavior outside
this worker. This process samples the same candidate frames and tags them with
the bundled Qwen3-VL GGUF model through llama.cpp. It prefers a persistent
llama-server process so the model loads once, and falls back to llama-mtmd-cli
when the server path is unavailable.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import concurrent.futures
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT_DIR / "bin"
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def _find_default_llama_dir() -> Path:
    bundled = BIN_DIR / "llama-bin-win-vulkan-x64"
    if bundled.exists():
        return bundled
    import shutil as _shutil
    found = _shutil.which("llama-server")
    return Path(found).resolve().parent if found else bundled


DEFAULT_LLAMA_DIR = _find_default_llama_dir()
DEFAULT_MODEL = BIN_DIR / "models" / "Qwen3VL-2B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ = BIN_DIR / "models" / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"

NUMERIC_KEYS = [
    "action_intensity",
    "beauty_score",
    "combat",
    "chase",
    "explosion",
    "character_focus",
    "camera_motion",
    "visual_quality",
]
ALLOWED_EMOTIONS = {"soft", "tension", "hype", "sad", "neutral"}
ALLOWED_USES = {"drop", "soft", "build", "transition", "flow", "filler"}
SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "number", "minimum": 0, "maximum": 1}
        for key in NUMERIC_KEYS
    },
    "required": NUMERIC_KEYS + ["emotion", "recommended_use", "description"],
    "additionalProperties": False,
}
SEMANTIC_SCHEMA["properties"].update({
    "emotion": {"type": "string", "enum": sorted(ALLOWED_EMOTIONS)},
    "recommended_use": {"type": "string", "enum": sorted(ALLOWED_USES)},
    "description": {"type": "string"},
})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _request_timeout() -> float:
    """Per-request watchdog cap once a server has completed one request."""
    if "BEATSYNC_QWEN_REQUEST_TIMEOUT" in os.environ:
        return float(_env_int("BEATSYNC_QWEN_REQUEST_TIMEOUT", 60, lo=10, hi=1800))
    # Back-compat: honor the pre-watchdog knob if someone tuned it.
    if "BEATSYNC_QWEN_LLAMA_HTTP_TIMEOUT" in os.environ:
        return float(_env_int("BEATSYNC_QWEN_LLAMA_HTTP_TIMEOUT", 240, lo=30, hi=1800))
    return 60.0


def _warmup_timeout() -> float:
    """First request per server start: model/mmproj warmup can be legitimately slow."""
    return float(_env_int("BEATSYNC_QWEN_WARMUP_TIMEOUT", 180, lo=30, hi=1800))


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        return "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


class QwenRequestTimeout(RuntimeError):
    """A single /v1/chat/completions call exceeded its watchdog timeout."""


class ServerWedgedError(RuntimeError):
    """Circuit breaker verdict: llama-server accepts requests but never completes them."""


class RequestTimeoutBreaker:
    """Trips after N consecutive llama-server request timeouts.

    Motivation (2026-07-07 incident): llama-server stayed 'healthy' on /health
    while its first chat completion hung forever, freezing a render for ~a day.
    With this breaker a wedged server costs minutes: one restart is allowed
    after the threshold is first reached; if timeouts continue, the breaker
    opens for good and the worker exits with whatever succeeded instead of
    grinding through every remaining candidate at the per-request timeout.
    Any successful request resets the consecutive count, so healthy-but-slow
    traffic never trips it.
    """

    def __init__(self) -> None:
        self.threshold = _env_int("BEATSYNC_QWEN_TIMEOUT_BREAKER", 6, lo=1, hi=100)
        self._lock = threading.Lock()
        self._consecutive = 0
        self._restart_used = False
        self._tripped = False

    def record_timeout(self) -> bool:
        with self._lock:
            self._consecutive += 1
            return self._tripped or self._consecutive >= self.threshold

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def at_threshold(self) -> bool:
        with self._lock:
            return self._tripped or self._consecutive >= self.threshold

    def consume_restart(self) -> bool:
        with self._lock:
            if self._restart_used or self._tripped:
                return False
            self._restart_used = True
            return True

    def reset_after_restart(self) -> None:
        with self._lock:
            self._consecutive = 0

    def trip(self) -> None:
        with self._lock:
            self._tripped = True

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped


# Every live llama-server Popen is registered here so termination signals and
# atexit can always reach it: the server must never outlive this worker.
_SERVER_PROCESSES: set = set()
_SERVER_PROCESSES_LOCK = threading.Lock()


def _register_server_process(process: subprocess.Popen) -> None:
    with _SERVER_PROCESSES_LOCK:
        _SERVER_PROCESSES.add(process)


def _unregister_server_process(process: subprocess.Popen) -> None:
    with _SERVER_PROCESSES_LOCK:
        _SERVER_PROCESSES.discard(process)


def _shutdown_server_processes() -> None:
    with _SERVER_PROCESSES_LOCK:
        processes = list(_SERVER_PROCESSES)
    for process in processes:
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 10.0
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _handle_termination_signal(signum, _frame) -> None:
    # Inference threads may be blocked inside urlopen(), so an ordinary
    # SystemExit could hang on executor shutdown. Stop llama-server first,
    # then exit hard so the server can never be orphaned by a worker kill.
    print(
        f"Qwen worker received signal {signum}; stopping llama-server and exiting",
        file=sys.stderr,
        flush=True,
    )
    _shutdown_server_processes()
    os._exit(128 + int(signum))


def _install_signal_handlers() -> None:
    # SIGBREAK is Windows-only; SIGHUP is POSIX-only. Register whatever exists.
    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_termination_signal)
        except (ValueError, OSError, RuntimeError):
            pass


def _parse_json_object(text: str) -> Dict:
    if not text:
        return {}
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _qwen_frame_width() -> int:
    return _env_int("BEATSYNC_QWEN_FRAME_WIDTH", 512, lo=224, hi=768)


def _max_new_tokens() -> int:
    return _env_int("BEATSYNC_QWEN_MAX_NEW_TOKENS", 128, lo=32, hi=256)


def _ctx_sizes() -> List[int]:
    primary = _env_int("BEATSYNC_QWEN_LLAMA_CTX", 8192, lo=1024, hi=262144)
    fallback = _env_int("BEATSYNC_QWEN_LLAMA_CTX_FALLBACK", 4096, lo=1024, hi=262144)
    sizes = []
    for value in [primary, fallback]:
        if value not in sizes:
            sizes.append(value)
    return sizes


def _llama_slots() -> int:
    return _env_int("BEATSYNC_QWEN_LLAMA_SLOTS", 16, lo=1, hi=32)


def _resize_frame(frame, max_width: int = 512):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)


def _candidate_mid_frame(fps: float, candidate: Dict) -> int:
    start = float(candidate.get("start", 0.0))
    end = float(candidate.get("end", start))
    t = start + max(0.01, end - start) * 0.5
    return max(0, int(round(t * fps)))


def _frame_to_image(frame, max_width: int) -> Image.Image | None:
    if frame is None:
        return None
    frame = _resize_frame(frame, max_width=max_width)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_frame(cap, fps: float, candidate: Dict, max_width: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, _candidate_mid_frame(fps, candidate))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _frame_to_image(frame, max_width=max_width)


def _prefetch_candidate_frames(cap, fps: float, candidates: List[Dict], max_width: int) -> List[Dict]:
    """Decode Qwen sample frames in timeline order and keep them in RAM."""
    started = time.perf_counter()
    if os.environ.get("BEATSYNC_QWEN_PREFETCH_FRAMES", "1") == "0":
        items = [{"candidate": c, "image": _extract_frame(cap, fps, c, max_width)} for c in candidates]
        ready = [item for item in items if item.get("image") is not None]
        elapsed = max(0.001, time.perf_counter() - started)
        print(
            f"Qwen frame prefetch disabled: {len(ready)}/{len(candidates)} frames "
            f"via direct seek at max width {max_width} in {elapsed:.1f}s",
            flush=True,
        )
        return ready

    try:
        max_gap = int(os.environ.get("BEATSYNC_QWEN_ORDERED_MAX_GAP", "96"))
    except ValueError:
        max_gap = 96
    max_gap = max(0, max_gap)

    plans = []
    for original_index, candidate in enumerate(candidates):
        plans.append({
            "original_index": original_index,
            "candidate": candidate,
            "frame_idx": _candidate_mid_frame(fps, candidate),
            "image": None,
        })

    seek_count = 0
    grab_count = 0
    current = None
    for plan in sorted(plans, key=lambda item: item["frame_idx"]):
        target = int(plan["frame_idx"])
        if current is None or target < current or target - current > max_gap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
            seek_count += 1
        while current < target:
            if not cap.grab():
                break
            current += 1
            grab_count += 1
        if current != target:
            continue
        ok, frame = cap.read()
        current += 1
        if ok and frame is not None:
            plan["image"] = _frame_to_image(frame, max_width=max_width)

    plans.sort(key=lambda item: item["original_index"])
    ready = [p for p in plans if p.get("image") is not None]
    elapsed = max(0.001, time.perf_counter() - started)
    approx_ram_mb = sum(item["image"].width * item["image"].height * 3 for item in ready) / (1024 * 1024)
    print(
        f"Qwen frame prefetch: {len(ready)}/{len(candidates)} frames in RAM "
        f"(~{approx_ram_mb:.0f} MB, {seek_count} seeks, {grab_count} grabs, "
        f"max gap {max_gap}, max width {max_width}) in {elapsed:.1f}s",
        flush=True,
    )
    return ready


def _configure_opencv_ffmpeg_threads() -> int:
    """Let OpenCV/FFmpeg use more decoder threads for frame prefetch."""
    cpu = os.cpu_count() or 4
    try:
        default_threads = min(8, max(2, cpu // 2))
        threads = int(os.environ.get("BEATSYNC_OPENCV_FFMPEG_THREADS", str(default_threads)))
    except ValueError:
        threads = min(8, max(2, cpu // 2))
    threads = max(1, min(threads, max(1, cpu)))

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"threads;{threads}")
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    return threads


def _normalize_semantic(data: Dict) -> Dict:
    if not isinstance(data, dict):
        return {}

    out = {}
    for key in NUMERIC_KEYS:
        if key not in data:
            return {}
        out[key] = _clamp(data[key])

    emotion = str(data.get("emotion", "")).strip().lower()
    if emotion not in ALLOWED_EMOTIONS:
        return {}
    recommended_use = str(data.get("recommended_use", "")).strip().lower()
    if recommended_use not in ALLOWED_USES:
        return {}

    description = str(data.get("description", "")).strip()
    if not description:
        return {}
    out["emotion"] = emotion
    out["recommended_use"] = recommended_use
    out["description"] = description[:160]
    return out


def _semantic_from_text(text: str) -> Dict:
    return _normalize_semantic(_parse_json_object(text))


def _build_prompt(audio_profile: Dict) -> str:
    style_hint = audio_profile.get("smart_preset", "rhythmic_gmv_amv")
    return (
        "You are tagging one source-video moment for professional AMV/GMV editing. "
        f"The music edit style is {style_hint}. "
        "Return JSON only. Keys: action_intensity, beauty_score, combat, chase, explosion, "
        "character_focus, camera_motion, visual_quality as numbers 0..1; "
        "emotion as one of soft,tension,hype,sad,neutral; "
        "recommended_use as one of drop,soft,build,transition,flow,filler; "
        "description under 12 words. Do not include markdown."
    )


def _safe_file_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "item"


def _image_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_data_uri(image: Image.Image) -> str:
    encoded = base64.b64encode(_image_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_prefix(response_path: str) -> Path:
    return Path(response_path).with_suffix("")


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def _append_log(path: Path, title: str, text: str) -> None:
    if not text:
        return
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n----- {title} -----\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _is_context_or_memory_error(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in [
        "out of memory",
        "memory allocation",
        "failed to allocate",
        "context",
        "ctx",
        "kv cache",
        "vram",
    ])


class LlamaPaths:
    def __init__(self, model_path: str | None) -> None:
        self.llama_dir = Path(os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", str(DEFAULT_LLAMA_DIR)))
        self.server_exe = self.llama_dir / f"llama-server{EXE_SUFFIX}"
        self.mtmd_exe = self.llama_dir / f"llama-mtmd-cli{EXE_SUFFIX}"
        self.list_exe = self.llama_dir / f"llama-cli{EXE_SUFFIX}"
        self.model = self._resolve_model(model_path)
        self.mmproj = self._resolve_mmproj()

    def _resolve_model(self, model_path: str | None) -> Path:
        env_model = os.environ.get("BEATSYNC_QWEN_LLAMA_MODEL")
        if env_model:
            return Path(env_model)
        if model_path:
            requested = Path(model_path)
            if requested.is_file() and requested.suffix.lower() == ".gguf":
                return requested
            candidates = [
                requested / DEFAULT_MODEL.name,
                requested.parent / DEFAULT_MODEL.name,
                DEFAULT_MODEL,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return DEFAULT_MODEL

    def _resolve_mmproj(self) -> Path:
        env_mmproj = os.environ.get("BEATSYNC_QWEN_LLAMA_MMPROJ")
        if env_mmproj:
            return Path(env_mmproj)
        candidates = [
            self.model.parent / DEFAULT_MMPROJ.name,
            DEFAULT_MMPROJ,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return DEFAULT_MMPROJ

    def validate(self) -> None:
        missing = [
            str(path)
            for path in [self.server_exe, self.mtmd_exe, self.model, self.mmproj]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing llama.cpp/Qwen files: " + "; ".join(missing))

    @property
    def model_id(self) -> str:
        return self.model.stem


def _llama_env(paths: LlamaPaths) -> Dict[str, str]:
    env = os.environ.copy()
    path_parts = [str(paths.llama_dir)]
    for part in env.get("PATH", "").split(os.pathsep):
        if part and os.path.normcase(os.path.abspath(part)) != os.path.normcase(str(paths.llama_dir)):
            path_parts.append(part)
    env["PATH"] = os.pathsep.join(path_parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _parse_devices(text: str) -> List[Dict[str, Any]]:
    devices = []
    pattern = re.compile(r"^\s*(Vulkan\d+):\s*(.+?)\s*\((\d+)\s+MiB,\s*(\d+)\s+MiB free\)", re.I)
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(2).strip()
        lower = name.lower()
        integrated = any(token in lower for token in [
            "radeon(tm) graphics",
            "integrated",
            "uhd",
            "iris",
            "vega",
        ])
        discrete = any(token in lower for token in [
            "nvidia",
            "geforce",
            "rtx",
            "gtx",
            "quadro",
            "radeon rx",
            "arc",
        ]) and not integrated
        devices.append({
            "id": match.group(1),
            "name": name,
            "total_mib": int(match.group(3)),
            "free_mib": int(match.group(4)),
            "discrete": discrete,
            "integrated": integrated,
        })
    return devices


def _adaptive_llama_slots(device: Dict[str, Any] | None = None) -> int:
    if "BEATSYNC_QWEN_LLAMA_SLOTS" in os.environ:
        return _llama_slots()
    if not device:
        return 4
    free_mib = int(device.get("free_mib") or 0)
    if free_mib >= 14 * 1024:
        return 16
    if free_mib >= 10 * 1024:
        return 8
    if free_mib >= 6 * 1024:
        return 4
    return 2


def _select_vulkan_device_info(paths: LlamaPaths) -> Tuple[str | None, str, Dict[str, Any] | None]:
    override = os.environ.get("BEATSYNC_QWEN_LLAMA_DEVICE", "").strip()
    if override:
        if override.lower() in {"none", "cpu"}:
            return None, "CPU/no Vulkan override", None
        return override, f"{override} (env override)", None

    try:
        result = subprocess.run(
            [str(paths.list_exe), "--list-devices"],
            cwd=str(paths.llama_dir),
            env=_llama_env(paths),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return None, f"device list unavailable: {exc}", None

    devices = _parse_devices((result.stdout or "") + "\n" + (result.stderr or ""))
    if not devices:
        return None, "no Vulkan devices reported", None

    discrete = [device for device in devices if device["discrete"]]
    pool = discrete or devices
    selected = max(pool, key=lambda item: (item["free_mib"], item["total_mib"]))
    return selected["id"], f"{selected['id']}: {selected['name']}", selected


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(method: str, url: str, payload: Dict | None = None, timeout: float = 30.0) -> Dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    return json.loads(raw)


class LlamaServerClient:
    def __init__(
        self,
        paths: LlamaPaths,
        device: str | None,
        ctx_size: int,
        response_path: str,
        slots: int,
        breaker: RequestTimeoutBreaker | None = None,
    ) -> None:
        self.paths = paths
        self.device = device
        self.ctx_size = ctx_size
        self.port = _free_tcp_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.slots = max(1, min(32, int(slots)))
        self.breaker = breaker
        self._success_count = 0
        self._count_lock = threading.Lock()
        prefix = _response_prefix(response_path)
        self.stdout_path = prefix.with_name(prefix.name + f"_llama_server_ctx{ctx_size}_stdout.log")
        self.stderr_path = prefix.with_name(prefix.name + f"_llama_server_ctx{ctx_size}_stderr.log")
        self.server_log_path = prefix.with_name(prefix.name + f"_llama_server_ctx{ctx_size}.log")
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8", errors="replace")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8", errors="replace")
        self.process: subprocess.Popen | None = None
        self.load_seconds = 0.0
        self._start()

    def _start(self) -> None:
        args = [
            str(self.paths.server_exe),
            "-m", str(self.paths.model),
            "--mmproj", str(self.paths.mmproj),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--gpu-layers", "all",
            "--split-mode", "none",
            "--parallel", str(self.slots),
            "--cont-batching",
            "--timeout", "3600",
            "--mmproj-offload",
            "--reasoning", "off",
            # Defensive flags (2026-07-07 wedge incident): recent llama.cpp
            # builds (Homebrew b9870) enable prompt-cache RAM, context
            # checkpoints and flash-attn by default; upstream issues
            # ggml-org/llama.cpp #24265, #17297 and #20921 tie those
            # subsystems to intermittent mid-prompt stalls while /health
            # stays ok. Disabling them costs ~15% gen speed; wave throughput
            # is unchanged.
            "--cache-ram", "0",
            "--ctx-checkpoints", "0",
            "--flash-attn", "off",
            # llama-server stdout is block-buffered when piped, so a killed
            # server leaves 0-byte stdout logs. --log-file writes diagnostics
            # directly so a wedge is diagnosable after the fact.
            "--log-file", str(self.server_log_path),
            "--alias", "qwen3vl",
            "--log-verbosity", "1",
            "--no-log-prefix",
        ]
        if self.device:
            args.extend(["--device", self.device])

        started = time.perf_counter()
        self.process = subprocess.Popen(
            args,
            cwd=str(self.paths.llama_dir),
            env=_llama_env(self.paths),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _register_server_process(self.process)

        ready_timeout = _env_int("BEATSYNC_QWEN_LLAMA_SERVER_READY_TIMEOUT", 180, lo=15, hi=900)
        deadline = time.perf_counter() + ready_timeout
        last_error = ""
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.process.returncode}; "
                    f"stderr: {_tail(self.stderr_path)}"
                )
            for endpoint in ["/health", "/v1/models"]:
                try:
                    _http_json("GET", self.base_url + endpoint, timeout=2.0)
                    self.load_seconds = time.perf_counter() - started
                    return
                except Exception as exc:
                    last_error = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"llama-server readiness timed out after {ready_timeout}s: {last_error}")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process is not None:
            _unregister_server_process(process)
        self._stdout_handle.close()
        self._stderr_handle.close()

    def _next_request_timeout(self) -> float:
        # First request per server start gets the generous warmup allowance;
        # after one success the tight per-request cap applies.
        with self._count_lock:
            warmed = self._success_count > 0
        return _request_timeout() if warmed else _warmup_timeout()

    def _post_chat_completion(self, payload: Dict) -> Dict:
        # Per-request watchdog: one bounded retry on timeout, then the item
        # fails into the wave's failed-item path (deterministic tags remain).
        last_timeout = 0.0
        for attempt in (1, 2):
            if self.breaker and self.breaker.at_threshold():
                raise ServerWedgedError("llama-server circuit breaker is open")
            timeout = self._next_request_timeout()
            last_timeout = timeout
            try:
                data = _http_json(
                    "POST", self.base_url + "/v1/chat/completions", payload, timeout=timeout
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server HTTP {exc.code}: {body}") from exc
            except Exception as exc:
                if not _is_timeout_error(exc):
                    raise
                at_threshold = self.breaker.record_timeout() if self.breaker else False
                print(
                    f"Qwen watchdog: chat completion timed out after {timeout:.0f}s "
                    f"(attempt {attempt}/2)",
                    flush=True,
                )
                if attempt == 1 and not at_threshold:
                    continue
                raise QwenRequestTimeout(
                    f"chat completion timed out after {timeout:.0f}s"
                ) from exc
            if self.breaker:
                self.breaker.record_success()
            with self._count_lock:
                self._success_count += 1
            return data
        raise QwenRequestTimeout(f"chat completion timed out after {last_timeout:.0f}s")

    def generate(self, image: Image.Image, prompt: str) -> str:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("llama-server is not running")
        payload = {
            "model": "qwen3vl",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "max_tokens": _max_new_tokens(),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "beatsync_semantic_tag",
                    "strict": True,
                    "schema": SEMANTIC_SCHEMA,
                },
            },
        }
        data = self._post_chat_completion(payload)

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)


class LlamaMtmdClient:
    def __init__(self, paths: LlamaPaths, device: str | None, ctx_size: int, response_path: str) -> None:
        self.paths = paths
        self.device = device
        self.ctx_size = ctx_size
        prefix = _response_prefix(response_path)
        self.stdout_path = prefix.with_name(prefix.name + "_llama_cli_stdout.log")
        self.stderr_path = prefix.with_name(prefix.name + "_llama_cli_stderr.log")
        self.frame_dir = prefix.with_name(prefix.name + "_llama_frames")
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image: Image.Image, prompt: str, item_id: str) -> str:
        image_path = self.frame_dir / f"{_safe_file_token(item_id)}.png"
        image.save(image_path, format="PNG")
        args = [
            str(self.paths.mtmd_exe),
            "-m", str(self.paths.model),
            "--mmproj", str(self.paths.mmproj),
            "--image", str(image_path),
            "-p", prompt,
            "-n", str(_max_new_tokens()),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--gpu-layers", "all",
            "--split-mode", "none",
            "--mmproj-offload",
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "1",
            "--min-p", "0",
            "--json-schema", json.dumps(SEMANTIC_SCHEMA, separators=(",", ":")),
            "--no-warmup",
            "--log-verbosity", "1",
            "--no-log-prefix",
        ]
        if self.device:
            args.extend(["--device", self.device])

        timeout = _env_int("BEATSYNC_QWEN_LLAMA_CLI_TIMEOUT", 300, lo=30, hi=3600)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                args,
                cwd=str(self.paths.llama_dir),
                env=_llama_env(self.paths),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            _append_log(self.stderr_path, f"{item_id} timeout", f"Timed out after {timeout}s")
            return ""

        elapsed = time.perf_counter() - started
        _append_log(self.stdout_path, f"{item_id} stdout ({elapsed:.1f}s)", result.stdout or "")
        _append_log(self.stderr_path, f"{item_id} stderr ({elapsed:.1f}s)", result.stderr or "")
        if result.returncode == 3221225786:
            _append_log(
                self.stderr_path,
                f"{item_id} interrupted",
                "llama-mtmd-cli exited with 3221225786 / 0xC000013A, usually an interruption.",
            )
            return ""
        if result.returncode != 0:
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            fallback_ctx = _ctx_sizes()[-1]
            if self.ctx_size != fallback_ctx and _is_context_or_memory_error(combined):
                _append_log(
                    self.stderr_path,
                    f"{item_id} ctx fallback",
                    f"Retrying llama-mtmd-cli with ctx {fallback_ctx} after exit code {result.returncode}.",
                )
                self.ctx_size = fallback_ctx
                return self.generate(image, prompt, item_id)
            _append_log(self.stderr_path, f"{item_id} exit", f"Exit code {result.returncode}")
            return ""
        return result.stdout or ""


class QwenLlamaClient:
    def __init__(self, model_path: str | None, response_path: str) -> None:
        self.paths = LlamaPaths(model_path)
        self.paths.validate()
        self.device, self.device_label, self.device_info = _select_vulkan_device_info(self.paths)
        self.target_slots = _adaptive_llama_slots(self.device_info)
        self.ctx_size = _ctx_sizes()[0]
        self.server: LlamaServerClient | None = None
        self.cli: LlamaMtmdClient | None = None
        self.load_seconds = 0.0
        self.batch_size = self.target_slots
        self.response_path = response_path
        self.breaker = RequestTimeoutBreaker()
        print(f"Qwen llama.cpp model: {self.paths.model.name}", flush=True)
        print(f"Qwen llama.cpp mmproj: {self.paths.mmproj.name}", flush=True)
        print(f"Qwen llama.cpp Vulkan device: {self.device_label}", flush=True)
        print(f"Qwen llama.cpp target slots: {self.target_slots}", flush=True)
        self._start_server_or_prepare_cli()

    @property
    def model_id(self) -> str:
        return f"{self.paths.model_id} (llama.cpp Vulkan)"

    def _start_server_or_prepare_cli(self) -> None:
        if os.environ.get("BEATSYNC_QWEN_LLAMA_DISABLE_SERVER", "0") == "1":
            self._prepare_cli(_ctx_sizes()[0])
            print("Qwen llama.cpp server disabled; using llama-mtmd-cli fallback", flush=True)
            return

        last_error = ""
        for ctx_size in _ctx_sizes():
            try:
                self.server = LlamaServerClient(
                    self.paths,
                    self.device,
                    ctx_size,
                    self.response_path,
                    self.target_slots,
                    breaker=self.breaker,
                )
                self.ctx_size = ctx_size
                self.load_seconds = self.server.load_seconds
                self.batch_size = self.server.slots
                print(
                    f"Qwen llama-server ready: ctx {ctx_size}, slots {self.batch_size}, "
                    f"load {self.load_seconds:.1f}s",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = str(exc)
                print(f"Qwen llama-server ctx {ctx_size} failed: {last_error[-600:]}", flush=True)
                if self.server:
                    self.server.close()
                    self.server = None
        self._prepare_cli(_ctx_sizes()[-1])
        print(f"Qwen falling back to llama-mtmd-cli: {last_error[-600:]}", flush=True)

    def _prepare_cli(self, ctx_size: int) -> None:
        self.server = None
        self.ctx_size = ctx_size
        self.batch_size = 1
        self.cli = LlamaMtmdClient(self.paths, self.device, ctx_size, self.response_path)

    def restart_server_with_slots(self, slots: int) -> bool:
        if os.environ.get("BEATSYNC_QWEN_LLAMA_DISABLE_SERVER", "0") == "1":
            return False
        if self.server:
            self.server.close()
            self.server = None
        self.cli = None
        self.target_slots = max(1, min(32, int(slots)))
        try:
            self.server = LlamaServerClient(
                self.paths,
                self.device,
                self.ctx_size,
                self.response_path,
                self.target_slots,
                breaker=self.breaker,
            )
            self.load_seconds += self.server.load_seconds
            self.batch_size = self.server.slots
            print(
                f"Qwen llama-server restarted: ctx {self.ctx_size}, slots {self.batch_size}, "
                f"load {self.server.load_seconds:.1f}s",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"Qwen llama-server restart failed: {str(exc)[-600:]}", flush=True)
            if self.server:
                self.server.close()
                self.server = None
            self._prepare_cli(self.ctx_size)
            return False

    def generate(self, image: Image.Image, prompt: str, item_id: str) -> str:
        if self.server:
            try:
                return self.server.generate(image, prompt)
            except (ServerWedgedError, QwenRequestTimeout):
                # Watchdog timeouts are the circuit breaker's business
                # (handled at the wave level). Falling back to llama-mtmd-cli
                # here would grind every candidate at the CLI timeout instead.
                raise
            except Exception as exc:
                fallback_ctx = _ctx_sizes()[-1]
                if self.ctx_size != fallback_ctx and _is_context_or_memory_error(str(exc)):
                    print(
                        f"Qwen llama-server request hit context/VRAM limits; retrying ctx {fallback_ctx}",
                        flush=True,
                    )
                    self.server.close()
                    self.server = None
                    try:
                        self.server = LlamaServerClient(
                            self.paths,
                            self.device,
                            fallback_ctx,
                            self.response_path,
                            self.target_slots,
                            breaker=self.breaker,
                        )
                        self.ctx_size = fallback_ctx
                        self.load_seconds += self.server.load_seconds
                        self.batch_size = self.server.slots
                        return self.server.generate(image, prompt)
                    except Exception as retry_exc:
                        print(f"Qwen llama-server ctx {fallback_ctx} retry failed: {retry_exc}", flush=True)
                print(f"Qwen llama-server request failed; switching to CLI fallback: {exc}", flush=True)
                if self.server:
                    self.server.close()
                self.server = None
                self._prepare_cli(self.ctx_size)
        if not self.cli:
            self._prepare_cli(self.ctx_size)
        return self.cli.generate(image, prompt, item_id)

    def close(self) -> None:
        if self.server:
            self.server.close()
            self.server = None


def _candidate_id(item: Dict, fallback_index: int = 0) -> str:
    return str(item["candidate"].get("id") or fallback_index)


def _generate_with_server(
    client: QwenLlamaClient,
    item: Dict,
    prompt: str,
    fallback_index: int = 0,
) -> Tuple[str, Dict, str]:
    item_id = _candidate_id(item, fallback_index)
    if not client.server:
        return item_id, {}, "llama-server is not running"
    try:
        text = client.server.generate(item["image"], prompt)
    except Exception as exc:
        return item_id, {}, str(exc)
    return item_id, _semantic_from_text(text), ""


def _generate_serial(
    client: QwenLlamaClient,
    item: Dict,
    prompt: str,
    fallback_index: int = 0,
) -> Tuple[str, Dict, str]:
    item_id = _candidate_id(item, fallback_index)
    try:
        text = client.generate(item["image"], prompt, item_id)
    except Exception as exc:
        return item_id, {}, str(exc)
    return item_id, _semantic_from_text(text), ""


def _error_category(error: str) -> str:
    """Bucket a failed-item error string for the end-of-video Counter log."""
    if not error:
        return "invalid_response"
    lowered = error.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "circuit breaker" in lowered or "wedged" in lowered:
        return "breaker_open"
    if "not running" in lowered:
        return "server_unavailable"
    if "http" in lowered:
        return "http_error"
    return "other"


class _TagWindow:
    """Retry/ratio accounting unit: one 'wave' of the old wave scheduler.

    The persistent-executor scheduler pipelines submissions across windows so
    a slow item can no longer idle the other llama-server slots, but retry
    grants, the breaker group, and the valid-ratio trigger still operate on
    windows of `batch_size` items so the recovery semantics stay those of the
    old wave loop.
    """

    __slots__ = ("items", "unsubmitted", "outstanding", "results", "failed", "retried")

    def __init__(self, items: List[Tuple[int, Dict]]) -> None:
        self.items = list(items)
        self.unsubmitted: deque = deque(self.items)
        self.outstanding = 0
        self.results: Dict[str, Dict] = {}
        self.failed: List[Tuple[int, Dict, str]] = []
        self.retried = False

    def unfinished_items(self) -> List[Tuple[int, Dict]]:
        return [
            (fallback_index, item)
            for fallback_index, item in self.items
            if _candidate_id(item, fallback_index) not in self.results
        ]


def _serial_sweep(
    client: QwenLlamaClient,
    window: _TagWindow,
    prompt: str,
    error_counts: Counter,
) -> List[Tuple[int, Dict, str]]:
    """No-llama-server path: one llama-mtmd-cli attempt per failed item."""
    still_failed: List[Tuple[int, Dict, str]] = []
    for fallback_index, item, _error in window.failed:
        item_id, semantic, error = _generate_serial(client, item, prompt, fallback_index)
        if semantic:
            window.results[item_id] = semantic
        else:
            error_counts[_error_category(error)] += 1
            still_failed.append((fallback_index, item, error))
    return still_failed


def _serial_tag_item(
    client: QwenLlamaClient,
    fallback_index: int,
    item: Dict,
    prompt: str,
    error_counts: Counter,
) -> Tuple[Dict[str, Dict], bool]:
    """One item through the old serial wave-of-one logic (batch size 1 / CLI).

    Returns ({item_id: semantic} or {}, requeue): requeue means the breaker's
    one-shot server restart succeeded and the item must be re-run. Raises
    ServerWedgedError when the breaker trips for good.
    """
    item_id, semantic, error = _generate_serial(client, item, prompt, fallback_index)
    if semantic:
        return {item_id: semantic}, False
    error_counts[_error_category(error)] += 1

    if client.server and not client.breaker.at_threshold():
        item_id, semantic, error = _generate_with_server(client, item, prompt, fallback_index)
        if semantic:
            return {item_id: semantic}, False
        error_counts[_error_category(error)] += 1

    if client.breaker.at_threshold():
        if client.server and client.breaker.consume_restart():
            reduced_slots = max(1, int(client.batch_size or 1) // 2)
            print(
                f"Qwen watchdog: {client.breaker.threshold} consecutive request timeouts; "
                f"restarting llama-server once with {reduced_slots} slot(s) before giving up",
                flush=True,
            )
            if client.restart_server_with_slots(reduced_slots):
                client.breaker.reset_after_restart()
                return {}, True
        client.breaker.trip()
        client.close()
        print(
            "Qwen watchdog: llama-server is wedged (accepts requests but never completes them); "
            "aborting semantic tagging with partial results",
            flush=True,
        )
        raise ServerWedgedError("llama-server wedged: consecutive request timeouts")

    if not client.server:
        item_id, semantic, error = _generate_serial(client, item, prompt, fallback_index)
        if semantic:
            return {item_id: semantic}, False
        error_counts[_error_category(error)] += 1
    return {}, False


def _run_concurrent_phase(
    client: QwenLlamaClient,
    executor: concurrent.futures.ThreadPoolExecutor,
    pending_items: deque,
    pending_groups: deque,
    prompt: str,
    semantics: Dict[str, Dict],
    error_counts: Counter,
    progress: Dict[str, int],
    report,
) -> None:
    """Run queued items at the current slot count until done or a restart.

    Submission is continuous: a BoundedSemaphore sized to the slot count is
    acquired before each submit and released as each request finishes, so the
    server always has work without unbounded queueing (the old code had a
    hard barrier after every `batch_size` items). Returns normally when every
    queued item is finalized, or after a breaker/ratio server restart with
    the re-queued work left on pending_items/pending_groups for the caller's
    loop (this loop replaces the old recursion). Raises ServerWedgedError
    when the breaker trips for good.
    """
    concurrency = max(1, int(client.batch_size or 1))
    sem = threading.BoundedSemaphore(concurrency)
    in_flight: Dict[concurrent.futures.Future, Tuple[_TagWindow, int, Dict]] = {}
    retry_queue: deque = deque()
    open_windows: List[_TagWindow] = []
    breaker_windows: List[_TagWindow] = []
    halve_windows: List[_TagWindow] = []
    fill_state: Dict[str, _TagWindow | None] = {"window": None}
    stop = {"new_submissions": False}

    def _next_submission():
        if retry_queue:
            return retry_queue.popleft()
        window = fill_state["window"]
        if window is None or not window.unsubmitted:
            if pending_groups:
                window = _TagWindow(pending_groups.popleft())
            elif pending_items:
                window = _TagWindow([
                    pending_items.popleft()
                    for _ in range(min(concurrency, len(pending_items)))
                ])
            else:
                return None
            open_windows.append(window)
            fill_state["window"] = window
        fallback_index, item = window.unsubmitted.popleft()
        return window, fallback_index, item

    def _submit(window: _TagWindow, fallback_index: int, item: Dict) -> None:
        window.outstanding += 1

        def _task():
            try:
                return _generate_with_server(client, item, prompt, fallback_index)
            finally:
                sem.release()

        in_flight[executor.submit(_task)] = (window, fallback_index, item)

    def _evaluate(window: _TagWindow) -> None:
        # First round done: grant the single per-item retry under the same
        # condition the old wave scheduler checked at wave end.
        if window.failed and not window.retried:
            window.retried = True
            if client.server and not client.breaker.at_threshold():
                retries, window.failed = window.failed, []
                for fallback_index, item, _error in retries:
                    retry_queue.append((window, fallback_index, item))
                return
        open_windows.remove(window)
        # Same evaluation order as the old wave: breaker group first so a
        # wedged server cannot ping-pong through slot-halving restarts.
        if window.failed and client.breaker.at_threshold():
            breaker_windows.append(window)
            stop["new_submissions"] = True
            return
        valid_ratio = len(window.results) / max(1, len(window.items))
        if window.failed and client.server and client.batch_size > 1 and valid_ratio < 0.70:
            halve_windows.append(window)
            stop["new_submissions"] = True
            return
        if window.failed and not client.server:
            window.failed = _serial_sweep(client, window, prompt, error_counts)
        report(len(window.items), window.results)

    while True:
        while not stop["new_submissions"]:
            if not sem.acquire(blocking=False):
                break
            submission = _next_submission()
            if submission is None:
                sem.release()
                break
            _submit(*submission)
        if not in_flight:
            break
        done, _ = concurrent.futures.wait(
            set(in_flight), return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            window, fallback_index, item = in_flight.pop(future)
            try:
                item_id, semantic, error = future.result()
            except Exception as exc:  # defensive: _generate_with_server catches
                item_id, semantic, error = _candidate_id(item, fallback_index), {}, str(exc)
            window.outstanding -= 1
            if semantic:
                window.results[item_id] = semantic
            else:
                error_counts[_error_category(error)] += 1
                window.failed.append((fallback_index, item, error))
            if (
                window.outstanding == 0
                and not window.unsubmitted
                and not any(entry[0] is window for entry in retry_queue)
            ):
                _evaluate(window)

    if not breaker_windows and not halve_windows:
        return  # every queued item finalized at this slot count

    # A restart is needed and in-flight work is drained. Windows interrupted
    # mid-round keep their successes and re-queue the rest; the old scheduler
    # had not started those items yet, and generation is deterministic per
    # candidate, so the outcome is the same either way.
    interrupted = list(open_windows)
    open_windows.clear()
    retry_queue.clear()

    def _commit_and_requeue_interrupted() -> None:
        for window in reversed(interrupted):
            semantics.update(window.results)
            progress["finalized"] += len(window.results)
            for pair in reversed(window.unfinished_items()):
                pending_items.appendleft(pair)

    if breaker_windows:
        # Old step order preserved: one restart at half slots, re-queue the
        # unfinished items (successes kept, like the old recursion on the
        # failed list); a second threshold event trips for good.
        if client.server and client.breaker.consume_restart():
            reduced_slots = max(1, int(client.batch_size or 1) // 2)
            print(
                f"Qwen watchdog: {client.breaker.threshold} consecutive request timeouts; "
                f"restarting llama-server once with {reduced_slots} slot(s) before giving up",
                flush=True,
            )
            if client.restart_server_with_slots(reduced_slots):
                client.breaker.reset_after_restart()
                _commit_and_requeue_interrupted()
                for window in reversed(halve_windows):
                    # Piggyback on the breaker restart's slot reduction.
                    pending_groups.appendleft(list(window.items))
                for window in reversed(breaker_windows):
                    semantics.update(window.results)
                    progress["finalized"] += len(window.results)
                    pending_groups.appendleft([
                        (fallback_index, item)
                        for fallback_index, item, _error in window.failed
                    ])
                return
        client.breaker.trip()
        client.close()
        print(
            "Qwen watchdog: llama-server is wedged (accepts requests but never completes them); "
            "aborting semantic tagging with partial results",
            flush=True,
        )
        raise ServerWedgedError("llama-server wedged: consecutive request timeouts")

    trigger = halve_windows[0]
    valid_ratio = len(trigger.results) / max(1, len(trigger.items))
    reduced_slots = max(1, int(client.batch_size) // 2)
    print(
        f"Qwen llama.cpp wave valid ratio {valid_ratio:.0%}; retrying with {reduced_slots} slots",
        flush=True,
    )
    if client.restart_server_with_slots(reduced_slots):
        _commit_and_requeue_interrupted()
        for window in reversed(halve_windows):
            # Re-run the whole window; the old recursion also recomputed the
            # window's successes after a ratio restart.
            pending_groups.appendleft(list(window.items))
        return
    # Restart failed: the client already fell back to llama-mtmd-cli. The old
    # code kept the wave's successes and swept its failures serially once.
    _commit_and_requeue_interrupted()
    for window in halve_windows:
        window.failed = _serial_sweep(client, window, prompt, error_counts)
        report(len(window.items), window.results)


def _run_inference_windowed(
    client: QwenLlamaClient,
    frame_items: List[Dict],
    prompt: str,
    semantics: Dict[str, Dict],
    error_counts: Counter,
    progress: Dict[str, int],
) -> None:
    """Tag every candidate frame using one persistent executor per video.

    Replaces the old wave loop (a fresh ThreadPoolExecutor and a hard barrier
    every `batch_size` items). Scheduling changed; results are keyed by
    candidate id so completion order cannot affect the response JSON, and all
    recovery machinery keeps its old wave-level semantics via _TagWindow
    groups (see _run_concurrent_phase). Finalized tags are written straight
    into `semantics` so a breaker trip still returns partial results.
    """
    total = len(frame_items)
    started = time.perf_counter()
    pending_items: deque = deque(enumerate(frame_items, 1))
    pending_groups: deque = deque()
    executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _report(finalized_count: int, results: Dict[str, Dict]) -> None:
        semantics.update(results)
        progress["finalized"] += finalized_count
        elapsed = max(0.001, time.perf_counter() - started)
        print(
            f"Qwen llama.cpp tagged {progress['finalized']}/{total} "
            f"({progress['finalized'] / elapsed:.2f}/s, batch {client.batch_size})",
            flush=True,
        )

    try:
        while pending_items or pending_groups:
            concurrency = max(1, int(client.batch_size or 1))
            if client.server and concurrency > 1:
                if executor is None:
                    # Sized once at the video's initial slot count; restarts
                    # only ever reduce slots and the per-phase semaphore is
                    # the actual in-flight bound, so spare threads just idle.
                    executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=concurrency
                    )
                _run_concurrent_phase(
                    client, executor, pending_items, pending_groups, prompt,
                    semantics, error_counts, progress, _report,
                )
            else:
                if pending_groups:
                    regrouped = [pair for group in pending_groups for pair in group]
                    pending_groups.clear()
                    pending_items.extendleft(reversed(regrouped))
                fallback_index, item = pending_items.popleft()
                results, requeue = _serial_tag_item(
                    client, fallback_index, item, prompt, error_counts
                )
                if requeue:
                    pending_items.appendleft((fallback_index, item))
                    continue
                _report(1, results)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _run_semantics_for_video(
    *,
    client: QwenLlamaClient,
    video_file: str,
    fps: float,
    candidates: List[Dict],
    prompt: str,
) -> tuple[Dict[str, Dict], Dict]:
    decode_threads = _configure_opencv_ffmpeg_threads()
    print(f"Qwen OpenCV decode threads: {decode_threads}", flush=True)
    frame_width = _qwen_frame_width()
    print(f"Qwen frame max width: {frame_width}", flush=True)
    cap = cv2.VideoCapture(video_file)
    semantics: Dict[str, Dict] = {}
    timings = {
        "prefetch_seconds": 0.0,
        "inference_seconds": 0.0,
        "frame_count": 0,
        "tag_count": 0,
    }
    prefetch_started = time.perf_counter()
    try:
        frame_items = _prefetch_candidate_frames(cap, fps, candidates, frame_width)
        timings["prefetch_seconds"] = time.perf_counter() - prefetch_started
        timings["frame_count"] = len(frame_items)
        inference_started = time.perf_counter()
        error_counts: Counter = Counter()
        progress = {"finalized": 0}
        try:
            _run_inference_windowed(
                client, frame_items, prompt, semantics, error_counts, progress
            )
        except ServerWedgedError:
            timings["wedged"] = True
            print(
                f"Qwen watchdog: skipping {len(frame_items) - progress['finalized']} "
                "remaining candidates for this video (server wedged)",
                flush=True,
            )
        elapsed = max(0.001, time.perf_counter() - inference_started)
        timings["inference_seconds"] = elapsed
        timings["tag_count"] = len(semantics)
        print(
            f"Qwen llama.cpp semantic inference total: {len(semantics)}/{len(frame_items)} tags "
            f"in {elapsed:.1f}s ({len(frame_items) / elapsed:.2f} candidates/s)",
            flush=True,
        )
        if error_counts:
            # Surface the previously-dropped per-failure error strings as one
            # line of categories (log-only; audit finding R2).
            print(
                "Qwen tagging errors by category: "
                + ", ".join(f"{name}={count}" for name, count in error_counts.most_common()),
                flush=True,
            )
    finally:
        cap.release()
    return semantics, timings


def main() -> None:
    _install_signal_handlers()
    atexit.register(_shutdown_server_processes)
    args = _parse_args()
    whole_started = time.perf_counter()
    with open(args.request, "r", encoding="utf-8") as f:
        request = json.load(f)

    model_path = request.get("qwen_model_path")
    audio_profile = request.get("audio_profile") or {}
    prompt = _build_prompt(audio_profile)

    client = QwenLlamaClient(model_path, args.response)
    jobs = request.get("jobs")
    if not jobs:
        jobs = [{
            "job_id": "single",
            "video_file": request["video_file"],
            "fps": float(request.get("fps") or 24.0),
            "candidates": request.get("candidates") or [],
        }]
        legacy_single = True
    else:
        legacy_single = False

    semantics_by_job: Dict[str, Dict[str, Dict]] = {}
    timings_by_job: Dict[str, Dict] = {}
    try:
        for job_index, job in enumerate(jobs, 1):
            job_id = str(job.get("job_id", job_index))
            video_file = job["video_file"]
            fps = float(job.get("fps") or 24.0)
            candidates: List[Dict] = job.get("candidates") or []
            print(
                f"Qwen llama.cpp job {job_index}/{len(jobs)}: {os.path.basename(video_file)} "
                f"({len(candidates)} candidates)",
                flush=True,
            )
            semantics, timings = _run_semantics_for_video(
                client=client,
                video_file=video_file,
                fps=fps,
                candidates=candidates,
                prompt=prompt,
            )
            semantics_by_job[job_id] = semantics
            timings_by_job[job_id] = timings
            if client.breaker.tripped:
                remaining_jobs = len(jobs) - job_index
                print(
                    "Qwen circuit breaker tripped: llama-server wedged; writing partial "
                    f"results and skipping {remaining_jobs} remaining job(s)",
                    flush=True,
                )
                break
    finally:
        client.close()

    total_seconds = time.perf_counter() - whole_started
    response = {
        "model_load_seconds": client.load_seconds,
        "model_id": client.model_id,
        "batch_size": client.batch_size,
        "peak_vram_gb": 0.0,
        "total_seconds": total_seconds,
        "timings_by_job": timings_by_job,
        "wedged": client.breaker.tripped,
    }
    if legacy_single:
        response["semantics"] = semantics_by_job.get("single", {})
    else:
        response["semantics_by_job"] = semantics_by_job

    with open(args.response, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
