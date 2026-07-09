"""Render-time logging: the per-run log file, the CMD stage logger, and the
stage-summary formatters.

Split out of gui.py so the orchestrator (which does the actual rendering) owns
its diagnostics independently of the Gradio UI. Pure stdlib — no first-party
imports — so importing this never triggers CUDA/FFmpeg environment setup.
"""

import os
import datetime
import re
import time
from typing import Dict


class RenderLogConsole:
    """Capture legacy verbose prints to a log file while the Gradio worker runs.

    The pipeline's diagnostics (including FFmpeg stderr on failures) used to be
    discarded entirely, leaving "N clip(s) failed" with no way to see why. Now
    everything lands in a per-run render log the error message can point at.
    """

    def __init__(self, log_dir: str, prefix: str = 'render'):
        self.log_path = os.path.join(
            log_dir, f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._file = None
        self._failed = False

    def _ensure_file(self):
        if self._file is None and not self._failed:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._file = open(self.log_path, 'a', encoding='utf-8', errors='replace')
            except OSError:
                self._failed = True
        return self._file

    def write(self, text: str) -> int:
        f = self._ensure_file()
        if f is not None:
            try:
                f.write(text)
                f.flush()
            except OSError:
                self._failed = True
        return len(text)

    def flush(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


class StageConsoleLogger:
    """Small CMD logger: stage start, up to 5 useful lines, stage end."""

    def __init__(self, stream, max_lines_per_stage: int = 5):
        self.stream = stream
        self.max_lines_per_stage = max(1, int(max_lines_per_stage))
        self.stage_number: int | None = None
        self.stage_started = 0.0
        self.stage_line_count = 0
        self.total_started = time.perf_counter()

    def start_stage(self, stage_number: int) -> None:
        if self.stage_number == stage_number:
            return
        self.end_stage()
        self.stage_number = stage_number
        self.stage_started = time.perf_counter()
        self.stage_line_count = 0
        self._write(f"Stage {stage_number} processing started:\n")

    def stage_line(self, stage_number: int, message: str) -> None:
        if self.stage_number != stage_number:
            self.start_stage(stage_number)
        self.line(message)

    def line(self, message: str) -> None:
        if self.stage_number is None:
            return
        if self.stage_line_count >= self.max_lines_per_stage:
            return
        message = self._clean(message)
        if message:
            self._write(f"  {message}\n")
            self.stage_line_count += 1

    def end_stage(self) -> None:
        if self.stage_number is None:
            return
        elapsed = int(round(time.perf_counter() - self.stage_started))
        self._write(f"Stage {self.stage_number} ended in {elapsed} seconds.\n\n")
        self.stage_number = None
        self.stage_started = 0.0
        self.stage_line_count = 0

    def finish(self) -> None:
        self.end_stage()
        elapsed = int(round(time.perf_counter() - self.total_started))
        self._write(f"Total time processing: {elapsed} seconds\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clean(self, text: str) -> str:
        text = str(text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", text).strip()


def _fmt_stage_seconds(seconds: float | int | None) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return "0.0s"


def _short_model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return os.path.basename(str(model_id).rstrip("/\\")) or str(model_id)


def _stage5_summary(console_logger: StageConsoleLogger | None, video_analysis: Dict | None) -> None:
    if console_logger is None or not isinstance(video_analysis, dict):
        return

    source_count = int(video_analysis.get("source_count") or len(video_analysis.get("videos") or []))
    worker_count = int(video_analysis.get("worker_count") or 1)
    cache_hits = int(video_analysis.get("cache_hits") or 0)
    ai_enabled = bool(video_analysis.get("ai_enabled"))
    qwen_total = int(video_analysis.get("qwen_frame_count") or 0)
    qwen_tags = int(video_analysis.get("qwen_tag_count") or 0)
    model_id = _short_model_name(video_analysis.get("qwen_model_id"))
    batch_size = int(video_analysis.get("qwen_concurrency") or 0)

    console_logger.line(f"Source videos: {source_count}, visual workers: {worker_count}")
    if ai_enabled:
        qwen_bits = ["Qwen: enabled"]
        if model_id:
            qwen_bits.append(f"model {model_id}")
        if batch_size:
            qwen_bits.append(f"batch {batch_size}")
        console_logger.line(", ".join(qwen_bits))
        if qwen_total:
            inference_seconds = float(video_analysis.get("qwen_inference_seconds") or video_analysis.get("qwen_seconds") or 0.0)
            qwen_rate = (qwen_total / inference_seconds) if inference_seconds > 0 else 0.0
            peak_vram = float(video_analysis.get("qwen_peak_vram_gb") or 0.0)
            perf_bits = []
            if batch_size:
                perf_bits.append(f"batch {batch_size}")
            if peak_vram > 0:
                perf_bits.append(f"~{peak_vram:.2f} GB VRAM")
            if qwen_rate > 0:
                perf_bits.append(f"{qwen_rate:.2f} candidates/s")
            if perf_bits:
                console_logger.line(f"Qwen performance: {', '.join(perf_bits)}")
            console_logger.line(
                f"Qwen tags: {qwen_tags}/{qwen_total} in {_fmt_stage_seconds(video_analysis.get('qwen_seconds'))}"
            )
    else:
        console_logger.line("Qwen: disabled")

    summary = video_analysis.get("summary")
    if summary:
        console_logger.line(f"Visual library: {summary}")
    console_logger.line(
        f"Analysis time: {_fmt_stage_seconds(video_analysis.get('analysis_seconds'))}, cache {cache_hits}/{source_count}"
    )


def _stage6_summary(console_logger: StageConsoleLogger | None, beat_info: Dict | None) -> None:
    if console_logger is None or not isinstance(beat_info, dict):
        return

    render_info = beat_info.get("render_info") or {}
    if not render_info:
        return

    cuts = int(render_info.get("render_cuts") or 0)
    frames = int(render_info.get("timeline_frames") or 0)
    fps = render_info.get("output_fps")
    if cuts or frames:
        fps_text = f" @ {float(fps):.1f} FPS" if fps is not None else ""
        console_logger.line(f"Render timeline: {cuts} cuts, {frames} frames{fps_text}")

    clip_workers = render_info.get("clip_workers")
    requested_workers = render_info.get("requested_workers")
    worker_text = ""
    if clip_workers:
        worker_text = f", workers {clip_workers}"
        if requested_workers and requested_workers != clip_workers:
            worker_text += f"/{requested_workers}"
    encoder = render_info.get("encoder")
    if encoder or worker_text:
        console_logger.line(f"Encoder: {encoder or 'unknown'}{worker_text}")

    if render_info.get("audio_duration") is not None:
        console_logger.line(f"Audio duration: {float(render_info['audio_duration']):.2f} seconds")

    plan_summary = render_info.get("plan_summary") or {}
    if plan_summary:
        console_logger.line(
            "Planner: "
            f"{int(plan_summary.get('clip_count') or 0)} clips, "
            f"{int(plan_summary.get('source_count') or 0)} sources, "
            f"AI moments {int(plan_summary.get('ai_tagged') or 0)}"
        )

    final_bits = []
    if render_info.get("target_resolution"):
        final_bits.append(f"resolution {render_info['target_resolution']}")
    if render_info.get("final_assembly_seconds") is not None:
        final_bits.append(f"assembly {_fmt_stage_seconds(render_info['final_assembly_seconds'])}")
    if final_bits:
        console_logger.line("Final: " + ", ".join(final_bits))
