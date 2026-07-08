#!/usr/bin/env python3
"""Stage 1: load-facing beat grid detection."""

import os
from typing import Optional, Tuple
import librosa
import numpy as np

from . import AutoWaveConfig
from . import _normalize, _smooth, _to_float


def _beat_this_grid(y: np.ndarray, sr: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Beats + downbeats from the Beat This! transformer (MIT, CPJKU).

    Returns (beat_times, downbeat_times) or None on any failure so the caller
    can fall back to librosa. CPU is the default device: for song-length audio
    the small model runs in well under a second, while MPS pays several seconds
    of warmup for identical output (measured 0.17s CPU vs 5.9s MPS on 20s).
    """
    try:
        from beat_this.inference import Audio2Beats
    except Exception as e:
        print(f"   ⚠️  BEATSYNC_BEAT_BACKEND=beat_this but the model is unavailable ({e}); using librosa")
        return None

    checkpoint = os.environ.get("BEATSYNC_BEAT_THIS_CHECKPOINT", "small0")
    device = os.environ.get("BEATSYNC_BEAT_THIS_DEVICE", "cpu")
    last_error: Exception | None = None
    for dev in dict.fromkeys([device, "cpu"]):
        try:
            audio2beats = Audio2Beats(checkpoint_path=checkpoint, device=dev, dbn=False)
            beats, downbeats = audio2beats(np.asarray(y, dtype=np.float32), sr)
            beats = np.asarray(beats, dtype=float).reshape(-1)
            downbeats = np.asarray(downbeats, dtype=float).reshape(-1)
            return beats[np.isfinite(beats)], downbeats[np.isfinite(downbeats)]
        except Exception as e:
            last_error = e
    print(f"   ⚠️  beat-this inference failed ({last_error}); using librosa")
    return None


def detect_master_beat_grid(y_percussive: np.ndarray, sr: int, cfg: AutoWaveConfig,
                            y_full: np.ndarray = None,
                            mel_S: np.ndarray = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (beat_times, tempo, beat_frames, onset_env, downbeat_times).

    downbeat_times is empty for the librosa backend; the beat_this backend
    (BEATSYNC_BEAT_BACKEND=beat_this) fills it from the transformer model.

    ``mel_S`` is an optional precomputed log-power mel spectrogram of
    ``y_percussive`` (onset_strength's own default feature); the aggregation
    happens after the mel, so sharing it is bit-identical to the y= path.
    """
    if mel_S is not None:
        onset_env = librosa.onset.onset_strength(
            S=mel_S,
            sr=sr,
            hop_length=cfg.hop_length,
            aggregate=np.median,
        )
    else:
        onset_env = librosa.onset.onset_strength(
            y=y_percussive,
            sr=sr,
            hop_length=cfg.hop_length,
            aggregate=np.median,
        )
    onset_env = _normalize(_smooth(onset_env, 3))

    no_downbeats = np.asarray([], dtype=float)
    backend = os.environ.get("BEATSYNC_BEAT_BACKEND", "librosa").strip().lower()
    if backend == "beat_this":
        result = _beat_this_grid(y_full if y_full is not None else y_percussive, sr)
        if result is not None:
            beat_times, downbeat_times = result
            if beat_times.size >= 2:
                diffs = np.diff(beat_times)
                tempo = 60.0 / float(np.median(diffs)) if diffs.size else 120.0
                beat_frames = np.asarray(
                    librosa.time_to_frames(beat_times, sr=sr, hop_length=cfg.hop_length), dtype=int
                )
                return beat_times, tempo, beat_frames, onset_env, downbeat_times
            print("   ⚠️  beat-this returned too few beats; using librosa")

    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=cfg.hop_length,
            units="frames",
            start_bpm=120,
            tightness=120,
            trim=False,
        )
    except TypeError:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            y=y_percussive,
            sr=sr,
            hop_length=cfg.hop_length,
            units="frames",
            start_bpm=120,
            tightness=120,
        )

    tempo = _to_float(tempo_raw, 120.0)
    beat_frames = np.asarray(beat_frames, dtype=int)

    if beat_frames.size < 2:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=cfg.hop_length,
            backtrack=True,
            wait=4,
        )
        beat_frames = np.asarray(onset_frames, dtype=int)
        tempo = 120.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=cfg.hop_length)
    beat_times = np.asarray(beat_times, dtype=float)
    valid = np.isfinite(beat_times) & (beat_times >= 0.0)
    return beat_times[valid], tempo, beat_frames[valid], onset_env, no_downbeats
