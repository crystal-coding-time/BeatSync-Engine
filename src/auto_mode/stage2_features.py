#!/usr/bin/env python3
"""Stage 2: beat-synchronous energy wave and rhythm feature extraction."""

import re
import subprocess
from typing import Dict, Optional, Tuple
import librosa
import numpy as np

from gpu_cpu_utils import GPU_AVAILABLE, cp
from . import AutoWaveConfig
from . import _interp_to_beats, _normalize, _safe_percentile, _smooth

def analyze_wave_features(y: np.ndarray, y_percussive: np.ndarray, sr: int,
                          beat_times: np.ndarray, beat_frames: np.ndarray,
                          onset_env: np.ndarray, cfg: AutoWaveConfig,
                          use_gpu: bool = False,
                          y_harmonic: Optional[np.ndarray] = None,
                          audio_file: Optional[str] = None,
                          start_time: float = 0.0,
                          duration: Optional[float] = None,
                          mel_S: Optional[np.ndarray] = None) -> Dict:
    """Extract beat-synchronous energy/rhythm data with smooth wave behavior."""
    duration = len(y) / sr

    # W2-5: the EBU R128 loudness pass is a full-song ffmpeg decode that shares
    # no state with the librosa work below, so start it now and let it run in
    # the background; compute_loudness() joins and parses it at the point the
    # result is first needed. Command, parse, and failure behavior (zeros + one
    # ⚠️ at that point) are identical to the old blocking call.
    loudness_probe = _start_loudness_probe(audio_file, start_time, duration)

    # One shared full-mix STFT: spectral_centroid's y= path is literally
    # np.abs(stft(y, same n_fft/hop/window))**1, and analyze_rhythm_bands runs
    # the identical stft call, so both can reuse this one (bit-identical).
    # rms deliberately stays on its y= path: librosa computes time-domain frame
    # RMS from y (no STFT at all) but windowed spectral RMS from S — different
    # numbers, and there is no redundant transform to save.
    stft_full = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    rms_curve = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    centroid_curve = librosa.feature.spectral_centroid(S=np.abs(stft_full), sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)[0]
    # mel_S is the same log-power mel onset_strength would build internally
    # (stage1 shares it too); flux differs from stage1's onset_env only by the
    # post-mel aggregation (mean here vs median there).
    if mel_S is not None:
        flux_curve = librosa.onset.onset_strength(S=mel_S, sr=sr, hop_length=cfg.hop_length)
    else:
        flux_curve = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=cfg.hop_length)

    rms_curve_n = _normalize(_smooth(rms_curve, 7))
    centroid_curve_n = _normalize(_smooth(centroid_curve, 7))
    flux_curve_n = _normalize(_smooth(flux_curve, 5))
    onset_n = _normalize(onset_env)

    rms = _interp_to_beats(rms_curve_n, beat_times, sr, cfg.hop_length)
    centroid = _interp_to_beats(centroid_curve_n, beat_times, sr, cfg.hop_length)
    flux = _interp_to_beats(flux_curve_n, beat_times, sr, cfg.hop_length)
    onset = _interp_to_beats(onset_n, beat_times, sr, cfg.hop_length)

    kick, bass, clap, hihat = analyze_rhythm_bands(y, sr, beat_times, cfg, use_gpu, stft=stft_full)

    # Musical novelty, but smoothed so it does not create twitchy cuts.
    novelty = _normalize(0.50 * flux + 0.35 * onset + 0.15 * np.abs(np.gradient(rms)))
    novelty = _smooth(novelty, 3)

    brightness = _normalize(0.65 * centroid + 0.35 * hihat)
    energy_raw = _normalize(0.55 * rms + 0.20 * flux + 0.15 * brightness + 0.10 * bass)

    # The important V3.2 change: a slower energy wave controls density.
    wave = _smooth(energy_raw, cfg.wave_smooth_beats)
    wave = _normalize(0.70 * wave + 0.30 * energy_raw)

    # Middle/finale arc lets the edit grow naturally without overcutting intro/outro.
    position = beat_times / max(duration, 1e-6)
    arc = np.sin(np.clip(position, 0.0, 1.0) * np.pi) ** 0.65
    arc = _normalize(0.58 * arc + 0.42 * wave)

    rhythm_score = _normalize(0.34 * kick + 0.26 * bass + 0.27 * clap + 0.08 * hihat + 0.05 * onset)
    impact_score = _normalize(0.42 * rhythm_score + 0.24 * novelty + 0.24 * wave + 0.10 * arc)

    high_thr = _safe_percentile(wave, 72, 0.72)
    peak_thr = _safe_percentile(wave, 88, 0.88)
    low_thr = _safe_percentile(wave, 30, 0.30)
    energy_levels = np.asarray([
        "peak" if e >= peak_thr else "high" if e >= high_thr else "low" if e <= low_thr else "medium"
        for e in wave
    ], dtype=object)

    kick_thr = _safe_percentile(kick, 72, 0.68)
    clap_thr = _safe_percentile(clap, 74, 0.68)
    hihat_thr = _safe_percentile(hihat, 82, 0.72)
    bass_thr = _safe_percentile(bass, 74, 0.68)

    is_phrase_anchor = np.zeros(len(beat_times), dtype=bool)
    is_bar_anchor = np.zeros(len(beat_times), dtype=bool)
    is_phrase_anchor[::max(1, cfg.phrase_beats)] = True
    is_bar_anchor[::max(1, cfg.bar_beats)] = True

    # Wave-13: three extra per-beat features consumed by stage4 scoring and
    # stage6 planning. Each is ALWAYS present (zero-filled + ⚠️ on any failure),
    # deterministic (pure librosa/numpy or a deterministic ffmpeg parse), and
    # normalized 0..1 to len(beat_times) exactly like the curves above.

    # SuperFlux onset strength (librosa SuperFlux recipe: lag=2, max_size=3 on
    # the percussive component, package hop kept for frame-time alignment).
    try:
        # SuperFlux's lag/max_size tweaks also apply after the mel, so the
        # shared spectrogram feeds this call unchanged as well.
        if mel_S is not None:
            superflux_curve = librosa.onset.onset_strength(
                S=mel_S, sr=sr, hop_length=cfg.hop_length, lag=2, max_size=3
            )
        else:
            superflux_curve = librosa.onset.onset_strength(
                y=y_percussive, sr=sr, hop_length=cfg.hop_length, lag=2, max_size=3
            )
        superflux_curve = _normalize(_smooth(np.asarray(superflux_curve, dtype=float), 3))
        onset_superflux = _normalize(_interp_to_beats(superflux_curve, beat_times, sr, cfg.hop_length))
    except Exception as e:
        print(f"      ⚠️ SuperFlux onset extraction failed; using zeros: {e}")
        onset_superflux = np.zeros(len(beat_times), dtype=float)

    # Harmonic change (HCDF): tonnetz-distance between adjacent frames.
    harmonic_change = compute_harmonic_change(
        y_harmonic if y_harmonic is not None else y, sr, beat_times, cfg
    )

    # EBU R128 momentary loudness at each beat (deterministic ffmpeg parse).
    loudness = compute_loudness(audio_file, start_time, duration, beat_times,
                                probe=loudness_probe)

    return {
        "kick": kick,
        "bass": bass,
        "clap": clap,
        "hihat": hihat,
        "rms": rms,
        "centroid": centroid,
        "flux": flux,
        "onset": onset,
        "novelty": novelty,
        "energy": energy_raw,
        "wave": wave,
        "brightness": brightness,
        "rhythm_score": rhythm_score,
        "impact_score": impact_score,
        "arc": arc,
        "energy_levels": energy_levels,
        "is_strong_kick": kick >= kick_thr,
        "is_strong_clap": clap >= clap_thr,
        "is_strong_hihat": hihat >= hihat_thr,
        "is_strong_bass": bass >= bass_thr,
        "is_bar_anchor": is_bar_anchor,
        "is_phrase_anchor": is_phrase_anchor,
        "rms_curve": rms_curve_n,
        "centroid_curve": centroid_curve_n,
        "flux_curve": flux_curve_n,
        # Raw (pre-normalization) flux curve so stage3's heuristic path can
        # reuse it instead of recomputing the identical onset_strength call.
        # Internal to the stage pipeline: not packaged into beat_info.
        "flux_curve_raw": flux_curve,
        "onset_superflux": onset_superflux,
        "harmonic_change": harmonic_change,
        "loudness": loudness,
    }


def compute_harmonic_change(y_harmonic: np.ndarray, sr: int, beat_times: np.ndarray,
                            cfg: AutoWaveConfig) -> np.ndarray:
    """HCDF: per-frame tonnetz-distance (chord/key-change strength), interp to beats.

    CQT chroma is the expensive step; if it fails we fall back to zeros with a
    ⚠️ log so the key is ALWAYS present at len(beat_times), normalized 0..1.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=cfg.hop_length)
        ton = librosa.feature.tonnetz(chroma=chroma)
        diff = np.linalg.norm(np.diff(ton, axis=1), axis=0)
        if diff.size == 0:
            return np.zeros(len(beat_times), dtype=float)
        # np.diff drops one frame; pad at the front so the curve realigns to the
        # original frame grid (frame i -> time i) used by _interp_to_beats.
        hcdf = np.concatenate([diff[:1], diff])
        hcdf = _smooth(hcdf, 5)
        return _normalize(_interp_to_beats(hcdf, beat_times, sr, cfg.hop_length))
    except Exception as e:
        print(f"      ⚠️ Harmonic-change (HCDF) extraction failed; using zeros: {e}")
        return np.zeros(len(beat_times), dtype=float)


def _start_loudness_probe(audio_file: Optional[str], start_time: float,
                          duration: Optional[float]):
    """Spawn the ebur128 ffmpeg pass in the background (W2-5 overlap).

    Returns a ``subprocess.Popen`` to be joined by ``compute_loudness``, None
    when there is no audio path (compute_loudness logs that case itself), or
    the raised exception when spawning failed — compute_loudness re-raises it
    at the join point so the failure log and zeros contract stay exactly where
    (and what) they were with the old blocking call.
    """
    if not audio_file:
        return None
    try:
        from ffmpeg_processing import FFMPEG_PATH

        cmd = [FFMPEG_PATH, '-nostdin', '-hide_banner']
        # Input seeking (-ss before -i) resets output PTS to 0, so the parsed
        # t: values share the same 0-based timeline as beat_times.
        if start_time and start_time > 0:
            cmd += ['-ss', f'{float(start_time):.6f}']
        cmd += ['-i', audio_file, '-map', 'a:0']
        if duration and duration > 0:
            cmd += ['-t', f'{float(duration):.6f}']
        cmd += ['-af', 'ebur128', '-f', 'null', '-']

        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
    except Exception as e:
        return e


def compute_loudness(audio_file: Optional[str], start_time: float,
                     duration: Optional[float], beat_times: np.ndarray,
                     probe=None) -> np.ndarray:
    """EBU R128 momentary loudness sampled at each beat.

    Runs ffmpeg's ebur128 filter once and parses the deterministic ``t:``/``M:``
    stderr trace (one sample per 100 ms). Momentary values are clamped at -60
    LUFS (silence reports ~-120) and mapped to 0..1 with a robust 5th->95th
    percentile scale so a single loud spike does not flatten everything else.
    Any failure -> zeros + ⚠️, so the key is ALWAYS present.

    ``probe`` is an already-started ``_start_loudness_probe`` result to join
    (the overlap path); when None the pass is spawned here and this call blocks
    on it exactly like before.
    """
    n = len(beat_times)
    if not audio_file:
        print("      ⚠️ Loudness: no audio path threaded to stage2; using zeros")
        return np.zeros(n, dtype=float)
    try:
        if probe is None:
            probe = _start_loudness_probe(audio_file, start_time, duration)
        if isinstance(probe, BaseException):
            raise probe  # deferred spawn failure -> identical ⚠️ + zeros below
        # No timeout: the old subprocess.run call had no deadline either.
        _stdout, stderr = probe.communicate()
        stderr = stderr or ""

        times: list = []
        mom: list = []
        for m in re.finditer(r't:\s*([-\d.]+).*?M:\s*([-\d.]+)', stderr):
            try:
                t = float(m.group(1))
                mval = float(m.group(2))
            except ValueError:
                continue
            times.append(t)
            mom.append(max(mval, -60.0))

        if len(times) < 2:
            print("      ⚠️ Loudness (ebur128) produced no parseable frames; using zeros")
            return np.zeros(n, dtype=float)

        times_arr = np.asarray(times, dtype=float)
        mom_arr = np.asarray(mom, dtype=float)
        sampled = np.interp(beat_times, times_arr, mom_arr,
                            left=float(mom_arr[0]), right=float(mom_arr[-1]))

        lo = float(np.percentile(mom_arr, 5))
        hi = float(np.percentile(mom_arr, 95))
        if hi - lo < 1e-8:
            return np.zeros(n, dtype=float)
        return np.clip((sampled - lo) / (hi - lo), 0.0, 1.0)
    except Exception as e:
        print(f"      ⚠️ Loudness (ebur128) extraction failed; using zeros: {e}")
        return np.zeros(n, dtype=float)


def analyze_rhythm_bands(y: np.ndarray, sr: int, beat_times: np.ndarray,
                         cfg: AutoWaveConfig, use_gpu: bool = False,
                         stft: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beat-level kick/bass/clap/hihat strength.

    ``stft`` lets the caller pass a precomputed complex STFT of ``y`` at
    cfg.n_fft/cfg.hop_length; when omitted it is computed here as before.
    """
    if stft is None:
        stft = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)

    use_cupy = bool(use_gpu and GPU_AVAILABLE and cp is not None)
    xp = cp if use_cupy else np

    if use_cupy:
        try:
            stft_x = cp.asarray(stft)
            freqs_x = cp.asarray(freqs)
        except Exception:
            use_cupy = False
            xp = np
            stft_x = stft
            freqs_x = freqs
    else:
        stft_x = stft
        freqs_x = freqs

    magnitude = xp.abs(stft_x)
    bands = {
        "kick": (freqs_x >= 35) & (freqs_x <= 145),
        "bass": (freqs_x >= 35) & (freqs_x <= 220),
        "clap": (freqs_x >= 150) & (freqs_x <= 4200),
        "hihat": freqs_x >= 4200,
    }

    frame_times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=cfg.hop_length)
    outputs: Dict[str, np.ndarray] = {}

    for name, mask in bands.items():
        try:
            curve = xp.sum(magnitude[mask, :], axis=0)
            if use_cupy:
                curve = cp.asnumpy(curve)
            curve = _normalize(_smooth(np.asarray(curve, dtype=float), 3))
            outputs[name] = np.interp(beat_times, frame_times, curve, left=float(curve[0]), right=float(curve[-1]))
        except Exception as e:
            print(f"      ⚠️ Rhythm band '{name}' extraction failed; using zeros: {e}")
            outputs[name] = np.zeros(len(beat_times), dtype=float)

    return outputs["kick"], outputs["bass"], outputs["clap"], outputs["hihat"]
