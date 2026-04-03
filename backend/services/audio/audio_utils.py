# backend/services/audio/audio_utils.py
"""
Audio preprocessing utilities for the X-DetectRT pipeline.

- Resamples to 16 kHz (required by WavLM Base+)
- Pads / trims to exactly 64,600 samples (~4 seconds)
- Downsamples waveform to a compact array for Canvas rendering
- Computes STFT power spectrogram matrix for the 3-layer Canvas spectrogram
"""

import io
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# WavLM requires 16 kHz mono; 64,600 ≈ 4.0375 s
TARGET_SR   = 16_000
NUM_SAMPLES = 64_600

# Canvas waveform resolution (must be lightweight for JSON)
WAVEFORM_CANVAS_POINTS = 2_000

# STFT parameters
STFT_N_FFT     = 512   # FFT size → 257 frequency bins
STFT_HOP_LEN   = 256   # hop → ~201 frames for 4 s at 16 kHz


# ---------------------------------------------------------------------------
# Preprocessing — raw bytes → fixed-length tensor
# ---------------------------------------------------------------------------

def load_and_preprocess_audio(
    file_bytes: bytes,
    target_sr: int  = TARGET_SR,
    num_samples: int = NUM_SAMPLES,
) -> torch.FloatTensor:
    """
    Convert raw audio file bytes into a fixed-length float tensor.

    Steps:
        1. Decode with librosa (supports wav, flac, mp3, ogg …)
        2. Resample to *target_sr* if needed
        3. Convert to mono by averaging channels
        4. Pad (by repeating) or trim to exactly *num_samples*

    Args:
        file_bytes:  Raw bytes of the uploaded audio file.
        target_sr:   Target sample rate (default 16 000 Hz).
        num_samples: Fixed output length in samples (default 64 600).

    Returns:
        A 1-D ``torch.FloatTensor`` of shape ``(num_samples,)``.
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required: pip install librosa soundfile")

    buf = io.BytesIO(file_bytes)

    try:
        waveform, sr = librosa.load(buf, sr=None, mono=True)
    except Exception as e:
        logger.error(f"[AudioUtils] librosa.load failed: {e}", exc_info=True)
        raise ValueError(f"Cannot decode audio: {e}") from e

    if sr != target_sr:
        logger.info(f"[AudioUtils] Resampling from {sr} Hz → {target_sr} Hz")
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    length = len(waveform)
    if length < num_samples:
        repeats = (num_samples // length) + 1
        waveform = np.tile(waveform, repeats)[:num_samples]
    elif length > num_samples:
        waveform = waveform[:num_samples]

    return torch.FloatTensor(waveform)


# ---------------------------------------------------------------------------
# Canvas waveform — compact downsampled array
# ---------------------------------------------------------------------------

def downsample_waveform(
    audio_tensor: torch.FloatTensor,
    n_points: int = WAVEFORM_CANVAS_POINTS,
) -> List[float]:
    """
    Downsample the full-resolution waveform to *n_points* for Canvas rendering.

    Uses chunk-max (per block take the max absolute value, preserving sign of
    the dominant sample) so that the envelope shape is faithfully represented.

    Returns:
        List of floats in [-1.0, 1.0] with length *n_points*.
    """
    wv = audio_tensor.numpy()
    total = len(wv)
    block = max(total // n_points, 1)
    out: List[float] = []
    for i in range(n_points):
        start = i * block
        end   = min(start + block, total)
        chunk = wv[start:end]
        if len(chunk) == 0:
            out.append(0.0)
        else:
            idx = int(np.argmax(np.abs(chunk)))
            out.append(float(chunk[idx]))
    return out


# ---------------------------------------------------------------------------
# STFT spectrogram matrix
# ---------------------------------------------------------------------------

def compute_stft_matrix(
    audio_tensor: torch.FloatTensor,
    sample_rate: int = TARGET_SR,
    n_fft: int       = STFT_N_FFT,
    hop_length: int  = STFT_HOP_LEN,
) -> Dict[str, object]:
    """
    Compute a power spectrogram (in dB) from the audio tensor.

    The result is returned as plain Python lists so it can be JSON-serialised
    directly by FastAPI without any intermediate numpy conversion.

    Args:
        audio_tensor: 1-D float tensor of audio samples at *sample_rate* Hz.
        sample_rate:  Audio sample rate (Hz).
        n_fft:        FFT window size.
        hop_length:   Hop size between frames.

    Returns:
        dict with keys:
            "matrix"  — 2-D list [freq_bins][time_frames] of dB values (float32)
            "times"   — 1-D list of time positions (seconds) for each frame
            "freqs"   — 1-D list of frequency values (Hz) for each bin
            "db_min"  — global dB minimum (for frontend colormap normalisation)
            "db_max"  — global dB maximum
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required: pip install librosa")

    wv = audio_tensor.numpy()

    # Compute magnitude spectrogram
    stft = librosa.stft(wv, n_fft=n_fft, hop_length=hop_length)
    mag  = np.abs(stft)                           # (n_fft//2 + 1, T)

    # Convert to dB with a floor to avoid log(0)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)  # (F, T), max → 0 dB

    freq_bins, time_frames = mag_db.shape

    # Time and frequency axes
    times = librosa.frames_to_time(
        np.arange(time_frames), sr=sample_rate, hop_length=hop_length
    ).tolist()
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).tolist()

    db_min = float(mag_db.min())
    db_max = float(mag_db.max())

    # Serialise — truncate to top 128 freq bins (0–8 kHz) for compactness
    MAX_FREQ_BINS = 128
    matrix_2d = mag_db[:MAX_FREQ_BINS, :].tolist()
    freqs_out  = freqs[:MAX_FREQ_BINS]

    logger.debug(
        f"[AudioUtils] STFT shape=({MAX_FREQ_BINS}, {time_frames}), "
        f"db_min={db_min:.1f}, db_max={db_max:.1f}"
    )

    return {
        "matrix": matrix_2d,
        "times":  times,
        "freqs":  freqs_out,
        "db_min": db_min,
        "db_max": db_max,
    }