# backend/services/audio/audio_utils.py
"""
Audio preprocessing utilities for the READ pipeline.

- Resamples to 16 kHz (required by WavLM Base+)
- Pads / trims to exactly 64,600 samples (~4 seconds)
- Generates a base64 waveform PNG for frontend display
"""

import io
import base64
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# WavLM requires 16 kHz mono audio; 64,600 ≈ 4.0375 s
TARGET_SR = 16_000
NUM_SAMPLES = 64_600


def load_and_preprocess_audio(
    file_bytes: bytes,
    target_sr: int = TARGET_SR,
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
        file_bytes: Raw bytes of the uploaded audio file.
        target_sr:  Target sample rate (default 16 000 Hz).
        num_samples: Fixed output length in samples (default 64 600).

    Returns:
        A 1-D ``torch.FloatTensor`` of shape ``(num_samples,)``.
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "librosa and soundfile are required. "
            "Run: pip install librosa soundfile"
        )

    buf = io.BytesIO(file_bytes)

    try:
        # librosa returns float32 array and the native sample rate
        waveform, sr = librosa.load(buf, sr=None, mono=True)
    except Exception as e:
        logger.error(f"[AudioUtils] librosa.load failed: {e}", exc_info=True)
        raise ValueError(f"Cannot decode audio: {e}") from e

    # Resample if necessary
    if sr != target_sr:
        logger.info(f"[AudioUtils] Resampling from {sr} Hz → {target_sr} Hz")
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Pad or trim
    length = len(waveform)
    if length < num_samples:
        # Repeat the clip until long enough, then slice
        repeats = (num_samples // length) + 1
        waveform = np.tile(waveform, repeats)[:num_samples]
        logger.debug(f"[AudioUtils] Padded {length} → {num_samples} samples")
    elif length > num_samples:
        waveform = waveform[:num_samples]
        logger.debug(f"[AudioUtils] Trimmed {length} → {num_samples} samples")

    return torch.FloatTensor(waveform)


def waveform_to_base64_png(
    audio_tensor: torch.FloatTensor,
    sample_rate: int = TARGET_SR,
) -> str:
    """
    Render a waveform plot and return it as a base64-encoded PNG string.

    Args:
        audio_tensor: 1-D float tensor of audio samples.
        sample_rate:  Sample rate used to compute the time axis (seconds).

    Returns:
        Base64-encoded PNG string (no data-URI prefix).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required. Run: pip install matplotlib")

    waveform = audio_tensor.numpy()
    times = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))

    fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
    ax.plot(times, waveform, color="#4A90E2", linewidth=0.5, alpha=0.9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title("Audio Waveform", fontsize=9)
    ax.set_xlim(0, times[-1])
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
