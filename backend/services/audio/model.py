# backend/services/audio/model.py
"""
READ – Audio Deepfake Detection Model

Architecture:
    Module A (Upstream / Frozen): WavLM Base+ feature extractor
    Module B (Downstream)       : Self-Attention Classifier (DeepFakeDetector)

WavLM converts raw 16 kHz waveform into rich 768-dim contextual vectors.
The classifier uses multi-head attention to locate artifact patterns.
"""

from __future__ import annotations
from typing import Optional, Tuple

import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Fixed paths
# --------------------------------------------------
WAVLM_HF_ID = "microsoft/wavlm-base-plus"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "ml_models", "audio_deepfake_classifier.pt")
)

# --------------------------------------------------
# Global model state  (mirrors services/detection/model.py pattern)
# --------------------------------------------------
_WAVLM_MODEL = None
_WAVLM_PROCESSOR = None
_DETECTOR_MODEL = None
_AUDIO_DEVICE = None


# ==================================================
# Module B: Forensic Back-End Classifier
# ==================================================

class DeepFakeDetector(nn.Module):
    """
    Lightweight Self-Attention classifier that sits on top of WavLM features.

    Input:  (batch, seq_len, 768) — WavLM last hidden state
    Output: (batch, 2)           — [real_logit, fake_logit]
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=False)
        self.fc_final = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: WavLM feature tensor of shape (batch, seq_len, 768)
        Returns:
            Logits of shape (batch, 2)
        """
        x = self.relu(self.fc1(x))              # (B, T, 128)
        x = x.permute(1, 0, 2)                  # (T, B, 128) — required by nn.MultiheadAttention
        attn_out, _ = self.attention(x, x, x)   # (T, B, 128)
        x = attn_out.mean(dim=0)                 # global avg pool → (B, 128)
        return self.fc_final(x)                  # (B, 2)


# ==================================================
# Loaders
# ==================================================

def load_audio_models(device: Optional[str] = None):
    """
    Load WavLM (frozen) and DeepFakeDetector (from checkpoint if available).

    Subsequent calls return the already-loaded models from the globals.
    If the classifier checkpoint does not exist, random weights are used
    and a WARNING is emitted — inference will work structurally but will
    produce meaningless results until a trained checkpoint is placed at
    CLASSIFIER_PATH.

    Returns:
        tuple: (wavlm_model, wavlm_processor, detector_model)
    """
    global _WAVLM_MODEL, _WAVLM_PROCESSOR, _DETECTOR_MODEL, _AUDIO_DEVICE

    if _WAVLM_MODEL is not None:
        return _WAVLM_MODEL, _WAVLM_PROCESSOR, _DETECTOR_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _AUDIO_DEVICE = device

    # --------------------------------------------------
    # Module A: WavLM Base+ (frozen feature extractor)
    # --------------------------------------------------
    try:
        from transformers import WavLMModel, Wav2Vec2Processor
        logger.info(f"[AudioModel] Loading WavLM Base+ from HuggingFace ({WAVLM_HF_ID}) …")
        wavlm = WavLMModel.from_pretrained(WAVLM_HF_ID)
        processor = Wav2Vec2Processor.from_pretrained(WAVLM_HF_ID)
        wavlm.to(device)
        wavlm.eval()
        # Freeze all parameters — we never retrain WavLM
        for param in wavlm.parameters():
            param.requires_grad_(False)
        logger.info("[AudioModel]  WavLM Base+ loaded and frozen.")
    except Exception as e:
        logger.error(f"[AudioModel]  Failed to load WavLM: {e}", exc_info=True)
        raise RuntimeError(f"Cannot load WavLM: {e}") from e

    # --------------------------------------------------
    # Module B: DeepFakeDetector classifier
    # --------------------------------------------------
    detector = DeepFakeDetector().to(device)

    if os.path.isfile(CLASSIFIER_PATH):
        try:
            state = torch.load(CLASSIFIER_PATH, map_location=device)
            detector.load_state_dict(state)
            logger.info(f"[AudioModel] ✅ Classifier loaded from {CLASSIFIER_PATH}")
        except Exception as e:
            logger.error(
                f"[AudioModel] ❌ Failed to load classifier weights from {CLASSIFIER_PATH}: {e}",
                exc_info=True,
            )
    else:
        logger.warning(
            f"[AudioModel] ⚠️  Classifier checkpoint not found at {CLASSIFIER_PATH}. "
            "Using random weights — place a trained .pt file there to get real results."
        )

    detector.eval()
    _WAVLM_MODEL = wavlm
    _WAVLM_PROCESSOR = processor
    _DETECTOR_MODEL = detector

    return _WAVLM_MODEL, _WAVLM_PROCESSOR, _DETECTOR_MODEL


# ==================================================
# Inference
# ==================================================

def extract_wavlm_features(audio_tensor: torch.FloatTensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Run the frozen WavLM model and return the last hidden state.

    Args:
        audio_tensor: 1-D float tensor of shape (num_samples,) at 16 kHz.
        device:       Torch device string. Defaults to _AUDIO_DEVICE.

    Returns:
        Tensor of shape (1, seq_len, 768).
    """
    wavlm, _, _ = load_audio_models(device)
    dev = device or _AUDIO_DEVICE or "cpu"

    with torch.no_grad():
        inp = audio_tensor.unsqueeze(0).to(dev)    # (1, num_samples)
        outputs = wavlm(input_values=inp)
        return outputs.last_hidden_state           # (1, seq_len, 768)


def run_audio_inference(task_id: str, audio_tensor: torch.FloatTensor) -> dict:
    """
    Full Module A → B pipeline: raw audio → verdict probabilities.

    Args:
        task_id:      Pipeline identifier (for logging consistency).
        audio_tensor: Pre-processed 1-D float tensor (64 600 samples, 16 kHz).

    Returns:
        dict with keys: task_id, real_prob, fake_prob
    """
    _, _, detector = load_audio_models()
    dev = _AUDIO_DEVICE or "cpu"

    features = extract_wavlm_features(audio_tensor, dev)   # (1, T, 768)

    with torch.no_grad():
        logits = detector(features)                         # (1, 2)
        probs = torch.softmax(logits, dim=-1)

    return {
        "task_id":   task_id,
        "real_prob": probs[0, 0].item(),
        "fake_prob": probs[0, 1].item(),
    }
