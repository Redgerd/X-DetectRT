import torch
from PIL import Image
from .modeling_gend import GenD

# --------------------------------------------------
# Global model state
# --------------------------------------------------
_GEND_MODEL = None
_GEND_DEVICE = None
_GEND_MODEL_PATH = "/app/GenD_PE_L"


def load_gend_model(device=None):
    """
    Load GenD model once and store it globally.
    Subsequent calls reuse the same model.
    """
    global _GEND_MODEL, _GEND_DEVICE

    if _GEND_MODEL is not None:
        return _GEND_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GenD.from_pretrained(_GEND_MODEL_PATH)
    model.to(device)
    model.eval()

    _GEND_MODEL = model
    _GEND_DEVICE = device

    return _GEND_MODEL


def run_gend_inference(task_id, frame):
    """
    Run GenD inference on a single frame.

    Args:
        task_id: Identifier (for pipeline consistency)
        frame: PIL.Image or ndarray

    Returns:
        dict: { "task_id", "real_prob", "fake_prob" }
    """
    model = load_gend_model()
    device = _GEND_DEVICE

    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)

    tensor = model.feature_extractor.preprocess(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)

    return {
        "task_id": task_id,
        "real_prob": probs[0, 0].item(),
        "fake_prob": probs[0, 1].item()
    }


if __name__ == "__main__":
    # Quick test
    from PIL import Image
    import numpy as np

    dummy_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = run_gend_inference("test", dummy_frame)
    print(result)
