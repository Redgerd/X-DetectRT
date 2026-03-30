# backend/services/detection/xai_methods.py
"""
Explainable AI (XAI) helpers for GenD deepfake detection.

Implements:
    - Grad-CAM++ : Visual heatmaps highlighting suspicious regions.
    - LIME        : Local superpixel-based feature attribution.

Reference: DYP.pdf – "Integrating Dual XAI Techniques" (Step 3)
"""

import io
import base64
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional model preloading for XAI usage
# ---------------------------------------------------------------------------

from services.detection.model import load_gend_model

_XAI_MODEL = None


def init_xai_model(device: str | None = None):
    """
    Preload GenD model for XAI usage.
    Call this once during FastAPI startup.

    Args:
        device: Optional torch device string ("cpu" or "cuda").
    """
    global _XAI_MODEL

    if _XAI_MODEL is None:
        _XAI_MODEL = load_gend_model(device=device)

    return _XAI_MODEL


def get_xai_model():
    """
    Returns the preloaded GenD model.
    If not initialized, loads it lazily.
    """
    global _XAI_MODEL

    if _XAI_MODEL is None:
        _XAI_MODEL = load_gend_model()

    return _XAI_MODEL

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _numpy_bgr_to_base64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _overlay_heatmap_on_image(heatmap: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """
    Overlay a normalised float heatmap [0,1] on an RGB numpy image.
    Returns a BGR uint8 image ready for imencode / imshow.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)         # BGR
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    original_bgr_resized = cv2.resize(original_bgr, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
    superimposed = cv2.addWeighted(original_bgr_resized, 0.5, heatmap_colored, 0.5, 0)
    return superimposed

# ---------------------------------------------------------------------------
# Grad-CAM++ for Vision Transformers (patch-based)
# ---------------------------------------------------------------------------

def _get_vit_target_layer(model):
    """
    Dynamically resolve the last transformer block of the backbone.
    Works for PerceptionEncoder (Eva / timm), CLIPEncoder, and DINOEncoder.
    """
    feature_extractor = model.feature_extractor

    # PerceptionEncoder (timm Eva backbone)
    if hasattr(feature_extractor, "backbone"):
        backbone = feature_extractor.backbone
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            return backbone.blocks[-1]
        if hasattr(backbone, "layers") and len(backbone.layers) > 0:
            return backbone.layers[-1]

    # CLIPEncoder
    if hasattr(feature_extractor, "vision_model"):
        enc = feature_extractor.vision_model.encoder
        if hasattr(enc, "layers") and len(enc.layers) > 0:
            return enc.layers[-1]

    # DINOEncoder
    if hasattr(feature_extractor, "backbone"):
        enc = feature_extractor.backbone.encoder
        if hasattr(enc, "layer") and len(enc.layer) > 0:
            return enc.layer[-1]

    raise ValueError("Could not resolve a target layer for Grad-CAM++ from the current GenD backbone.")


class _GradCAMPlusPlus:
    """
    Lightweight Grad-CAM++ implementation for Vision Transformers.
    Hooks into a transformer block and reshapes 1-D patch activations into
    a 2-D spatial grid before computing the heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = []

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            # out may be a tuple (hidden_state, ...) – take first element
            self._activations = out[0] if isinstance(out, tuple) else out

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0] if isinstance(grad_out, tuple) else grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generate a Grad-CAM++ heatmap for the given class index.

        Args:
            input_tensor: Pre-processed image tensor of shape (1, C, H, W).
            class_idx: Target class index (1 = Fake).

        Returns:
            Normalised float32 heatmap of shape (H, W) in [0, 1].
        """
        self._register_hooks()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)           # (1, num_classes)
        score = logits[0, class_idx]

        # Backward pass
        score.backward(retain_graph=False)

        activations = self._activations.detach()   # (1, seq_len, C)
        gradients = self._gradients.detach()        # (1, seq_len, C)

        self._remove_hooks()

        # Grad-CAM++ weight calculation
        relu_grads = F.relu(gradients)
        alpha_num = relu_grads ** 2
        alpha_denom = 2.0 * relu_grads ** 2 + (activations * relu_grads ** 3).sum(dim=1, keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(gradients)).sum(dim=1, keepdim=True)  # (1, 1, C)

        # Weighted combination of activations
        # activations: (1, seq_len, C)  weights: (1, 1, C)
        cam = (weights * activations).sum(dim=-1)   # (1, seq_len)
        cam = F.relu(cam)                            # keep only positive
        cam = cam[0]                                 # (seq_len,)

        # Reshape to 2-D grid
        seq_len = cam.shape[0]
        # Vision Transformers often prepend a [CLS] token; drop it if present
        if int(seq_len ** 0.5) ** 2 != seq_len:
            cam = cam[1:]  # strip [CLS]
            seq_len = cam.shape[0]

        grid_size = int(seq_len ** 0.5)
        cam_2d = cam.reshape(grid_size, grid_size).cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam_2d.min(), cam_2d.max()
        if cam_max - cam_min > 1e-8:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)
        else:
            cam_2d = np.zeros_like(cam_2d)

        return cam_2d


def generate_gradcam(model, image_tensor: torch.Tensor, original_pil: Image.Image) -> str:
    """
    Generate a Grad-CAM++ heatmap overlay for the GenD model.

    Args:
        model: The loaded GenD model (eval mode).
        image_tensor: Pre-processed tensor of shape (1, C, H, W) on the correct device.
        original_pil: The raw PIL image (used for the overlay background).

    Returns:
        base64-encoded JPEG string of the heatmap-overlaid image.
    """
    try:
        target_layer = _get_vit_target_layer(model)
        cam_engine = _GradCAMPlusPlus(model, target_layer)

        # Grad-CAM requires gradients
        image_tensor = image_tensor.clone().requires_grad_(True)
        cam_2d = cam_engine.generate(image_tensor, class_idx=1)   # class 1 = Fake

        # Resize heatmap to original image size
        orig_w, orig_h = original_pil.size
        heatmap_resized = cv2.resize(cam_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        original_rgb = np.array(original_pil.convert("RGB"))
        overlay_bgr = _overlay_heatmap_on_image(heatmap_resized, original_rgb)

        return _numpy_bgr_to_base64(overlay_bgr)

    except Exception as e:
        logger.error(f"[XAI] Grad-CAM++ generation failed: {e}", exc_info=True)
        # Return transparent placeholder (original image un-annotated)
        return _pil_to_base64(original_pil.convert("RGB"))


# ---------------------------------------------------------------------------
# LIME – Local Interpretable Model-Agnostic Explanations
# ---------------------------------------------------------------------------

def generate_lime(model, original_pil: Image.Image, device: str) -> str:
    """
    Generate a LIME superpixel explanation for the GenD model.

    Args:
        model: The loaded GenD model (eval mode).
        original_pil: The raw PIL image.
        device: Torch device string ('cpu' or 'cuda').

    Returns:
        base64-encoded JPEG string of the LIME boundary overlay image.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        explainer = lime_image.LimeImageExplainer(verbose=False)

        def _predict_batch(images: np.ndarray) -> np.ndarray:
            """
            Prediction function expected by LIME: takes (N, H, W, 3) uint8 numpy array,
            returns (N, num_classes) probability array.
            """
            probs_list = []
            with torch.no_grad():
                for img_arr in images:
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    tensor = model.feature_extractor.preprocess(pil_img).unsqueeze(0).to(device)
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    probs_list.append(probs)
            return np.array(probs_list)

        original_rgb = np.array(original_pil.convert("RGB"))

        explanation = explainer.explain_instance(
            original_rgb,
            _predict_batch,
            top_labels=1,
            hide_color=0,
            num_samples=100,      # balance quality vs. speed for real-time use
            batch_size=10,
        )

        # Visualise the top label superpixels (positive contributions only)
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=10,
            hide_rest=False,
        )

        lime_img = mark_boundaries(temp / 255.0, mask)      # float [0,1] RGB
        lime_img_uint8 = (lime_img * 255).astype(np.uint8)
        lime_bgr = cv2.cvtColor(lime_img_uint8, cv2.COLOR_RGB2BGR)

        return _numpy_bgr_to_base64(lime_bgr)

    except Exception as e:
        logger.error(f"[XAI] LIME generation failed: {e}", exc_info=True)
        return _pil_to_base64(original_pil.convert("RGB"))
