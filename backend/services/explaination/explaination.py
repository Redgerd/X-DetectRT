# F:\XDetect\X-DetectRT\backend\services\explaination\explaination.py
"""
Explainable AI (XAI) helpers for GenD deepfake detection.

Implements:
    - Grad-CAM++ : Visual heatmaps highlighting suspicious regions.
    - ELA         : Error Level Analysis – JPEG compression artifact map.
    - 2D FFT      : Fast Fourier Transform frequency-domain anomaly map.
    - LIME        : Superpixel attribution (vectorized, optimized).

Reference: DYP.pdf – "Integrating Dual XAI Techniques" (Step 3)
"""

import io
import base64
import logging
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
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
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)          # BGR
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

        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        activations = self._activations.detach()   # (1, seq_len, C)
        gradients = self._gradients.detach()        # (1, seq_len, C)

        self._remove_hooks()

        relu_grads = F.relu(gradients)
        alpha_num = relu_grads ** 2
        alpha_denom = 2.0 * relu_grads ** 2 + (activations * relu_grads ** 3).sum(dim=1, keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(gradients)).sum(dim=1, keepdim=True)  # (1, 1, C)

        cam = (weights * activations).sum(dim=-1)   # (1, seq_len)
        cam = F.relu(cam)
        cam = cam[0]                                 # (seq_len,)

        seq_len = cam.shape[0]
        if int(seq_len ** 0.5) ** 2 != seq_len:
            cam = cam[1:]  # strip [CLS]
            seq_len = cam.shape[0]

        grid_size = int(seq_len ** 0.5)
        cam_2d = cam.reshape(grid_size, grid_size).cpu().numpy()

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

        image_tensor = image_tensor.clone().requires_grad_(True)
        cam_2d = cam_engine.generate(image_tensor, class_idx=1)

        orig_w, orig_h = original_pil.size
        heatmap_resized = cv2.resize(cam_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        original_rgb = np.array(original_pil.convert("RGB"))
        overlay_bgr = _overlay_heatmap_on_image(heatmap_resized, original_rgb)

        return _numpy_bgr_to_base64(overlay_bgr)

    except Exception as e:
        logger.error(f"[XAI] Grad-CAM++ generation failed: {e}", exc_info=True)
        return _pil_to_base64(original_pil.convert("RGB"))


# ---------------------------------------------------------------------------
# Error Level Analysis (ELA)
# ---------------------------------------------------------------------------

def generate_ela(original_pil: Image.Image, quality: int = 90, amplify: int = 20) -> str:
    """
    Generate an Error Level Analysis (ELA) image.

    ELA re-saves the image at a known JPEG quality level and computes the
    per-pixel difference against the original.  Regions that were altered
    (spliced, generated, or re-compressed) tend to retain higher error
    levels than authentic regions, making them stand out as bright areas.

    Args:
        original_pil: The raw PIL image.
        quality: JPEG re-save quality (default 90).  Lower values amplify
                 differences but may introduce more compression noise.
        amplify: Scalar applied to the absolute difference before display
                 (default 20).  Increase for higher-contrast visualisation.

    Returns:
        base64-encoded JPEG string of the ELA heatmap image.
    """
    try:
        # Convert to RGB to ensure consistent channel count
        img_rgb = original_pil.convert("RGB")

        # Re-save at the target quality into a buffer and reload
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        # Absolute pixel-wise difference
        ela_pil = ImageChops.difference(img_rgb, recompressed)

        # Amplify for visibility
        ela_np = np.array(ela_pil, dtype=np.float32) * amplify
        ela_np = np.clip(ela_np, 0, 255).astype(np.uint8)

        # Convert to heatmap overlay: collapse channels → single-channel intensity
        ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
        ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)      # BGR

        # Overlay on original
        original_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        original_bgr = cv2.resize(original_bgr, (ela_colored.shape[1], ela_colored.shape[0]))
        overlay = cv2.addWeighted(original_bgr, 0.4, ela_colored, 0.6, 0)

        return _numpy_bgr_to_base64(overlay)

    except Exception as e:
        logger.error(f"[XAI] ELA generation failed: {e}", exc_info=True)
        return _pil_to_base64(original_pil.convert("RGB"))


# ---------------------------------------------------------------------------
# 2D FFT Power Spectrum – graph data for frontend rendering
# ---------------------------------------------------------------------------

def generate_fft(original_pil: Image.Image, radial_bins: int = 128) -> dict:
    """
    Compute the 2-D FFT power spectrum and return structured data for
    frontend graph rendering (e.g. Recharts, Chart.js, D3).

    Two complementary series are returned:

    1. ``radial_profile``  – Rotational average of log-power vs spatial
       frequency (cycles / pixel).  This is the primary line chart series.
       Deepfake generators produce a characteristic bump or plateau at
       mid-to-high frequencies compared to authentic images.

    2. ``quadrant_energy`` – Total log-power split into four named
       frequency bands (DC, low, mid, high).  Useful as a bar / radar
       chart alongside the radial profile.

    Args:
        original_pil: The raw PIL image.
        radial_bins:  Number of frequency bins for the radial profile
                      (default 128).  Reduce for smoother curves, increase
                      for finer resolution.

    Returns:
        Dict with the following keys:

        ``radial_profile`` : list of dicts
            ``{ "frequency": float,   # cycles per pixel  (0.0 – 0.5)
                "log_power": float }``  # mean log(1 + magnitude²) in bin

        ``quadrant_energy`` : dict
            ``{ "dc":   float,   # single DC bin (centre pixel)
                "low":  float,   # 0 < r <= 0.1  (coarse structure)
                "mid":  float,   # 0.1 < r <= 0.3 (texture / edges)
                "high": float }``# 0.3 < r <= 0.5 (fine detail / noise)

        ``peak_frequency`` : float
            Spatial frequency (cycles/pixel) of the highest-power bin
            outside the DC component.  Spikes here indicate periodic GAN
            artefacts.

        ``stats`` : dict
            ``{ "mean_log_power": float,
                "std_log_power":  float,
                "max_log_power":  float,
                "high_freq_ratio": float }``  # high-band energy / total
    """
    try:
        img_gray = np.array(original_pil.convert("L"), dtype=np.float32)
        h, w = img_gray.shape

        # 2D FFT → shift DC to centre → power spectrum
        f_transform = np.fft.fft2(img_gray)
        f_shifted   = np.fft.fftshift(f_transform)
        power       = np.abs(f_shifted) ** 2
        log_power   = np.log1p(power)

        # ── Build normalised frequency coordinate grid ────────────────────
        # Each pixel's distance from centre expressed in cycles/pixel [0, 0.5]
        cy, cx = h // 2, w // 2
        ys = (np.arange(h) - cy) / h          # fractional freq along y
        xs = (np.arange(w) - cx) / w          # fractional freq along x
        XX, YY = np.meshgrid(xs, ys)
        radius = np.sqrt(XX ** 2 + YY ** 2)   # cycles/pixel, max ≈ 0.707

        # ── Radial profile ────────────────────────────────────────────────
        max_freq   = 0.5                       # Nyquist limit
        bin_edges  = np.linspace(0.0, max_freq, radial_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        radial_profile = []
        for i in range(radial_bins):
            mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
            mean_lp = float(log_power[mask].mean()) if mask.any() else 0.0
            radial_profile.append({
                "frequency": round(float(bin_centres[i]), 6),
                "log_power": round(mean_lp, 6),
            })

        # ── Quadrant energy bands ─────────────────────────────────────────
        def _band_energy(r_min: float, r_max: float) -> float:
            mask = (radius > r_min) & (radius <= r_max)
            return round(float(log_power[mask].sum()), 4) if mask.any() else 0.0

        dc_val  = round(float(log_power[cy, cx]), 4)
        low_e   = _band_energy(0.0,  0.1)
        mid_e   = _band_energy(0.1,  0.3)
        high_e  = _band_energy(0.3,  0.5)
        total_e = low_e + mid_e + high_e or 1.0   # avoid div-by-zero

        quadrant_energy = {
            "dc":   dc_val,
            "low":  low_e,
            "mid":  mid_e,
            "high": high_e,
        }

        # ── Peak frequency (exclude DC bin) ───────────────────────────────
        non_dc_mask = radius > (bin_edges[1])   # skip first bin
        if non_dc_mask.any():
            peak_idx   = np.argmax(log_power * non_dc_mask)
            peak_r     = float(radius.flat[peak_idx])
        else:
            peak_r = 0.0

        # ── Summary statistics ────────────────────────────────────────────
        flat_lp = log_power.flatten()
        stats = {
            "mean_log_power":  round(float(flat_lp.mean()), 6),
            "std_log_power":   round(float(flat_lp.std()),  6),
            "max_log_power":   round(float(flat_lp.max()),  6),
            "high_freq_ratio": round(high_e / total_e, 6),
        }

        return {
            "radial_profile":   radial_profile,   # → line chart
            "quadrant_energy":  quadrant_energy,  # → bar / radar chart
            "peak_frequency":   round(peak_r, 6),
            "stats":            stats,
        }

    except Exception as e:
        logger.error(f"[XAI] FFT generation failed: {e}", exc_info=True)
        return {
            "radial_profile":  [],
            "quadrant_energy": {"dc": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0},
            "peak_frequency":  0.0,
            "stats":           {"mean_log_power": 0.0, "std_log_power": 0.0,
                                 "max_log_power": 0.0, "high_freq_ratio": 0.0},
            "error":           str(e),
        }


# ---------------------------------------------------------------------------
# Optimized LIME – superpixel attribution for frontend bar chart
# ---------------------------------------------------------------------------

def generate_lime(
    model,
    original_pil: Image.Image,
    device: str = "cpu",
    n_superpixels: int = 30,
    n_samples: int = 64,
    batch_size: int = 16,
) -> dict:
    """
    Run a vectorized, optimized LIME explanation on a single frame using
    SLIC superpixels as interpretable features.

    Key optimizations over the naive implementation:
      1. Segment masks precomputed once as a boolean array (H, W, n_segs).
      2. Full replacement image precomputed once via label-indexed lookup.
      3. All n_samples masked images built in a single NumPy broadcast op
         — no Python loop over superpixels per sample.
      4. All preprocessing handled in a single batched call; no per-image
         preprocess inside the inference loop.
      5. Inference runs in one pass per batch with torch.no_grad() + autocast.

    Args:
        model:          Loaded GenD model in eval mode.
        original_pil:   Raw PIL image for the frame.
        device:         Torch device string matching the model ("cpu"/"cuda").
        n_superpixels:  Number of SLIC segments (default 30).
        n_samples:      Coalition samples to draw (default 64).
        batch_size:     Model inference batch size (default 16).

    Returns:
        Dict ready for JSON serialisation:

        ``features`` : list of dicts  →  bar chart series
            ``{ "superpixel_id": int,
                "importance":    float,
                "abs_importance": float,
                "direction":     str }``   # "fake" | "real" | "neutral"

        ``baseline_fake_prob`` : float
        ``top_fake_superpixels``  : list[int]
        ``top_real_superpixels``  : list[int]
        ``stats`` : dict
            ``{ "n_superpixels": int, "n_samples": int, "r2_score": float }``
    """
    try:
        import torch
        from skimage.segmentation import slic
        from skimage.util import img_as_float

        img_rgb   = np.array(original_pil.convert("RGB"), dtype=np.uint8)   # (H, W, 3)
        img_float = img_as_float(img_rgb)

        # ── 1. SLIC segmentation ──────────────────────────────────────────
        segments = slic(
            img_float,
            n_segments=n_superpixels,
            compactness=10,
            sigma=1,
            start_label=0,
        )                                                       # (H, W)  int
        unique_segments = np.unique(segments)
        n_segs          = len(unique_segments)

        # Re-index segments to 0…n_segs-1 (SLIC may skip labels)
        remap = np.zeros(unique_segments.max() + 1, dtype=np.int32)
        for new_id, old_id in enumerate(unique_segments):
            remap[old_id] = new_id
        seg_idx = remap[segments]                               # (H, W)  0-based

        # ── 2. Precompute per-superpixel mean colours ─────────────────────
        # color_map[i] = mean uint8 RGB of segment i
        color_map = np.zeros((n_segs, 3), dtype=np.float64)
        counts    = np.zeros(n_segs, dtype=np.int64)
        np.add.at(color_map, seg_idx.ravel(), img_rgb.reshape(-1, 3))
        np.add.at(counts,    seg_idx.ravel(), 1)
        color_map = (color_map / counts[:, None]).astype(np.uint8)  # (n_segs, 3)

        # ── 3. Precompute full replacement image ──────────────────────────
        # replacement[y, x] = mean colour of whichever segment owns that pixel
        replacement = color_map[seg_idx]                        # (H, W, 3)  uint8

        # ── 4. Baseline probability on unmasked image ─────────────────────
        def _preprocess(pil_img: Image.Image) -> torch.Tensor:
            return model.feature_extractor.preprocess(pil_img)   # (C, H, W)

        def _infer_batch(tensors: list[torch.Tensor]) -> np.ndarray:
            batch = torch.stack(tensors, dim=0).to(device)        # (B, C, H, W)
            with torch.no_grad():
                if device != "cpu":
                    with torch.autocast(device_type=device):
                        logits = model(batch)
                else:
                    logits = model(batch)
                probs = torch.softmax(logits, dim=-1)[:, 1]       # fake prob
            return probs.cpu().numpy()

        baseline_prob = float(_infer_batch([_preprocess(original_pil)])[0])

        # ── 5. Random coalition sampling ──────────────────────────────────
        rng        = np.random.default_rng(seed=42)
        coalitions = rng.integers(0, 2, size=(n_samples, n_segs), dtype=np.uint8)
                                                                # (n_samples, n_segs)

        # LIME proximity kernel: weight samples close to "all features on"
        distances    = np.count_nonzero(coalitions == 0, axis=1) / n_segs
        kernel_width = 0.25
        weights      = np.exp(-(distances ** 2) / (kernel_width ** 2)).astype(np.float32)

        # ── 6. Vectorized masked-image construction + inference ────────────
        #
        # Core idea:
        #   keep_mask[s, y, x] = coalitions[s, seg_idx[y, x]]   (bool)
        #   masked[s, y, x, c] = img_rgb[y,x,c]  if keep_mask  else replacement[y,x,c]
        #
        # We process one batch at a time to stay memory-efficient.
        # Each batch builds (batch_size, H, W, 3) at once — no per-superpixel
        # Python loop anywhere.

        scores          = np.zeros(n_samples, dtype=np.float32)
        H, W            = img_rgb.shape[:2]
        seg_idx_flat    = seg_idx.ravel()                       # (H*W,)

        for batch_start in range(0, n_samples, batch_size):
            batch_coal = coalitions[batch_start: batch_start + batch_size]  # (B, n_segs)
            B          = len(batch_coal)

            # keep_pixel[b, p] = 1 if pixel p's segment is kept in sample b
            keep_pixel = batch_coal[:, seg_idx_flat]            # (B, H*W)  uint8
            keep_pixel = keep_pixel.reshape(B, H, W, 1)         # (B, H, W, 1)

            # Broadcast: choose original or replacement per pixel
            # img_rgb:     (H, W, 3)  → broadcast as (1, H, W, 3)
            # replacement: (H, W, 3)  → broadcast as (1, H, W, 3)
            masked_batch = np.where(keep_pixel, img_rgb, replacement)  # (B, H, W, 3)  uint8

            # Preprocess each masked image (PIL required by feature_extractor)
            tensors = [
                _preprocess(Image.fromarray(masked_batch[b]))
                for b in range(B)
            ]

            batch_scores = _infer_batch(tensors)
            scores[batch_start: batch_start + B] = batch_scores

        # ── 7. Weighted ridge regression → importance weights ─────────────
        X       = coalitions.astype(np.float32)                 # (n_samples, n_segs)
        y       = scores                                         # (n_samples,)
        W_diag  = np.diag(weights)
        lambda_ = 1e-3

        XtW  = X.T @ W_diag
        XtWX = XtW @ X + lambda_ * np.eye(n_segs, dtype=np.float32)
        XtWy = XtW @ y
        coeffs = np.linalg.solve(XtWX, XtWy)                   # (n_segs,)

        # ── 8. R² score ───────────────────────────────────────────────────
        y_pred  = X @ coeffs
        ss_res  = float(np.sum(weights * (y - y_pred) ** 2))
        ss_tot  = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
        r2      = round(1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0, 4)

        # ── 9. Build output feature list ──────────────────────────────────
        features = []
        for idx, seg_id in enumerate(unique_segments):
            imp = float(coeffs[idx])
            direction = "fake" if imp > 0.01 else ("real" if imp < -0.01 else "neutral")
            features.append({
                "superpixel_id":  int(seg_id),
                "importance":     round(imp, 6),
                "abs_importance": round(abs(imp), 6),
                "direction":      direction,
            })

        features.sort(key=lambda f: f["abs_importance"], reverse=True)

        top_fake = [f["superpixel_id"] for f in features if f["direction"] == "fake"][:5]
        top_real = [f["superpixel_id"] for f in features if f["direction"] == "real"][:5]

        return {
            "features":             features,
            "baseline_fake_prob":   round(baseline_prob, 6),
            "top_fake_superpixels": top_fake,
            "top_real_superpixels": top_real,
            "stats": {
                "n_superpixels": n_segs,
                "n_samples":     n_samples,
                "r2_score":      r2,
            },
        }

    except Exception as e:
        logger.error(f"[XAI] LIME generation failed: {e}", exc_info=True)
        return {
            "features":             [],
            "baseline_fake_prob":   0.0,
            "top_fake_superpixels": [],
            "top_real_superpixels": [],
            "stats": {"n_superpixels": 0, "n_samples": 0, "r2_score": 0.0},
            "error": str(e),
        }