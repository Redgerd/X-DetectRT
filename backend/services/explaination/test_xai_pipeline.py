#!/usr/bin/env python3
"""
XAI Pipeline Test Runner
========================
Tests all 7 XAI techniques with a real or dummy input image.

Usage:
    python test_xai_pipeline.py --image path/to/face.jpg
    python test_xai_pipeline.py                      # uses a random dummy image

Output:
    xai_test_outputs/
        01_shap_timeshap.png
        07_lime_superpixels.png
        08_integrated_gradients.png
        09_sam_attribution.png
        10_counterfactual.png
        11_tcav.png
        12_prototype_analysis.png
"""

import argparse
import base64
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── ensure backend/ is on path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explaination.pipeline import run_xai_pipeline

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-7s %(name)s — %(message)s"
)

# ── Output filename mapping ─────────────────────────────────────────────────
OUTPUT_FILENAMES = {
    "shap_timeshap":        "01_shap_timeshap.png",
    "lime_superpixels":     "07_lime_superpixels.png",
    "integrated_gradients": "08_integrated_gradients.png",
    "sam_attribution":      "09_sam_attribution.png",
    "counterfactual":       "10_counterfactual.png",
    "tcav":                 "11_tcav.png",
    "prototype_analysis":   "12_prototype_analysis.png",
}

# ── Mock Model (mirrors the real GenD API) ──────────────────────────────────
class _MockFeatureExtractor:
    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Return a random (C,H,W) tensor for any input image."""
        return torch.rand(3, 224, 224)

class MockModel(nn.Module):
    """
    Minimal stand-in for the real GenD model.
    Provides the same interface used by all XAI modules:
        model.feature_extractor.preprocess(pil)  -> tensor
        model(tensor)                             -> logits
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = _MockFeatureExtractor()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Argument Parsing ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all XAI techniques on a face image and save professional PNG charts."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the input face image (JPEG/PNG). "
             "If omitted, a random dummy image is generated.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Directory to save output PNG files (default: outputs/).",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=50,
        help="Number of LIME perturbation samples (default: 50, use 300+ for accuracy).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=20,
        help="Integrated Gradients interpolation steps (default: 20, use 50+ for accuracy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device to run inference on (default: cpu).",
    )
    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load / generate input image ──────────────────────────────────────
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)
        try:
            pil_image = Image.open(image_path).convert("RGB")
            print(f"[INFO] Loaded image: {image_path}  ({pil_image.size[0]}×{pil_image.size[1]} px)")
        except Exception as e:
            print(f"[ERROR] Could not open image '{image_path}': {e}")
            sys.exit(1)
    else:
        print("[INFO] No --image provided — using a random dummy 224×224 image.")
        pil_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

    # ── Set up mock model ──────────────────────────────────────────────────
    print("[INFO] Initializing mock model …")
    model = MockModel()
    model.eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    # ── Dummy temporal data for SHAP TimeShap ─────────────────────────────
    frame_tensors = [torch.rand(3, 224, 224) for _ in range(5)]
    frame_probs   = [0.15, 0.42, 0.78, 0.61, 0.33]

    # ── Pipeline config ───────────────────────────────────────────────────
    config = {
        "techniques":     list(OUTPUT_FILENAMES.keys()),
        "lime_num_samples": args.lime_samples,
        "ig_steps":         args.ig_steps,
    }

    # ── Run XAI pipeline ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" Running XAI Pipeline …")
    print("─" * 60)
    t_start = time.perf_counter()

    results = run_xai_pipeline(
        model=model,
        pil_image=pil_image,
        device=args.device,
        frame_tensors=frame_tensors,
        frame_probs=frame_probs,
        config=config,
    )

    total_elapsed = round(time.perf_counter() - t_start, 2)

    # ── Save outputs & print summary ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "─" * 60)
    print(f" {'Technique':<28} {'Status':<10} {'Time':>6}   Output file")
    print("─" * 60)

    for res in results:
        tech     = res["technique"]
        err      = res.get("error")
        elapsed  = res.get("elapsed_seconds", "??")
        filename = OUTPUT_FILENAMES.get(tech, f"{tech}.png")
        out_path = os.path.join(args.output_dir, filename)

        if err:
            print(f" {tech:<28} {'FAILED':<10} {str(elapsed):>6}s  {err[:55]}")
        else:
            # Save chart PNG
            fig_b64 = res.get("figure_base64")
            if fig_b64:
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(fig_b64))
                status = "OK"
            else:
                status = "no fig"

            print(f" {tech:<28} {status:<10} {str(elapsed):>6}s  {out_path}")

            # Print scores / narrative
            scores = res.get("scores")
            if scores:
                print(f"   {'Scores':}")
                for k, v in list(scores.items())[:5]:
                    print(f"     {k}: {v}")
                if len(scores) > 5:
                    print(f"     … and {len(scores) - 5} more")

            narrative = res.get("narrative") or res.get("top_match")
            if isinstance(narrative, list):
                print(f"   Narrative: {narrative[0]}")
            elif isinstance(narrative, str):
                print(f"   Top match: {narrative}")

        print()

    print("─" * 60)
    print(f" Total pipeline time: {total_elapsed}s")
    print(f" Output directory: {os.path.abspath(args.output_dir)}/")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
