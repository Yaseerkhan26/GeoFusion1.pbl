"""
===============================================================================
File: src/models/inference.py
Purpose: Safe AI Model Inference, Softmax Confidence, and Shannon Entropy Uncertainty
===============================================================================
"""

import math
from typing import Any, Dict, Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

from src.models.config import NUM_CLASSES
from src.utils.config import CLASS_COLORMAP, CLASSES


def compute_entropy_uncertainty(probs: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    """
    Computes normalized Shannon entropy uncertainty map from class probabilities:
        H(p) = - sum(p_i * ln(p_i)) / ln(K)
    Normalized to [0.0, 1.0], where:
      0.0 -> Total certainty (one class probability is 1.0)
      1.0 -> Complete uncertainty (uniform distribution across all K classes)
    """
    # probs shape: [num_classes, H, W]
    num_classes = probs.shape[0]
    log_k = math.log(num_classes)

    p_clamped = torch.clamp(probs, min=eps, max=1.0)
    entropy = -torch.sum(p_clamped * torch.log(p_clamped), dim=0)
    normalized_entropy = entropy / log_k
    return torch.clamp(normalized_entropy, 0.0, 1.0).cpu().numpy()


def run_model_inference(
    model: torch.nn.Module,
    s1_tensor: torch.Tensor,
    s2_tensor: torch.Tensor,
    device: torch.device,
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Performs verified model inference on a patch triplet with strict numerical safety checks.
    Inputs:
      s1_tensor: [2, 256, 256] or [1, 2, 256, 256]
      s2_tensor: [6, 256, 256] or [1, 6, 256, 256]
    Returns dictionary with:
      - predictions: argmax class indices [256, 256]
      - confidence_map: max class probability [256, 256]
      - uncertainty_map: normalized Shannon entropy [256, 256]
      - filtered_predictions: class indices with -1 for pixels < confidence_threshold
      - mean_confidence: float (0..1)
      - mean_uncertainty: float (0..1)
      - uncertainty_tiers: dict of low/medium/high counts and percentages
      - class_counts: dict of class name -> count
    """
    # 1. Numerical safety: validate tensor integrity
    if not isinstance(s1_tensor, torch.Tensor) or not isinstance(s2_tensor, torch.Tensor):
        raise TypeError("Inputs s1_tensor and s2_tensor must be torch.Tensor instances.")

    if torch.isnan(s1_tensor).any() or torch.isinf(s1_tensor).any():
        raise ValueError("Input s1_tensor contains NaN or Infinite values.")

    if torch.isnan(s2_tensor).any() or torch.isinf(s2_tensor).any():
        raise ValueError("Input s2_tensor contains NaN or Infinite values.")

    # 2. Shape validation
    s1_shape = s1_tensor.shape
    s2_shape = s2_tensor.shape

    if s1_tensor.ndim == 3:
        if s1_shape[0] != 2 or s1_shape[1] != 256 or s1_shape[2] != 256:
            raise ValueError(f"Expected s1_tensor [2, 256, 256], got {list(s1_shape)}")
        s1_in = s1_tensor.unsqueeze(0).to(device)
    elif s1_tensor.ndim == 4:
        if s1_shape[1] != 2 or s1_shape[2] != 256 or s1_shape[3] != 256:
            raise ValueError(f"Expected s1_tensor [B, 2, 256, 256], got {list(s1_shape)}")
        s1_in = s1_tensor.to(device)
    else:
        raise ValueError(f"Invalid dimensions for s1_tensor: {s1_tensor.ndim}")

    if s2_tensor.ndim == 3:
        if s2_shape[0] != 6 or s2_shape[1] != 256 or s2_shape[2] != 256:
            raise ValueError(f"Expected s2_tensor [6, 256, 256], got {list(s2_shape)}")
        s2_in = s2_tensor.unsqueeze(0).to(device)
    elif s2_tensor.ndim == 4:
        if s2_shape[1] != 6 or s2_shape[2] != 256 or s2_shape[3] != 256:
            raise ValueError(f"Expected s2_tensor [B, 6, 256, 256], got {list(s2_shape)}")
        s2_in = s2_tensor.to(device)
    else:
        raise ValueError(f"Invalid dimensions for s2_tensor: {s2_tensor.ndim}")

    model.eval()

    with torch.no_grad():
        logits = model(s1_in, s2_in)  # Expected: [1, 8, 256, 256]
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Model output classes {logits.shape[1]} does not match NUM_CLASSES {NUM_CLASSES}")

        probs = F.softmax(logits, dim=1).squeeze(0)  # [8, 256, 256]

        # Validate softmax properties
        prob_sum = torch.sum(probs, dim=0)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
            raise ValueError("Softmax probabilities do not sum to 1.0 across classes.")

        conf_tensor, preds_tensor = torch.max(probs, dim=0)

        preds = preds_tensor.cpu().numpy()  # [256, 256]
        conf = conf_tensor.cpu().numpy()    # [256, 256]
        uncertainty = compute_entropy_uncertainty(probs)

    # Filtered predictions: pixels below confidence_threshold marked as -1 (Uncertain)
    filtered = preds.copy()
    if confidence_threshold > 0.0:
        filtered[conf < confidence_threshold] = -1

    # Class pixel counts
    total_pixels = int(preds.size)
    class_counts = {}
    for c_id in range(NUM_CLASSES):
        cnt = int(np.sum(preds == c_id))
        cls_name = CLASSES.get(c_id, f"Class {c_id}")
        class_counts[cls_name] = {
            "pixels": cnt,
            "percentage": round((cnt / total_pixels) * 100.0, 2),
            "color": CLASS_COLORMAP.get(c_id, "#888888"),
        }

    # Uncertainty tier categorization
    low_unc = int(np.sum(uncertainty < 0.30))
    med_unc = int(np.sum((uncertainty >= 0.30) & (uncertainty <= 0.60)))
    high_unc = int(np.sum(uncertainty > 0.60))

    uncertainty_tiers = {
        "Low (<0.30)": {"count": low_unc, "percentage": round(low_unc / total_pixels * 100.0, 2)},
        "Medium (0.30-0.60)": {"count": med_unc, "percentage": round(med_unc / total_pixels * 100.0, 2)},
        "High (>0.60)": {"count": high_unc, "percentage": round(high_unc / total_pixels * 100.0, 2)},
    }

    return {
        "predictions": preds,
        "confidence_map": conf,
        "uncertainty_map": uncertainty,
        "filtered_predictions": filtered,
        "mean_confidence": float(np.mean(conf)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "uncertainty_tiers": uncertainty_tiers,
        "class_counts": class_counts,
        "uncertain_pixel_count": int(np.sum(conf < confidence_threshold)),
    }


def colorize_categorical_map(
    class_map: np.ndarray,
    colormap: Dict[int, str] = CLASS_COLORMAP,
    uncertain_color: str = "#4A5568",
) -> np.ndarray:
    """
    Renders an integer land-cover map as an RGB image array [H, W, 3] in [0, 1].
    Handles -1 (Uncertain) using a distinct neutral gray.
    """
    h, w = class_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.float32)

    uncertain_rgb = mcolors.to_rgb(uncertain_color)

    for cls_id, hex_code in colormap.items():
        mask = (class_map == cls_id)
        if np.any(mask):
            rgb_img[mask] = mcolors.to_rgb(hex_code)

    # Uncertain pixels
    unc_mask = (class_map == -1)
    if np.any(unc_mask):
        rgb_img[unc_mask] = uncertain_rgb

    return rgb_img


def compute_prediction_error_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computes pixel-wise agreement between model prediction and test ground truth:
      - agreement_mask: True where prediction == ground_truth
      - error_mask: True where prediction != ground_truth
    """
    correct_mask = (prediction == ground_truth)
    error_mask = ~correct_mask
    total_pixels = int(prediction.size)
    correct_pixels = int(np.sum(correct_mask))
    incorrect_pixels = int(np.sum(error_mask))

    stats = {
        "total_pixels": total_pixels,
        "correct_pixels": correct_pixels,
        "incorrect_pixels": incorrect_pixels,
        "patch_accuracy": round((correct_pixels / total_pixels) * 100.0, 2),
    }

    return error_mask, stats
