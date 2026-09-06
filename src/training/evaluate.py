"""
===============================================================================
File: src/training/evaluate.py
Purpose: Model Performance Evaluation, Accuracy Assessment, & Metrics Reporting

Description:
    Computes statistical evaluation metrics for land cover classification:
    - Overall Accuracy (OA)
    - Cohen's Kappa Coefficient (k)
    - Per-Class Intersection over Union (IoU / Jaccard Index)
    - Precision, Recall, and F1-score
    Generates and saves confusion matrix charts to outputs/graphs/ and
    metrics summary reports to outputs/reports/.
===============================================================================
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import CLASSES, GRAPHS_DIR, REPORTS_DIR


def calculate_metrics(y_true, y_pred):
    """
    Calculates key remote sensing evaluation metrics.

    Args:
        y_true (np.ndarray): Flattened array of ground truth class labels.
        y_pred (np.ndarray): Flattened array of model predicted class labels.

    Returns:
        dict: Summary dictionary containing OA, Kappa, and IoU values.
    """
    print("[STARTER] Calculating Overall Accuracy, Kappa, and Per-Class IoU...")
    
    # Calculate Overall Accuracy
    oa = np.mean(y_true == y_pred)
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "overall_accuracy": oa,
        "cohen_kappa": kappa,
        "confusion_matrix": cm
    }
    return metrics


def save_confusion_matrix_plot(cm, output_path=GRAPHS_DIR / "confusion_matrix.png"):
    """
    Renders and exports a visual confusion matrix graphic.

    Args:
        cm (np.ndarray): Confusion matrix array.
        output_path (Path): Path to output image file in outputs/graphs/.
    """
    print(f"[STARTER] Exporting confusion matrix graphic to: {output_path}")
    # TODO: Use Matplotlib/Seaborn to generate heatmaps with class labels and save figure


def generate_evaluation_report(metrics, output_path=REPORTS_DIR / "evaluation_report.txt"):
    """
    Writes formatted evaluation report text file to outputs/reports/.

    Args:
        metrics (dict): Metrics dictionary containing OA, Kappa, and per-class stats.
        output_path (Path): Path to output text report file.
    """
    print(f"[STARTER] Generating metrics text summary report at: {output_path}")
    # TODO: Format report text and write to output file


if __name__ == "__main__":
    print("=== Multimodal Model Evaluation Module ===")
    
    # Starter dummy predictions for metric test verification
    dummy_true = np.random.randint(1, 6, size=1000)
    dummy_pred = np.random.randint(1, 6, size=1000)
    
    results = calculate_metrics(dummy_true, dummy_pred)
    print(f"[TEST RESULTS] Overall Accuracy: {results['overall_accuracy'] * 100:.2f}%")
    print(f"[TEST RESULTS] Cohen's Kappa: {results['cohen_kappa']:.4f}")
