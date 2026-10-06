"""
===============================================================================
File: src/utils/provenance.py
Purpose: Scientific Provenance, Checkpoint Fingerprinting, and Evaluation Verification
===============================================================================
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

# Ensure output directories exist
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    """
    Computes a deterministic fingerprint of the dataset splits and class mapping.
    Uses SHA-256 of train.csv, val.csv, test.csv, and class_mapping.json.
    """
    splits_dir = data_dir / "splits"
    mapping_path = data_dir / "class_mapping.json"

    components = {}
    h = hashlib.sha256()

    for name in ["train.csv", "val.csv", "test.csv"]:
        fpath = splits_dir / name
        if fpath.exists():
            file_hash = compute_file_sha256(fpath)
            # Count lines for quick verification with proper context manager
            with open(fpath, "rb") as f:
                line_count = sum(1 for _ in f) - 1  # exclude header
            components[name] = {"sha256": file_hash, "samples": line_count}
            h.update(f"{name}:{file_hash}".encode())
        else:
            components[name] = {"sha256": None, "samples": 0}

    if mapping_path.exists():
        map_hash = compute_file_sha256(mapping_path)
        components["class_mapping.json"] = {"sha256": map_hash}
        h.update(f"class_mapping:{map_hash}".encode())

    combined_fingerprint = h.hexdigest()[:16]
    return {
        "dataset_fingerprint": combined_fingerprint,
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }


def get_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Extracts deep metadata, parameter counts, compatibility, and hash for a model checkpoint.
    """
    if not checkpoint_path.exists():
        return {"exists": False, "filename": checkpoint_path.name}

    stat = checkpoint_path.stat()
    file_size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = compute_file_sha256(checkpoint_path)

    # Inspect weights
    meta = {
        "exists": True,
        "filename": checkpoint_path.name,
        "path": str(checkpoint_path),
        "size_bytes": file_size,
        "size_kb": round(file_size / 1024, 1),
        "modified": mtime,
        "sha256": sha256_hash,
        "sha256_short": sha256_hash[:16],
        "compatible": False,
        "architecture": "Unknown",
        "num_classes": None,
        "keys_count": 0,
        "provenance_notes": "",
    }

    try:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict):
            meta["keys_count"] = len(sd.keys())
            # Check if this is MultimodalFusionNet
            has_s1_enc = any(k.startswith("s1_encoder") for k in sd.keys())
            has_s2_enc = any(k.startswith("s2_encoder") for k in sd.keys())
            has_classifier = any(k.startswith("classifier") for k in sd.keys())

            if has_s1_enc and has_s2_enc and has_classifier:
                meta["architecture"] = "MultimodalFusionNet"
                # Check output classes from classifier.6.weight shape [num_classes, 16, 1, 1]
                if "classifier.6.weight" in sd:
                    shape = sd["classifier.6.weight"].shape
                    meta["num_classes"] = shape[0]
                    if shape[0] == 8:
                        meta["compatible"] = True
                elif "classifier.6.bias" in sd:
                    meta["num_classes"] = sd["classifier.6.bias"].shape[0]
                    if meta["num_classes"] == 8:
                        meta["compatible"] = True
            elif has_s1_enc:
                meta["architecture"] = "Sentinel1OnlyNet"
            elif has_s2_enc:
                meta["architecture"] = "Sentinel2OnlyNet"

            # Assign provenance notes based on known checkpoint training history
            if "colab" in checkpoint_path.name.lower():
                meta["provenance_notes"] = "Trained on Google Colab (20 Epochs, Weighted CE Loss)"
            elif checkpoint_path.name == "fusion_best.pth":
                meta["provenance_notes"] = "Local 2-epoch dry run"
            elif checkpoint_path.name == "fusion_model_best.pth":
                meta["provenance_notes"] = "Legacy baseline checkpoint"
    except Exception as e:
        meta["error"] = str(e)

    return meta


def get_all_checkpoints(models_dir: Path = MODELS_DIR) -> List[Dict[str, Any]]:
    """Discovers and returns metadata for all .pth checkpoints in models directory."""
    checkpoints = []
    if not models_dir.exists():
        return checkpoints

    for p in sorted(models_dir.glob("*.pth")):
        checkpoints.append(get_checkpoint_metadata(p))
    return checkpoints


def get_evaluation_filepath(checkpoint_name: str, eval_dir: Path = EVAL_DIR) -> Path:
    """Returns the dedicated evaluation JSON path for a given checkpoint."""
    stem = Path(checkpoint_name).stem
    return eval_dir / f"{stem}_evaluation.json"


def load_evaluation_for_checkpoint(checkpoint_path: Path, eval_dir: Path = EVAL_DIR) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Loads evaluation artifacts for a checkpoint and determines its status:
    Returns (status, eval_data):
      - status: 'VALIDATED', 'STALE', 'PENDING', or 'UNKNOWN'
    """
    if not checkpoint_path.exists():
        return "UNKNOWN", None

    eval_file = get_evaluation_filepath(checkpoint_path.name, eval_dir)
    if not eval_file.exists():
        # Check if legacy evaluation_results.txt exists
        legacy_txt = eval_dir / "evaluation_results.txt"
        if checkpoint_path.name == "fusion_model_best.pth" and legacy_txt.exists():
            # Legacy evaluation is present for fusion_model_best.pth
            return "STALE", None
        return "PENDING", None

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        current_hash = compute_file_sha256(checkpoint_path)
        recorded_hash = eval_data.get("checkpoint_sha256", "")

        current_ds_fingerprint = compute_dataset_fingerprint()["dataset_fingerprint"]
        recorded_ds_fingerprint = eval_data.get("dataset_fingerprint", "")

        # Strict provenance verification
        if current_hash != recorded_hash:
            return "STALE", eval_data

        if current_ds_fingerprint != recorded_ds_fingerprint:
            return "STALE", eval_data

        return "VALIDATED", eval_data

    except Exception:
        return "UNKNOWN", None


def save_evaluation_record(
    checkpoint_path: Path,
    metrics: Dict[str, Any],
    per_class_metrics: List[Dict[str, Any]],
    confusion_matrix: List[List[int]],
    eval_dir: Path = EVAL_DIR,
) -> Path:
    """
    Saves a cryptographically verifiable evaluation record for a checkpoint.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta = get_checkpoint_metadata(checkpoint_path)
    ds_meta = compute_dataset_fingerprint()

    record = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": ckpt_meta["sha256"],
        "checkpoint_modified": ckpt_meta["modified"],
        "checkpoint_size_bytes": ckpt_meta["size_bytes"],
        "architecture": ckpt_meta["architecture"],
        "dataset_fingerprint": ds_meta["dataset_fingerprint"],
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
    }

    eval_file = get_evaluation_filepath(checkpoint_path.name, eval_dir)
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)

    # Also save per-class CSV for backward compatibility / table inspection
    csv_file = eval_dir / f"{checkpoint_path.stem}_per_class.csv"
    import pandas as pd
    pd.DataFrame(per_class_metrics).to_csv(csv_file, index=False)

    return eval_file
