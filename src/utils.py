import os
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int) -> None:
    """Fix random seeds across numpy, random, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> List[List[int]]:
    """Return 2x2 confusion matrix [[tn, fp], [fn, tp]]."""
    if predictions.size == 0:
        return []
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    return cm.tolist()


ERROR_BUCKETS: List[Tuple[str, int, int]] = [
    ("short (<80 tokens)", 0, 80),
    ("medium (80-160 tokens)", 80, 160),
    ("long (>=160 tokens)", 160, 10_000),
]


def compute_error_buckets(
    lengths: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Group accuracy stats by length buckets."""
    if lengths.size == 0 or lengths.shape[0] != labels.shape[0]:
        return {}
    buckets: Dict[str, Dict[str, float]] = {}
    for name, low, high in ERROR_BUCKETS:
        mask = (lengths >= low) & (lengths < high)
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((predictions[mask] == labels[mask]).sum())
        errors = total - correct
        buckets[name] = {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": float(correct / total),
            "error_rate": float(errors / total),
        }
    return buckets


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)