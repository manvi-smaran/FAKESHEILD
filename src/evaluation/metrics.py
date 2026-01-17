from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def parse_prediction(response: str) -> int:
    response_lower = response.lower().strip()
    
    # Check for direct yes/no at the start (common for short model responses)
    # "Is this fake?" -> Yes = Fake, No = Real
    # We assume the prompt asks "is this fake/deepfake?"
    if response_lower.startswith("yes"):
        return 1  # Fake
    if response_lower.startswith("no"):
        return 0  # Real
    
    fake_keywords = [
        "fake", "manipulated", "deepfake", "synthetic", 
        "generated", "altered", "forged", "artificial",
        "not real", "not authentic", "not genuine",
    ]
    real_keywords = [
        "real", "authentic", "genuine", "original", 
        "unaltered", "natural", "not fake", "not manipulated",
    ]
    
    # Check for negations first (more specific)
    for keyword in fake_keywords:
        if keyword.startswith("not ") and keyword in response_lower:
            return 1
    for keyword in real_keywords:
        if keyword.startswith("not ") and keyword in response_lower:
            return 0
    
    # Then check positive keywords
    for keyword in fake_keywords:
        if not keyword.startswith("not ") and keyword in response_lower:
            return 1
    
    for keyword in real_keywords:
        if not keyword.startswith("not ") and keyword in response_lower:
            return 0
    
    return -1


def extract_confidence(response: str) -> float:
    response_lower = response.lower()
    
    if "high" in response_lower:
        return 0.9
    elif "medium" in response_lower:
        return 0.6
    elif "low" in response_lower:
        return 0.3
    
    import re
    match = re.search(r'(\d{1,3})%', response)
    if match:
        return float(match.group(1)) / 100.0
    
    return 0.5


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    confidences: List[float] = None,
) -> Dict[str, float]:
    valid_mask = [p != -1 for p in predictions]
    
    filtered_preds = [p for p, v in zip(predictions, valid_mask) if v]
    filtered_labels = [l for l, v in zip(labels, valid_mask) if v]
    
    if not filtered_preds:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "valid_ratio": 0.0,
        }
    
    metrics = {
        "accuracy": accuracy_score(filtered_labels, filtered_preds),
        "precision": precision_score(filtered_labels, filtered_preds, zero_division=0),
        "recall": recall_score(filtered_labels, filtered_preds, zero_division=0),
        "f1": f1_score(filtered_labels, filtered_preds, zero_division=0),
        "valid_ratio": len(filtered_preds) / len(predictions),
    }
    
    if confidences:
        filtered_conf = [c for c, v in zip(confidences, valid_mask) if v]
        scores = [c if p == 1 else 1 - c for p, c in zip(filtered_preds, filtered_conf)]
        
        if len(set(filtered_labels)) > 1:
            metrics["auc"] = roc_auc_score(filtered_labels, scores)
        else:
            metrics["auc"] = 0.0
    else:
        if len(set(filtered_labels)) > 1:
            metrics["auc"] = roc_auc_score(filtered_labels, filtered_preds)
        else:
            metrics["auc"] = 0.0
    
    return metrics


def compute_per_manipulation_metrics(
    predictions: List[int],
    labels: List[int],
    manipulation_types: List[str],
) -> Dict[str, Dict[str, float]]:
    unique_types = set(manipulation_types)
    results = {}
    
    for manip_type in unique_types:
        mask = [m == manip_type for m in manipulation_types]
        
        type_preds = [p for p, m in zip(predictions, mask) if m]
        type_labels = [l for l, m in zip(labels, mask) if m]
        
        if type_preds:
            results[manip_type] = compute_metrics(type_preds, type_labels)
            results[manip_type]["count"] = len(type_preds)
    
    return results


def get_confusion_matrix(
    predictions: List[int],
    labels: List[int],
) -> np.ndarray:
    valid_mask = [p != -1 for p in predictions]
    
    filtered_preds = [p for p, v in zip(predictions, valid_mask) if v]
    filtered_labels = [l for l, v in zip(labels, valid_mask) if v]
    
    if not filtered_preds:
        return np.zeros((2, 2), dtype=int)
    
    return confusion_matrix(filtered_labels, filtered_preds, labels=[0, 1])


def format_results(
    metrics: Dict[str, float],
    model_name: str,
    dataset_name: str,
    eval_type: str,
) -> str:
    lines = [
        f"\n{'='*60}",
        f"Model: {model_name}",
        f"Dataset: {dataset_name}",
        f"Evaluation: {eval_type}",
        f"{'='*60}",
        f"AUC:       {metrics.get('auc', 0):.4f}",
        f"Accuracy:  {metrics.get('accuracy', 0):.4f}",
        f"Precision: {metrics.get('precision', 0):.4f}",
        f"Recall:    {metrics.get('recall', 0):.4f}",
        f"F1 Score:  {metrics.get('f1', 0):.4f}",
        f"Valid %:   {metrics.get('valid_ratio', 0)*100:.1f}%",
        f"{'='*60}\n",
    ]
    return "\n".join(lines)
