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
    """
    Parse prediction from model response with proper negation handling.
    Uses boundary-aware regex to avoid false matches.
    """
    import re
    
    response_lower = response.lower().strip()
    
    # Check for direct yes/no at the start (common for short model responses)
    # "Is this fake?" -> Yes = Fake, No = Real
    if response_lower.startswith("yes"):
        return 1  # Fake
    if response_lower.startswith("no ") or response_lower == "no":
        return 0  # Real (but not "not" which starts with "no")
    
    # Boundary-aware negation patterns (check FIRST - more specific)
    # Matches: "not fake", "not a fake", "isn't fake", "is not fake"
    not_fake_pattern = r'\b(not\s+(a\s+)?fake|isn\'?t\s+(a\s+)?fake|is\s+not\s+(a\s+)?fake)\b'
    not_real_pattern = r'\b(not\s+(a\s+)?real|isn\'?t\s+real|is\s+not\s+real)\b'
    not_deepfake_pattern = r'\b(not\s+(a\s+)?deepfake|isn\'?t\s+(a\s+)?deepfake)\b'
    not_manipulated_pattern = r'\b(not\s+manipulated|isn\'?t\s+manipulated)\b'
    
    # Check negations first
    if re.search(not_fake_pattern, response_lower):
        return 0  # "not fake" = Real
    if re.search(not_deepfake_pattern, response_lower):
        return 0  # "not deepfake" = Real
    if re.search(not_manipulated_pattern, response_lower):
        return 0  # "not manipulated" = Real
    if re.search(not_real_pattern, response_lower):
        return 1  # "not real" = Fake
    
    # Boundary-aware positive patterns (word boundaries prevent partial matches)
    fake_patterns = [
        r'\b(fake|deepfake|manipulated|synthetic|generated|altered|forged|artificial)\b',
    ]
    real_patterns = [
        r'\b(real|authentic|genuine|original|unaltered|natural)\b',
    ]
    
    # Check for fake indicators
    for pattern in fake_patterns:
        if re.search(pattern, response_lower):
            return 1
    
    # Check for real indicators
    for pattern in real_patterns:
        if re.search(pattern, response_lower):
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
    scores_are_proba: bool = False,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Binary predictions (0=real, 1=fake)
        labels: Ground truth labels
        confidences: Confidence/probability scores
        scores_are_proba: If True, treat confidences as P(fake) directly for AUC.
                          If False (legacy), transform based on predicted class.
    """
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
            "auc_valid_ratio": 0.0,
        }
    
    # Classification metrics on ALL valid predictions (pred != -1)
    metrics = {
        "accuracy": accuracy_score(filtered_labels, filtered_preds),
        "precision": precision_score(filtered_labels, filtered_preds, zero_division=0),
        "recall": recall_score(filtered_labels, filtered_preds, zero_division=0),
        "f1": f1_score(filtered_labels, filtered_preds, zero_division=0),
        "valid_ratio": len(filtered_preds) / len(predictions),
    }
    
    # AUC computed SEPARATELY on samples with valid (non-None) confidence scores
    if confidences:
        # Filter to only samples with valid predictions AND non-None confidence
        auc_mask = [v and (c is not None) for v, c in zip(valid_mask, confidences)]
        auc_preds = [p for p, m in zip(predictions, auc_mask) if m]
        auc_labels = [l for l, m in zip(labels, auc_mask) if m]
        auc_scores = [c for c, m in zip(confidences, auc_mask) if m]
        
        metrics["auc_valid_ratio"] = len(auc_scores) / len(predictions) if predictions else 0.0
        
        if auc_scores and len(set(auc_labels)) > 1:
            if scores_are_proba:
                # Use confidence directly as P(fake) for AUC
                scores = auc_scores
            else:
                # Legacy: transform confidence based on predicted class
                scores = [c if p == 1 else 1 - c for p, c in zip(auc_preds, auc_scores)]
            
            metrics["auc"] = roc_auc_score(auc_labels, scores)
        else:
            metrics["auc"] = 0.0
    else:
        # No confidences - use predictions for AUC
        metrics["auc_valid_ratio"] = metrics["valid_ratio"]
        if len(set(filtered_labels)) > 1:
            metrics["auc"] = roc_auc_score(filtered_labels, filtered_preds)
        else:
            metrics["auc"] = 0.0
    
    return metrics


def compute_per_manipulation_metrics(
    predictions: List[int],
    labels: List[int],
    manipulation_types: List[str],
    confidences: List[float] = None,
    scores_are_proba: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per manipulation type.
    
    Args:
        predictions: Binary predictions
        labels: Ground truth labels
        manipulation_types: Type of manipulation for each sample
        confidences: Confidence/probability scores (optional)
        scores_are_proba: If True, confidences are P(fake) for AUC
    """
    unique_types = set(manipulation_types)
    results = {}
    
    for manip_type in unique_types:
        mask = [m == manip_type for m in manipulation_types]
        
        type_preds = [p for p, m in zip(predictions, mask) if m]
        type_labels = [l for l, m in zip(labels, mask) if m]
        
        # Also filter confidences if provided
        type_conf = None
        if confidences is not None:
            type_conf = [c for c, m in zip(confidences, mask) if m]
        
        if type_preds:
            results[manip_type] = compute_metrics(
                type_preds, type_labels, type_conf, scores_are_proba
            )
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
