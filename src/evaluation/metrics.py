from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
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


# =============================================================================
# Score Preparation for AUC_all vs AUC_valid Contract
# =============================================================================

def build_all_and_valid(
    labels: List[int],
    scores: List[Optional[float]],
    fill_value: float = 0.5,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float]:
    """
    Prepare scores for AUC_all and AUC_valid computation.
    
    Paper contract:
    - AUC_all: fill missing scores with 0.5
    - AUC_valid: only valid p_fake samples
    
    Args:
        labels: Ground truth labels
        scores: P(fake) scores (may contain None)
        fill_value: Value to fill for missing scores (default 0.5)
    
    Returns:
        (y_all, s_all): Arrays for AUC_all computation
        (y_valid, s_valid): Arrays for AUC_valid computation
        valid_ratio: Fraction of valid scores
    """
    y = np.asarray(labels)
    
    # Build valid mask before filling
    valid_mask = np.asarray([s is not None for s in scores], dtype=bool)
    
    # Fill missing with fill_value for _all metrics
    s_all = np.asarray(
        [fill_value if s is None else float(s) for s in scores], 
        dtype=float
    )
    
    # Extract valid-only arrays
    y_valid = y[valid_mask]
    s_valid = s_all[valid_mask]  # Already filled, but only valid indices
    
    valid_ratio = float(valid_mask.mean()) if len(valid_mask) > 0 else 0.0
    
    return (y, s_all), (y_valid, s_valid), valid_ratio


# =============================================================================
# Deployment Metrics (TPR@FPR, EER, PR-AUC)
# =============================================================================

def compute_tpr_at_fpr(
    labels: List[int],
    scores: List[float],
    fpr_target: float = 0.01,
) -> float:
    """
    Compute TPR at a given FPR threshold.
    
    Args:
        labels: Ground truth (0=real, 1=fake)
        scores: P(fake) scores
        fpr_target: Target FPR (default 0.01 = 1%)
    
    Returns:
        TPR at the given FPR level
    """
    if len(set(labels)) < 2 or len(scores) == 0:
        return 0.0
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find the TPR at the largest FPR <= fpr_target
    valid_idx = np.where(fpr <= fpr_target)[0]
    if len(valid_idx) == 0:
        return 0.0
    
    return float(tpr[valid_idx[-1]])


def compute_eer(labels: List[int], scores: List[float]) -> float:
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where FPR = FNR (1 - TPR).
    """
    if len(set(labels)) < 2 or len(scores) == 0:
        return 0.5
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the point where FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer)


def compute_pr_auc(labels: List[int], scores: List[float]) -> float:
    """Compute Area Under Precision-Recall Curve (Average Precision)."""
    if len(set(labels)) < 2 or len(scores) == 0:
        return 0.0
    
    return float(average_precision_score(labels, scores))


def compute_full_metrics(
    labels: List[int],
    scores: List[float],
) -> Dict[str, float]:
    """
    Compute all deployment-relevant metrics on a single (labels, scores) pair.
    
    Note: For _all vs _valid contract, use compute_full_metrics_contract() instead.
    
    Returns:
        Dict with ROC-AUC, PR-AUC, TPR@FPR=0.1%, TPR@FPR=1%, EER
    """
    if len(set(labels)) < 2 or len(scores) == 0:
        return {
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "tpr_at_fpr_0.1%": 0.0,
            "tpr_at_fpr_1%": 0.0,
            "eer": 0.5,
        }
    
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": compute_pr_auc(labels, scores),
        "tpr_at_fpr_0.1%": compute_tpr_at_fpr(labels, scores, 0.001),
        "tpr_at_fpr_1%": compute_tpr_at_fpr(labels, scores, 0.01),
        "eer": compute_eer(labels, scores),
    }


def compute_full_metrics_contract(
    labels: List[int],
    scores: List[Optional[float]],
    fill_value: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all metrics with the paper contract:
    - *_all: fill missing scores with 0.5
    - *_valid: only valid p_fake samples
    
    Args:
        labels: Ground truth labels
        scores: P(fake) scores (may contain None)
        fill_value: Value to fill for missing scores
    
    Returns:
        Dict with metrics_all, metrics_valid, and valid_ratio
    """
    (y_all, s_all), (y_valid, s_valid), valid_ratio = build_all_and_valid(
        labels, scores, fill_value
    )
    
    # Compute _all metrics
    metrics_all = compute_full_metrics(y_all.tolist(), s_all.tolist())
    metrics_all = {f"{k}_all": v for k, v in metrics_all.items()}
    
    # Compute _valid metrics
    if len(y_valid) > 0 and len(set(y_valid)) > 1:
        metrics_valid = compute_full_metrics(y_valid.tolist(), s_valid.tolist())
    else:
        metrics_valid = {
            "roc_auc": 0.0, "pr_auc": 0.0, 
            "tpr_at_fpr_0.1%": 0.0, "tpr_at_fpr_1%": 0.0, "eer": 0.5
        }
    metrics_valid = {f"{k}_valid": v for k, v in metrics_valid.items()}
    
    # Combine
    result = {**metrics_all, **metrics_valid, "valid_ratio": valid_ratio}
    return result


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    labels: List[int],
    scores: List[float],
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        labels: Ground truth labels
        scores: Predicted scores
        metric_fn: Function(labels, scores) -> float
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95)
        seed: Random seed
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    n = len(labels)
    
    if n == 0:
        return 0.0, 0.0, 0.0
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Point estimate
    point_estimate = metric_fn(labels, scores)
    
    # Bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_scores = scores[idx]
        
        # Skip if only one class in bootstrap sample
        if len(set(boot_labels)) < 2:
            continue
        
        try:
            val = metric_fn(boot_labels, boot_scores)
            bootstrap_values.append(val)
        except:
            continue
    
    if len(bootstrap_values) == 0:
        return point_estimate, point_estimate, point_estimate
    
    # Compute CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return float(point_estimate), float(lower), float(upper)


def compute_metrics_with_ci(
    labels: List[int],
    scores: List[float],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute all metrics with bootstrap 95% confidence intervals.
    
    Returns:
        Dict with metric: (point, lower, upper) for each metric
    """
    results = {}
    
    # ROC-AUC
    results["roc_auc"] = bootstrap_ci(
        labels, scores, 
        lambda y, s: roc_auc_score(y, s) if len(set(y)) > 1 else 0.0,
        n_bootstrap
    )
    
    # PR-AUC
    results["pr_auc"] = bootstrap_ci(
        labels, scores,
        lambda y, s: average_precision_score(y, s) if len(set(y)) > 1 else 0.0,
        n_bootstrap
    )
    
    # TPR@FPR=1%
    results["tpr_at_fpr_1%"] = bootstrap_ci(
        labels, scores,
        lambda y, s: compute_tpr_at_fpr(y.tolist(), s.tolist(), 0.01),
        n_bootstrap
    )
    
    # TPR@FPR=0.1%
    results["tpr_at_fpr_0.1%"] = bootstrap_ci(
        labels, scores,
        lambda y, s: compute_tpr_at_fpr(y.tolist(), s.tolist(), 0.001),
        n_bootstrap
    )
    
    # EER
    results["eer"] = bootstrap_ci(
        labels, scores,
        lambda y, s: compute_eer(y.tolist(), s.tolist()),
        n_bootstrap
    )
    
    return results


def compute_contract_metrics_with_ci(
    labels: List[int],
    scores: List[Optional[float]],
    fill_value: float = 0.5,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute all metrics with bootstrap CIs using the paper contract:
    - all_ci: metrics on scores filled with 0.5 for missing
    - valid_ci: metrics on valid scores only
    
    This is the stats-hygiene-approved approach for reviewers.
    
    Args:
        labels: Ground truth labels
        scores: P(fake) scores (may contain None)
        fill_value: Value to fill for missing scores
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dict with 'valid_ratio', 'all_ci', 'valid_ci'
    """
    (y_all, s_all), (y_valid, s_valid), valid_ratio = build_all_and_valid(
        labels, scores, fill_value
    )
    
    result = {
        "valid_ratio": valid_ratio,
        "n_total": len(labels),
        "n_valid": len(y_valid),
    }
    
    # CI on _all scores (filled)
    result["all_ci"] = compute_metrics_with_ci(
        y_all.tolist(), s_all.tolist(), n_bootstrap
    )
    
    # CI on _valid scores only
    if len(y_valid) > 0 and len(set(y_valid)) > 1:
        result["valid_ci"] = compute_metrics_with_ci(
            y_valid.tolist(), s_valid.tolist(), n_bootstrap
        )
    else:
        result["valid_ci"] = {}
    
    return result


