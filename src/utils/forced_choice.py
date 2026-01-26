"""
Robust forced-choice scoring for p_fake computation.

This module provides the correct way to compute p_fake from VLM logits:
- Full sequence log-probability (not just first token)
- Handles multi-token completions
- Handles padding correctly (left or right padding)
- Scores multiple whitespace variants for tokenizer robustness
- No dependence on generated text

Usage:
    from src.utils.forced_choice import forced_choice_p_fake
    
    p_fake, scores = forced_choice_p_fake(model, inputs, tokenizer)
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np


# =============================================================================
# WHITESPACE VARIANTS - Critical for cross-model robustness
# =============================================================================
REAL_CANDIDATES = [" Real", "Real", " real", "real"]
FAKE_CANDIDATES = [" Fake", "Fake", " fake", "fake"]

# Keys to drop when building extended inputs (avoid position/cache mismatch)
DROP_KEYS = {"position_ids", "cache_position", "past_key_values"}


def trim_to_context(base_inputs: dict) -> torch.Tensor:
    """
    Trim input_ids to only tokens where attention_mask == 1.
    Works for both left and right padding.
    
    Returns:
        input_ids_trimmed [1, L_ctx]
    """
    # Enforce batch=1 to avoid silent wrong scoring
    assert base_inputs["input_ids"].shape[0] == 1, (
        f"forced-choice expects batch=1 for correctness, got batch={base_inputs['input_ids'].shape[0]}. "
        "Batch scoring should be done at a higher level."
    )
    
    if "attention_mask" not in base_inputs:
        return base_inputs["input_ids"]
    
    attn = base_inputs["attention_mask"][0].bool()  # [L]
    input_ids = base_inputs["input_ids"][0, attn]   # [L_ctx]
    return input_ids.unsqueeze(0)


def score_completion(
    model, 
    base_inputs: dict, 
    tokenizer, 
    completion_text: str
) -> float:
    """
    Compute log P(completion_text | prompt, image) using teacher forcing.
    
    Works for multi-token completions and handles padding correctly.
    Robust to both left and right padding.
    
    Args:
        model: The VLM model
        base_inputs: Tokenized prompt + image inputs (from processor)
        tokenizer: The tokenizer
        completion_text: The text to score (e.g., "Real" or "Fake")
    
    Returns:
        Log-probability of the completion given the prompt
    """
    # Tokenize completion
    comp_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    device = base_inputs["input_ids"].device
    comp = torch.tensor(comp_ids, device=device)[None, :]  # [1, T]
    
    # Trim to context (handles left/right padding robustly)
    input_ids_trim = trim_to_context(base_inputs).to(device)
    prompt_len = input_ids_trim.shape[1]
    
    # Append completion tokens
    input_ids_ext = torch.cat([input_ids_trim, comp], dim=1)
    attn_ext = torch.ones(1, input_ids_ext.shape[1], device=device, dtype=torch.long)
    
    # Build extended inputs
    # - Update input_ids and attention_mask
    # - Drop position_ids/cache_position to avoid mismatch
    # - Keep pixel_values etc unchanged
    ext_inputs = {}
    for k, v in base_inputs.items():
        if k == "input_ids":
            ext_inputs[k] = input_ids_ext
        elif k == "attention_mask":
            ext_inputs[k] = attn_ext
        elif k in DROP_KEYS:
            continue  # Let model recompute these
        else:
            ext_inputs[k] = v
    
    with torch.no_grad():
        # Disable KV cache for reproducibility (guard for models that don't support it)
        try:
            outputs = model(**ext_inputs, use_cache=False)
        except TypeError:
            outputs = model(**ext_inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # Safety check: logits should match extended input length
    assert logits.shape[1] == input_ids_ext.shape[1], (
        f"Logits alignment mismatch: logits length {logits.shape[1]} != input length {input_ids_ext.shape[1]}. "
        "Check whether model returns shifted logits or has different forward signature."
    )
    
    # For token t in completion, predictive logits are at position (prompt_len + t - 1)
    # Because logits[i] predicts token[i+1]
    start = prompt_len - 1
    end = start + comp.shape[1]
    logits_slice = logits[0, start:end, :]  # [T, vocab_size]
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits_slice, dim=-1)  # [T, vocab_size]
    
    # Get log-prob for each completion token
    token_indices = comp[0]  # [T]
    token_logp = log_probs[torch.arange(len(token_indices), device=device), token_indices]
    
    # Sum log-probs for full sequence probability
    total_logp = token_logp.sum().item()
    
    return total_logp


def score_best_candidate(
    model,
    inputs: dict,
    tokenizer,
    candidates: List[str],
) -> Tuple[float, str]:
    """
    Score multiple candidate completions and return the best (highest log-prob).
    
    This handles tokenizer whitespace conventions robustly.
    
    Returns:
        (best_log_prob, best_candidate_text)
    """
    best_score = float('-inf')
    best_cand = candidates[0]
    
    for cand in candidates:
        try:
            score = score_completion(model, inputs, tokenizer, cand)
            if score > best_score:
                best_score = score
                best_cand = cand
        except Exception:
            continue
    
    return best_score, best_cand


def forced_choice_p_fake(
    model,
    inputs: dict,
    tokenizer,
    real_candidates: List[str] = None,
    fake_candidates: List[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute p_fake using forced-choice scoring.
    
    Uses sigmoid(log_p_fake - log_p_real) for numerical stability.
    Scores multiple whitespace variants for tokenizer robustness.
    
    Args:
        model: The VLM model
        inputs: Tokenized prompt + image (prompt should end with "Answer: ")
        tokenizer: The tokenizer
        real_candidates: List of "real" text variants (default: [" Real", "Real", ...])
        fake_candidates: List of "fake" text variants (default: [" Fake", "Fake", ...])
    
    Returns:
        p_fake: Probability that image is fake
        scores: Dict with 'log_p_real', 'log_p_fake', 'p_fake', 'pred', 'best_real', 'best_fake'
    """
    real_cands = real_candidates or REAL_CANDIDATES
    fake_cands = fake_candidates or FAKE_CANDIDATES
    
    # Score best variant for each class
    log_p_real, best_real = score_best_candidate(model, inputs, tokenizer, real_cands)
    log_p_fake, best_fake = score_best_candidate(model, inputs, tokenizer, fake_cands)
    
    # Stable: sigmoid(log_p_fake - log_p_real)
    p_fake = torch.sigmoid(torch.tensor(log_p_fake - log_p_real)).item()
    
    # Derive label from p_fake for consistency
    pred = int(p_fake >= 0.5)
    
    return p_fake, {
        "log_p_real": log_p_real,
        "log_p_fake": log_p_fake,
        "p_fake": p_fake,
        "pred": pred,
        "best_real": best_real,
        "best_fake": best_fake,
        "method": "forced_choice_full_sequence",
    }


# Classification prompt suffix
_CLASSIFICATION_SUFFIX = "\n\nAnswer with exactly one word: Real or Fake.\n\nAnswer:"


def get_classification_prompt_suffix() -> str:
    """
    Returns the prompt suffix that forces single-word output.
    Append this to your detection prompt to enable forced-choice scoring.
    """
    return _CLASSIFICATION_SUFFIX


def append_classification_suffix(prompt: str) -> str:
    """
    Append classification suffix if not already present.
    Checks for exact suffix, not partial match.
    """
    if prompt.rstrip().endswith(_CLASSIFICATION_SUFFIX.rstrip()):
        return prompt
    return prompt + _CLASSIFICATION_SUFFIX


# =============================================================================
# JS Divergence for Consistency Loss
# =============================================================================

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon divergence for Bernoulli distributions.
    
    Args:
        p: Tensor of probabilities in [0, 1]
        q: Tensor of probabilities in [0, 1] (same shape as p, or scalar)
        eps: Small value for numerical stability
    
    Returns:
        JS divergence for each element
    """
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    
    # Convert to 2-class distributions [1-p, p]
    p2 = torch.stack([1 - p, p], dim=-1)
    q2 = torch.stack([1 - q, q], dim=-1)
    
    # Midpoint
    m = 0.5 * (p2 + q2)
    
    # KL divergences
    kl_pm = torch.sum(p2 * (p2.log() - m.log()), dim=-1)
    kl_qm = torch.sum(q2 * (q2.log() - m.log()), dim=-1)
    
    return 0.5 * (kl_pm + kl_qm)


def consistency_loss_js(p_fakes: torch.Tensor) -> torch.Tensor:
    """
    JS-divergence based consistency loss across augmentations.
    
    Args:
        p_fakes: Tensor of p_fake values from different augmentations [n_aug]
    
    Returns:
        Mean JS divergence from the mean prediction
    """
    p_mean = p_fakes.mean()
    js_divs = js_divergence(p_fakes, p_mean.expand_as(p_fakes))
    return js_divs.mean()


# =============================================================================
# Sanity Checks
# =============================================================================

def verify_p_fake_sanity(p_fake_values: list, labels: list) -> dict:
    """
    Verify that p_fake values make sense.
    
    Checks:
    1. mean(p_fake | fake) > mean(p_fake | real)
    2. std(p_fake) > 0.05 (no template copying)
    3. Unique values > 10 (variance exists)
    """
    # Filter out None values
    valid_pairs = [(p, l) for p, l in zip(p_fake_values, labels) if p is not None]
    if len(valid_pairs) == 0:
        return {"error": "No valid p_fake values", "all_pass": False}
    
    p_fake = np.array([p for p, l in valid_pairs])
    labels_arr = np.array([l for p, l in valid_pairs])
    
    real_scores = p_fake[labels_arr == 0]
    fake_scores = p_fake[labels_arr == 1]
    
    checks = {
        "n_valid": len(valid_pairs),
        "n_total": len(p_fake_values),
        "valid_ratio": len(valid_pairs) / len(p_fake_values) if p_fake_values else 0,
        "mean_fake": float(fake_scores.mean()) if len(fake_scores) > 0 else None,
        "mean_real": float(real_scores.mean()) if len(real_scores) > 0 else None,
        "std_all": float(p_fake.std()),
        "std_real": float(real_scores.std()) if len(real_scores) > 0 else None,
        "std_fake": float(fake_scores.std()) if len(fake_scores) > 0 else None,
        "mean_gap": float(fake_scores.mean() - real_scores.mean()) if len(fake_scores) > 0 and len(real_scores) > 0 else None,
        "min": float(p_fake.min()),
        "max": float(p_fake.max()),
        "p1": float(np.percentile(p_fake, 1)),
        "p99": float(np.percentile(p_fake, 99)),
        "unique_values": int(len(np.unique(np.round(p_fake, 3)))),
    }
    
    # Sanity checks
    checks["pass_mean_ordering"] = (
        checks["mean_fake"] is not None and 
        checks["mean_real"] is not None and
        checks["mean_fake"] > checks["mean_real"]
    )
    checks["pass_variance"] = checks["std_all"] > 0.05
    checks["pass_unique"] = checks["unique_values"] > 10
    checks["all_pass"] = all([
        checks["pass_mean_ordering"],
        checks["pass_variance"],
        checks["pass_unique"],
    ])
    
    return checks
