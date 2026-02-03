"""
Robust Forced-Choice Scoring for VLM Binary Classification

Key features:
- Full-sequence teacher forcing (not single-token)
- Whitespace variant robustness
- Proper handling of multi-token completions
- Works with left/right padding
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

# Whitespace variants for tokenizer robustness
REAL_CANDIDATES = [" Real", "Real", " real", "real"]
FAKE_CANDIDATES = [" Fake", "Fake", " fake", "fake"]

# Keys to drop (avoid position mismatch / cache weirdness)
DROP_KEYS = {"position_ids", "cache_position", "past_key_values"}


def trim_to_context(base_inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trim input_ids to only tokens where attention_mask == 1.
    Works for both left and right padding.
    Requires batch == 1.
    """
    assert base_inputs["input_ids"].shape[0] == 1, "forced-choice expects batch==1"
    if "attention_mask" not in base_inputs:
        ids = base_inputs["input_ids"]
        attn = torch.ones_like(ids)
        return ids, attn

    attn = base_inputs["attention_mask"][0].bool()           # [L]
    input_ids = base_inputs["input_ids"][0, attn]            # [L_ctx]
    input_ids = input_ids.unsqueeze(0)                       # [1, L_ctx]
    attn_out = torch.ones_like(input_ids)
    return input_ids, attn_out


def score_completion_tokens(
    model,
    base_inputs: dict,
    completion_ids: List[int],
) -> float:
    """
    Teacher-forced log P(completion | prompt, image) using full sequence.
    Robust to padding and multi-token completions.
    """
    device = base_inputs["input_ids"].device
    comp = torch.tensor(completion_ids, device=device).unsqueeze(0)  # [1, T]

    # Trim context (works for left/right padding)
    input_ids_trim, attn_trim = trim_to_context(base_inputs)
    input_ids_trim = input_ids_trim.to(device)

    prompt_len = input_ids_trim.shape[1]
    assert prompt_len >= 1, "prompt_len must be >= 1"

    # Append completion
    input_ids_ext = torch.cat([input_ids_trim, comp], dim=1)            # [1, L+T]
    attn_ext = torch.ones_like(input_ids_ext, device=device)

    # Build ext inputs, dropping problematic keys
    ext_inputs = {}
    for k, v in base_inputs.items():
        if k in DROP_KEYS:
            continue
        if k == "input_ids":
            ext_inputs[k] = input_ids_ext
        elif k == "attention_mask":
            ext_inputs[k] = attn_ext
        else:
            ext_inputs[k] = v

    with torch.no_grad():
        outputs = model(**ext_inputs, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab]

    # Assert alignment
    assert logits.shape[1] == input_ids_ext.shape[1], (
        f"logits seq_len {logits.shape[1]} != input_ids seq_len {input_ids_ext.shape[1]}"
    )

    # logits[i] predicts token[i+1]
    # For completion token t, predictive logits at position (prompt_len + t - 1)
    start = prompt_len - 1
    end = start + comp.shape[1]
    logits_slice = logits[0, start:end, :]                   # [T, vocab]

    log_probs = F.log_softmax(logits_slice, dim=-1)          # [T, vocab]
    token_ids = comp[0]                                      # [T]

    token_logp = log_probs[torch.arange(token_ids.numel(), device=device), token_ids]
    return float(token_logp.sum().item())


def score_best_candidate(
    model,
    inputs: dict,
    tokenizer,
    candidates: List[str],
) -> Tuple[float, str]:
    """
    Score multiple whitespace variants and return the best log-prob candidate.
    """
    best_score = -1e30
    best_text = candidates[0]

    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            continue
        score = score_completion_tokens(model, inputs, ids)
        if score > best_score:
            best_score = score
            best_text = text

    return best_score, best_text


def forced_choice_p_fake_best(
    model,
    inputs: dict,
    tokenizer,
) -> Tuple[float, Dict[str, float]]:
    """
    p_fake = sigmoid(logP(fake) - logP(real)) using best whitespace variants.
    """
    log_p_real, best_real = score_best_candidate(model, inputs, tokenizer, REAL_CANDIDATES)
    log_p_fake, best_fake = score_best_candidate(model, inputs, tokenizer, FAKE_CANDIDATES)

    p_fake = torch.sigmoid(torch.tensor(log_p_fake - log_p_real)).item()
    return p_fake, {
        "log_p_real": log_p_real,
        "log_p_fake": log_p_fake,
        "best_real": best_real,
        "best_fake": best_fake,
        "method": "forced_choice_full_sequence_best_variant",
    }


# Legacy compatibility
def forced_choice_p_fake(model, inputs, tokenizer, real_text=" Real", fake_text=" Fake"):
    """Legacy wrapper that uses the robust implementation."""
    return forced_choice_p_fake_best(model, inputs, tokenizer)
