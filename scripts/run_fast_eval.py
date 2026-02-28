"""
FAST Zero-Shot & Few-Shot Evaluation for Qwen2-VL.

Uses single-forward-pass first-token logit scoring instead of
full-sequence forced-choice (which needs 9 forward passes per sample).

Speed: ~1 forward pass per sample vs ~9 → ~9x faster.

Usage:
    # Zero-shot on 500 samples (~15 min)
    python scripts/run_fast_eval.py --dataset faceforensics --max_samples 500

    # Few-shot with k=4 on 500 samples (~15 min)
    python scripts/run_fast_eval.py --dataset faceforensics --max_samples 500 --k 4

    # Both zero-shot and few-shot
    python scripts/run_fast_eval.py --dataset faceforensics --max_samples 500 --mode both
"""

import argparse
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def load_model_and_processor(config_path="configs/model_configs.yaml"):
    """Load Qwen2-VL model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["models"]["qwen_vl"]
    print(f"Loading {model_config['name']}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config["hub_id"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_config["hub_id"], trust_remote_code=True
    )

    return model, processor, config


def first_token_p_fake(model, inputs, tokenizer):
    """
    Single-forward-pass p_fake using first generated token logits.

    Instead of scoring full sequences for multiple candidates (9 passes),
    we do 1 forward pass and compare logits at the last position for
    "Real" vs "Fake" token IDs.

    Returns:
        p_fake: float, probability of fake
        debug: dict with log probs
    """
    # Get token IDs for Real/Fake (use the most common tokenization)
    real_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in [" Real", "Real", " real", "real"]]
    fake_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in [" Fake", "Fake", " fake", "fake"]]

    # Remove duplicates
    real_ids = list(set(real_ids))
    fake_ids = list(set(fake_ids))

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab]

    # Last token logits predict the first generated token
    last_logits = logits[0, -1, :]  # [vocab]

    # Get max log-prob across all Real/Fake token variants
    log_probs = F.log_softmax(last_logits, dim=-1)

    log_p_real = max(log_probs[tid].item() for tid in real_ids)
    log_p_fake = max(log_probs[tid].item() for tid in fake_ids)

    p_fake = torch.sigmoid(torch.tensor(log_p_fake - log_p_real)).item()

    return p_fake, {"log_p_real": log_p_real, "log_p_fake": log_p_fake}


def build_inputs(model, processor, image, prompt):
    """Build model inputs from image + prompt."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}


def run_eval(model, processor, config, dataset_name, max_samples, prompt, label="eval"):
    """Run evaluation with single-forward-pass scoring."""
    from src.data.dataset_loader import create_dataset

    dataset = create_dataset(config, dataset_name, max_samples)
    n = len(dataset)
    print(f"\n{'='*50}")
    print(f"  {label} | {n} samples | {dataset_name}")
    print(f"{'='*50}")

    predictions, p_fakes, labels, manip_types = [], [], [], []
    tokenizer = processor.tokenizer

    start_time = time.time()

    for idx, (image, gt_label, manip_type) in enumerate(tqdm(dataset, desc=label)):
        try:
            inputs = build_inputs(model, processor, image, prompt)
            p_fake, debug = first_token_p_fake(model, inputs, tokenizer)
        except Exception as e:
            if idx < 3:
                print(f"  [WARN] sample {idx}: {e}")
            p_fake = 0.5

        pred = 1 if p_fake >= 0.5 else 0
        predictions.append(pred)
        p_fakes.append(p_fake)
        labels.append(gt_label)
        manip_types.append(manip_type)

        if idx < 5:
            print(f"  [DEBUG] i={idx} label={gt_label} p_fake={p_fake:.3f} pred={pred}")

    elapsed = time.time() - start_time

    # Compute metrics
    import numpy as np
    p = np.array(p_fakes, dtype=float)
    y = np.array(labels, dtype=int)

    correct = sum(1 for pr, l in zip(predictions, labels) if pr == l)
    accuracy = correct / n if n > 0 else 0

    real_p = p[y == 0]
    fake_p = p[y == 1]
    separation = float(fake_p.mean() - real_p.mean()) if len(fake_p) > 0 and len(real_p) > 0 else 0

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
    except Exception:
        auc = None

    fake_count = sum(labels)
    real_count = n - fake_count
    pred_fake = sum(predictions)
    pred_real = n - pred_fake

    # Real-class accuracy and Fake-class accuracy
    real_correct = sum(1 for pr, l in zip(predictions, labels) if l == 0 and pr == 0)
    fake_correct = sum(1 for pr, l in zip(predictions, labels) if l == 1 and pr == 1)
    real_acc = real_correct / real_count if real_count > 0 else 0
    fake_acc = fake_correct / fake_count if fake_count > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Results: {label}")
    print(f"{'='*50}")
    print(f"  Accuracy:     {accuracy:.2%} ({correct}/{n})")
    if auc is not None:
        print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  p_fake sep:   {separation:.4f}")
    print(f"  mean_fake:    {fake_p.mean():.3f}" if len(fake_p) > 0 else "")
    print(f"  mean_real:    {real_p.mean():.3f}" if len(real_p) > 0 else "")
    print(f"  Real acc:     {real_acc:.2%} ({real_correct}/{real_count})")
    print(f"  Fake acc:     {fake_acc:.2%} ({fake_correct}/{fake_count})")
    print(f"  Pred dist:    Real={pred_real}, Fake={pred_fake}")
    print(f"  Time:         {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")
    print(f"{'='*50}")

    return {
        "label": label,
        "dataset": dataset_name,
        "n_samples": n,
        "accuracy": accuracy,
        "auc": auc,
        "separation": separation,
        "mean_p_fake_fake": float(fake_p.mean()) if len(fake_p) > 0 else None,
        "mean_p_fake_real": float(real_p.mean()) if len(real_p) > 0 else None,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "pred_real": pred_real,
        "pred_fake": pred_fake,
        "elapsed_seconds": elapsed,
    }


# ─── Prompts ──────────────────────────────────────────────────────────
ZERO_SHOT_PROMPT = """Analyze this facial image carefully for signs of deepfake manipulation.

Look for: facial inconsistencies, boundary artifacts, unnatural lighting, texture anomalies.

Is this image Real or Fake? Answer with one word:"""


def make_few_shot_prompt(k):
    """Build a text-only few-shot prompt (no extra images needed)."""
    examples = []
    for i in range(k // 2):
        examples.append(f"Example {2*i+1}: [real face image] → Real")
        examples.append(f"Example {2*i+2}: [manipulated face image] → Fake")

    return f"""You are an expert deepfake detector. Here are {k} labeled examples:

{chr(10).join(examples)}

Key patterns:
- Real faces: consistent lighting, natural skin texture, sharp boundaries
- Fake faces: blending artifacts, inconsistent lighting, unnatural smoothness

Now analyze this new image. Is it Real or Fake? Answer with one word:"""


def main():
    parser = argparse.ArgumentParser(description="Fast Zero/Few-Shot Eval")
    parser.add_argument("--dataset", default="faceforensics", choices=["celebdf", "faceforensics"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--mode", default="both", choices=["zero_shot", "few_shot", "both"])
    parser.add_argument("--k", type=int, nargs="+", default=[4], help="Few-shot k values")
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output", default="results/fast_eval_results.json")
    args = parser.parse_args()

    model, processor, config = load_model_and_processor(args.config)
    all_results = {}

    if args.mode in ("zero_shot", "both"):
        result = run_eval(
            model, processor, config, args.dataset, args.max_samples,
            ZERO_SHOT_PROMPT, label="Zero-Shot"
        )
        all_results["zero_shot"] = result

    if args.mode in ("few_shot", "both"):
        for k in args.k:
            result = run_eval(
                model, processor, config, args.dataset, args.max_samples,
                make_few_shot_prompt(k), label=f"Few-Shot (k={k})"
            )
            all_results[f"few_shot_k{k}"] = result

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'Accuracy':>10} {'AUC':>8} {'Separation':>12}")
    print(f"  {'-'*50}")
    for key, r in all_results.items():
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"  {r['label']:<20} {r['accuracy']:>9.2%} {auc_str:>8} {r['separation']:>11.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
