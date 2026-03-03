"""
Generate publication-quality visual results figures for FAKESHEILD.

Shows dataset samples with model predictions across different approaches,
demonstrating that VLMs require fine-tuning for deepfake detection.

Two approaches:
  A) Sample Grid:    8 images (4 real + 4 fake) with Zero-Shot vs CNN+LoRA scores
  B) Pipeline Flow:  2 images (1 real + 1 fake) through all 4 methods

Usage (on remote GPU machine):
  # Approach A — needs only CNN+LoRA checkpoint
  python scripts/generate_visual_results.py --approach A \
      --checkpoint_cnn_lora checkpoints/qwen_vl-cnn-lora/<timestamp>/best

  # Approach B — needs both LoRA and CNN+LoRA checkpoints
  python scripts/generate_visual_results.py --approach B \
      --checkpoint_cnn_lora checkpoints/qwen_vl-cnn-lora/<timestamp>/best \
      --checkpoint_lora checkpoints/qwen_vl-lora/<timestamp>/best

  # Both approaches
  python scripts/generate_visual_results.py --approach both \
      --checkpoint_cnn_lora checkpoints/qwen_vl-cnn-lora/<timestamp>/best \
      --checkpoint_lora checkpoints/qwen_vl-lora/<timestamp>/best

Output: results/charts/
"""

import os
import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import yaml

from src.data.dataset_loader import create_dataset
from src.models.forensic_cnn import ForensicCNN
from src.utils.forced_choice import forced_choice_p_fake_best

OUTPUT_DIR = Path("results/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Publication style (matches generate_results_charts.py)
# =============================================================================
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#333333",
    "axes.labelcolor":    "#222222",
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "text.color":         "#222222",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# Colours
CLR_REAL    = "#2E8B57"   # SeaGreen
CLR_FAKE    = "#CC3333"   # Crimson
CLR_CORRECT = "#2E8B57"
CLR_WRONG   = "#CC3333"
CLR_ZS      = "#B0B0B0"
CLR_FS      = "#9E9E9E"
CLR_LORA    = "#5B9BD5"
CLR_CNN     = "#ED7D31"


# =============================================================================
# Model Loading Helpers
# =============================================================================

def load_base_model():
    """Load Qwen2-VL base model + processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("Loading Qwen2-VL-2B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
    )
    return model, processor


def _validate_checkpoint(path, label, require_cnn=False):
    """Validate checkpoint path exists and has required files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {p}")
    adapter_cfg = p / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(
            f"{label}: adapter_config.json not found in {p}\n"
            f"  Contents: {[x.name for x in p.iterdir()]}"
        )
    if require_cnn:
        cnn_pt = p / "forensic_cnn.pt"
        if not cnn_pt.exists():
            raise FileNotFoundError(
                f"{label}: forensic_cnn.pt not found in {p}"
            )
    print(f"  [OK] {label} checkpoint validated: {p}")
    return str(p)  # return as-is (matches working eval script)


def load_lora_model(checkpoint_path):
    """Load base model with LoRA adapter applied."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    ckpt = _validate_checkpoint(checkpoint_path, "LoRA")
    print(f"Loading LoRA from {ckpt}...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
    )
    return model, processor


def load_cnn_lora_model(checkpoint_path, num_tokens=4):
    """Load base model with LoRA adapter + ForensicCNN."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    ckpt = _validate_checkpoint(checkpoint_path, "CNN+LoRA", require_cnn=True)
    print(f"Loading CNN+LoRA from {ckpt}...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
    )

    # Load ForensicCNN
    cnn_path = Path(ckpt) / "forensic_cnn.pt"
    hidden_size = 1536  # Qwen2-VL-2B
    cnn = ForensicCNN(hidden_size=hidden_size, num_tokens=num_tokens)
    cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    device = next(model.parameters()).device
    cnn = cnn.to(device).to(torch.float32)
    cnn.eval()
    print(f"  ForensicCNN loaded ({sum(p.numel() for p in cnn.parameters()):,} params)")

    return model, processor, cnn


# =============================================================================
# Scoring Helpers
# =============================================================================

SCORING_PROMPT = (
    "Classify authenticity.\n\n"
    "Answer with exactly one word: Real or Fake.\n\n"
    "Answer:"
)

ZERO_SHOT_PROMPT = (
    "Analyze this facial image carefully for signs of deepfake manipulation.\n\n"
    "Look for: facial inconsistencies, boundary artifacts, unnatural lighting, "
    "texture anomalies.\n\n"
    "Is this image Real or Fake? Answer with one word:"
)


def build_scoring_inputs(model, processor, image, prompt):
    """Build model inputs for forced-choice scoring."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=[image], return_tensors="pt")
    return {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}


def score_zero_shot(model, processor, image):
    """Get p_fake using zero-shot prompt (no fine-tuning)."""
    inputs = build_scoring_inputs(model, processor, image, ZERO_SHOT_PROMPT)
    p_fake, debug = forced_choice_p_fake_best(model, inputs, processor.tokenizer)
    return p_fake


def score_finetuned(model, processor, image):
    """Get p_fake using fine-tuned model with training prompt."""
    inputs = build_scoring_inputs(model, processor, image, SCORING_PROMPT)
    p_fake, debug = forced_choice_p_fake_best(model, inputs, processor.tokenizer)
    return p_fake


def _get_language_model(model):
    """Find the language_model sub-module for hook injection."""
    for name, module in model.named_modules():
        if name.endswith(".language_model") and hasattr(module, "layers"):
            return module
    # Fallback paths
    for get_lm in [
        lambda: model.base_model.model.model.language_model,
        lambda: model.model.model.language_model,
    ]:
        try:
            lm = get_lm()
            if lm is not None and hasattr(lm, "layers"):
                return lm
        except AttributeError:
            continue
    raise RuntimeError("Cannot find language_model sub-module")


def score_cnn_lora(model, processor, cnn, image, num_tokens=4):
    """Get p_fake with CNN+LoRA (forensic tokens injected via hook)."""
    inputs = build_scoring_inputs(model, processor, image, SCORING_PROMPT)

    # Compute forensic tokens
    device = next(model.parameters()).device
    cnn_input = cnn.transform(image).unsqueeze(0).to(device).to(torch.float32)
    forensic_tokens = cnn(cnn_input)  # [1, num_tokens, hidden_size]

    # Setup persistent hook for all forward calls during scoring
    language_model = _get_language_model(model)
    _state = {"ft": forensic_tokens, "active": True}

    def inject_hook(module, args, kwargs):
        if not _state["active"] or _state["ft"] is None:
            return args, kwargs
        ft = _state["ft"]
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            return args, kwargs

        B = inputs_embeds.shape[0]
        num_ft = ft.shape[1]
        ft = ft.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        if ft.shape[0] != B:
            ft = ft.expand(B, -1, -1)

        kwargs["inputs_embeds"] = torch.cat([ft, inputs_embeds], dim=1)

        attn_mask = kwargs.get("attention_mask", None)
        if attn_mask is not None:
            fa = torch.ones(B, num_ft, dtype=attn_mask.dtype, device=attn_mask.device)
            kwargs["attention_mask"] = torch.cat([fa, attn_mask], dim=1)

        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None and position_ids.dim() == 3:
            ft_pos = torch.zeros(3, B, num_ft, dtype=position_ids.dtype, device=position_ids.device)
            for i in range(num_ft):
                ft_pos[:, :, i] = i
            kwargs["position_ids"] = torch.cat([ft_pos, position_ids + num_ft], dim=2)
        elif position_ids is not None and position_ids.dim() == 2:
            ft_pos = torch.arange(num_ft, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0).expand(B, -1)
            kwargs["position_ids"] = torch.cat([ft_pos, position_ids + num_ft], dim=1)

        return args, kwargs

    handle = language_model.register_forward_pre_hook(inject_hook, with_kwargs=True)
    try:
        p_fake, debug = forced_choice_p_fake_best(model, inputs, processor.tokenizer)
    finally:
        handle.remove()
        _state["active"] = False

    return p_fake


def make_few_shot_prompt(k=8):
    """Build text-only few-shot prompt (same as run_fast_eval.py)."""
    from src.utils.prompts import FEW_SHOT_EXAMPLE_REAL, FEW_SHOT_EXAMPLE_FAKE
    examples = []
    idx = 1
    for _ in range(k // 2):
        examples.append(FEW_SHOT_EXAMPLE_REAL.format(idx=idx))
        idx += 1
        examples.append(FEW_SHOT_EXAMPLE_FAKE.format(idx=idx))
        idx += 1
    examples_str = "\n\n".join(examples)
    return (
        f"You are analyzing facial images to detect deepfakes. "
        f"Here are {k} labeled examples:\n\n{examples_str}\n\n"
        f"Now analyze this new image.\n"
        f"Is this a real photograph or a deepfake? Answer with one word: Real or Fake."
    )


def score_few_shot(model, processor, image, k=8):
    """Get p_fake using few-shot prompt."""
    prompt = make_few_shot_prompt(k)
    inputs = build_scoring_inputs(model, processor, image, prompt)
    p_fake, debug = forced_choice_p_fake_best(model, inputs, processor.tokenizer)
    return p_fake


# =============================================================================
# Sample Selection
# =============================================================================

def select_samples(config, dataset_name, n_real=4, n_fake=4, seed=42):
    """Select balanced, seeded samples from dataset."""
    dataset = create_dataset(config, dataset_name, max_samples=None)
    all_samples = dataset.samples

    real_samples = [s for s in all_samples if s[1] == 0]
    fake_samples = [s for s in all_samples if s[1] == 1]

    random.seed(seed)
    random.shuffle(real_samples)
    random.shuffle(fake_samples)

    selected_real = real_samples[:n_real]
    selected_fake = fake_samples[:n_fake]

    print(f"Selected {len(selected_real)} real + {len(selected_fake)} fake samples")
    return selected_real, selected_fake


def load_image(frame_path):
    """Load and resize image for display."""
    img = Image.open(frame_path).convert("RGB")
    max_dim = 640
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    return img


# =============================================================================
# APPROACH A: Sample Grid
# =============================================================================

def generate_approach_a(
    base_model, base_processor,
    cnn_model, cnn_processor, cnn,
    real_samples, fake_samples,
    num_tokens=4,
    figure_idx=0,
):
    """
    Generate Fig: 2×4 grid of dataset samples.
    Top row: 4 real images. Bottom row: 4 fake images.
    Each cell shows the image, ground truth, zero-shot p_fake, CNN+LoRA p_fake.
    """
    n_cols = max(len(real_samples), len(fake_samples))
    fig = plt.figure(figsize=(3.2 * n_cols, 8.5))

    # Title
    fig.suptitle(
        "Fig. 9:  Qualitative Results — Zero-Shot vs CNN+LoRA Predictions",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Subtitle
    fig.text(
        0.5, 0.945,
        "Zero-shot assigns near-identical scores to real and fake; "
        "CNN+LoRA produces well-separated predictions",
        ha="center", fontsize=9, style="italic", color="#666666",
    )

    gs = gridspec.GridSpec(2, n_cols, hspace=0.45, wspace=0.25, top=0.90, bottom=0.04)

    all_samples = [(real_samples, 0, "REAL"), (fake_samples, 1, "FAKE")]

    for row_idx, (samples, gt_label, gt_text) in enumerate(all_samples):
        for col_idx, (frame_path, label, manip_type) in enumerate(samples):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            image = load_image(frame_path)

            # Score with both methods
            print(f"  Scoring [{gt_text}] sample {col_idx + 1}...")
            with torch.no_grad():
                p_zs = score_zero_shot(base_model, base_processor, image)
                p_cnn = score_cnn_lora(cnn_model, cnn_processor, cnn, image, num_tokens)

            pred_zs = "Fake" if p_zs >= 0.5 else "Real"
            pred_cnn = "Fake" if p_cnn >= 0.5 else "Real"
            correct_zs = (p_zs >= 0.5) == (gt_label == 1)
            correct_cnn = (p_cnn >= 0.5) == (gt_label == 1)

            # Display image
            ax.imshow(np.array(image))
            ax.set_xticks([])
            ax.set_yticks([])

            # Ground truth border
            border_color = CLR_REAL if gt_label == 0 else CLR_FAKE
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

            # Ground truth label
            gt_color = CLR_REAL if gt_label == 0 else CLR_FAKE
            ax.set_title(
                f"{gt_text}",
                fontsize=10, fontweight="bold", color=gt_color, pad=4,
            )

            # Scores below image
            zs_mark = "✓" if correct_zs else "✗"
            cnn_mark = "✓" if correct_cnn else "✗"
            zs_clr = CLR_CORRECT if correct_zs else CLR_WRONG
            cnn_clr = CLR_CORRECT if correct_cnn else CLR_WRONG

            score_text = (
                f"Zero-Shot: $p_{{fake}}$={p_zs:.2f} → {pred_zs} {zs_mark}\n"
                f"CNN+LoRA: $p_{{fake}}$={p_cnn:.2f} → {pred_cnn} {cnn_mark}"
            )

            ax.text(
                0.5, -0.02, score_text,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.95),
            )

    # Legend annotation
    fig.text(
        0.5, 0.01,
        "Green border = Real  |  Red border = Fake  |  "
        "✓ = correct prediction  |  ✗ = incorrect prediction",
        ha="center", fontsize=8, color="#777777",
    )

    suffix = chr(ord('a') + figure_idx)  # a, b, c, ...
    output_path = OUTPUT_DIR / f"9{suffix}_visual_results_grid.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n  [SAVED] {output_path}")
    return output_path


# =============================================================================
# APPROACH B: Pipeline Flow
# =============================================================================

def generate_approach_b(
    base_model, base_processor,
    lora_model, lora_processor,
    cnn_model, cnn_processor, cnn,
    real_samples, fake_samples,
    num_tokens=4,
    figure_idx=0,
):
    """
    Generate Fig: 2×4 pipeline showing 1 real + 1 fake image through all 4 methods.
    Columns: Zero-Shot | Few-Shot (k=8) | LoRA | CNN+LoRA
    Rows: Real image | Fake image
    """
    methods = ["Zero-Shot", "Few-Shot\n(k=8)", "LoRA\n(Fine-Tuned)", "CNN+LoRA\n(Fine-Tuned)"]
    method_colors = [CLR_ZS, CLR_FS, CLR_LORA, CLR_CNN]

    real_sample = real_samples[0]
    fake_sample = fake_samples[0]

    real_image = load_image(real_sample[0])
    fake_image = load_image(fake_sample[0])

    # Score all methods
    results = {}
    print("  Scoring real image across all methods...")
    with torch.no_grad():
        results["real"] = {
            "zs": score_zero_shot(base_model, base_processor, real_image),
            "fs": score_few_shot(base_model, base_processor, real_image, k=8),
            "lora": score_finetuned(lora_model, lora_processor, real_image),
            "cnn": score_cnn_lora(cnn_model, cnn_processor, cnn, real_image, num_tokens),
        }

    print("  Scoring fake image across all methods...")
    with torch.no_grad():
        results["fake"] = {
            "zs": score_zero_shot(base_model, base_processor, fake_image),
            "fs": score_few_shot(base_model, base_processor, fake_image, k=8),
            "lora": score_finetuned(lora_model, lora_processor, fake_image),
            "cnn": score_cnn_lora(cnn_model, cnn_processor, cnn, fake_image, num_tokens),
        }

    # Print debug
    for cls_name, scores in results.items():
        print(f"  [{cls_name.upper()}] ZS={scores['zs']:.3f}  FS={scores['fs']:.3f}  "
              f"LoRA={scores['lora']:.3f}  CNN={scores['cnn']:.3f}")

    # Build figure
    fig = plt.figure(figsize=(14, 7.5))

    fig.suptitle(
        "Fig. 10:  Pipeline Comparison — Same Image Across Detection Methods",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.925,
        "Only fine-tuned methods correctly separate real from fake; "
        "zero-shot and few-shot fail regardless of prompt",
        ha="center", fontsize=9, style="italic", color="#666666",
    )

    gs = gridspec.GridSpec(
        2, 4, hspace=0.50, wspace=0.20,
        top=0.87, bottom=0.08, left=0.04, right=0.96,
    )

    rows = [
        ("real", real_image, 0, "Ground Truth: REAL", CLR_REAL),
        ("fake", fake_image, 1, "Ground Truth: FAKE", CLR_FAKE),
    ]
    method_keys = ["zs", "fs", "lora", "cnn"]

    for row_idx, (cls_key, image, gt_label, gt_text, gt_color) in enumerate(rows):
        for col_idx, (m_key, m_name, m_color) in enumerate(
            zip(method_keys, methods, method_colors)
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            p_fake = results[cls_key][m_key]
            pred = "Fake" if p_fake >= 0.5 else "Real"
            correct = (p_fake >= 0.5) == (gt_label == 1)

            # Show image
            ax.imshow(np.array(image))
            ax.set_xticks([])
            ax.set_yticks([])

            # Method column header (top row only)
            if row_idx == 0:
                ax.set_title(m_name, fontsize=10, fontweight="bold",
                             color=m_color, pad=6)

            # Ground truth label on left (first column only)
            if col_idx == 0:
                ax.set_ylabel(gt_text, fontsize=10, fontweight="bold",
                              color=gt_color, labelpad=8)

            # Border color based on correctness
            border = CLR_CORRECT if correct else CLR_WRONG
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

            # Score overlay
            mark = "✓" if correct else "✗"
            mark_color = CLR_CORRECT if correct else CLR_WRONG
            pred_label = f"$p_{{fake}}$ = {p_fake:.2f}"

            ax.text(
                0.5, -0.02, f"{pred_label}\nPrediction: {pred} {mark}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold",
                color=mark_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=mark_color, alpha=0.92, linewidth=1.2),
            )

    # Legend
    fig.text(
        0.5, 0.02,
        "Green border/text = correct prediction  |  Red border/text = incorrect prediction  |  "
        "Decision boundary: $p_{fake}$ ≥ 0.5 → Fake",
        ha="center", fontsize=8, color="#777777",
    )

    # Arrow annotation between no-training and fine-tuned
    fig.text(
        0.50, 0.91,
        "◄── No Training ──┤── Fine-Tuned ──►",
        ha="center", fontsize=8, color="#999999", fontweight="bold",
    )

    suffix = chr(ord('a') + figure_idx)
    output_path = OUTPUT_DIR / f"10{suffix}_visual_results_pipeline.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n  [SAVED] {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate visual results figures for FAKESHEILD"
    )
    parser.add_argument(
        "--approach", type=str, default="both", choices=["A", "B", "both"],
        help="Which figure approach to generate",
    )
    parser.add_argument(
        "--checkpoint_cnn_lora", type=str, required=True,
        help="Path to CNN+LoRA checkpoint (contains adapter_config.json + forensic_cnn.pt)",
    )
    parser.add_argument(
        "--checkpoint_lora", type=str, default=None,
        help="Path to LoRA-only checkpoint (required for Approach B)",
    )
    parser.add_argument(
        "--dataset", type=str, default="faceforensics",
        choices=["faceforensics", "celebdf"],
    )
    parser.add_argument("--config", type=str, default="configs/model_configs.yaml")
    parser.add_argument("--n_real", type=int, default=4, help="Number of real samples (Approach A)")
    parser.add_argument("--n_fake", type=int, default=4, help="Number of fake samples (Approach A)")
    parser.add_argument("--num_tokens", type=int, default=4, help="Forensic token count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_figures", type=int, default=3,
                        help="Number of figures to generate per approach (each uses different samples)")
    args = parser.parse_args()

    do_a = args.approach in ("A", "both")
    do_b = args.approach in ("B", "both")

    if do_b and args.checkpoint_lora is None:
        parser.error("Approach B requires --checkpoint_lora")

    print("\n" + "=" * 56)
    print("  FAKESHEILD — Visual Results Generator")
    print("=" * 56)
    print(f"  Approach:       {'A + B' if args.approach == 'both' else args.approach}")
    print(f"  CNN+LoRA ckpt:  {args.checkpoint_cnn_lora}")
    if args.checkpoint_lora:
        print(f"  LoRA ckpt:      {args.checkpoint_lora}")
    print(f"  Num figures:    {args.num_figures}")
    print(f"  Dataset:        {args.dataset}")
    print("=" * 56 + "\n")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load models (load once, reuse for all figures)
    print("\n--- Loading Models ---")
    base_model, base_processor = load_base_model()
    cnn_model, cnn_processor, cnn = load_cnn_lora_model(
        args.checkpoint_cnn_lora, num_tokens=args.num_tokens
    )

    lora_model, lora_processor = None, None
    if do_b:
        lora_model, lora_processor = load_lora_model(args.checkpoint_lora)

    # Generate multiple figures with different seeds
    for fig_idx in range(args.num_figures):
        current_seed = args.seed + fig_idx
        print(f"\n{'='*56}")
        print(f"  Figure set {fig_idx + 1}/{args.num_figures} (seed={current_seed})")
        print(f"{'='*56}")

        # For Approach A: pick n_real + n_fake fresh samples
        # For Approach B: pick 1 real + 1 fake (use offset into shuffled list)
        real_samples, fake_samples = select_samples(
            config, args.dataset,
            n_real=max(args.n_real, 1), n_fake=max(args.n_fake, 1),
            seed=current_seed,
        )

        if do_a:
            print(f"\n--- Approach A: Sample Grid (set {fig_idx + 1}) ---")
            generate_approach_a(
                base_model, base_processor,
                cnn_model, cnn_processor, cnn,
                real_samples[:args.n_real], fake_samples[:args.n_fake],
                num_tokens=args.num_tokens,
                figure_idx=fig_idx,
            )

        if do_b:
            print(f"\n--- Approach B: Pipeline Flow (set {fig_idx + 1}) ---")
            generate_approach_b(
                base_model, base_processor,
                lora_model, lora_processor,
                cnn_model, cnn_processor, cnn,
                real_samples, fake_samples,
                num_tokens=args.num_tokens,
                figure_idx=fig_idx,
            )

    print("\n" + "=" * 56)
    print("  All visual results saved to results/charts/")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
