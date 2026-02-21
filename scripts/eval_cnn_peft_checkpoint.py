"""
Evaluate a CNN + PEFT checkpoint without retraining.

Usage:
    python scripts/eval_cnn_peft_checkpoint.py --checkpoint checkpoints/qwen_vl-cnn-lora/20260221_183715/best
    python scripts/eval_cnn_peft_checkpoint.py --checkpoint checkpoints/qwen_vl-cnn-lora/20260221_183715/final
    python scripts/eval_cnn_peft_checkpoint.py --checkpoint checkpoints/qwen_vl-cnn-lora/20260221_183715/best --max_samples 100
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.models.forensic_cnn import ForensicCNN
from src.training.cnn_peft_finetune import CNNPEFTTrainer, MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a CNN + PEFT checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint dir (contains adapter_config.json + forensic_cnn.pt)",
    )
    parser.add_argument(
        "--dataset", type=str, default="faceforensics",
        choices=["celebdf", "faceforensics", "combined"],
    )
    parser.add_argument(
        "--config", type=str, default="configs/model_configs.yaml",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max eval samples (None = all)",
    )
    parser.add_argument(
        "--num_tokens", type=int, default=4,
        help="Number of forensic tokens (must match training)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen_vl",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    model_config = MODEL_CONFIGS[args.model]
    hidden_size = model_config["hidden_size"]

    print(f"\n{'='*60}")
    print(f"CNN + PEFT Checkpoint Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Dataset:     {args.dataset}")
    print(f"Max Samples: {args.max_samples or 'ALL'}")
    print(f"{'='*60}\n")

    # 1. Load base model
    print("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config["model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. Load LoRA adapter from checkpoint
    print(f"Loading LoRA adapter from {ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    # 3. Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_config["model_name"],
        trust_remote_code=True,
    )

    # 4. Load ForensicCNN weights
    cnn_path = ckpt_path / "forensic_cnn.pt"
    print(f"Loading ForensicCNN from {cnn_path}...")
    forensic_cnn = ForensicCNN(
        hidden_size=hidden_size,
        num_tokens=args.num_tokens,
    )
    forensic_cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    device = next(model.parameters()).device
    forensic_cnn = forensic_cnn.to(device).to(torch.float32)
    forensic_cnn.eval()
    print(f"  CNN loaded ({sum(p.numel() for p in forensic_cnn.parameters()):,} params)")

    # 5. Create a trainer instance for evaluation (reuse the evaluate logic)
    trainer = CNNPEFTTrainer(
        model_type=args.model,
        num_forensic_tokens=args.num_tokens,
        output_dir=str(ckpt_path.parent),
    )
    trainer.model = model
    trainer.processor = processor
    trainer.forensic_cnn = forensic_cnn
    trainer.config_path = args.config
    trainer.dataset_name = args.dataset

    # 6. Run evaluation
    metrics = trainer.evaluate(max_samples=args.max_samples)

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.2%}")
    print(f"p_fake sep:  {metrics['p_fake_stats'].get('separation', 'N/A')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
