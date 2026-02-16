"""
Run CNN + PEFT Hybrid Fine-Tuning for Deepfake Detection

Usage:
    python scripts/run_cnn_peft_finetune.py --dataset faceforensics --epochs 10
    python scripts/run_cnn_peft_finetune.py --dataset combined --epochs 10 --cnn_lr 1e-4
    python scripts/run_cnn_peft_finetune.py --dataset faceforensics --epochs 10 --num_tokens 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.cnn_peft_finetune import run_cnn_peft_finetuning


def main():
    parser = argparse.ArgumentParser(
        description="CNN + PEFT Hybrid Fine-Tuning for Deepfake Detection"
    )
    
    parser.add_argument(
        "--model", type=str, default="qwen_vl",
        choices=["qwen_vl"],
        help="Model to fine-tune (currently only qwen_vl supported).",
    )
    parser.add_argument(
        "--dataset", type=str, default="faceforensics",
        choices=["celebdf", "faceforensics", "combined"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--peft_method", type=str, default="lora",
        choices=["lora", "dora"],
        help="PEFT method to use alongside CNN.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--cnn_lr", type=float, default=1e-4,
        help="Learning rate for CNN parameters (higher since training from scratch).",
    )
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--alpha", type=int, default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples for quick testing.",
    )
    parser.add_argument(
        "--num_tokens", type=int, default=4,
        help="Number of forensic tokens to inject.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/model_configs.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Custom output directory.",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"CNN + PEFT Hybrid Fine-Tuning")
    print(f"{'='*60}")
    print(f"Model:            {args.model}")
    print(f"Dataset:          {args.dataset}")
    print(f"PEFT Method:      {args.peft_method}")
    print(f"Epochs:           {args.epochs}")
    print(f"LoRA LR:          {args.lr}")
    print(f"CNN LR:           {args.cnn_lr}")
    print(f"LoRA Rank:        {args.rank}")
    print(f"Forensic Tokens:  {args.num_tokens}")
    print(f"Max Samples:      {args.max_samples or 'ALL'}")
    print(f"{'='*60}\n")
    
    trainer, results = run_cnn_peft_finetuning(
        config_path=args.config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        cnn_lr=args.cnn_lr,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        batch_size=args.batch_size,
        model_type=args.model,
        peft_method=args.peft_method,
        num_forensic_tokens=args.num_tokens,
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Loss:    {results.get('best_loss', 'N/A'):.4f}")
    print(f"Final Loss:   {results.get('final_loss', 'N/A'):.4f}")
    print(f"Accuracy:     {results.get('accuracy', 'N/A'):.2%}")
    print(f"Checkpoints:  {trainer.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
