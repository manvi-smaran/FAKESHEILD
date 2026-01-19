#!/usr/bin/env python3
"""
Run LoRA Fine-Tuning for SmolVLM on Deepfake Detection

Usage:
    python scripts/run_lora_finetune.py --data_dir ./data/celeb_df --epochs 3 --max_samples 1000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.lora_finetune import run_lora_finetuning


def main():
    parser = argparse.ArgumentParser(description="Run LoRA Fine-Tuning for SmolVLM")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/celeb_df",
        help="Path to CelebDF dataset directory",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/smolvlm-lora",
        help="Directory to save fine-tuned model",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (None = use all)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA training",
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more parameters)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SmolVLM LoRA Fine-Tuning for Deepfake Detection")
    print("="*60)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Samples: {args.max_samples or 'ALL'}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_rank}")
    print("="*60 + "\n")
    
    # Run training
    trainer, metrics = run_lora_finetuning(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final AUC: {metrics['auc']:.4f}")
    print(f"Model saved to: {args.output_dir}/final")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    main()
