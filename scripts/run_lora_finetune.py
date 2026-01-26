#!/usr/bin/env python3
"""
Run LoRA Fine-Tuning for SmolVLM on Deepfake Detection

Usage:
    python scripts/run_lora_finetune.py --dataset celebdf --epochs 3 --max_samples 1000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.lora_finetune import run_lora_finetuning


def main():
    parser = argparse.ArgumentParser(description="Run LoRA Fine-Tuning for SmolVLM")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_configs.yaml",
        help="Path to config file",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="celebdf",
        choices=["celebdf", "faceforensics"],
        help="Dataset to train on",
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
    
    parser.add_argument(
        "--model",
        type=str,
        default="smolvlm",
        choices=["smolvlm", "qwen_vl", "moondream"],
        help="Model to fine-tune (smolvlm, qwen_vl, moondream)",
    )
    
    args = parser.parse_args()
    
    # Set default output dir based on model if not specified
    if args.output_dir == "checkpoints/smolvlm-lora":
        args.output_dir = f"checkpoints/{args.model}-lora"
    
    print("\n" + "="*60)
    print(f"{args.model.upper()} LoRA Fine-Tuning for Deepfake Detection")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Samples: {args.max_samples or 'ALL'}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_rank}")
    print("="*60 + "\n")
    
    # Run training
    trainer, metrics = run_lora_finetuning(
        config_path=args.config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        model_type=args.model,
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

