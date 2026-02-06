#!/usr/bin/env python3
"""
Run PEFT Fine-Tuning for VLMs on Deepfake Detection

Supports multiple PEFT methods: LoRA, QLoRA, DoRA, AdaLoRA, Prefix Tuning

Examples:
    # LoRA (baseline)
    python scripts/run_peft_finetune.py --model smolvlm --peft_method lora --dataset celebdf --epochs 3

    # QLoRA (4-bit quantization)
    python scripts/run_peft_finetune.py --model smolvlm --peft_method qlora --dataset celebdf --epochs 3

    # DoRA (weight-decomposed)
    python scripts/run_peft_finetune.py --model smolvlm --peft_method dora --dataset celebdf --epochs 3

    # AdaLoRA (adaptive rank)
    python scripts/run_peft_finetune.py --model smolvlm --peft_method adalora --dataset celebdf --epochs 3

    # Prefix Tuning
    python scripts/run_peft_finetune.py --model smolvlm --peft_method prefix --dataset celebdf --epochs 3
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.peft_finetune import run_peft_finetuning
from src.training.peft_configs import list_available_methods, get_method_info


def main():
    parser = argparse.ArgumentParser(
        description="Run PEFT Fine-Tuning for VLMs on Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PEFT Methods Available:
  lora     - Low-Rank Adaptation (baseline)
  qlora    - Quantized LoRA (4-bit, memory efficient)
  dora     - Weight-Decomposed LoRA (improved performance)
  adalora  - Adaptive LoRA (dynamic rank allocation)
  prefix   - Prefix Tuning (learnable prefix tokens)
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="smolvlm",
        choices=["smolvlm", "qwen_vl"],
        help="VLM model to fine-tune (moondream not supported - uses custom API)",
    )
    
    # PEFT method selection
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=list_available_methods(),
        help="PEFT method to use",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="faceforensics",
        help="Dataset name from configs/model_configs.yaml",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_configs.yaml",
        help="Path to config file",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use (for quick testing)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (5e-5 recommended for VLM LoRA)",
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (start with 8, increase if needed)",
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (usually 2x rank)",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: checkpoints/{model}-{peft_method})",
    )
    
    # AdaLoRA specific
    parser.add_argument(
        "--adalora_init_r",
        type=int,
        default=12,
        help="AdaLoRA initial rank",
    )
    
    parser.add_argument(
        "--adalora_target_r",
        type=int,
        default=8,
        help="AdaLoRA target rank after pruning",
    )
    
    # Prefix Tuning specific
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="Number of virtual tokens for Prefix Tuning",
    )
    
    args = parser.parse_args()
    
    # Build output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.model}-{args.peft_method}"
    
    # Print configuration
    print("=" * 60)
    print(f"PEFT Fine-Tuning for Deepfake Detection")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"PEFT Method: {get_method_info(args.peft_method)}")
    print(f"Dataset:     {args.dataset}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"LR:          {args.learning_rate}")
    print(f"LoRA Rank:   {args.lora_rank}")
    print(f"LoRA Alpha:  {args.lora_alpha}")
    print(f"Output:      {args.output_dir}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("=" * 60)
    
    # Build PEFT-specific kwargs
    peft_kwargs = {}
    
    if args.peft_method == "adalora":
        peft_kwargs["init_r"] = args.adalora_init_r
        peft_kwargs["target_r"] = args.adalora_target_r
        print(f"AdaLoRA Config: init_r={args.adalora_init_r}, target_r={args.adalora_target_r}")
    
    if args.peft_method == "prefix":
        peft_kwargs["num_virtual_tokens"] = args.num_virtual_tokens
        print(f"Prefix Config: num_virtual_tokens={args.num_virtual_tokens}")
    
    print()
    
    # Run training
    trainer, metrics = run_peft_finetuning(
        config_path=args.config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        model_type=args.model,
        peft_method=args.peft_method,
        **peft_kwargs,
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Loss:   {metrics.get('best_loss', 'N/A'):.4f}")
    print(f"Final Loss:  {metrics.get('final_loss', 'N/A'):.4f}")
    print(f"Accuracy:    {metrics.get('accuracy', 0):.2%}")
    print(f"Checkpoints: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
