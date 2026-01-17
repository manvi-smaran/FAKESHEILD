import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.zero_shot import run_zero_shot_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run Zero-Shot Deepfake Detection Evaluation")
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["qwen_vl", "smolvlm", "moondream", "florence2", "llava_next"],
        help="Specific model to evaluate. If not provided, evaluates all models.",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="celebdf",
        choices=["celebdf", "faceforensics"],
        help="Dataset to evaluate on.",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate. Useful for quick testing.",
    )
    
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="zero_shot_binary",
        choices=["zero_shot_binary", "zero_shot_detection", "zero_shot_simple", "chain_of_thought"],
        help="Type of prompt to use.",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_configs.yaml",
        help="Path to configuration file.",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"VSLM Zero-Shot Deepfake Detection Evaluation")
    print(f"{'='*60}")
    print(f"Model:      {args.model or 'ALL'}")
    print(f"Dataset:    {args.dataset}")
    print(f"Prompt:     {args.prompt_type}")
    print(f"Max Samples: {args.max_samples or 'ALL'}")
    print(f"{'='*60}\n")
    
    results = run_zero_shot_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        prompt_type=args.prompt_type,
        config_path=args.config,
    )
    
    return results


if __name__ == "__main__":
    main()
