import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.few_shot import run_few_shot_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run Few-Shot Deepfake Detection Evaluation")
    
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
        "--k",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="TOTAL number of few-shot examples (k//2 per class). E.g., k=4 means 2 real + 2 fake examples.",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_configs.yaml",
        help="Path to configuration file.",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Use JSON-based prompts with structured output.",
    )
    
    parser.add_argument(
        "--forced_choice",
        action="store_true",
        help="Use forced-choice logit-based p_fake scoring (recommended, eliminates template copying).",
    )
    
    args = parser.parse_args()
    
    if args.forced_choice:
        eval_type = "Forced-Choice"
    elif args.json:
        eval_type = "JSON Few-Shot"
    else:
        eval_type = "Few-Shot"
    
    print(f"\n{'='*60}")
    print(f"VSLM {eval_type} Deepfake Detection Evaluation")
    print(f"{'='*60}")
    print(f"Model:         {args.model or 'ALL'}")
    print(f"Dataset:       {args.dataset}")
    print(f"K values:      {args.k}")
    print(f"Max Samples:   {args.max_samples or 'ALL'}")
    print(f"Forced-Choice: {args.forced_choice}")
    print(f"JSON Mode:     {args.json}")
    print(f"{'='*60}\n")
    
    # Use unified function with use_json parameter
    results = run_few_shot_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        k_values=args.k,
        max_samples=args.max_samples,
        config_path=args.config,
        use_json=args.json,
        use_forced_choice=args.forced_choice,
    )
    
    return results


if __name__ == "__main__":
    main()


