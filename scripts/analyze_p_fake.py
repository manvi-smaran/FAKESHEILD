#!/usr/bin/env python3
"""
Analyze p_fake distribution from evaluation results.
Shows histogram of p_fake for real vs fake samples.

Usage:
    python scripts/analyze_p_fake.py
    
After running evaluation with --json flag, this will show:
1. Whether p_fake values have variance (or are template-copied)
2. Histogram of p_fake for real vs fake samples
3. AUC diagnosis
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_and_plot(results_dir: str = "results", output_dir: str = "results/figures"):
    """Load all JSON results and plot p_fake distributions."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for result_file in results_path.glob("json_few_shot_*.json"):
        print(f"\n{'='*60}")
        print(f"Analyzing: {result_file.name}")
        print(f"{'='*60}")
        
        with open(result_file) as f:
            data = json.load(f)
        
        model = data.get("model_full_name", data.get("model"))
        dataset = data.get("dataset")
        
        for k_key, k_data in data.get("results_by_k", {}).items():
            p_fake_values = k_data.get("p_fake_values", [])
            labels = k_data.get("ground_truth_labels", [])
            method = k_data.get("p_fake_method", "unknown")
            
            if not p_fake_values or not labels:
                print(f"  {k_key}: No p_fake data saved. Re-run evaluation.")
                continue
            
            p_fake = np.array(p_fake_values)
            labels = np.array(labels)
            
            real_scores = p_fake[labels == 0]
            fake_scores = p_fake[labels == 1]
            
            # Statistics
            print(f"\n{k_key} (method: {method}):")
            print(f"  Real samples: n={len(real_scores)}, p_fake mean={real_scores.mean():.3f}, std={real_scores.std():.4f}")
            print(f"  Fake samples: n={len(fake_scores)}, p_fake mean={fake_scores.mean():.3f}, std={fake_scores.std():.4f}")
            
            # Unique values check
            unique_vals = len(np.unique(p_fake))
            print(f"  Unique p_fake values: {unique_vals}")
            
            if unique_vals <= 5:
                print(f"  ⚠️  WARNING: Only {unique_vals} unique values!")
                print(f"  Values: {sorted(np.unique(p_fake))}")
                print(f"  This indicates TEMPLATE COPYING - models just output example values!")
            
            # AUC diagnosis
            auc = k_data.get("metrics", {}).get("auc", 0.5)
            if real_scores.std() < 0.01 and fake_scores.std() < 0.01:
                print(f"  ⚠️  AUC={auc:.4f} is MEANINGLESS (no score variance)")
            elif auc < 0.55:
                print(f"  ⚠️  AUC={auc:.4f} indicates no class separability")
            else:
                print(f"  ✓ AUC={auc:.4f}")
            
            # Plot histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(real_scores, bins=20, alpha=0.7, label=f'Real (n={len(real_scores)})', color='green', density=True)
            ax.hist(fake_scores, bins=20, alpha=0.7, label=f'Fake (n={len(fake_scores)})', color='red', density=True)
            
            ax.set_xlabel('p_fake (Probability of being fake)')
            ax.set_ylabel('Density')
            ax.set_title(f'{model} on {dataset} ({k_key})\nAUC={auc:.3f}, Method={method}')
            ax.legend()
            ax.set_xlim(0, 1)
            
            # Add vertical lines for means
            ax.axvline(real_scores.mean(), color='green', linestyle='--', label='Real mean')
            ax.axvline(fake_scores.mean(), color='red', linestyle='--', label='Fake mean')
            
            # Save figure
            fig_name = f"p_fake_dist_{model.replace(' ', '_')}_{dataset}_{k_key.replace('=', '')}.png"
            fig.savefig(output_path / fig_name, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {output_path / fig_name}")


if __name__ == "__main__":
    analyze_and_plot()
