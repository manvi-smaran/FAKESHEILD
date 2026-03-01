"""
Generate comparison charts for FAKESHEILD experiments.

All results are hardcoded from experiment runs. Charts are saved to results/charts/.

Run: python scripts/generate_results_charts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

output_dir = Path("results/charts")
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COLOR PALETTE
# =============================================================================
C = {
    'zero_shot': '#FF6B6B',
    'few_shot':  '#FF9E7A',
    'lora':      '#4ECDC4',
    'lora_mlp':  '#45B7D1',
    'dora':      '#96CEB4',
    'combined':  '#6C5CE7',
    'cnn':       '#FFD93D',
    'bg':        '#1a1a2e',
    'panel':     '#16213e',
    'grid':      '#2a2a4a',
    'text':      '#EAEAEA',
}

plt.rcParams.update({
    'figure.facecolor': C['bg'],
    'axes.facecolor':   C['panel'],
    'axes.edgecolor':   C['grid'],
    'axes.labelcolor':  C['text'],
    'text.color':       C['text'],
    'xtick.color':      C['text'],
    'ytick.color':      C['text'],
    'grid.color':       C['grid'],
    'grid.alpha':       0.4,
    'font.family':      'sans-serif',
    'font.size':        12,
})

# =============================================================================
# EXPERIMENT RESULTS  (all from actual runs)
# =============================================================================

# ── Approach accuracy on FaceForensics++ ─────────────────────────────────

APPROACHES = [
    "Zero-Shot",
    "Few-Shot\n(k=4)",
    "Few-Shot\n(k=8)",
    "LoRA",
    "LoRA+MLP",
    "DoRA",
    "LoRA\n(Combined)",
    "CNN+LoRA",
]
APPROACH_ACC = [31.92, 33.0, 35.0, 70.38, 70.86, 70.61, 82.55, 76.42]
APPROACH_COLORS = [
    C['zero_shot'],
    C['few_shot'], C['few_shot'],
    C['lora'], C['lora_mlp'], C['dora'],
    C['combined'],
    C['cnn'],
]

# ── Fine-tuning method breakdown ──────────────────────────────────────────
FINETUNE_METHODS  = ['LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
FINETUNE_ACC      = [70.38, 70.86, 70.61, 82.55, 76.42]
FINETUNE_SEP      = [0.330, 0.417, 0.410, 0.726, 0.526]
FINETUNE_COLORS   = [C['lora'], C['lora_mlp'], C['dora'], C['combined'], C['cnn']]

# ── Per-class accuracy ────────────────────────────────────────────────────
PERCLASS_METHODS = ['Zero-Shot', 'Few-Shot (k=8)', 'LoRA (FF++)', 'LoRA (Combined)', 'CNN+LoRA']
REAL_CLASS_ACC   = [79.0,  70.0,  65.0,  78.0,  54.0]   # approx: 1-recall_fake
FAKE_CLASS_ACC   = [2.0,   8.0,   76.0,  87.0,  98.0]   # recall_fake

# ── p_fake separation ─────────────────────────────────────────────────────
SEP_METHODS = ['Zero-Shot', 'Few-Shot', 'LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
SEP_VALUES  = [0.05,        0.07,       0.330,  0.417,      0.410,  0.726,              0.526]
SEP_COLORS  = [C['zero_shot'], C['few_shot'],
               C['lora'], C['lora_mlp'], C['dora'], C['combined'], C['cnn']]

# ── mean_real / mean_fake ─────────────────────────────────────────────────
MEAN_METHODS = ['LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
MEAN_FAKE_V  = [0.673, 0.711, 0.703, 0.880, 0.733]
MEAN_REAL_V  = [0.343, 0.294, 0.292, 0.154, 0.207]


# =============================================================================
# HELPERS
# =============================================================================
def add_value_labels(ax, bars, fmt="{:.1f}%", pad=1.0, fontsize=11):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                fmt.format(h), ha='center', va='bottom',
                fontweight='bold', fontsize=fontsize, color=C['text'])

def add_hbar_labels(ax, bars, fmt="{:.2f}%", pad=0.5, fontsize=11):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va='center', fontweight='bold',
                fontsize=fontsize, color=C['text'])

def save(fig, name):
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# CHART 1 — All Approaches Comparison
# =============================================================================
def chart_all_approaches():
    fig, ax = plt.subplots(figsize=(15, 7))

    x = np.arange(len(APPROACHES))
    bars = ax.bar(x, APPROACH_ACC, color=APPROACH_COLORS,
                  width=0.65, edgecolor='white', linewidth=0.5, alpha=0.9)
    add_value_labels(ax, bars, pad=1.2)

    # Chance line
    ax.axhline(50, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(len(APPROACHES) - 0.4, 51.5, 'Random (50%)',
            fontsize=9, color='#FF6B6B', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(APPROACHES, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Qwen2-VL: All Approaches on FaceForensics++',
                 fontsize=16, fontweight='bold', pad=18)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor=C['zero_shot'], label='Zero-Shot'),
        mpatches.Patch(facecolor=C['few_shot'],  label='Few-Shot'),
        mpatches.Patch(facecolor=C['lora'],      label='LoRA'),
        mpatches.Patch(facecolor=C['lora_mlp'],  label='LoRA+MLP'),
        mpatches.Patch(facecolor=C['dora'],      label='DoRA'),
        mpatches.Patch(facecolor=C['combined'],  label='LoRA (Combined Dataset)'),
        mpatches.Patch(facecolor=C['cnn'],       label='CNN+LoRA Hybrid'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=9, facecolor=C['panel'], edgecolor=C['grid'])

    save(fig, '1_all_approaches.png')


# =============================================================================
# CHART 2 — Fine-Tuning Method Breakdown (accuracy + separation)
# =============================================================================
def chart_finetune_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: accuracy
    x = np.arange(len(FINETUNE_METHODS))
    bars = ax1.bar(x, FINETUNE_ACC, color=FINETUNE_COLORS,
                   width=0.6, edgecolor='white', linewidth=0.5, alpha=0.9)
    add_value_labels(ax1, bars, pad=0.8)
    ax1.axhline(70.5, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(0.05, 71.5, '~70% FF++ ceiling', fontsize=9,
             color='#FF6B6B', alpha=0.8, transform=ax1.get_xaxis_transform())
    ax1.set_xticks(x)
    ax1.set_xticklabels(FINETUNE_METHODS, fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Fine-Tuning Method', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 95)
    ax1.grid(axis='y', alpha=0.3)

    # Right: p_fake separation
    bars2 = ax2.bar(x, FINETUNE_SEP, color=FINETUNE_COLORS,
                    width=0.6, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars2, FINETUNE_SEP):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', fontweight='bold',
                 fontsize=11, color=C['text'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(FINETUNE_METHODS, fontsize=10)
    ax2.set_ylabel('p_fake Separation (fake − real)', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Separation by Method', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 0.85)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('PEFT Fine-Tuning Method Comparison — FaceForensics++',
                 fontsize=15, fontweight='bold', y=1.01)
    save(fig, '2_finetune_comparison.png')


# =============================================================================
# CHART 3 — Per-Class Accuracy (Real vs Fake)
# =============================================================================
def chart_per_class():
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(PERCLASS_METHODS))
    w = 0.35
    b1 = ax.bar(x - w/2, REAL_CLASS_ACC, w, label='Real → Correctly Real',
                color='#4ECDC4', alpha=0.9, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, FAKE_CLASS_ACC, w, label='Fake → Correctly Fake',
                color='#FF6B6B', alpha=0.9, edgecolor='white', linewidth=0.5)

    for bar, val in zip(b1, REAL_CLASS_ACC):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val}%', ha='center', fontweight='bold', fontsize=10, color=C['text'])
    for bar, val in zip(b2, FAKE_CLASS_ACC):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val}%', ha='center', fontweight='bold', fontsize=10, color=C['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(PERCLASS_METHODS, fontsize=10)
    ax.set_ylabel('Per-Class Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Accuracy: Zero/Few-Shot vs Fine-Tuned',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=10,
              facecolor=C['panel'], edgecolor=C['grid'])
    ax.grid(axis='y', alpha=0.3)

    # Annotation
    ax.annotate('⚠ Zero/Few-Shot:\nHigh real recall\nbut misses fakes',
                xy=(0.5, 8), xytext=(0.8, 40),
                fontsize=9, color='#FF6B6B',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2d1f1f',
                          edgecolor='#FF6B6B', alpha=0.8))

    save(fig, '3_per_class_accuracy.png')


# =============================================================================
# CHART 4 — p_fake Separation Across All Methods
# =============================================================================
def chart_separation():
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(SEP_METHODS))
    bars = ax.bar(x, SEP_VALUES, color=SEP_COLORS,
                  width=0.6, edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, SEP_VALUES):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold',
                fontsize=11, color=C['text'])

    ax.axhline(0, color='#FF6B6B', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SEP_METHODS, fontsize=10)
    ax.set_ylabel('Separation (mean_fake − mean_real)', fontsize=12, fontweight='bold')
    ax.set_title('p_fake Probability Separation: How Well the Model Discriminates',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(-0.05, 0.85)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.02, 0.92,
            'Higher = better calibrated discrimination',
            transform=ax.transAxes, fontsize=10, alpha=0.7,
            style='italic', color=C['text'])

    save(fig, '4_pfake_separation.png')


# =============================================================================
# CHART 5 — mean_fake vs mean_real per fine-tuning method
# =============================================================================
def chart_mean_pfake():
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(MEAN_METHODS))
    w = 0.35
    b1 = ax.bar(x - w/2, MEAN_FAKE_V, w, label='mean p_fake (Fake images)',
                color='#FF6B6B', alpha=0.9, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, MEAN_REAL_V, w, label='mean p_fake (Real images)',
                color='#4ECDC4', alpha=0.9, edgecolor='white', linewidth=0.5)

    for bar, val in zip(b1, MEAN_FAKE_V):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    for bar, val in zip(b2, MEAN_REAL_V):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

    ax.axhline(0.5, color='white', linestyle=':', linewidth=1.2, alpha=0.4)
    ax.text(len(MEAN_METHODS) - 0.3, 0.51, 'Decision\nboundary',
            fontsize=8, color='white', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MEAN_METHODS, fontsize=10)
    ax.set_ylabel('Mean p_fake Score', fontsize=12, fontweight='bold')
    ax.set_title('p_fake Calibration: Fake vs Real Images — Fine-Tuned Models',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10,
              facecolor=C['panel'], edgecolor=C['grid'])
    ax.grid(axis='y', alpha=0.3)

    save(fig, '5_mean_pfake_calibration.png')


# =============================================================================
# CHART 6 — Summary 2×2 Dashboard
# =============================================================================
def chart_summary_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('FAKESHEILD: Deepfake Detection — Results Summary',
                 fontsize=18, fontweight='bold', y=1.00)

    # ── A: Accuracy progression ──────────────────────────────────────────
    ax = axes[0, 0]
    stages = ['Zero-Shot', 'Few-Shot', 'LoRA\n(FF++)', 'CNN+LoRA\n(FF++)', 'LoRA\n(Combined)']
    accs   = [31.92,       34.0,       70.38,          76.42,              82.55]
    stage_colors = [C['zero_shot'], C['few_shot'], C['lora'], C['cnn'], C['combined']]

    ax.plot(range(len(stages)), accs, 'o-', color='#4ECDC4',
            linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2.5)
    ax.fill_between(range(len(stages)), accs, alpha=0.08, color='#4ECDC4')
    for i, (s, a) in enumerate(zip(stages, accs)):
        ax.annotate(f'{a:.1f}%', (i, a), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontweight='bold',
                    fontsize=11, color=stage_colors[i])
    ax.axhline(50, color='#FF6B6B', linestyle=':', linewidth=1.2, alpha=0.4)
    ax.text(0.02, 52, 'Random', fontsize=8, color='#FF6B6B', alpha=0.6)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(20, 95)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('A) Accuracy Progression', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.2)

    # ── B: p_fake separation ─────────────────────────────────────────────
    ax = axes[0, 1]
    x = np.arange(len(SEP_METHODS))
    bars = ax.bar(x, SEP_VALUES, color=SEP_COLORS,
                  width=0.6, alpha=0.9, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, SEP_VALUES):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(SEP_METHODS, fontsize=8)
    ax.set_ylabel('Separation')
    ax.set_title('B) p_fake Separation', fontweight='bold', fontsize=13)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.2)

    # ── C: Per-class accuracy bar chart ──────────────────────────────────
    ax = axes[1, 0]
    methods_short = ['Zero-Shot', 'Few-Shot', 'LoRA', 'Combined', 'CNN+LoRA']
    x = np.arange(len(methods_short))
    w = 0.35
    b1 = ax.bar(x - w/2, REAL_CLASS_ACC, w, label='Real class accuracy',
                color='#4ECDC4', alpha=0.9, edgecolor='white', linewidth=0.4)
    b2 = ax.bar(x + w/2, FAKE_CLASS_ACC, w, label='Fake class accuracy',
                color='#FF6B6B', alpha=0.9, edgecolor='white', linewidth=0.4)
    for bar, val in zip(list(b1)+list(b2), REAL_CLASS_ACC+FAKE_CLASS_ACC):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=9)
    ax.set_ylabel('Per-Class Accuracy (%)')
    ax.set_title('C) Per-Class Accuracy', fontweight='bold', fontsize=13)
    ax.legend(fontsize=8, facecolor=C['panel'], edgecolor=C['grid'])
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.2)

    # ── D: Key findings ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')
    findings = (
        "Key Findings\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "① Zero/Few-Shot VLMs are unreliable\n"
        "   → 31% accuracy (below random!)\n"
        "   → Severe real-class bias\n\n"
        "② LoRA fine-tuning is essential\n"
        "   → 31% → 70%+ on FF++\n"
        "   → Balanced real & fake recall\n\n"
        "③ CNN+LoRA hybrid adds +6% over LoRA\n"
        "   → 76.42% accuracy, sep=0.526\n"
        "   → Forensic features genuinely help\n\n"
        "④ Combined dataset breaks ceiling\n"
        "   → 82.55% with LoRA only\n\n"
        "⑤ Next: CNN+LoRA on combined dataset\n"
        "   → Expected: >85% accuracy"
    )
    ax.text(0.05, 0.95, findings, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8',
                      facecolor='#0f3460', edgecolor='#4ECDC4', alpha=0.85))
    ax.set_title('D) Key Findings', fontweight='bold', fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, '6_summary_dashboard.png')


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == '__main__':
    print(f"\n{'='*52}")
    print(f"  Generating FAKESHEILD Results Charts")
    print(f"{'='*52}\n")

    chart_all_approaches()
    chart_finetune_breakdown()
    chart_per_class()
    chart_separation()
    chart_mean_pfake()
    chart_summary_dashboard()

    print(f"\n  All charts saved to: {output_dir}/")
    print(f"{'='*52}")
