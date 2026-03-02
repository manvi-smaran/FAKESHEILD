"""
Generate publication-quality comparison charts for FAKESHEILD experiments.

ALL VALUES ARE FROM ACTUAL EXPERIMENT RUNS — no estimates.

Style: white background, serif labels, hatching for B&W printing, 300 DPI.

Run: python scripts/generate_results_charts.py
Output: results/charts/ directory with PNG charts
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("results/charts")
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Publication style
# =============================================================================
plt.rcParams.update({
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     '#333333',
    'axes.labelcolor':    '#222222',
    'axes.grid':          True,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'text.color':         '#222222',
    'xtick.color':        '#333333',
    'ytick.color':        '#333333',
    'grid.color':         '#DDDDDD',
    'grid.alpha':         0.7,
    'grid.linewidth':     0.5,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'figure.dpi':         100,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
})

# Muted colours suitable for publication / B&W printing
CLR_ZS   = '#B0B0B0'
CLR_FS   = '#9E9E9E'
CLR_LORA = '#5B9BD5'
CLR_MLP  = '#4472C4'
CLR_DORA = '#70AD47'
CLR_COMB = '#7030A0'
CLR_CNN  = '#ED7D31'
CLR_REAL = '#5B9BD5'
CLR_FAKE = '#ED7D31'

# =============================================================================
# ALL VERIFIED EXPERIMENT DATA
# =============================================================================
#
# Zero-shot & Few-shot: run_fast_eval.py on 20,000 samples (2839 real, 17161 fake)
#   NOTE: Dataset is 85.8% fake. "Accuracy" is heavily affected by class imbalance.
#   Balanced accuracy = (real_acc + fake_acc) / 2 is the fair metric.
#
# Fine-tuned models: evaluated via forced-choice on ~4000 samples (~50/50 split)
#

# ── Zero-shot / Few-shot results (fast_eval, 20K samples) ────────────
ZFS_METHODS   = ['Zero-Shot', 'Few-Shot\n(k=2)', 'Few-Shot\n(k=4)', 'Few-Shot\n(k=8)']
ZFS_ACC       = [20.21, 14.21, 14.19, 14.19]
ZFS_AUC       = [0.5125, 0.5238, 0.5115, 0.5347]
ZFS_SEP       = [0.0056, 0.0048, 0.0018, 0.0051]
ZFS_MEAN_FAKE = [0.332, 0.256, 0.227, 0.252]
ZFS_MEAN_REAL = [0.326, 0.251, 0.225, 0.247]
ZFS_REAL_ACC  = [93.06, 100.0, 100.0, 100.0]
ZFS_FAKE_ACC  = [8.16, 0.02, 0.0, 0.0]
ZFS_BAL_ACC   = [(r + f) / 2 for r, f in zip(ZFS_REAL_ACC, ZFS_FAKE_ACC)]
# Balanced: [50.61, 50.01, 50.0, 50.0]

# ── Fine-tuned results (forced-choice eval, ~4K samples, ~50/50) ─────
FT_METHODS    = ['LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
FT_ACC        = [70.38, 70.86, 70.61, 82.55, 76.42]
FT_SEP        = [0.330, 0.417, 0.410, 0.726, 0.526]
FT_MEAN_FAKE  = [0.673, 0.711, 0.703, 0.880, 0.733]
FT_MEAN_REAL  = [0.343, 0.294, 0.292, 0.154, 0.207]
FT_CLR        = [CLR_LORA, CLR_MLP, CLR_DORA, CLR_COMB, CLR_CNN]
FT_HATCH      = ['///', '\\\\\\', '...', 'xxx', '|||']

# CNN+LoRA per-class (derived from: 3050/3991 correct, R=1979, F=2012, predR=2110, predF=1881)
# TN=1574, TP=1476, FP=405, FN=536
CNN_REAL_ACC = 79.5   # 1574/1979
CNN_FAKE_ACC = 73.3   # 1476/2012


# =============================================================================
# HELPERS
# =============================================================================
def _label_bars(ax, bars, fmt='{:.1f}', ypad=0.8, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + ypad,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fontsize, color='#333333', fontweight='bold')


def _save(fig, name):
    path = output_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f'  [saved] {path}')


# =============================================================================
# FIGURE 1: Balanced Accuracy — Fair Comparison Across All Approaches
# =============================================================================
def fig_balanced_accuracy():
    """Use balanced accuracy for fair comparison across different eval datasets."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    methods = ['Zero-Shot', 'Few-Shot\n(k=2)', 'Few-Shot\n(k=4)', 'Few-Shot\n(k=8)',
               'LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
    # Zero/few-shot: balanced acc from actual per-class numbers
    # Fine-tuned: accuracy on balanced dataset (~50/50) ≈ balanced accuracy
    bal_acc = ZFS_BAL_ACC + FT_ACC
    colors = [CLR_ZS, CLR_FS, CLR_FS, CLR_FS] + FT_CLR

    x = np.arange(len(methods))
    bars = ax.bar(x, bal_acc, color=colors, width=0.65,
                  edgecolor='#555555', linewidth=0.6)
    _label_bars(ax, bars, fmt='{:.1f}%', ypad=0.8, fontsize=9)

    ax.axhline(50, color='#CC0000', linewidth=0.8, linestyle='--', label='Random baseline (50%)')

    # Separator line between zero/few-shot and fine-tuned
    ax.axvline(3.5, color='#999999', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.text(1.5, 88, 'No Training', ha='center', fontsize=9, style='italic', color='#777')
    ax.text(6.5, 88, 'Fine-Tuned (10 epochs)', ha='center', fontsize=9, style='italic', color='#777')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('Fig. 1:  Balanced Accuracy Across All Approaches  (FaceForensics++)')
    ax.set_ylim(0, 95)
    ax.legend(loc='upper left', framealpha=0.9)
    _save(fig, '1_balanced_accuracy.png')


# =============================================================================
# FIGURE 2: Fine-Tuning Method Comparison (accuracy + separation)
# =============================================================================
def fig_finetune_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(FT_METHODS))

    # (a) Accuracy
    bars = ax1.bar(x, FT_ACC, color=FT_CLR, width=0.6,
                   hatch=FT_HATCH, edgecolor='#444', linewidth=0.6)
    _label_bars(ax1, bars, fmt='{:.2f}%', ypad=0.6)
    ax1.axhline(70.5, color='#CC0000', linewidth=0.7, linestyle=':',
                alpha=0.7, label='~70% FF++ ceiling')
    ax1.set_xticks(x); ax1.set_xticklabels(FT_METHODS)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Accuracy')
    ax1.set_ylim(55, 90)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # (b) Separation
    bars2 = ax2.bar(x, FT_SEP, color=FT_CLR, width=0.6,
                    hatch=FT_HATCH, edgecolor='#444', linewidth=0.6)
    _label_bars(ax2, bars2, fmt='{:.3f}', ypad=0.008)
    ax2.set_xticks(x); ax2.set_xticklabels(FT_METHODS)
    ax2.set_ylabel(r'$\Delta p_{\mathrm{fake}}$  (mean fake $-$ mean real)')
    ax2.set_title('(b) Probability Separation')
    ax2.set_ylim(0, 0.85)

    fig.suptitle('Fig. 2:  PEFT Fine-Tuning Method Comparison  (FaceForensics++, 10 epochs)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, '2_finetune_comparison.png')


# =============================================================================
# FIGURE 3: Per-Class Accuracy — Verified Data Only
# =============================================================================
def fig_per_class():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    methods = ['Zero-Shot', 'Few-Shot (k=8)', 'CNN+LoRA']
    real_acc = [93.06, 100.0, CNN_REAL_ACC]
    fake_acc = [8.16, 0.0, CNN_FAKE_ACC]

    x = np.arange(len(methods))
    w = 0.35
    b1 = ax.bar(x - w/2, real_acc, w, label='Real-class accuracy',
                color=CLR_REAL, edgecolor='#444', linewidth=0.5, hatch='///')
    b2 = ax.bar(x + w/2, fake_acc, w, label='Fake-class accuracy',
                color=CLR_FAKE, edgecolor='#444', linewidth=0.5, hatch='\\\\\\')

    for bar, val in zip(list(b1) + list(b2), real_acc + fake_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig. 3:  Per-Class Accuracy — Class Bias in Zero/Few-Shot vs Fine-Tuned')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=0.9)

    # Annotation for the bias
    ax.annotate('Predicts 100% as "Real"\n(0% fake detection)',
                xy=(1, 0), xytext=(1.3, 30),
                fontsize=9, color='#CC0000',
                arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5',
                          edgecolor='#CC0000', alpha=0.9))

    _save(fig, '3_per_class_accuracy.png')


# =============================================================================
# FIGURE 4: AUC-ROC — All Methods
# =============================================================================
def fig_auc():
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['Zero-Shot', 'Few-Shot\n(k=2)', 'Few-Shot\n(k=4)', 'Few-Shot\n(k=8)']
    aucs = ZFS_AUC
    colors = [CLR_ZS, CLR_FS, CLR_FS, CLR_FS]

    x = np.arange(len(methods))
    bars = ax.bar(x, aucs, color=colors, width=0.55,
                  edgecolor='#444', linewidth=0.5)
    _label_bars(ax, bars, fmt='{:.4f}', ypad=0.005)

    ax.axhline(0.5, color='#CC0000', linewidth=0.8, linestyle='--',
               label='Random baseline (AUC = 0.5)')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Fig. 4:  AUC-ROC — Zero-Shot and Few-Shot  (Qwen2-VL-2B)')
    ax.set_ylim(0.45, 0.58)
    ax.legend(loc='upper left', framealpha=0.9)

    ax.text(0.98, 0.05, 'All AUC values within 0.01-0.03 of random',
            transform=ax.transAxes, ha='right', fontsize=9,
            style='italic', color='#777')

    _save(fig, '4_auc_roc.png')


# =============================================================================
# FIGURE 5: p_fake Separation — All Methods
# =============================================================================
def fig_separation():
    fig, ax = plt.subplots(figsize=(12, 5))

    methods = ['Zero-Shot', 'Few-Shot\n(k=8)',
               'LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
    seps = [0.0056, 0.0051] + FT_SEP
    colors = [CLR_ZS, CLR_FS] + FT_CLR

    x = np.arange(len(methods))
    bars = ax.bar(x, seps, color=colors, width=0.6,
                  edgecolor='#444', linewidth=0.5)
    _label_bars(ax, bars, fmt='{:.3f}', ypad=0.008)

    ax.axhline(0, color='#333', linewidth=0.5)
    ax.axvline(1.5, color='#999', linewidth=0.8, linestyle='-', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(r'$\Delta p_{\mathrm{fake}}$  (mean fake $-$ mean real)')
    ax.set_title(r'Fig. 5:  $p_{\mathrm{fake}}$ Separation — Discrimination Quality')
    ax.set_ylim(-0.03, 0.82)

    ax.text(0.5, 0.75, 'Near-zero:\nno discrimination',
            fontsize=8, color='#999', style='italic')

    _save(fig, '5_pfake_separation.png')


# =============================================================================
# FIGURE 6: p_fake Calibration (mean_fake vs mean_real)
# =============================================================================
def fig_calibration():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FT_METHODS))
    w = 0.35

    b1 = ax.bar(x - w/2, FT_MEAN_FAKE, w,
                label=r'Mean $p_{\mathrm{fake}}$ (fake images)',
                color=CLR_FAKE, edgecolor='#444', linewidth=0.5, hatch='\\\\\\')
    b2 = ax.bar(x + w/2, FT_MEAN_REAL, w,
                label=r'Mean $p_{\mathrm{fake}}$ (real images)',
                color=CLR_REAL, edgecolor='#444', linewidth=0.5, hatch='///')

    for bar, val in zip(list(b1) + list(b2), FT_MEAN_FAKE + FT_MEAN_REAL):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.axhline(0.5, color='#CC0000', linewidth=0.8, linestyle='--',
               label='Decision boundary (0.5)')

    ax.set_xticks(x)
    ax.set_xticklabels(FT_METHODS)
    ax.set_ylabel(r'Mean $p_{\mathrm{fake}}$')
    ax.set_title(r'Fig. 6:  $p_{\mathrm{fake}}$ Calibration — Fine-Tuned Models')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    _save(fig, '6_pfake_calibration.png')


# =============================================================================
# FIGURE 7: Accuracy Progression Line Plot
# =============================================================================
def fig_progression():
    fig, ax = plt.subplots(figsize=(9, 5))

    stages = ['Zero-Shot', 'Few-Shot\n(k=8)', 'LoRA\n(FF++)',
              'CNN+LoRA\n(FF++)', 'LoRA\n(Combined)']
    # Using balanced accuracy for zero/few-shot, accuracy for fine-tuned (~balanced eval set)
    accs = [50.6, 50.0, 70.38, 76.42, 82.55]
    colors = [CLR_ZS, CLR_FS, CLR_LORA, CLR_CNN, CLR_COMB]

    ax.plot(range(len(stages)), accs, 'o-', color='#444444',
            linewidth=2, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#333', zorder=3)

    for i, (s, a, c) in enumerate(zip(stages, accs, colors)):
        ax.scatter(i, a, color=c, s=100, zorder=4,
                   edgecolors='#333', linewidths=0.8)
        offset = 12 if i < 3 else -16
        ax.annotate(f'{a:.1f}%', (i, a),
                    textcoords='offset points', xytext=(0, offset),
                    ha='center', fontweight='bold', fontsize=11, color='#333')

    ax.fill_between(range(len(stages)), accs, alpha=0.06, color=CLR_LORA)
    ax.axhline(50, color='#CC0000', linewidth=0.7, linestyle=':', alpha=0.6)
    ax.text(0.01, 51, 'Random baseline', fontsize=8, color='#CC0000', alpha=0.7)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('Fig. 7:  Accuracy Progression — Zero-Shot to CNN-Augmented Fine-Tuning')
    ax.set_ylim(40, 90)
    _save(fig, '7_accuracy_progression.png')


# =============================================================================
# FIGURE 8: Zero/Few-Shot Detailed — Shows the "Real" Prediction Bias
# =============================================================================
def fig_zero_few_detail():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = ['Zero-Shot', 'Few-Shot\n(k=2)', 'Few-Shot\n(k=4)', 'Few-Shot\n(k=8)']
    x = np.arange(len(labels))
    w = 0.35

    # (a) Per-class accuracy
    b1 = ax1.bar(x - w/2, ZFS_REAL_ACC, w, label='Real-class accuracy',
                 color=CLR_REAL, edgecolor='#444', linewidth=0.5, hatch='///')
    b2 = ax1.bar(x + w/2, ZFS_FAKE_ACC, w, label='Fake-class accuracy',
                 color=CLR_FAKE, edgecolor='#444', linewidth=0.5, hatch='\\\\\\')
    for bar, val in zip(list(b1) + list(b2), ZFS_REAL_ACC + ZFS_FAKE_ACC):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold', color='#333')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Per-Class Accuracy')
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=8, framealpha=0.9)

    # (b) AUC
    bars = ax2.bar(x, ZFS_AUC, color=[CLR_ZS, CLR_FS, CLR_FS, CLR_FS],
                   width=0.55, edgecolor='#444', linewidth=0.5)
    _label_bars(ax2, bars, fmt='{:.4f}', ypad=0.003, fontsize=9)
    ax2.axhline(0.5, color='#CC0000', linewidth=0.8, linestyle='--',
                label='Random (AUC=0.5)')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('(b) AUC-ROC')
    ax2.set_ylim(0.48, 0.56)
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.suptitle('Fig. 8:  Zero-Shot and Few-Shot Analysis — Severe "Real" Prediction Bias',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, '8_zero_few_shot_detail.png')


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    print('\n' + '=' * 52)
    print('  Generating FAKESHEILD Publication Charts')
    print('=' * 52 + '\n')

    fig_balanced_accuracy()
    fig_finetune_comparison()
    fig_per_class()
    fig_auc()
    fig_separation()
    fig_calibration()
    fig_progression()
    fig_zero_few_detail()

    print(f'\n  All charts saved to: {output_dir}/')
    print('=' * 52)
