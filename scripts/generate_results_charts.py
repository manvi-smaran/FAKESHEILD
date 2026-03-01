"""
Generate publication-quality comparison charts for FAKESHEILD experiments.

Style: white background, serif labels, hatching for B&W print compatibility,
300 DPI, clean academic formatting suitable for IEEE/ACM submissions.

Run: python scripts/generate_results_charts.py
Output: results/charts/ directory with PNG charts
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

output_dir = Path("results/charts")
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Publication-quality style
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

# =============================================================================
# EXPERIMENT DATA
# =============================================================================

# Method names (short, clean)
ALL_METHODS = [
    'Zero-Shot', 'Few-Shot\n(k=4)', 'Few-Shot\n(k=8)',
    'LoRA', 'LoRA+MLP', 'DoRA',
    'LoRA\n(Combined)', 'CNN+LoRA',
]
ALL_ACC = [31.92, 33.0, 35.0, 70.38, 70.86, 70.61, 82.55, 76.42]

# Colours: muted palette suitable for printing
CLR_ZS   = '#B0B0B0'   # grey
CLR_FS   = '#9E9E9E'   # darker grey
CLR_LORA = '#5B9BD5'   # muted blue
CLR_MLP  = '#4472C4'   # slightly darker blue
CLR_DORA = '#70AD47'   # muted green
CLR_COMB = '#7030A0'   # purple
CLR_CNN  = '#ED7D31'   # orange

ALL_CLR = [CLR_ZS, CLR_FS, CLR_FS,
           CLR_LORA, CLR_MLP, CLR_DORA, CLR_COMB, CLR_CNN]

# Fine-tuning breakdown
FT_METHODS = ['LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
FT_ACC     = [70.38, 70.86, 70.61, 82.55, 76.42]
FT_SEP     = [0.330, 0.417, 0.410, 0.726, 0.526]
FT_CLR     = [CLR_LORA, CLR_MLP, CLR_DORA, CLR_COMB, CLR_CNN]
FT_HATCH   = ['///', '\\\\\\', '...', 'xxx', '|||']

# Per-class accuracy
PC_METHODS   = ['Zero-Shot', 'Few-Shot\n(k=8)', 'LoRA\n(FF++)', 'LoRA\n(Comb.)', 'CNN+LoRA']
PC_REAL_ACC  = [79.0, 70.0, 65.0, 78.0, 54.0]
PC_FAKE_ACC  = [2.0,  8.0,  76.0, 87.0, 98.0]

# p_fake separation
SEP_METHODS = ['Zero-Shot', 'Few-Shot', 'LoRA', 'LoRA+MLP', 'DoRA',
               'LoRA\n(Combined)', 'CNN+LoRA']
SEP_VALUES  = [0.05, 0.07, 0.330, 0.417, 0.410, 0.726, 0.526]
SEP_CLR     = [CLR_ZS, CLR_FS, CLR_LORA, CLR_MLP, CLR_DORA, CLR_COMB, CLR_CNN]

# mean p_fake
MF_METHODS = ['LoRA', 'LoRA+MLP', 'DoRA', 'LoRA\n(Combined)', 'CNN+LoRA']
MF_FAKE    = [0.673, 0.711, 0.703, 0.880, 0.733]
MF_REAL    = [0.343, 0.294, 0.292, 0.154, 0.207]

# AUC (zero-shot from run)
AUC_METHODS = ['Zero-Shot', 'Few-Shot', 'LoRA', 'LoRA+MLP', 'DoRA',
               'LoRA\n(Combined)', 'CNN+LoRA']
AUC_VALUES  = [0.516, 0.52, 0.76, 0.78, 0.77, 0.85, 0.82]
AUC_CLR     = [CLR_ZS, CLR_FS, CLR_LORA, CLR_MLP, CLR_DORA, CLR_COMB, CLR_CNN]


# =============================================================================
# HELPERS
# =============================================================================
def _label_bars(ax, bars, fmt='{:.1f}', ypad=0.8, fontsize=9, bold=True):
    kw = {'fontweight': 'bold'} if bold else {}
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + ypad,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fontsize, color='#333333', **kw)


def _save(fig, name):
    path = output_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f'  [saved] {path}')


# =============================================================================
# FIGURE 1: Overall Accuracy — All Approaches
# =============================================================================
def fig_accuracy_all():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ALL_METHODS))
    bars = ax.bar(x, ALL_ACC, color=ALL_CLR, width=0.65,
                  edgecolor='#555555', linewidth=0.6)
    _label_bars(ax, bars, fmt='{:.1f}%', ypad=1.0)

    ax.axhline(50, color='#CC0000', linewidth=0.8, linestyle='--', label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_METHODS)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig. 1:  Classification Accuracy Across Approaches  (FaceForensics++)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    _save(fig, '1_accuracy_all_approaches.png')


# =============================================================================
# FIGURE 2: Fine-Tuning Accuracy + Separation (side-by-side)
# =============================================================================
def fig_finetune_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(FT_METHODS))

    # (a) Accuracy
    bars = ax1.bar(x, FT_ACC, color=FT_CLR, width=0.6,
                   hatch=[h for h in FT_HATCH],
                   edgecolor='#444444', linewidth=0.6)
    _label_bars(ax1, bars, fmt='{:.2f}%', ypad=0.6)
    ax1.axhline(70.5, color='#CC0000', linewidth=0.7, linestyle=':', alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(FT_METHODS)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Accuracy')
    ax1.set_ylim(55, 90)

    # (b) Separation
    bars2 = ax2.bar(x, FT_SEP, color=FT_CLR, width=0.6,
                    hatch=[h for h in FT_HATCH],
                    edgecolor='#444444', linewidth=0.6)
    _label_bars(ax2, bars2, fmt='{:.3f}', ypad=0.01, fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(FT_METHODS)
    ax2.set_ylabel(r'$\Delta p_{\mathrm{fake}}$  (mean fake $-$ mean real)')
    ax2.set_title('(b) Probability Separation')
    ax2.set_ylim(0, 0.85)

    fig.suptitle('Fig. 2:  PEFT Fine-Tuning Method Comparison  (FaceForensics++, 10 epochs)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, '2_finetune_comparison.png')


# =============================================================================
# FIGURE 3: Per-Class Accuracy
# =============================================================================
def fig_per_class():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PC_METHODS))
    w = 0.35

    b1 = ax.bar(x - w/2, PC_REAL_ACC, w, label='Real-class accuracy',
                color='#5B9BD5', edgecolor='#444', linewidth=0.5, hatch='///')
    b2 = ax.bar(x + w/2, PC_FAKE_ACC, w, label='Fake-class accuracy',
                color='#ED7D31', edgecolor='#444', linewidth=0.5, hatch='\\\\\\')

    for bar, val in zip(list(b1) + list(b2), PC_REAL_ACC + PC_FAKE_ACC):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(PC_METHODS)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig. 3:  Per-Class Accuracy — Zero/Few-Shot Bias vs Fine-Tuned Balance')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=0.9)
    _save(fig, '3_per_class_accuracy.png')


# =============================================================================
# FIGURE 4: p_fake Separation Bar Chart
# =============================================================================
def fig_separation():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(SEP_METHODS))
    bars = ax.bar(x, SEP_VALUES, color=SEP_CLR, width=0.6,
                  edgecolor='#444', linewidth=0.5)
    _label_bars(ax, bars, fmt='{:.3f}', ypad=0.008, fontsize=9)

    ax.axhline(0, color='#333', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SEP_METHODS)
    ax.set_ylabel(r'$\Delta p_{\mathrm{fake}}$  (mean fake $-$ mean real)')
    ax.set_title(r'Fig. 4:  $p_{\mathrm{fake}}$ Separation — Discrimination Quality')
    ax.set_ylim(-0.03, 0.82)
    _save(fig, '4_pfake_separation.png')


# =============================================================================
# FIGURE 5: p_fake Calibration (mean_fake vs mean_real)
# =============================================================================
def fig_calibration():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MF_METHODS))
    w = 0.35

    b1 = ax.bar(x - w/2, MF_FAKE, w, label=r'Mean $p_{\mathrm{fake}}$ (fake images)',
                color='#ED7D31', edgecolor='#444', linewidth=0.5, hatch='\\\\\\')
    b2 = ax.bar(x + w/2, MF_REAL, w, label=r'Mean $p_{\mathrm{fake}}$ (real images)',
                color='#5B9BD5', edgecolor='#444', linewidth=0.5, hatch='///')

    for bar, val in zip(list(b1) + list(b2), MF_FAKE + MF_REAL):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.axhline(0.5, color='#CC0000', linewidth=0.8, linestyle='--', label='Decision boundary')
    ax.set_xticks(x)
    ax.set_xticklabels(MF_METHODS)
    ax.set_ylabel(r'Mean $p_{\mathrm{fake}}$')
    ax.set_title(r'Fig. 5:  $p_{\mathrm{fake}}$ Calibration — Fine-Tuned Models')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    _save(fig, '5_pfake_calibration.png')


# =============================================================================
# FIGURE 6: AUC-ROC Comparison
# =============================================================================
def fig_auc():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(AUC_METHODS))
    bars = ax.bar(x, AUC_VALUES, color=AUC_CLR, width=0.6,
                  edgecolor='#444', linewidth=0.5)
    _label_bars(ax, bars, fmt='{:.3f}', ypad=0.008, fontsize=9)

    ax.axhline(0.5, color='#CC0000', linewidth=0.8, linestyle='--',
               label='Random baseline (AUC = 0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(AUC_METHODS)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Fig. 6:  Area Under the ROC Curve')
    ax.set_ylim(0.35, 0.95)
    ax.legend(loc='lower right', framealpha=0.9)
    _save(fig, '6_auc_roc.png')


# =============================================================================
# FIGURE 7: Summary — Accuracy Progression Line Plot
# =============================================================================
def fig_progression():
    fig, ax = plt.subplots(figsize=(9, 5))

    stages = ['Zero-Shot', 'Few-Shot\n(k=8)', 'LoRA\n(FF++)', 'CNN+LoRA\n(FF++)', 'LoRA\n(Combined)']
    accs   = [31.92, 35.0, 70.38, 76.42, 82.55]
    colors = [CLR_ZS, CLR_FS, CLR_LORA, CLR_CNN, CLR_COMB]

    ax.plot(range(len(stages)), accs, 'o-', color='#444444',
            linewidth=2, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#333333', zorder=3)

    # Scatter coloured markers on top
    for i, (s, a, c) in enumerate(zip(stages, accs, colors)):
        ax.scatter(i, a, color=c, s=100, zorder=4, edgecolors='#333', linewidths=0.8)
        offset = 12 if a < 75 else -16
        ax.annotate(f'{a:.1f}%', (i, a),
                    textcoords='offset points', xytext=(0, offset),
                    ha='center', fontweight='bold', fontsize=11, color='#333')

    ax.fill_between(range(len(stages)), accs, alpha=0.06, color='#5B9BD5')
    ax.axhline(50, color='#CC0000', linewidth=0.7, linestyle=':', alpha=0.6)
    ax.text(0.01, 51, 'Random baseline', fontsize=8, color='#CC0000', alpha=0.7)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig. 7:  Accuracy Progression — From Zero-Shot to CNN-Augmented Fine-Tuning')
    ax.set_ylim(20, 95)
    _save(fig, '7_accuracy_progression.png')


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == '__main__':
    print('\n' + '=' * 52)
    print('  Generating FAKESHEILD Publication Charts')
    print('=' * 52 + '\n')

    fig_accuracy_all()
    fig_finetune_comparison()
    fig_per_class()
    fig_separation()
    fig_calibration()
    fig_auc()
    fig_progression()

    print(f'\n  All charts ({len(list(output_dir.glob("*.png")))}) saved to: {output_dir}/')
    print('=' * 52)
