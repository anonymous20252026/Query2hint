"""
Adaptsteer — Learning Curve Figure Generator
Generates the key meta-learning figure for the paper.
Run this script to produce publication-ready plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# DATA — from your pairwise_results_summary.csv
# ============================================================

K           = [0,      5,      10,     20,     50    ]

# AUROC
C_auroc_mean = [0.7830, 0.7650, 0.8004, 0.8172, 0.8557]
C_auroc_std  = [0.0000, 0.0591, 0.0361, 0.0459, 0.0115]
R_auroc_mean = [0.7239, 0.7964, 0.8183, 0.8596, 0.8623]
R_auroc_std  = [0.0000, 0.0394, 0.0486, 0.0278, 0.0262]

# Ranking Accuracy
C_rank_mean  = [0.7831, 0.7644, 0.7953, 0.8140, 0.8497]
C_rank_std   = [0.0000, 0.0547, 0.0224, 0.0454, 0.0193]
R_rank_mean  = [0.7386, 0.8033, 0.8199, 0.8566, 0.8619]
R_rank_std   = [0.0000, 0.0369, 0.0350, 0.0318, 0.0287]

# Win rates
win_rates    = [0.0,   0.8,   0.8,   1.0,   0.8  ]

# References
Optimal_AUROC = 0.9939

# ============================================================
# FIGURE 1 — AUROC Learning Curve (Main Paper Figure)
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

K_arr = np.array(K)
C_arr = np.array(C_auroc_mean)
R_arr = np.array(R_auroc_mean)
C_std = np.array(C_auroc_std)
R_std = np.array(R_auroc_std)

# ── Lines ────────────────────────────────────────────────────
ax.plot(K_arr, C_arr, 'o-',
        color='#4472C4', linewidth=2.0, markersize=7,
        label='AdaptSteer-C (Contrastive only)')

ax.plot(K_arr, R_arr, 's-',
        color='#ED7D31', linewidth=2.0, markersize=7,
        label='AdaptSteer-R (Reptile meta-learned)')

# ── Shaded std bands ─────────────────────────────────────────
ax.fill_between(K_arr, C_arr - C_std, C_arr + C_std,
                alpha=0.15, color='#4472C4')
ax.fill_between(K_arr, R_arr - R_std, R_arr + R_std,
                alpha=0.15, color='#ED7D31')

# ── Optimal reference line ────────────────────────────────────
ax.axhline(y=Optimal_AUROC, color='#70AD47',
           linestyle='--', linewidth=1.5,
           label=f'Optimal')

# ── Annotate crossover point ─────────────────────────────────
ax.annotate('Reptile overtakes\nContrastive at K=5',
            xy=(5, 0.796), xytext=(12, 0.76),
            fontsize=8, color='#666666',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

# ── Annotate K=20 gain ───────────────────────────────────────
ax.annotate(f'+0.042\nAUROC',
            xy=(20, 0.860), xytext=(23, 0.830),
            fontsize=8, color='#ED7D31', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.2))

# ── Formatting ───────────────────────────────────────────────
ax.set_xlabel('Number of Adaptation Queries (K)', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Few-Shot Workload Adaptation: CEB → JOB\n'
             'Adaptsteer-R vs Adaptsteer-C (5 seeds, shaded = ±1 std)',
             fontsize=11)
ax.set_xticks(K)
ax.set_xlim(-2, 55)
ax.set_ylim(0.68, 1.02)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure_learning_curve_auroc.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_learning_curve_auroc.png', dpi=300, bbox_inches='tight')
print("Saved: figure_learning_curve_auroc.pdf + .png")
plt.show()


# ============================================================
# FIGURE 2 — AUROC + Ranking Accuracy (Side by Side)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (y_C, y_R, std_C, std_R, ylabel, title) in zip(axes, [
    (C_auroc_mean, R_auroc_mean, C_auroc_std, R_auroc_std,
     'AUROC', 'AUROC — CEB → JOB Adaptation'),
    (C_rank_mean, R_rank_mean, C_rank_std, R_rank_std,
     'Ranking Accuracy', 'Ranking Accuracy — CEB → JOB Adaptation'),
]):
    y_C = np.array(y_C)
    y_R = np.array(y_R)
    s_C = np.array(std_C)
    s_R = np.array(std_R)

    ax.plot(K_arr, y_C, 'o-', color='#4472C4', linewidth=2.0,
            markersize=7, label='Adaptsteer-C (Contrastive)')
    ax.plot(K_arr, y_R, 's-', color='#ED7D31', linewidth=2.0,
            markersize=7, label='Adaptsteer-R (Reptile)')

    ax.fill_between(K_arr, y_C - s_C, y_C + s_C,
                    alpha=0.15, color='#4472C4')
    ax.fill_between(K_arr, y_R - s_R, y_R + s_R,
                    alpha=0.15, color='#ED7D31')

    ax.axhline(y=Optimal_AUROC, color='#70AD47',
               linestyle='--', linewidth=1.5,
               label=f'Optimal ({Optimal_AUROC:.3f})')

    ax.set_xlabel('Number of Adaptation Queries (K)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(K)
    ax.set_xlim(-2, 55)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure_learning_curve_both.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_learning_curve_both.png', dpi=300, bbox_inches='tight')
print("Saved: figure_learning_curve_both.pdf + .png")
plt.show()


# ============================================================
# FIGURE 3 — Win Rate Bar Chart
# ============================================================

fig, ax = plt.subplots(figsize=(6, 3.5))

K_labels = [f'K={k}' for k in K]
colors   = ['#FF4444' if w == 0 else
            '#FFA500' if w < 1.0 else
            '#00AA44' for w in win_rates]

bars = ax.bar(K_labels, [w * 100 for w in win_rates],
              color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

# annotate bars
for bar, w in zip(bars, win_rates):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 1,
            f'{int(w*100)}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Adaptation Queries (K)', fontsize=11)
ax.set_ylabel('Reptile Win Rate (%)', fontsize=11)
ax.set_title('Adaptsteer-R vs Adaptsteer-C\nReptile Win Rate by K (5 seeds)',
             fontsize=11)
ax.set_ylim(0, 115)
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(4.55, 52, '50% baseline', fontsize=8, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure_win_rate.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_win_rate.png', dpi=300, bbox_inches='tight')
print("Saved: figure_win_rate.pdf + .png")
plt.show()


# ============================================================
# FIGURE 4 — Stability (Std Dev) Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(6, 3.5))

x     = np.arange(len(K))
width = 0.35

b1 = ax.bar(x - width/2, C_auroc_std, width,
            label='Adaptsteer-C', color='#4472C4', alpha=0.85)
b2 = ax.bar(x + width/2, R_auroc_std, width,
            label='Adaptsteer-R', color='#ED7D31', alpha=0.85)

ax.set_xlabel('Adaptation Queries (K)', fontsize=11)
ax.set_ylabel('AUROC Standard Deviation', fontsize=11)
ax.set_title('Adaptation Stability\n(Lower std = more reliable)',
             fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K])
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# highlight K=20 where Reptile is 39% more stable
ax.annotate('39% lower\nvariance', xy=(3 + width/2, 0.0278),
            xytext=(3.5, 0.042),
            fontsize=8, color='#ED7D31', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.2))

plt.tight_layout()
plt.savefig('figure_stability.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_stability.png', dpi=300, bbox_inches='tight')
print("Saved: figure_stability.pdf + .png")
plt.show()


# ============================================================
# PRINT LATEX TABLE FOR PAPER
# ============================================================

print("\n" + "=" * 70)
print("LATEX TABLE — Copy directly into your paper")
print("=" * 70)

print(r"""
\begin{table}[h]
\centering
\small
\caption{Few-shot workload adaptation (CEB $\rightarrow$ JOB).
AUROC averaged over 5 random seeds. Higher is better.}
\label{tab:fewshot}
\begin{tabular}{ccccc}
\toprule
$K$ & Adaptsteer-C & Adaptsteer-R & $\Delta$ AUROC & Win Rate \\
\midrule""")

rows = [
    (0,  0.7830, 0.0000, 0.7239, 0.0000, -0.059,  "0\\%"),
    (5,  0.7650, 0.0591, 0.7964, 0.0394, +0.031,  "80\\%"),
    (10, 0.8004, 0.0361, 0.8183, 0.0486, +0.018,  "80\\%"),
    (20, 0.8172, 0.0459, 0.8596, 0.0278, +0.042, "100\\%"),
    (50, 0.8557, 0.0115, 0.8623, 0.0262, +0.007,  "80\\%"),
]

for k, cm, cs, rm, rs, gain, wr in rows:
    sign  = "+" if gain > 0 else ""
    bold  = r"\textbf{" if gain > 0 else ""
    ebold = r"}" if gain > 0 else ""
    print(f"{k} & {cm:.4f}$\\pm${cs:.4f} & "
          f"{bold}{rm:.4f}$\\pm${rs:.4f}{ebold} & "
          f"{sign}{gain:.3f} & {wr} \\\\")

print(r"""\midrule
Optimal & \multicolumn{4}{c}{0.9939 (upper bound — trained on CEB+JOB)} \\
\bottomrule
\end{tabular}
\end{table}
""")

print("=" * 70)
print("All figures saved as PDF and PNG ✅")
print("LaTeX table printed above ✅")
