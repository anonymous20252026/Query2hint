# ============================================================
# Adaptsteer — Learning Curve Figure
# ============================================================
#
# PURPOSE:
#   Generate publication-quality learning curve figure
#   showing few-shot adaptation results from paper table.
#
# FIGURE:
#   X axis → K (number of JOB queries used for adaptation)
#   Y axis → AUROC / Ranking Accuracy
#   Lines  → Contrastive vs Reptile vs Oracle
#   Shaded → ±1 std deviation across 5 seeds
#
# OUTPUT:
#   results/figure_learning_curve.pdf  ← for paper
#   results/figure_learning_curve.png  ← for slides
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed — saves directly to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Input data ───────────────────────────────────────────
    RAW_RESULTS_PATH = "pairwise_results_raw.csv"   # per-seed raw results

    # ── Output ───────────────────────────────────────────────
    RESULTS_DIR      = "results"
    FIG_PDF          = "results/figure_learning_curve.pdf"
    FIG_PNG          = "results/figure_learning_curve.png"

    # ── Oracle scores (fixed, from evaluation output) ────────
    ORACLE_AUROC     = 0.9939
    ORACLE_RANK_ACC  = 0.9958

    # ── Plot settings ────────────────────────────────────────
    FIGURE_WIDTH     = 12      # inches
    FIGURE_HEIGHT    = 5       # inches
    DPI              = 300     # publication quality
    FONT_SIZE        = 13
    TITLE_SIZE       = 15
    LEGEND_SIZE      = 11

    # ── Colors ───────────────────────────────────────────────
    COLOR_CONTRASTIVE = "#2196F3"   # blue
    COLOR_REPTILE     = "#F44336"   # red
    COLOR_ORACLE      = "#4CAF50"   # green

    # ── K values ────────────────────────────────────────────
    K_VALUES         = [0, 5, 10, 20, 50]
    K_LABELS         = ["0\n(zero-shot)", "5", "10", "20", "50"]

    # ── Seeds used ──────────────────────────────────────────
    SEEDS            = [42, 123, 456, 789, 999]


cfg = Config()
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Adaptsteer — Learning Curve Figure Generator")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD OR BUILD DATA
# ============================================================

print("\nLoading results data...")

# try to load raw CSV first
if os.path.exists(cfg.RAW_RESULTS_PATH):
    df = pd.read_csv(cfg.RAW_RESULTS_PATH)
    print(f"Loaded: {cfg.RAW_RESULTS_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

else:
    # fallback — build from hardcoded results
    print(f"⚠️  {cfg.RAW_RESULTS_PATH} not found")
    print("Building from hardcoded results...")

    rows = []

    # Seed 42
    rows += [
        {"seed": 42, "k": 0,  "auroc_c": 0.7830, "auroc_r": 0.7239, "rank_c": 0.7831, "rank_r": 0.7386},
        {"seed": 42, "k": 5,  "auroc_c": 0.7954, "auroc_r": 0.8314, "rank_c": 0.7837, "rank_r": 0.8355},
        {"seed": 42, "k": 10, "auroc_c": 0.8054, "auroc_r": 0.8396, "rank_c": 0.7965, "rank_r": 0.8314},
        {"seed": 42, "k": 20, "auroc_c": 0.8462, "auroc_r": 0.8524, "rank_c": 0.8295, "rank_r": 0.8510},
        {"seed": 42, "k": 50, "auroc_c": 0.8732, "auroc_r": 0.8778, "rank_c": 0.8782, "rank_r": 0.8872},
    ]
    # Seed 123
    rows += [
        {"seed": 123, "k": 0,  "auroc_c": 0.7830, "auroc_r": 0.7239, "rank_c": 0.7831, "rank_r": 0.7386},
        {"seed": 123, "k": 5,  "auroc_c": 0.8354, "auroc_r": 0.8288, "rank_c": 0.8346, "rank_r": 0.8371},
        {"seed": 123, "k": 10, "auroc_c": 0.8305, "auroc_r": 0.8438, "rank_c": 0.8251, "rank_r": 0.8561},
        {"seed": 123, "k": 20, "auroc_c": 0.7960, "auroc_r": 0.8822, "rank_c": 0.7985, "rank_r": 0.8872},
        {"seed": 123, "k": 50, "auroc_c": 0.8448, "auroc_r": 0.8501, "rank_c": 0.8337, "rank_r": 0.8467},
    ]
    # Seed 456
    rows += [
        {"seed": 456, "k": 0,  "auroc_c": 0.7830, "auroc_r": 0.7239, "rank_c": 0.7831, "rank_r": 0.7386},
        {"seed": 456, "k": 5,  "auroc_c": 0.7084, "auroc_r": 0.7704, "rank_c": 0.7084, "rank_r": 0.7735},
        {"seed": 456, "k": 10, "auroc_c": 0.8231, "auroc_r": 0.8368, "rank_c": 0.8034, "rank_r": 0.8295},
        {"seed": 456, "k": 20, "auroc_c": 0.8802, "auroc_r": 0.8949, "rank_c": 0.8809, "rank_r": 0.8918},
        {"seed": 456, "k": 50, "auroc_c": 0.8567, "auroc_r": 0.8733, "rank_c": 0.8523, "rank_r": 0.8740},
    ]
    # Seed 789
    rows += [
        {"seed": 789, "k": 0,  "auroc_c": 0.7830, "auroc_r": 0.7239, "rank_c": 0.7831, "rank_r": 0.7386},
        {"seed": 789, "k": 5,  "auroc_c": 0.6987, "auroc_r": 0.7410, "rank_c": 0.7091, "rank_r": 0.7558},
        {"seed": 789, "k": 10, "auroc_c": 0.7391, "auroc_r": 0.7315, "rank_c": 0.7639, "rank_r": 0.7619},
        {"seed": 789, "k": 20, "auroc_c": 0.7638, "auroc_r": 0.8344, "rank_c": 0.7575, "rank_r": 0.8237},
        {"seed": 789, "k": 50, "auroc_c": 0.8458, "auroc_r": 0.8224, "rank_c": 0.8299, "rank_r": 0.8189},
    ]
    # Seed 999
    rows += [
        {"seed": 999, "k": 0,  "auroc_c": 0.7830, "auroc_r": 0.7239, "rank_c": 0.7831, "rank_r": 0.7386},
        {"seed": 999, "k": 5,  "auroc_c": 0.7873, "auroc_r": 0.8106, "rank_c": 0.7864, "rank_r": 0.8147},
        {"seed": 999, "k": 10, "auroc_c": 0.8038, "auroc_r": 0.8396, "rank_c": 0.7875, "rank_r": 0.8208},
        {"seed": 999, "k": 20, "auroc_c": 0.7997, "auroc_r": 0.8340, "rank_c": 0.8036, "rank_r": 0.8292},
        {"seed": 999, "k": 50, "auroc_c": 0.8581, "auroc_r": 0.8877, "rank_c": 0.8543, "rank_r": 0.8826},
    ]

    df = pd.DataFrame(rows)
    print("Hardcoded data loaded successfully")


# ============================================================
# SECTION 3: COMPUTE STATISTICS
# ============================================================

print("\nComputing statistics...")

# detect column names from CSV or use hardcoded names
if "auroc_contrastive" in df.columns:
    # from pairwise_results_raw.csv
    auroc_c_col  = "auroc_contrastive"
    auroc_r_col  = "auroc_reptile"
    rank_c_col   = "rank_contrastive"
    rank_r_col   = "rank_reptile"
    k_col        = "n_shots"
else:
    # from hardcoded fallback
    auroc_c_col  = "auroc_c"
    auroc_r_col  = "auroc_r"
    rank_c_col   = "rank_c"
    rank_r_col   = "rank_r"
    k_col        = "k"

# compute mean and std per K value
stats = []
for k in cfg.K_VALUES:
    subset = df[df[k_col] == k]

    stats.append({
        "k"          : k,
        "auroc_c_mean": subset[auroc_c_col].mean(),
        "auroc_c_std" : subset[auroc_c_col].std(),
        "auroc_r_mean": subset[auroc_r_col].mean(),
        "auroc_r_std" : subset[auroc_r_col].std(),
        "rank_c_mean" : subset[rank_c_col].mean(),
        "rank_c_std"  : subset[rank_c_col].std(),
        "rank_r_mean" : subset[rank_r_col].mean(),
        "rank_r_std"  : subset[rank_r_col].std(),
    })

stats_df = pd.DataFrame(stats)

print("\nComputed statistics:")
print(f"{'K':>5} | {'AUROC Contrastive':^20} | {'AUROC Reptile':^20} | {'Gain':>8}")
print("-" * 65)
for _, row in stats_df.iterrows():
    gain = row["auroc_r_mean"] - row["auroc_c_mean"]
    flag = "✅" if gain > 0 else "❌"
    print(f"{int(row['k']):>5} | "
          f"{row['auroc_c_mean']:.4f} ± {row['auroc_c_std']:.4f}    | "
          f"{row['auroc_r_mean']:.4f} ± {row['auroc_r_std']:.4f}    | "
          f"{gain:>+.4f} {flag}")


# ============================================================
# SECTION 4: PLOT
# ============================================================

print("\nGenerating figure...")

# global font settings
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : cfg.FONT_SIZE,
    "axes.titlesize" : cfg.TITLE_SIZE,
    "axes.labelsize" : cfg.FONT_SIZE,
    "xtick.labelsize": cfg.FONT_SIZE - 1,
    "ytick.labelsize": cfg.FONT_SIZE - 1,
    "legend.fontsize": cfg.LEGEND_SIZE,
    "figure.dpi"     : cfg.DPI,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(cfg.FIGURE_WIDTH, cfg.FIGURE_HEIGHT),
    sharey=False
)

k_vals    = stats_df["k"].values
k_indices = np.arange(len(k_vals))   # x positions for bars/lines


# ── Helper: draw one subplot ─────────────────────────────────

def draw_subplot(ax, c_mean, c_std, r_mean, r_std,
                 oracle_val, metric_name, subplot_label):
    """
    Draw learning curve with:
      - Lines for Contrastive and Reptile
      - Shaded ±1 std band
      - Dashed oracle line
      - Crossover annotation
      - Zero-shot marker
    """

    # ── Lines + shaded std bands ─────────────────────────────
    ax.plot(k_indices, c_mean,
            color     = cfg.COLOR_CONTRASTIVE,
            marker    = "o",
            linewidth = 2.5,
            markersize= 8,
            label     = "Contrastive (CEB only)",
            zorder    = 3)

    ax.fill_between(k_indices,
                    c_mean - c_std,
                    c_mean + c_std,
                    alpha = 0.15,
                    color = cfg.COLOR_CONTRASTIVE,
                    zorder= 2)

    ax.plot(k_indices, r_mean,
            color     = cfg.COLOR_REPTILE,
            marker    = "s",
            linewidth = 2.5,
            markersize= 8,
            label     = "Reptile (CEB only)",
            zorder    = 3)

    ax.fill_between(k_indices,
                    r_mean - r_std,
                    r_mean + r_std,
                    alpha = 0.15,
                    color = cfg.COLOR_REPTILE,
                    zorder= 2)

    # ── Oracle dashed line ───────────────────────────────────
    ax.axhline(y         = oracle_val,
               color     = cfg.COLOR_ORACLE,
               linestyle = "--",
               linewidth = 2.0,
               alpha     = 0.8,
               label     = f"Oracle (CEB+JOB) = {oracle_val:.4f}",
               zorder    = 1)

    # ── Zero-shot divider ────────────────────────────────────
    ax.axvline(x         = 0.5,
               color     = "gray",
               linestyle = ":",
               linewidth = 1.5,
               alpha     = 0.6,
               zorder    = 1)

    ax.text(0.02, 0.97, "zero-shot",
            transform   = ax.transAxes,
            fontsize    = 9,
            color       = "gray",
            verticalalignment = "top")

    ax.text(0.18, 0.97, "few-shot →",
            transform   = ax.transAxes,
            fontsize    = 9,
            color       = "gray",
            verticalalignment = "top")

    # ── Crossover annotation ─────────────────────────────────
    # find first K where Reptile beats Contrastive
    crossover_idx = None
    for idx in range(1, len(k_vals)):
        if r_mean[idx] > c_mean[idx]:
            crossover_idx = idx
            break

    if crossover_idx is not None:
        ax.annotate(
            f"Reptile wins\nfrom K={k_vals[crossover_idx]}",
            xy         = (crossover_idx, r_mean[crossover_idx]),
            xytext     = (crossover_idx + 0.3,
                          r_mean[crossover_idx] - 0.025),
            fontsize   = 9,
            color      = cfg.COLOR_REPTILE,
            arrowprops = dict(
                arrowstyle = "->",
                color      = cfg.COLOR_REPTILE,
                lw         = 1.5
            ),
            zorder     = 5
        )

    # ── Gain labels at each K ────────────────────────────────
    for idx in range(1, len(k_vals)):
        gain = r_mean[idx] - c_mean[idx]
        if gain > 0:
            y_pos  = max(r_mean[idx], c_mean[idx]) + 0.008
            color  = cfg.COLOR_REPTILE
            symbol = f"+{gain:.3f}"
        else:
            y_pos  = max(r_mean[idx], c_mean[idx]) + 0.008
            color  = cfg.COLOR_CONTRASTIVE
            symbol = f"{gain:.3f}"

        ax.text(idx, y_pos, symbol,
                ha        = "center",
                fontsize  = 8,
                color     = color,
                fontweight= "bold",
                zorder    = 5)

    # ── Axes formatting ──────────────────────────────────────
    ax.set_xticks(k_indices)
    ax.set_xticklabels(cfg.K_LABELS)
    ax.set_xlabel("Number of JOB Queries for Adaptation (K)",
                  labelpad=8)
    ax.set_ylabel(metric_name, labelpad=8)

    # y axis range — give room for annotations
    all_vals  = np.concatenate([c_mean, r_mean])
    y_min     = max(0.60, all_vals.min() - 0.04)
    y_max     = min(1.02, oracle_val   + 0.04)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.3, len(k_vals) - 0.7)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2f")
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.grid(axis="x", linestyle=":",  alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # subplot label (a) / (b)
    ax.text(-0.08, 1.05, subplot_label,
            transform   = ax.transAxes,
            fontsize    = cfg.TITLE_SIZE,
            fontweight  = "bold")

    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")


# ── Draw subplot (a) — AUROC ─────────────────────────────────
draw_subplot(
    ax            = ax1,
    c_mean        = stats_df["auroc_c_mean"].values,
    c_std         = stats_df["auroc_c_std"].values,
    r_mean        = stats_df["auroc_r_mean"].values,
    r_std         = stats_df["auroc_r_std"].values,
    oracle_val    = cfg.ORACLE_AUROC,
    metric_name   = "AUROC",
    subplot_label = "(a)"
)
ax1.set_title("AUROC — Few-Shot Adaptation (CEB → JOB)")


# ── Draw subplot (b) — Ranking Accuracy ──────────────────────
draw_subplot(
    ax            = ax2,
    c_mean        = stats_df["rank_c_mean"].values,
    c_std         = stats_df["rank_c_std"].values,
    r_mean        = stats_df["rank_r_mean"].values,
    r_std         = stats_df["rank_r_std"].values,
    oracle_val    = cfg.ORACLE_RANK_ACC,
    metric_name   = "Ranking Accuracy",
    subplot_label = "(b)"
)
ax2.set_title("Ranking Accuracy — Few-Shot Adaptation (CEB → JOB)")


# ── Main title ───────────────────────────────────────────────
fig.suptitle(
    "Adaptsteer: Reptile Meta-Learning Enables Faster Adaptation to New Workloads",
    fontsize   = cfg.TITLE_SIZE + 1,
    fontweight = "bold",
    y          = 1.02
)

plt.tight_layout(pad=2.0)


# ============================================================
# SECTION 5: SAVE FIGURE
# ============================================================

plt.savefig(cfg.FIG_PDF,
            format      = "pdf",
            dpi         = cfg.DPI,
            bbox_inches = "tight")

plt.savefig(cfg.FIG_PNG,
            format      = "png",
            dpi         = cfg.DPI,
            bbox_inches = "tight")

print(f"\n✅ Figures saved:")
print(f"   {cfg.FIG_PDF}  ← use in paper (LaTeX)")
print(f"   {cfg.FIG_PNG}  ← use in slides/preview")


# ============================================================
# SECTION 6: PRINT PAPER TABLE
# ============================================================

print("\n" + "=" * 60)
print("PAPER TABLE — for copy-paste into LaTeX")
print("=" * 60)
print()
print("AUROC:")
print(f"{'K':<6} | {'Contrastive':^22} | {'Reptile':^22} | {'Gain':>8} | Win%")
print("-" * 75)

for _, row in stats_df.iterrows():
    gain     = row["auroc_r_mean"] - row["auroc_c_mean"]
    # compute win rate across seeds
    subset   = df[df[k_col] == row["k"]]
    wins     = (subset[auroc_r_col] > subset[auroc_c_col]).sum()
    win_rate = wins / len(subset) * 100
    flag     = "✅" if gain > 0 else "❌"

    print(f"{int(row['k']):<6} | "
          f"{row['auroc_c_mean']:.4f} ± {row['auroc_c_std']:.4f}   | "
          f"{row['auroc_r_mean']:.4f} ± {row['auroc_r_std']:.4f}   | "
          f"{gain:>+.4f}  | "
          f"{win_rate:.0f}% {flag}")

print()
print(f"Oracle AUROC = {cfg.ORACLE_AUROC:.4f}  ← upper bound (CEB+JOB full training)")

print()
print("Ranking Accuracy:")
print(f"{'K':<6} | {'Contrastive':^22} | {'Reptile':^22} | {'Gain':>8} | Win%")
print("-" * 75)

for _, row in stats_df.iterrows():
    gain     = row["rank_r_mean"] - row["rank_c_mean"]
    subset   = df[df[k_col] == row["k"]]
    wins     = (subset[rank_r_col] > subset[rank_c_col]).sum()
    win_rate = wins / len(subset) * 100
    flag     = "✅" if gain > 0 else "❌"

    print(f"{int(row['k']):<6} | "
          f"{row['rank_c_mean']:.4f} ± {row['rank_c_std']:.4f}   | "
          f"{row['rank_r_mean']:.4f} ± {row['rank_r_std']:.4f}   | "
          f"{gain:>+.4f}  | "
          f"{win_rate:.0f}% {flag}")

print()
print(f"Oracle Rank Acc = {cfg.ORACLE_RANK_ACC:.4f}  ← upper bound")


# ============================================================
# SECTION 7: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE")
print("=" * 60)
print()
print("Files produced:")
print(f"  {cfg.FIG_PDF}")
print(f"  {cfg.FIG_PNG}")
print()
print("Key findings for paper:")

# find crossover K for AUROC
for i, row in stats_df.iterrows():
    if row["auroc_r_mean"] > row["auroc_c_mean"] and row["k"] > 0:
        print(f"  ✅ Reptile outperforms Contrastive from K={int(row['k'])} onwards (AUROC)")
        break

# find max gain
max_gain_row = stats_df.loc[
    (stats_df["auroc_r_mean"] - stats_df["auroc_c_mean"]).idxmax()
]
max_gain = max_gain_row["auroc_r_mean"] - max_gain_row["auroc_c_mean"]
print(f"  ✅ Largest gain at K={int(max_gain_row['k'])}: +{max_gain:.4f} AUROC")
print(f"  ✅ Oracle gap: {cfg.ORACLE_AUROC - stats_df['auroc_r_mean'].max():.4f} AUROC")
print(f"  ✅ Averaged over {len(cfg.SEEDS)} seeds: {cfg.SEEDS}")
print("=" * 60)