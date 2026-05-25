"""
generate_paper_figures.py
=========================
Generates all programmatic figures required by mainfile.tex.

Output layout (all files go to figures/):
  figures/fig_main_comparison.pdf/png  — Table 3 visual (§4.4)
  figures/learning_curve.pdf/png       — few-shot 3-method curve (§4.5)
  figures/fig_stability.pdf/png        — adaptation std-dev bar (§4.5)

Run from the project root:
    python generate_paper_figures.py
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("figures", exist_ok=True)

DPI = 300
COLOR_PG   = "#022B4D"   # deep red  — PostgreSQL Default
COLOR_FIXED= "#7030A0"   # purple    — Fixed Alternative
COLOR_LLM  = "#ED7D31"   # orange    — LLMSteer
COLOR_ADA_C= "#4472C4"   # blue      — AdaptSteer-C
COLOR_ADA_R= "#70AD47"   # green     — AdaptSteer-R (best)
COLOR_OPT  = "#808080"   # grey      — Optimal upper bound


# ── Shared formatter ──────────────────────────────────────────────────────────
def _fmt_k(v, _):
    return f"{v:,.0f}"


# =============================================================================
# FIGURE A — Main end-to-end comparison (Option A: no AdaptSteer-F bar)
# =============================================================================

def make_main_comparison():
    """
    Two-panel bar chart: Total Workload (s) and P90 Tail Latency (s).
    Data from Table 3 of mainfile.tex (10-fold CV mean ± std).
    Fixed Alternative included on the workload axis (3,032 s per fold);
    excluded from P90 panel since the P90 comparison focuses on adaptive methods.
    """

    # --- Data -----------------------------------------------------------------
    methods_wl = [
        "PostgreSQL\nDefault",
        "Fixed\nAlternative",
        "LLMSteer",
        "AdaptSteer-C\n(ours)",
        "AdaptSteer-R\n(ours)",
        "Optimal\n(upper bound)",
    ]
    # All values are mean total workload over 20% test fold, averaged over 10 CV splits
    wl_mean = [8135, 3032, 2548, 2546, 2435, 1778]
    wl_std  = [1454.30, 311.38,  0,       267.70,  320.40,  63.40  ]
    wl_pct  = [None, "-62.7%", None, "-68.4%", "-70.1%", None]

    methods_p90 = [
        "PostgreSQL\nDefault",
        "LLMSteer",
        "AdaptSteer-C\n(ours)",
        "AdaptSteer-R\n(ours)",
        "Optimal\n(upper bound)",
    ]
    p90_mean = [19.60, 5.70, 5.62, 5.43, 4.8]
    p90_std  = [0,     0,    0.37, 0.35, 0   ]

    bar_colors_wl = [COLOR_PG, COLOR_FIXED, COLOR_LLM,
                     COLOR_ADA_C, COLOR_ADA_R, COLOR_OPT]
    bar_colors_p90 = [COLOR_PG, COLOR_LLM, COLOR_ADA_C, COLOR_ADA_R, COLOR_OPT]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Workload panel -------------------------------------------------------
    x1 = np.arange(len(methods_wl))
    bars1 = ax1.bar(x1, wl_mean, color=bar_colors_wl, alpha=0.88,
                    edgecolor="white", linewidth=0.6, zorder=3)

    # Error bars for variants with std
    for i, (m, s) in enumerate(zip(wl_mean, wl_std)):
        if s > 0:
            ax1.errorbar(x1[i], m, yerr=s, fmt="none",
                         ecolor="black", elinewidth=1.2, capsize=4, zorder=4)

    # Value labels on top of bars
    for i, (bar, m, pct) in enumerate(zip(bars1, wl_mean, wl_pct)):
        label = f"{m:,.0f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 200,
                 label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        if pct:
            ax1.text(bar.get_x() + bar.get_width() / 2, m / 2,
                     pct, ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(methods_wl, fontsize=8.5)
    ax1.set_ylabel("Total Workload (s)", fontsize=11)
    ax1.set_title("(a) Total Workload Latency", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # footnote about Fixed Alternative
    # ax1.annotate("Fixed Alternative (3,032 s) always applies\nhint_26 (best fixed hint).",
    #              xy=(1, 3031.53), xytext=(2.2, 5000),
    #              fontsize=6.5, color="#7030A0",
    #              arrowprops=dict(arrowstyle="->", color="#7030A0", lw=0.8))

    # --- P90 panel ------------------------------------------------------------
    x2 = np.arange(len(methods_p90))
    bars2 = ax2.bar(x2, p90_mean, color=bar_colors_p90, alpha=0.88,
                    edgecolor="white", linewidth=0.6, zorder=3)

    for i, (m, s) in enumerate(zip(p90_mean, p90_std)):
        if s > 0:
            ax2.errorbar(x2[i], m, yerr=s, fmt="none",
                         ecolor="black", elinewidth=1.2, capsize=4, zorder=4)

    for bar, m in zip(bars2, p90_mean):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 0.2,
                 f"{m:.2f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_p90, fontsize=8.5)
    ax2.set_ylabel("P90 Latency (s)", fontsize=11)
    ax2.set_title("(b) P90 Tail Latency", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "End-to-End Optimizer Steering — JOB + CEB Workload (3,246 queries)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("figures/fig_main_comparison.pdf", bbox_inches="tight", dpi=DPI)
    plt.savefig("figures/fig_main_comparison.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("Saved: figures/fig_main_comparison.{pdf,png}")


# =============================================================================
# FIGURE B — Few-shot learning curve (3 methods + NoMeta-SVC)
# =============================================================================

def make_fewshot_curve():
    """
    AUROC vs K for AdaptSteer-C, NoMeta-SVC, AdaptSteer-R.
    Data from results_v3/fewshot_nometa_summary.csv.
    Saved as figures/learning_curve.{pdf,png} (the path referenced in mainfile.tex).
    """
    K = np.array([0, 5, 10, 20, 50])

    # AdaptSteer-C
    C_mean = np.array([0.7830, 0.7650, 0.8004, 0.8172, 0.8557])
    C_std  = np.array([0.0000, 0.0591, 0.0361, 0.0459, 0.0115])

    # NoMeta-SVC (classifier-only adaptation on fixed contrastive encoder)
    N_mean = np.array([0.7825, 0.8009, 0.8079, 0.8540, 0.8858])
    N_std  = np.array([0.0000, 0.0395, 0.0860, 0.0135, 0.0038])

    # AdaptSteer-R (Reptile meta-learning)
    R_mean = np.array([0.7239, 0.7964, 0.8183, 0.8596, 0.8623])
    R_std  = np.array([0.0000, 0.0394, 0.0487, 0.0278, 0.0262])

    FULL_DATA_REF = 0.9939   # supervised ceiling (CEB+JOB)

    COLOR_C = "#4472C4"   # blue  — AdaptSteer-C
    COLOR_N = "#9E480E"   # brown — NoMeta-SVC
    COLOR_R = "#70AD47"   # green — AdaptSteer-R

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # --- Lines ---------------------------------------------------------------
    ax.plot(K, C_mean, "o-", color=COLOR_C, lw=2.0, ms=7,
            label="AdaptSteer-C (contrastive only)")
    ax.fill_between(K, C_mean - C_std, C_mean + C_std, alpha=0.13, color=COLOR_C)

    ax.plot(K, N_mean, "D--", color=COLOR_N, lw=1.8, ms=7,
            label="NoMeta-SVC (classifier-only adaptation)")
    ax.fill_between(K, N_mean - N_std, N_mean + N_std, alpha=0.13, color=COLOR_N)

    ax.plot(K, R_mean, "s-", color=COLOR_R, lw=2.2, ms=7,
            label="AdaptSteer-R (Reptile meta-learning)")
    ax.fill_between(K, R_mean - R_std, R_mean + R_std, alpha=0.15, color=COLOR_R)

    # --- Reference line ------------------------------------------------------
    ax.axhline(FULL_DATA_REF, color="#C00000", ls="--", lw=1.4,
               label=f"Full-data ref. (AUROC = {FULL_DATA_REF:.4f})")

    # --- Key annotations -----------------------------------------------------
    # AdaptSteer-R peak advantage at K=20
    ax.annotate(
        "AdaptSteer-R peak\nadvantage at K=20",
        xy=(20, 0.8596), xytext=(28, 0.830),
        fontsize=8, color=COLOR_R,
        arrowprops=dict(arrowstyle="->", color=COLOR_R, lw=1.0)
    )
    # NoMeta-SVC surpasses at K=50
    ax.annotate(
        "NoMeta-SVC surpasses\nAdaptSteer-R at K=50",
        xy=(50, 0.8858), xytext=(36, 0.900),
        fontsize=8, color=COLOR_N,
        arrowprops=dict(arrowstyle="->", color=COLOR_N, lw=1.0)
    )

    # --- Formatting ----------------------------------------------------------
    ax.set_xlabel("Number of Adaptation Triplets (K)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title("Few-Shot Workload Adaptation: CEB → JOB\n"
                 "(mean ± 1 std over 5 seeds)", fontsize=11)
    ax.set_xticks(K)
    ax.set_xlim(-2, 58)
    ax.set_ylim(0.68, 1.02)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/learning_curve.pdf", bbox_inches="tight", dpi=DPI)
    plt.savefig("figures/learning_curve.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("Saved: figures/learning_curve.{pdf,png}")


# =============================================================================
# FIGURE C — Adaptation stability (AUROC std per K)
# =============================================================================

def make_stability():
    """
    Grouped bar chart showing AUROC standard deviation per K.
    Lower std = more reliable adaptation.
    Data matches generate_figures.py (5 seeds).
    Saved as figures/fig_stability.{pdf,png}.
    """
    K      = [0, 5, 10, 20, 50]
    C_std  = [0.000, 0.0591, 0.0361, 0.0459, 0.0115]
    R_std  = [0.000, 0.0394, 0.0487, 0.0278, 0.0262]

    COLOR_C = "#4472C4"
    COLOR_R = "#70AD47"

    x = np.arange(len(K))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))

    b1 = ax.bar(x - w/2, C_std, w, label="AdaptSteer-C",
                color=COLOR_C, alpha=0.87, edgecolor="white")
    b2 = ax.bar(x + w/2, R_std, w, label="AdaptSteer-R",
                color=COLOR_R, alpha=0.87, edgecolor="white")

    # Annotate K=20 where Reptile variance is 39 % lower
    ax.annotate(
        "39% lower\nvariance",
        xy=(3 + w/2, 0.0278), xytext=(3.6, 0.042),
        fontsize=8.5, color=COLOR_R, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_R, lw=1.1)
    )

    ax.set_xlabel("Number of Adaptation Triplets (K)", fontsize=11)
    ax.set_ylabel("AUROC Standard Deviation", fontsize=11)
    ax.set_title("Adaptation Stability\n(5 seeds; lower = more reliable)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K])
    ax.legend(fontsize=9.5)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/fig_stability.pdf", bbox_inches="tight", dpi=DPI)
    plt.savefig("figures/fig_stability.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("Saved: figures/fig_stability.{pdf,png}")


# =============================================================================
# FIGURE D — PCA embedding scatter (copy from results_v3 with corrected label)
# =============================================================================

def copy_oracle_figure():
    """
    The oracle PCA scatter requires running experiments_oracle_analysis.py
    (loads encoder models from disk). If the PNG exists, copy it to figures/
    with the corrected name. The label fix (AdaptSteer-O → AdaptSteer-F) is
    applied at the source in experiments_oracle_analysis.py (enc_name 'Full').
    """
    src = "results_v3/oracle_analysis_figure.png"
    dst = "figures/fig_embedding_pca.png"
    src_pdf = "results_v3/oracle_analysis_figure.pdf"
    dst_pdf = "figures/fig_embedding_pca.pdf"

    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {src} → {dst}")
        print("  NOTE: Label still shows 'AdaptSteer-O' in copied file.")
        print("  Re-run experiments_oracle_analysis.py to get 'AdaptSteer-F' label.")
    else:
        print(f"SKIP: {src} not found — run experiments_oracle_analysis.py first.")

    if os.path.exists(src_pdf):
        shutil.copy2(src_pdf, dst_pdf)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating paper figures → figures/")
    print("=" * 60)

    make_main_comparison()
    make_fewshot_curve()
    make_stability()
    copy_oracle_figure()

    print("\n" + "=" * 60)
    print("Done.")
    print()
    print("Files written to figures/:")
    for f in sorted(os.listdir("figures")):
        path = os.path.join("figures", f)
        size = os.path.getsize(path)
        print(f"  {f:45s}  {size:>8,} bytes")
    print()
    print("Figures NOT yet in figures/ (require separate steps):")
    print("  figure1.pdf   — system architecture diagram (draw manually)")
    print("  figure2-7.png — robustness panels (run generate_robustness_figures.py)")
    print("  figure10.pdf  — data construction diagram")
    print("  figure11.pdf  — Stage 1 contrastive learning diagram")
    print("  figure12.pdf  — Stage 2 meta-learning diagram")
    print("  fig_embedding_pca (AdaptSteer-F label) — re-run experiments_oracle_analysis.py")
