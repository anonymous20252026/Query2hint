"""
exp_code_backbone_ablation.py
==============================
Backbone selection experiment: code-specialized models vs. general-purpose models.

Evaluates 4 code-specialized encoders in TWO conditions:
  (A) Off-the-shelf  : raw embeddings, no fine-tuning
  (B) Fine-tuned     : same contrastive TripletLoss training used for AdaptSteer-C

Protocol is identical to exp1_supervision_ablation.py:
  - data/job.csv + data/ceb.csv  (3,246 queries)
  - StratifiedShuffleSplit(n_splits=10, train_size=0.8, random_state=24508)
  - BENCHMARK_IDX=0 (PostgreSQL default), LONGTAIL_IDX=26 (fixed alternative)
  - SVC-120-S pipeline: StandardScaler → PCA(120) → StandardScaler → SVC(RBF)
  - Workload = sum of test-fold latencies at chosen hint per query

Models:
  CodeBERT         microsoft/codebert-base
  GraphCodeBERT    microsoft/graphcodebert-base
  CodeT5-base      Salesforce/codet5-base  (encoder-only, handled by sentence-transformers 3.x)
  UniXCoder        microsoft/unixcoder-base

Triplets:
  stage1_triplets_CEB.csv + stage1_triplets_JOB.csv  (same multi-config preference pairs
  used in AdaptSteer-C training)

Outputs:
  encoders/codebert_stage1/          fine-tuned CodeBERT
  encoders/graphcodebert_stage1/     fine-tuned GraphCodeBERT
  encoders/codet5base_stage1/        fine-tuned CodeT5-base (encoder)
  encoders/unixcoder_stage1/         fine-tuned UniXCoder
  results/code_backbone_ablation.csv  per-model metrics
  figures/fig_code_backbone.pdf/png
"""

import os
import gc
import warnings
import ast
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
os.makedirs("results",  exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("figures",  exist_ok=True)

# ── Protocol constants (must match exp1_supervision_ablation.py exactly) ──────
RANDOM_SEED   = 24508
BENCHMARK_IDX = 0
LONGTAIL_IDX  = 26
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
THRESHOLD     = 0.5

FINETUNE_EPOCHS    = 1       # 1 epoch sufficient for ablation comparison
FINETUNE_BATCH     = 16      # larger batch → faster training
FINETUNE_LR        = 2e-5
FINETUNE_WARMUP    = 50
MAX_TRIPLETS       = 8_000   # fast ablation: ~500 steps/model ≈ 2 min each
MAX_SEQ_LENGTH     = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: No CUDA GPU found — training will be very slow on CPU.")

# ── Code-specialized backbones ────────────────────────────────────────────────
CODE_MODELS = {
    "CodeBERT":      "microsoft/codebert-base",
    "GraphCodeBERT": "microsoft/graphcodebert-base",
    "CodeT5-base":   "Salesforce/codet5-base",     # sentence-transformers 3.x uses T5EncoderModel
    "UniXCoder":     "microsoft/unixcoder-base",
}

SAVE_PATHS = {
    "CodeBERT":      "encoders/codebert_stage1",
    "GraphCodeBERT": "encoders/graphcodebert_stage1",
    "CodeT5-base":   "encoders/codet5base_stage1",
    "UniXCoder":     "encoders/unixcoder_stage1",
}

# ── Existing results for comparison ─────────────────────────────────────────
REFERENCE_ROWS = [
    {"encoder": "PostgreSQL Default", "condition": "—",
     "workload_mean": 8134.70, "workload_std": 0.0,
     "p90_mean": 19.60, "p90_std": 0.0,
     "auroc_mean": float("nan"), "f1_mean": float("nan")},
    {"encoder": "MPNet-Raw",          "condition": "off-the-shelf",
     "workload_mean": 2353.90, "workload_std": 285.97,
     "p90_mean": 5.60,  "p90_std": 0.26,
     "auroc_mean": 0.848, "f1_mean": 0.671},
    {"encoder": "AdaptSteer-C",       "condition": "fine-tuned (AdaptSteer)",
     "workload_mean": 2567.60, "workload_std": 267.70,
     "p90_mean": 6.06,  "p90_std": 0.37,
     "auroc_mean": 0.810, "f1_mean": 0.632},
    {"encoder": "AdaptSteer-R",       "condition": "fine-tuned+meta",
     "workload_mean": 2434.90, "workload_std": 320.40,
     "p90_mean": 5.83,  "p90_std": 0.35,
     "auroc_mean": 0.811, "f1_mean": 0.647},
    {"encoder": "Optimal Oracle",     "condition": "—",
     "workload_mean": 1064.10, "workload_std": 0.0,
     "p90_mean": 3.40, "p90_std": 0.0,
     "auroc_mean": float("nan"), "f1_mean": float("nan")},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (identical to exp1)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []


def load_data():
    job = pd.read_csv("data/job.csv",
                      converters={"hint_list": _safe_eval, "runtime_list": _safe_eval})
    ceb = pd.read_csv("data/ceb.csv",
                      converters={"hint_list": _safe_eval, "runtime_list": _safe_eval})
    data = pd.concat([job, ceb]).reset_index(drop=True)
    data["mean_runtime"] = data["runtime_list"].apply(np.mean)
    data["sql"] = data["sql"].apply(lambda x: x.strip("\n"))
    return data


def prepare_model_df(data):
    df = data.copy()
    df = df.explode("hint_list").sort_values(["filename", "hint_list"])
    df = df.groupby(["filename", "sql"], as_index=False).agg(
        hint_list   =("hint_list",    list),
        mean_runtime=("mean_runtime", list),
    )
    df["opt_l"] = df["mean_runtime"].apply(min)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SentenceTransformer builder — handles BERT-like + T5 backbones
# ═══════════════════════════════════════════════════════════════════════════════

def build_st_model(model_id: str) -> SentenceTransformer:
    """
    Build a SentenceTransformer with mean-pool head for any backbone.
    sentence-transformers 3.x auto-routes T5 configs to T5EncoderModel.
    """
    word = models.Transformer(model_id, max_seq_length=MAX_SEQ_LENGTH)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    norm = models.Normalize()
    return SentenceTransformer(modules=[word, pool, norm], device=DEVICE)


def load_or_build_st(save_path: str, model_id: str) -> SentenceTransformer:
    """Load fine-tuned model if it exists, else build from pretrained."""
    if os.path.isdir(save_path):
        print(f"    Loading existing fine-tuned model: {save_path}")
        return SentenceTransformer(save_path, device=DEVICE)
    return build_st_model(model_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helper
# ═══════════════════════════════════════════════════════════════════════════════

def encode_all(sqls, st_model: SentenceTransformer) -> np.ndarray:
    st_model.max_seq_length = MAX_SEQ_LENGTH
    return st_model.encode(
        sqls, batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SVC-120-S evaluation (identical to exp1)
# ═══════════════════════════════════════════════════════════════════════════════

def make_svc120s():
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=120, random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
    ])


def evaluate(X_feats: np.ndarray, hint_l: torch.Tensor,
             binary_l: torch.Tensor, encoder_name: str) -> dict:
    bench_t = torch.LongTensor([BENCHMARK_IDX])
    lt_t    = torch.LongTensor([LONGTAIL_IDX])
    y       = binary_l.numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    workloads, p90s, aurocs, f1s = [], [], [], []
    for tr_idx, te_idx in splitter.split(X_feats, y):
        X_tr, X_te = X_feats[tr_idx], X_feats[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        cw = {0: n_pos / n_neg, 1: n_neg / n_pos}

        clf = make_svc120s()
        clf.named_steps["svc"].set_params(class_weight=cw)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        chosen = hint_l[te_idx].gather(
            1, torch.where(torch.tensor(y_pred) > THRESHOLD,
                           lt_t, bench_t).view(-1, 1))
        workloads.append(chosen.sum().item())
        p90s.append(chosen.quantile(0.90).item())
        aurocs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "encoder"      : encoder_name,
        "workload_mean": np.mean(workloads),  "workload_std": np.std(workloads),
        "p90_mean"     : np.mean(p90s),       "p90_std":      np.std(p90s),
        "auroc_mean"   : np.mean(aurocs),     "auroc_std":    np.std(aurocs),
        "f1_mean"      : np.mean(f1s),        "f1_std":       np.std(f1s),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Contrastive fine-tuning with multi-config execution-preference triplets
# ═══════════════════════════════════════════════════════════════════════════════

def load_triplets() -> list:
    """
    Load stage1 triplets (anchor, positive, negative) from CEB + JOB splits.
    These are the same multi-config preference pairs used for AdaptSteer-C.
    """
    dfs = []
    for path in ["stage1_triplets_CEB.csv", "stage1_triplets_JOB.csv"]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError("No stage1_triplets_*.csv found in project root.")
    df = pd.concat(dfs, ignore_index=True).dropna(
        subset=["anchor", "positive", "negative"])
    # Shuffle and cap
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(df) > MAX_TRIPLETS:
        df = df.iloc[:MAX_TRIPLETS]
    print(f"  Triplets loaded: {len(df)} (capped at {MAX_TRIPLETS})")
    return [
        InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
        for _, row in df.iterrows()
    ]


def finetune(name: str, model_id: str, save_path: str,
             triplet_examples: list) -> SentenceTransformer:
    """Fine-tune one backbone with TripletLoss and save."""
    print(f"\n  {'─'*55}")
    print(f"  Fine-tuning: {name}  ({model_id})")
    print(f"  {'─'*55}")

    st = build_st_model(model_id)

    loader  = DataLoader(triplet_examples, shuffle=True,
                         batch_size=FINETUNE_BATCH, drop_last=True)
    loss_fn = losses.TripletLoss(model=st)

    total_steps  = len(loader) * FINETUNE_EPOCHS
    warmup_steps = min(FINETUNE_WARMUP, total_steps // 10)

    t0 = time.time()
    st.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=FINETUNE_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": FINETUNE_LR},
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    st.save(save_path)
    print(f"  Saved → {save_path}")
    return st


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(results_df: pd.DataFrame):
    """
    Two-panel figure:
      (a) Workload latency: off-the-shelf + fine-tuned code models vs references
      (b) AUROC: same grouping
    """
    DPI = 300

    # Separate code models from reference baselines
    ref_names = {"PostgreSQL Default", "MPNet-Raw", "AdaptSteer-C",
                 "AdaptSteer-R", "Optimal Oracle"}

    code_rows  = results_df[~results_df["encoder"].isin(ref_names)].copy()
    ref_rows   = results_df[results_df["encoder"].isin(
                     ["PostgreSQL Default", "MPNet-Raw", "AdaptSteer-R",
                      "Optimal Oracle"])].copy()

    # Palette
    COLORS_OFF  = "#5B9BD5"   # blue: off-the-shelf
    COLORS_FT   = "#70AD47"   # green: fine-tuned
    COLOR_PG    = "#C00000"
    COLOR_MPNET = "#ED7D31"
    COLOR_ADA   = "#4472C4"
    COLOR_OPT   = "#A0A0A0"

    ref_colors = {
        "PostgreSQL Default": COLOR_PG,
        "MPNet-Raw":          COLOR_MPNET,
        "AdaptSteer-R":       COLOR_ADA,
        "Optimal Oracle":     COLOR_OPT,
    }

    model_names = ["CodeBERT", "GraphCodeBERT", "CodeT5-base", "UniXCoder"]
    n = len(model_names)
    x = np.arange(n)
    w = 0.38

    # Extract values
    def get(name, condition, col):
        row = results_df[(results_df["encoder"] == name) &
                         (results_df["condition"] == condition)]
        if len(row) == 0:
            return float("nan"), 0.0
        return row.iloc[0][col], row.iloc[0].get(col.replace("_mean", "_std"), 0.0)

    off_wl  = [get(m, "off-the-shelf",  "workload_mean") for m in model_names]
    ft_wl   = [get(m, "fine-tuned",     "workload_mean") for m in model_names]
    off_auc = [get(m, "off-the-shelf",  "auroc_mean")    for m in model_names]
    ft_auc  = [get(m, "fine-tuned",     "auroc_mean")    for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Workload ────────────────────────────────────────────────────
    b_off = ax1.bar(x - w/2, [v[0] for v in off_wl], w,
                    yerr=[v[1] for v in off_wl],
                    label="Off-the-shelf", color=COLORS_OFF, alpha=0.88,
                    capsize=3, error_kw={"elinewidth": 0.9})
    b_ft  = ax1.bar(x + w/2, [v[0] for v in ft_wl],  w,
                    yerr=[v[1] for v in ft_wl],
                    label="Fine-tuned (same as AdaptSteer-C)", color=COLORS_FT,
                    alpha=0.88, capsize=3, error_kw={"elinewidth": 0.9})

    # Reference lines
    for ref_name, row in zip(
            ["PostgreSQL Default", "MPNet-Raw", "AdaptSteer-R", "Optimal Oracle"],
            ref_rows.to_dict("records")):
        ls = "--" if "Optimal" not in ref_name else ":"
        ax1.axhline(row["workload_mean"],
                    color=ref_colors.get(ref_name, "#808080"),
                    ls=ls, lw=1.1, alpha=0.8, label=ref_name)

    # Value labels on fine-tuned bars
    for i, (bar, (m, s)) in enumerate(zip(b_ft, ft_wl)):
        if not np.isnan(m):
            ax1.text(bar.get_x() + bar.get_width()/2, m + 80,
                     f"{m:,.0f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=9)
    ax1.set_ylabel("Mean Test-Fold Total Workload (s)", fontsize=10)
    ax1.set_title("(a) Workload Latency", fontsize=10.5, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    # ── Panel (b): AUROC ──────────────────────────────────────────────────────
    ax2.bar(x - w/2, [v[0] for v in off_auc], w,
            yerr=[v[1] for v in off_auc],
            label="Off-the-shelf", color=COLORS_OFF, alpha=0.88,
            capsize=3, error_kw={"elinewidth": 0.9})
    ax2.bar(x + w/2, [v[0] for v in ft_auc],  w,
            yerr=[v[1] for v in ft_auc],
            label="Fine-tuned", color=COLORS_FT, alpha=0.88,
            capsize=3, error_kw={"elinewidth": 0.9})

    # Reference AUROC lines
    ref_auroc = {"MPNet-Raw": (0.848, COLOR_MPNET),
                 "AdaptSteer-R": (0.811, COLOR_ADA)}
    for rname, (val, col) in ref_auroc.items():
        ax2.axhline(val, color=col, ls="--", lw=1.1, alpha=0.8, label=rname)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=9)
    ax2.set_ylabel("AUROC", fontsize=10)
    ax2.set_title("(b) Ranking AUROC", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0.5, 1.0)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=7.5, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Code-Specialized Backbone Ablation: Off-the-Shelf vs. Fine-Tuned\n"
        "(JOB+CEB, 10-split StratifiedShuffleSplit, mean±std)",
        fontsize=10.5, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("figures/fig_code_backbone.pdf", bbox_inches="tight", dpi=DPI)
    plt.savefig("figures/fig_code_backbone.png", bbox_inches="tight", dpi=DPI)
    plt.close()
    print("\nSaved: figures/fig_code_backbone.{pdf,png}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("Code Backbone Ablation — AdaptSteer Contrastive Pipeline")
    print("=" * 65)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    raw   = load_data()
    mdf   = prepare_model_df(raw)
    sqls  = mdf["sql"].tolist()
    hint_l  = torch.stack(mdf["mean_runtime"].apply(torch.Tensor).tolist())
    binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).float()
    print(f"  Queries: {len(mdf)}  |  Positive labels: "
          f"{int(binary_l.sum())} ({100*binary_l.mean():.1f}%)")

    # ── 2. Load triplets once ─────────────────────────────────────────────────
    print("\nLoading contrastive triplets...")
    triplets = load_triplets()

    # ── 3. Run experiments ────────────────────────────────────────────────────
    results = []

    for name, model_id in CODE_MODELS.items():
        save_path = SAVE_PATHS[name]
        print(f"\n{'='*65}")
        print(f"  MODEL: {name}  ({model_id})")
        print(f"{'='*65}")

        # ── (A) Off-the-shelf ─────────────────────────────────────────────────
        print(f"\n  [A] Off-the-shelf: {name}")
        try:
            st_raw = build_st_model(model_id)
            X_raw  = encode_all(sqls, st_raw)
            del st_raw; gc.collect(); torch.cuda.empty_cache()

            res_raw = evaluate(X_raw, hint_l, binary_l, name)
            res_raw["condition"] = "off-the-shelf"
            results.append(res_raw)
            print(f"  Workload: {res_raw['workload_mean']:,.1f} ± {res_raw['workload_std']:,.1f}s"
                  f"  |  AUROC: {res_raw['auroc_mean']:.4f} ± {res_raw['auroc_std']:.4f}")
        except Exception as e:
            print(f"  ERROR (off-the-shelf): {e}")
            import traceback; traceback.print_exc()

        # ── (B) Fine-tuned ────────────────────────────────────────────────────
        print(f"\n  [B] Fine-tuning: {name}")
        try:
            if os.path.isdir(save_path):
                print(f"  Already exists — loading {save_path}")
                st_ft = SentenceTransformer(save_path, device=DEVICE)
            else:
                st_ft = finetune(name, model_id, save_path, triplets)

            X_ft = encode_all(sqls, st_ft)
            del st_ft; gc.collect(); torch.cuda.empty_cache()

            res_ft = evaluate(X_ft, hint_l, binary_l, name)
            res_ft["condition"] = "fine-tuned"
            results.append(res_ft)
            print(f"  Workload: {res_ft['workload_mean']:,.1f} ± {res_ft['workload_std']:,.1f}s"
                  f"  |  AUROC: {res_ft['auroc_mean']:.4f} ± {res_ft['auroc_std']:.4f}")
        except Exception as e:
            print(f"  ERROR (fine-tuned): {e}")
            import traceback; traceback.print_exc()

    # ── 4. Add reference rows ─────────────────────────────────────────────────
    for ref in REFERENCE_ROWS:
        ref = ref.copy()
        if "condition" not in ref:
            ref["condition"] = "—"
        results.append(ref)

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    # fill missing std columns gracefully
    for col in ["workload_std", "p90_std", "auroc_std", "f1_std"]:
        if col not in df_res.columns:
            df_res[col] = 0.0
        df_res[col] = df_res[col].fillna(0.0)

    df_res.to_csv("results/code_backbone_ablation.csv", index=False)
    print("\n" + "="*65)
    print("RESULTS SUMMARY")
    print("="*65)
    hdr = f"{'Encoder':<20} {'Cond':<18} {'Workload (s)':>22}  {'AUROC':>8}  {'F1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in df_res.iterrows():
        wl  = f"{row['workload_mean']:,.1f} ± {row.get('workload_std',0):,.1f}"
        auc = f"{row['auroc_mean']:.4f}" if not np.isnan(row.get('auroc_mean', float('nan'))) else "   —  "
        f1  = f"{row['f1_mean']:.4f}"   if not np.isnan(row.get('f1_mean',    float('nan'))) else "  —  "
        print(f"{row['encoder']:<20} {row.get('condition','—'):<18} {wl:>22}  {auc:>8}  {f1:>7}")
    print()
    print(f"Saved: results/code_backbone_ablation.csv")

    # ── 6. Generate figure ────────────────────────────────────────────────────
    try:
        make_figure(df_res)
    except Exception as e:
        print(f"Figure generation failed: {e}")
        import traceback; traceback.print_exc()

    print("\n✅ Done.")
