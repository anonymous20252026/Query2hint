"""
exp_fewshot_e2e.py
==================
End-to-end binary steering workload after K-shot label-efficient adaptation
on the full JOB+CEB dataset (same 10-fold StratifiedShuffleSplit as main results).

Experiment design
-----------------
The main binary steering result (AdaptSteer-R = 2434.9 ± 320.4 s) trains the
SVC-120-S classifier on ALL ~2597 training examples per fold.

This experiment asks: how does workload degrade when we restrict the SVC to
only K labeled examples per fold?

For each K in {10, 20, 50, 100, 200, 500, full}:
  - Use the frozen AdaptSteer-R encoder (Reptile-initialized, no re-fine-tuning)
  - Sample K examples uniformly at random from the training fold
  - Train SVC-120-S on those K examples
  - Evaluate binary steering workload on the test fold
  - Repeat SEEDS times for stable estimates (K < full only)

Also reports:
  - AdaptSteer-C (frozen contrastive encoder, same K-shot SVC)
  - MPNet-Raw (off-the-shelf, same K-shot SVC)

Outputs
-------
  results/fewshot_e2e_workload.csv
  figures/fig_fewshot_e2e.pdf  /  .png
"""

import os
import warnings
import random

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
from sklearn.metrics import roc_auc_score

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 24508
BENCHMARK_IDX = 0    # PostgreSQL default
LONGTAIL_IDX  = 26   # best fixed alternative
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
THRESHOLD     = 0.5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS         = [42, 123, 999]   # seeds for K < full

K_VALUES = [10, 20, 50, 100, 200, 500, "full"]

ENCODERS = {
    "AdaptSteer-C": "encoders/encoder_all-mpnet-base-v2_v1",
    "AdaptSteer-R": "encoders/encoder_reptile_mpnet_v4",
    "MPNet-Raw":    "sentence-transformers/all-mpnet-base-v2",
}

print("=" * 65)
print("exp_fewshot_e2e — End-to-End Workload vs K (JOB+CEB, binary)")
print(f"Device: {DEVICE}")
print("=" * 65)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    job = pd.read_csv("data/job.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    ceb = pd.read_csv("data/ceb.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    data = pd.concat([job, ceb]).reset_index(drop=True)
    data["mean_runtime"] = data["runtime_list"].apply(np.mean)
    data["sql"] = data["sql"].str.strip("\n")
    return data


def prepare_features(data):
    df = data.copy()
    df = df.explode("hint_list").sort_values(["filename", "hint_list"])
    df = df.groupby(["filename", "sql"], as_index=False).agg(
        hint_list    =("hint_list",    list),
        mean_runtime =("mean_runtime", list),
    )
    df["opt_l"] = df["mean_runtime"].apply(min)
    return df.reset_index(drop=True)


def build_tensors(model_df):
    hint_l   = torch.stack(
        model_df["mean_runtime"].apply(torch.Tensor).tolist())
    binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).float()
    return hint_l, binary_l


def encode_queries(model_df, encoder_path):
    st = SentenceTransformer(encoder_path, device=DEVICE)
    texts = model_df["sql"].tolist()
    X = st.encode(texts, batch_size=64, show_progress_bar=False,
                  convert_to_numpy=True)
    return torch.tensor(X, dtype=torch.float32)


# ── SVC-120-S pipeline ────────────────────────────────────────────────────────

def make_svc120s(class_weight=None, n_samples=None):
    """SVC-120-S pipeline; caps PCA at min(120, n_samples-1) for small K."""
    n_pca = 120 if n_samples is None else min(120, n_samples - 1)
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=n_pca, random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight=class_weight,
                        random_state=RANDOM_SEED)),
    ])


# ── K-shot evaluation on one CV fold ─────────────────────────────────────────

def eval_fold_k(X_tr, y_tr, X_te, y_te, hint_l_te, K, seed):
    """Train SVC on K samples from training fold; evaluate workload on test."""
    rng = np.random.RandomState(seed)

    if K == "full" or K >= len(X_tr):
        X_k, y_k = X_tr, y_tr
    else:
        # Stratified sample: try to preserve positive/negative ratio
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        k_pos = min(K // 2, len(pos_idx))
        k_neg = min(K - k_pos, len(neg_idx))
        k_pos = K - k_neg   # rebalance if neg was capped
        sel_pos = rng.choice(pos_idx, size=k_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=k_neg, replace=False)
        sel = np.concatenate([sel_pos, sel_neg])
        X_k, y_k = X_tr[sel], y_tr[sel]

    if len(np.unique(y_k)) < 2:
        return float("nan"), float("nan"), float("nan")

    n_pos = (y_k == 1).sum()
    n_neg = (y_k == 0).sum()
    cw = {0: n_pos / n_neg, 1: n_neg / n_pos} if n_neg > 0 and n_pos > 0 else None

    clf = make_svc120s(class_weight=cw, n_samples=len(X_k))
    clf.fit(X_k, y_k)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    benchmark = torch.LongTensor([BENCHMARK_IDX])
    longtail  = torch.LongTensor([LONGTAIL_IDX])
    chosen = hint_l_te.gather(
        1, torch.where(torch.tensor(y_pred) > THRESHOLD,
                       longtail, benchmark).view(-1, 1))
    workload = chosen.sum().item()
    p90      = chosen.quantile(0.90).item()
    auroc    = roc_auc_score(y_te, y_prob)

    return workload, p90, auroc


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_k_shots(X, hint_l, binary_l, encoder_name):
    X_np = X.numpy()
    y    = binary_l.numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
    folds = list(splitter.split(X_np, y))

    records = []
    for K in K_VALUES:
        k_label = "full" if K == "full" else int(K)
        seeds = [RANDOM_SEED] if K == "full" else SEEDS

        all_wl, all_p90, all_auroc = [], [], []
        for seed in seeds:
            fold_wl, fold_p90, fold_auroc = [], [], []
            for tr_idx, te_idx in folds:
                wl, p90, auroc = eval_fold_k(
                    X_np[tr_idx], y[tr_idx],
                    X_np[te_idx], y[te_idx],
                    hint_l[te_idx], K, seed)
                if not np.isnan(wl):
                    fold_wl.append(wl)
                    fold_p90.append(p90)
                    fold_auroc.append(auroc)

            all_wl.append(np.mean(fold_wl))
            all_p90.append(np.mean(fold_p90))
            all_auroc.append(np.mean(fold_auroc))

        rec = {
            "encoder":       encoder_name,
            "K":             k_label,
            "n_seeds":       len(seeds),
            "workload_mean": np.mean(all_wl),
            "workload_std":  np.std(all_wl),
            "p90_mean":      np.mean(all_p90),
            "p90_std":       np.std(all_p90),
            "auroc_mean":    np.mean(all_auroc),
            "auroc_std":     np.std(all_auroc),
        }
        records.append(rec)
        k_str = "full" if K == "full" else f"K={K:4d}"
        print(f"  [{encoder_name}] {k_str}: workload={rec['workload_mean']:.1f}"
              f"±{rec['workload_std']:.1f}s  AUROC={rec['auroc_mean']:.4f}")

    return records


# ── Load and encode ───────────────────────────────────────────────────────────

print("\n[1/5] Loading JOB + CEB data...")
raw_data = load_data()
model_df = prepare_features(raw_data)
hint_l, binary_l = build_tensors(model_df)

n_pos = binary_l.sum().int().item()
n_neg = (binary_l == 0).sum().int().item()
print(f"  Queries: {len(model_df)}  |  pos (hint_26 better): {n_pos}  "
      f"|  neg: {n_neg}  |  pos_rate: {n_pos/len(model_df):.3f}")

all_records = []

# PostgreSQL Default baseline (K-independent reference)
pg_wl = hint_l[:, BENCHMARK_IDX].sum().item() / K_FOLDS / (1 / (1 - TRAIN_SIZE))
# Approximate per-fold test workload = total * test_fraction / n_folds... use known value
PG_WL_MEAN  = 8134.70;  PG_WL_STD  = 1454.3
OPT_WL_MEAN = 1064.10;  OPT_WL_STD = 63.4
LLMSTEER_WL = 2547.70

print("\n[2/5] Encoding with MPNet-Raw (off-the-shelf)...")
X_raw = encode_queries(model_df, ENCODERS["MPNet-Raw"])
print(f"  Embedding shape: {X_raw.shape}")
recs_raw = evaluate_k_shots(X_raw, hint_l, binary_l, "MPNet-Raw")
all_records.extend(recs_raw)

print("\n[3/5] Encoding with AdaptSteer-C (contrastive)...")
X_c = encode_queries(model_df, ENCODERS["AdaptSteer-C"])
recs_c = evaluate_k_shots(X_c, hint_l, binary_l, "AdaptSteer-C")
all_records.extend(recs_c)

print("\n[4/5] Encoding with AdaptSteer-R (Reptile)...")
X_r = encode_queries(model_df, ENCODERS["AdaptSteer-R"])
recs_r = evaluate_k_shots(X_r, hint_l, binary_l, "AdaptSteer-R")
all_records.extend(recs_r)

# ── Save CSV ──────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(all_records)
df_out.to_csv("results/fewshot_e2e_workload.csv", index=False)
print(f"\n[5/5] Saved → results/fewshot_e2e_workload.csv ({len(df_out)} rows)")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Summary: Workload (s) vs K — JOB+CEB binary steering")
print(f"{'K':>8}  {'MPNet-Raw':>14}  {'AdaptSteer-C':>14}  {'AdaptSteer-R':>14}")
print("-" * 65)
for K in K_VALUES:
    def get(enc):
        row = df_out[(df_out.encoder == enc) & (df_out.K == ("full" if K == "full" else int(K)))]
        if len(row) == 0: return "  --  "
        m, s = row.iloc[0]["workload_mean"], row.iloc[0]["workload_std"]
        return f"{m:.0f}±{s:.0f}"
    print(f"{'full' if K=='full' else int(K):>8}  {get('MPNet-Raw'):>14}  "
          f"{get('AdaptSteer-C'):>14}  {get('AdaptSteer-R'):>14}")
print(f"\nReferences: PG Default={PG_WL_MEAN:.0f}s  "
      f"LLMSteer={LLMSTEER_WL:.0f}s  Oracle={OPT_WL_MEAN:.0f}s")


# ── Figure ────────────────────────────────────────────────────────────────────
DPI = 300
COLOR_RAW = "#ED7D31"   # orange — MPNet-Raw
COLOR_C   = "#4472C4"   # blue  — AdaptSteer-C
COLOR_R   = "#70AD47"   # green — AdaptSteer-R

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Numeric K values for x-axis (use 2600 for "full")
k_numeric = [10, 20, 50, 100, 200, 500, 2597]   # ~2597 = avg training set size

def extract_curve(enc, df):
    wl_m, wl_s, au_m, au_s = [], [], [], []
    for k_lab, k_x in zip(K_VALUES, k_numeric):
        k_key = "full" if k_lab == "full" else int(k_lab)
        row = df[(df.encoder == enc) & (df.K == k_key)]
        if len(row) == 0:
            wl_m.append(np.nan); wl_s.append(0)
            au_m.append(np.nan); au_s.append(0)
        else:
            wl_m.append(row.iloc[0]["workload_mean"])
            wl_s.append(row.iloc[0]["workload_std"])
            au_m.append(row.iloc[0]["auroc_mean"])
            au_s.append(row.iloc[0]["auroc_std"])
    return (np.array(wl_m), np.array(wl_s),
            np.array(au_m), np.array(au_s))

for enc, color, label in [
    ("MPNet-Raw",     COLOR_RAW, "MPNet-Raw"),
    ("AdaptSteer-C",  COLOR_C,   "AdaptSteer-C"),
    ("AdaptSteer-R",  COLOR_R,   "AdaptSteer-R"),
]:
    wl_m, wl_s, au_m, au_s = extract_curve(enc, df_out)

    ax1.plot(k_numeric, wl_m, "o-", color=color, lw=2.0, ms=6, label=label)
    ax1.fill_between(k_numeric, wl_m - wl_s, wl_m + wl_s,
                     alpha=0.15, color=color)
    ax2.plot(k_numeric, au_m, "o-", color=color, lw=2.0, ms=6, label=label)
    ax2.fill_between(k_numeric, au_m - au_s, au_m + au_s,
                     alpha=0.15, color=color)

# Reference lines — workload panel
ax1.axhline(PG_WL_MEAN,  color="#C00000", ls="--", lw=1.3, alpha=0.8,
            label=f"PostgreSQL Default ({PG_WL_MEAN:,.0f} s)")
ax1.axhline(LLMSTEER_WL, color="#9E480E", ls=":",  lw=1.2, alpha=0.7,
            label=f"LLMSteer ({LLMSTEER_WL:,.0f} s)")
ax1.axhline(OPT_WL_MEAN, color="#808080", ls=":",  lw=1.2, alpha=0.7,
            label=f"Oracle ({OPT_WL_MEAN:,.0f} s)")

# Reference lines — AUROC panel
ax2.axhline(0.848, color=COLOR_RAW, ls=":", lw=1.2, alpha=0.6)  # MPNet-Raw full
ax2.axhline(0.811, color=COLOR_R,   ls=":", lw=1.2, alpha=0.6)  # AdaptSteer-R full

# Formatting
for ax, ylabel, title in [
    (ax1, "Mean Test-Fold Workload (s)", "(a) Binary Steering Workload vs. K"),
    (ax2, "AUROC",                        "(b) Ranking AUROC vs. K"),
]:
    ax.set_xlabel("Training examples K (log scale)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(k_numeric)
    ax.set_xticklabels(["10", "20", "50", "100", "200", "500", "full"],
                       fontsize=8.5)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9)

ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

fig.suptitle(
    "Label Efficiency: Binary Steering Workload vs. K Training Examples\n"
    "(JOB+CEB, 10-fold StratifiedShuffleSplit, frozen encoder + SVC-120-S)",
    fontsize=10.5, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figures/fig_fewshot_e2e.pdf", bbox_inches="tight", dpi=DPI)
plt.savefig("figures/fig_fewshot_e2e.png", bbox_inches="tight", dpi=DPI)
plt.close()
print("\nSaved: figures/fig_fewshot_e2e.{pdf,png}")
print("\nDone.")
