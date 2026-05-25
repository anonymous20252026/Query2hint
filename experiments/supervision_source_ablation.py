"""
Experiment 1: Supervision Ablation
====================================
Proves that multi-configuration execution-preference supervision is the
source of AdaSteer's advantage, not the encoder architecture alone.

Four conditions under the same SVC-120-S binary steering protocol:
  MPNet-Raw    : off-the-shelf all-mpnet-base-v2, no fine-tuning
  MPNet-Binary : all-mpnet-base-v2 fine-tuned with BINARY-ONLY supervision
                 (triplets constructed from hint-0 vs hint-26 only)
  AdaSteer-C   : encoder trained with full multi-config preference supervision
  AdaSteer-R   : preference supervision + Reptile meta-learning

Outputs:
  results/exp1_supervision_ablation.csv
  results/exp1_supervision_ablation_figure.pdf / .png
"""

import os
import warnings
import random
import gc
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 24508
BENCHMARK_IDX  = 0   # default PostgreSQL
LONGTAIL_IDX   = 26  # fixed alternative hint set
K_FOLDS        = 10
TRAIN_SIZE     = 0.8
THRESHOLD      = 0.5
DEVICE         = "cuda"

ENCODERS = {
    "AdaSteer-C": "encoders/encoder_all-mpnet-base-v2_v1",
    "AdaSteer-R": "encoders/encoder_reptile_mpnet_v4",
}
MPNET_BASE     = "sentence-transformers/all-mpnet-base-v2"
BINARY_SAVE    = "encoders/encoder_mpnet_binary_supervised"

BINARY_FINETUNE_SAMPLES = 8000   # triplets for binary fine-tuning
BINARY_EPOCHS           = 3
BINARY_BATCH            = 8      # reduced from 32 to fit GPU (attention mem ∝ batch²)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. This experiment requires a CUDA GPU.")

print("=" * 65)
print("Experiment 1: Supervision Ablation")
print("=" * 65)
print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    job = pd.read_csv("data/job.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    ceb = pd.read_csv("data/ceb.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    data = pd.concat([job, ceb]).reset_index(drop=True)
    data["mean_runtime"] = data["runtime_list"].apply(np.mean)
    data["sql"] = data["sql"].apply(lambda x: x.strip("\n"))
    return data


def prepare_features(data):
    df = data.copy()
    df = df.drop(columns=["runtime_list", "plan_tree",
                           "mean_runtime"], errors="ignore")
    df["mean_runtime"] = data["mean_runtime"]
    df = df.explode("hint_list").sort_values(["filename", "hint_list"])
    df = df.groupby(["filename", "sql"], as_index=False).agg(
        hint_list   =("hint_list",    list),
        mean_runtime=("mean_runtime", list),
    )
    df["opt_l"] = df["mean_runtime"].apply(min)
    return df.reset_index(drop=True)


def build_tensors(model_df):
    X      = torch.tensor(np.stack(model_df["features"].tolist()),
                          dtype=torch.float32)
    hint_l = torch.stack(model_df["mean_runtime"].apply(torch.Tensor).tolist())
    opt_l  = torch.stack(
        model_df["opt_l"].apply(
            lambda x: torch.Tensor([x]).repeat(hint_l.size(1))
        ).tolist()
    )
    binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).float()
    return X, hint_l, opt_l, binary_l


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_queries(model_df, encoder_path_or_model):
    own_model = isinstance(encoder_path_or_model, str)
    if own_model:
        st = SentenceTransformer(encoder_path_or_model, device=DEVICE)
    else:
        st = encoder_path_or_model
    st.max_seq_length = 512
    sqls = model_df["sql"].tolist()
    embs = st.encode(sqls, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    if own_model:
        del st
        gc.collect()
        torch.cuda.empty_cache()
    df = model_df.copy()
    df["features"] = list(embs)
    return df


# ── Binary fine-tuning of MPNet ───────────────────────────────────────────────

def finetune_binary_mpnet(model_df, binary_labels):
    """Fine-tune all-mpnet-base-v2 using ONLY binary hint-0 vs hint-26 labels.
    Triplets: anchor + positive (same label) + negative (opposite label).
    """
    print("  Fine-tuning MPNet-Binary with binary-only supervision...")

    sqls   = model_df["sql"].tolist()
    labels = binary_labels.numpy()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    rng = np.random.RandomState(RANDOM_SEED)
    triplets = []
    half = BINARY_FINETUNE_SAMPLES // 2

    # anchor=pos, positive=another_pos, negative=neg
    for _ in range(half):
        a, p = rng.choice(pos_idx, 2, replace=False)
        n    = rng.choice(neg_idx)
        triplets.append(InputExample(texts=[sqls[a], sqls[p], sqls[n]]))

    # anchor=neg, positive=another_neg, negative=pos
    for _ in range(half):
        a, p = rng.choice(neg_idx, 2, replace=False)
        n    = rng.choice(pos_idx)
        triplets.append(InputExample(texts=[sqls[a], sqls[p], sqls[n]]))

    rng.shuffle(triplets)

    st = SentenceTransformer(MPNET_BASE, device=DEVICE)
    st.max_seq_length = 512

    loader = DataLoader(triplets, shuffle=True, batch_size=BINARY_BATCH)
    loss_fn = losses.TripletLoss(model=st)

    st.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=BINARY_EPOCHS,
        warmup_steps=100,
        show_progress_bar=False,
        output_path=None,
    )
    st.save(BINARY_SAVE)
    print(f"  Saved MPNet-Binary → {BINARY_SAVE}")
    return st


# ── SVC-120-S classifier (best known config) ─────────────────────────────────

def make_svc120s():
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=120, random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        random_state=RANDOM_SEED)),
    ])


# ── 10-fold CV evaluation ─────────────────────────────────────────────────────

def evaluate_encoder(X, hint_l, binary_l, encoder_name):
    benchmark = torch.LongTensor([BENCHMARK_IDX])
    longtail  = torch.LongTensor([LONGTAIL_IDX])

    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    X_np = X.numpy()
    y    = binary_l.numpy()

    workloads, p90s, aurocs, f1s = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_np, y)):
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        cw = {0: n_pos / n_neg, 1: n_neg / n_pos}

        clf = make_svc120s()
        clf.named_steps["svc"].set_params(class_weight=cw)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # workload: apply predicted hint per query
        chosen = hint_l[te_idx].gather(
            1, torch.where(torch.tensor(y_pred) > THRESHOLD,
                           longtail, benchmark).view(-1, 1))
        workloads.append(chosen.sum().item())
        p90s.append(chosen.quantile(0.90).item())
        aurocs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "encoder"       : encoder_name,
        "workload_mean" : np.mean(workloads),
        "workload_std"  : np.std(workloads),
        "p90_mean"      : np.mean(p90s),
        "p90_std"       : np.std(p90s),
        "auroc_mean"    : np.mean(aurocs),
        "auroc_std"     : np.std(aurocs),
        "f1_mean"       : np.mean(f1s),
        "f1_std"        : np.std(f1s),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

print("\nLoading data...")
raw_data   = load_data()
model_df   = prepare_features(raw_data)
print(f"Queries: {len(model_df)}")

results = []

# LLMSteer published baseline
results.append({
    "encoder": "LLMSteer (OpenAI)",
    "workload_mean": 2547.7, "workload_std": 0,
    "p90_mean": 5.7, "p90_std": 0,
    "auroc_mean": float("nan"), "auroc_std": 0,
    "f1_mean": float("nan"), "f1_std": 0,
})

# 1. MPNet-Raw: off-the-shelf, no fine-tuning
print("\n[1/4] MPNet-Raw (off-the-shelf, no fine-tuning)...")
df1 = encode_queries(model_df.copy(), MPNET_BASE)
X1, hint_l, opt_l, binary_l = build_tensors(df1)
results.append(evaluate_encoder(X1, hint_l, binary_l, "MPNet-Raw"))
print(f"  Workload: {results[-1]['workload_mean']:.1f} ± {results[-1]['workload_std']:.1f}s")

# 2. MPNet-Binary: fine-tune with binary-only supervision
# binary_l is the same for all encoders (depends on hint runtimes, not embeddings)
print("\n[2/4] MPNet-Binary (binary-only supervision fine-tuning)...")
if os.path.isdir(BINARY_SAVE):
    print(f"  Loading existing fine-tuned model from {BINARY_SAVE}...")
    binary_st = SentenceTransformer(BINARY_SAVE, device=DEVICE)
else:
    binary_st = finetune_binary_mpnet(model_df, binary_l)

df2 = encode_queries(model_df.copy(), binary_st)
X2, _, _, _ = build_tensors(df2)
results.append(evaluate_encoder(X2, hint_l, binary_l, "MPNet-Binary"))
print(f"  Workload: {results[-1]['workload_mean']:.1f} ± {results[-1]['workload_std']:.1f}s")

# 3. AdaSteer-C: multi-config preference supervision
print("\n[3/4] AdaSteer-C (multi-config execution-preference supervision)...")
df3 = encode_queries(model_df.copy(), ENCODERS["AdaSteer-C"])
X3, _, _, _ = build_tensors(df3)
results.append(evaluate_encoder(X3, hint_l, binary_l, "AdaSteer-C"))
print(f"  Workload: {results[-1]['workload_mean']:.1f} ± {results[-1]['workload_std']:.1f}s")

# 4. AdaSteer-R: preference supervision + Reptile meta-learning
print("\n[4/4] AdaSteer-R (preference supervision + meta-learning)...")
df4 = encode_queries(model_df.copy(), ENCODERS["AdaSteer-R"])
X4, _, _, _ = build_tensors(df4)
results.append(evaluate_encoder(X4, hint_l, binary_l, "AdaSteer-R"))
print(f"  Workload: {results[-1]['workload_mean']:.1f} ± {results[-1]['workload_std']:.1f}s")

# ── Results table ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv("results/exp1_supervision_ablation.csv", index=False)

print("\n" + "=" * 70)
print("SUPERVISION ABLATION — Results (SVC-120-S, 10-fold CV)")
print("=" * 70)
print(f"{'Encoder':<28} | {'Workload (s)':>16} | {'P90 (s)':>10} | {'AUROC':>8} | {'F1':>8}")
print("-" * 70)
for r in results:
    w = f"{r['workload_mean']:.1f}±{r['workload_std']:.0f}" if r['workload_std'] > 0 else f"{r['workload_mean']:.1f}"
    p = f"{r['p90_mean']:.4f}"
    a = f"{r['auroc_mean']:.4f}" if not np.isnan(r['auroc_mean']) else "   —"
    f = f"{r['f1_mean']:.4f}" if not np.isnan(r['f1_mean']) else "   —"
    print(f"{r['encoder']:<28} | {w:>16} | {p:>10} | {a:>8} | {f:>8}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
names  = [r["encoder"] for r in results]
colors = ["#aaaaaa", "#f4a261", "#2196F3", "#4CAF50", "#9C27B0"][:len(names)]

for ax, (metric, label, key) in zip(axes, [
    ("workload_mean", "Total Workload (s)", "workload_std"),
    ("p90_mean",      "P90 Latency (s)",    "p90_std"),
    ("auroc_mean",    "AUROC",              "auroc_std"),
]):
    vals = [r[metric] for r in results]
    errs = [r[key]    for r in results]
    bars = ax.bar(names, vals, color=colors, yerr=errs, capsize=4,
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel(label)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.axhline(results[0][metric], color="grey", linestyle="--",
               linewidth=1, label="LLMSteer" if metric == "workload_mean" else "")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_title("Total Workload ↓ (lower = better)")
axes[1].set_title("P90 Tail Latency ↓")
axes[2].set_title("AUROC ↑ (higher = better)")
fig.suptitle("Experiment 1: Supervision Ablation — Proving Execution-Preference Supervision Matters",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("results/exp1_supervision_ablation_figure.pdf", bbox_inches="tight")
plt.savefig("results/exp1_supervision_ablation_figure.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: results/exp1_supervision_ablation.csv")
print("✅ Saved: results/exp1_supervision_ablation_figure.{pdf,png}")
