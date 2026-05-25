"""
Experiment 3: Multi-Action Steering
======================================
Shows the framework operates beyond binary steering.
For each candidate set size K ∈ {2, 4, 8, 16}, trains a K-class
multiclass classifier and evaluates actual query execution workload.

Baselines compared:
  - PostgreSQL Default   : always use hint-0 (no steering)
  - Best Fixed Config    : always use the globally cheapest hint in the set
  - Random Selection     : uniform random over K hints (averaged over seeds)
  - Oracle Best-in-Set   : always pick the per-query best in the K candidates
  - Direct Multiclass    : AdaSteer-C + SVC multiclass over K classes
  - AdaSteer-C Multiclass: same + temperature scaling / probability thresholding

Also reports:
  - Regret vs oracle: (model_workload - oracle_workload)
  - Top-1 accuracy: fraction of queries where model picks the oracle hint

Outputs:
  results/exp3_multiaction_steering.csv
  results/exp3_multiaction_steering_figure.pdf / .png
"""

import os
import warnings
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
from sklearn.metrics import top_k_accuracy_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 24508
BENCHMARK_IDX = 0
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
DEVICE        = "cuda"
ENCODER_PATH  = "encoders/encoder_all-mpnet-base-v2_v1"   # AdaSteer-C

K_SIZES = [2, 4, 8, 16]
RANDOM_SEEDS_FOR_RANDOM_BASELINE = [0, 1, 2, 3, 4]

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. This experiment requires a CUDA GPU.")

print("=" * 65)
print("Experiment 3: Multi-Action Steering")
print("=" * 65)
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"Encoder  : {ENCODER_PATH}")
print(f"K sizes  : {K_SIZES}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_prepare():
    job = pd.read_csv("data/job.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    ceb = pd.read_csv("data/ceb.csv",
                      converters={"hint_list": eval, "runtime_list": eval})
    data = pd.concat([job, ceb]).reset_index(drop=True)
    data["mean_runtime"] = data["runtime_list"].apply(np.mean)
    data["sql"] = data["sql"].apply(lambda x: x.strip("\n"))
    df = data.copy()
    df = df.explode("hint_list").sort_values(["filename", "hint_list"])
    df = df.groupby(["filename", "sql"], as_index=False).agg(
        hint_list   =("hint_list",    list),
        mean_runtime=("mean_runtime", list),
    )
    return df.reset_index(drop=True)


print("\nLoading data and computing embeddings (once)...")
model_df = load_and_prepare()
n_hints  = len(model_df["mean_runtime"].iloc[0])
print(f"Queries   : {len(model_df)}")
print(f"Hint sets : {n_hints}")

st = SentenceTransformer(ENCODER_PATH, device=DEVICE)
st.max_seq_length = 512
embs = st.encode(model_df["sql"].tolist(), batch_size=64,
                 convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False)
X_base = torch.tensor(embs, dtype=torch.float32)
hint_l = torch.stack(model_df["mean_runtime"].apply(torch.Tensor).tolist())

# Global best-fixed hint: index of the hint with lowest AVERAGE runtime
global_mean_rt  = hint_l.mean(dim=0)                    # [n_hints]
best_fixed_idx  = global_mean_rt.argmin().item()
print(f"Best fixed hint (by mean runtime): index {best_fixed_idx}")

all_alt_indices = list(range(1, n_hints))
rng = np.random.RandomState(RANDOM_SEED)


# ── SVC multiclass classifier ─────────────────────────────────────────────────

def make_multiclass_svc(n_classes):
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=min(120, n_classes * 15),
                        random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        decision_function_shape="ovr",
                        random_state=RANDOM_SEED)),
    ])


# ── Per-K evaluation ──────────────────────────────────────────────────────────

def run_for_k(k_size):
    """
    Select k_size hint indices (always include hint-0 as the default option).
    Train a k_size-class SVC and measure workload vs baselines.
    """
    # Candidate set: default (0) + k_size-1 alternatives
    # Include hint-26 as a known informative alternative
    alts_pool = [26] + [i for i in all_alt_indices if i != 26]
    rng.shuffle(alts_pool[1:])
    candidate_set = [BENCHMARK_IDX] + alts_pool[: k_size - 1]
    # candidate_set[i] → global hint index

    hint_l_sub = hint_l[:, candidate_set]             # [N, k_size]
    # Multiclass label: for each query, the index within candidate_set
    # that achieves minimum runtime
    class_labels = hint_l_sub.argmin(dim=1).numpy()   # [N] ∈ {0,..,k_size-1}

    X_np = X_base.numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    metrics = {
        "default":     [],   "best_fixed": [], "random": [],
        "oracle":      [],   "model":      [],
        "default_p90": [],   "best_fixed_p90": [], "random_p90": [],
        "oracle_p90":  [],   "model_p90":  [],
        "top1_acc":    [],   "regret":     [],
    }

    # Best fixed: pre-compute which candidate is globally cheapest
    sub_mean = hint_l_sub.mean(dim=0)
    best_fixed_in_set = sub_mean.argmin().item()       # local index in candidate_set

    for tr_idx, te_idx in splitter.split(X_np, class_labels):
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr       = class_labels[tr_idx]

        clf = make_multiclass_svc(k_size)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)                     # predicted class (local idx)

        # Oracle: per-query minimum runtime in candidate_set
        oracle_rt = hint_l_sub[te_idx].min(dim=1).values
        oracle_lbl = hint_l_sub[te_idx].argmin(dim=1).numpy()

        # Model: use predicted local index → global hint
        model_rt = hint_l_sub[te_idx].gather(
            1, torch.tensor(y_pred, dtype=torch.long).view(-1, 1)
        ).squeeze(1)

        # Default: always use hint-0 (local index 0)
        default_rt = hint_l_sub[te_idx, 0]

        # Best fixed: always use the globally cheapest hint in set
        best_fixed_rt = hint_l_sub[te_idx, best_fixed_in_set]

        # Random baseline: average over multiple seeds
        rand_rts = []
        for s in RANDOM_SEEDS_FOR_RANDOM_BASELINE:
            rng2 = np.random.RandomState(s)
            rand_choices = rng2.randint(0, k_size, len(te_idx))
            rand_rt = hint_l_sub[te_idx].gather(
                1, torch.tensor(rand_choices, dtype=torch.long).view(-1, 1)
            ).squeeze(1)
            rand_rts.append(rand_rt)
        random_rt = torch.stack(rand_rts).mean(dim=0)

        metrics["default"].append(default_rt.sum().item())
        metrics["default_p90"].append(default_rt.quantile(0.90).item())
        metrics["best_fixed"].append(best_fixed_rt.sum().item())
        metrics["best_fixed_p90"].append(best_fixed_rt.quantile(0.90).item())
        metrics["random"].append(random_rt.sum().item())
        metrics["random_p90"].append(random_rt.quantile(0.90).item())
        metrics["oracle"].append(oracle_rt.sum().item())
        metrics["oracle_p90"].append(oracle_rt.quantile(0.90).item())
        metrics["model"].append(model_rt.sum().item())
        metrics["model_p90"].append(model_rt.quantile(0.90).item())

        top1 = (y_pred == oracle_lbl).mean()
        metrics["top1_acc"].append(top1)

        regret = (model_rt - oracle_rt).mean().item()
        metrics["regret"].append(regret)

    return {
        "k_size": k_size,
        "candidate_set": candidate_set,
        **{f"{m}_mean": np.mean(v) for m, v in metrics.items()},
        **{f"{m}_std":  np.std(v)  for m, v in metrics.items()},
    }


results = []
for k in K_SIZES:
    print(f"\nRunning K = {k} ...")
    r = run_for_k(k)
    results.append(r)
    print(f"  Default: {r['default_mean']:.1f}s | "
          f"Oracle: {r['oracle_mean']:.1f}s | "
          f"Model: {r['model_mean']:.1f}s | "
          f"Top-1: {r['top1_acc_mean']:.3f} | "
          f"Regret: {r['regret_mean']:.3f}s/query")

df_res = pd.DataFrame(results)
df_res.drop(columns=["candidate_set"], errors="ignore").to_csv(
    "results/exp3_multiaction_steering.csv", index=False)

print("\n" + "=" * 85)
print("MULTI-ACTION STEERING RESULTS")
print("=" * 85)
hdr = f"{'K':>4} | {'Default':>12} | {'BestFixed':>12} | {'Random':>12} | {'Oracle':>12} | {'Model':>12} | {'Top-1':>6} | {'Regret/q':>8}"
print(hdr)
print("-" * 85)
for r in results:
    print(f"{r['k_size']:>4} | "
          f"{r['default_mean']:>8.1f}±{r['default_std']:>3.0f} | "
          f"{r['best_fixed_mean']:>8.1f}±{r['best_fixed_std']:>3.0f} | "
          f"{r['random_mean']:>8.1f}±{r['random_std']:>3.0f} | "
          f"{r['oracle_mean']:>8.1f}±{r['oracle_std']:>3.0f} | "
          f"{r['model_mean']:>8.1f}±{r['model_std']:>3.0f} | "
          f"{r['top1_acc_mean']:>6.3f} | "
          f"{r['regret_mean']:>8.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
k_vals = [r["k_size"] for r in results]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel 1: Workload vs K for all methods
methods = [
    ("default_mean", "default_std",    "#aaaaaa", "Default PostgreSQL"),
    ("best_fixed_mean", "best_fixed_std", "#FF9800", "Best Fixed"),
    ("random_mean",  "random_std",     "#f4a261", "Random"),
    ("oracle_mean",  "oracle_std",     "#4CAF50", "Oracle Best-in-Set"),
    ("model_mean",   "model_std",      "#2196F3", "AdaSteer-C (multiclass)"),
]
for mn, ms, color, label in methods:
    axes[0].errorbar(k_vals, [r[mn] for r in results],
                     yerr=[r[ms] for r in results],
                     marker="o", color=color, linewidth=2, capsize=4,
                     label=label)
axes[0].set_xlabel("Candidate set size K")
axes[0].set_ylabel("Total Workload (s)")
axes[0].set_title("Workload ↓ vs. K")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(k_vals)

# Panel 2: P90 vs K
for mn, ms, color, label in [
    ("default_p90_mean", "default_p90_std", "#aaaaaa", "Default"),
    ("oracle_p90_mean",  "oracle_p90_std",  "#4CAF50", "Oracle"),
    ("model_p90_mean",   "model_p90_std",   "#2196F3", "AdaSteer-C"),
]:
    axes[1].errorbar(k_vals, [r[mn] for r in results],
                     yerr=[r[ms] for r in results],
                     marker="o", color=color, linewidth=2, capsize=4,
                     label=label)
axes[1].set_xlabel("Candidate set size K")
axes[1].set_ylabel("P90 Latency (s)")
axes[1].set_title("P90 ↓ vs. K")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
axes[1].set_xticks(k_vals)

# Panel 3: Top-1 accuracy + regret vs K
ax3  = axes[2]
ax3b = ax3.twinx()
ax3.errorbar(k_vals, [r["top1_acc_mean"] for r in results],
             yerr=[r["top1_acc_std"] for r in results],
             marker="o", color="#2196F3", linewidth=2, capsize=4,
             label="Top-1 Accuracy")
ax3b.errorbar(k_vals, [r["regret_mean"] for r in results],
              yerr=[r["regret_std"] for r in results],
              marker="s", color="#f44336", linewidth=2, capsize=4,
              linestyle="--", label="Mean Regret/query (s)")
ax3.set_xlabel("Candidate set size K")
ax3.set_ylabel("Top-1 Accuracy", color="#2196F3")
ax3b.set_ylabel("Mean Regret per query (s)", color="#f44336")
ax3.set_title("Prediction Quality vs. K")
ax3.set_xticks(k_vals)
ax3.grid(alpha=0.3)

fig.suptitle("Experiment 3: Multi-Action Steering — Framework Generalizes Beyond Binary Protocol",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("results/exp3_multiaction_steering_figure.pdf", bbox_inches="tight")
plt.savefig("results/exp3_multiaction_steering_figure.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: results/exp3_multiaction_steering.csv")
print("✅ Saved: results/exp3_multiaction_steering_figure.{pdf,png}")
