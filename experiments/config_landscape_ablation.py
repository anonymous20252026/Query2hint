"""
Experiment 2: Configuration Landscape Ablation
================================================
Validates that AdaSteer benefits from learning over a RICHER configuration
landscape, not merely from using a fine-tuned encoder.

Methodology:
  Fix the encoder (AdaSteer-C, pre-trained).
  Vary the number of hint configurations |H| used to:
    (a) define the "best alternative" binary steering target per query,
    (b) measure downstream binary steering workload quality.

  For each |H| in {2, 4, 8, 16, 32, 47}:
    - The binary label for a query = "is the best hint among |H| configs
      better than the default (hint-0)?"
    - The model learns: "should I steer away from default?"
    - At test time: select the globally best alternative found in the |H| set.

  Larger |H| → better oracle alternative → richer supervision signal.
  Key question: does steering quality improve monotonically with |H|?

Also reports per-|H|:
  - Oracle workload (if we always pick the best of the |H| configs)
  - Model workload (SVC-120-S prediction)
  - Supervision coverage (fraction of queries where an alternative beats default)

Outputs:
  results/exp2_config_landscape.csv
  results/exp2_config_landscape_figure.pdf / .png
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
from sklearn.metrics import roc_auc_score, f1_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 24508
BENCHMARK_IDX = 0    # default PostgreSQL hint (always included)
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
THRESHOLD     = 0.5
DEVICE        = "cuda"
ENCODER_PATH  = "encoders/encoder_all-mpnet-base-v2_v1"  # AdaSteer-C

H_SIZES = [2, 4, 8, 16, 32, 47]

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. This experiment requires a CUDA GPU.")

print("=" * 65)
print("Experiment 2: Configuration Landscape Ablation")
print("=" * 65)
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"Encoder        : {ENCODER_PATH}")
print(f"H sizes tested : {H_SIZES}")


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
    df["opt_l"] = df["mean_runtime"].apply(min)
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

# All non-default hint indices (exclude index 0 = default)
all_alt_indices = list(range(1, n_hints))
rng = np.random.RandomState(RANDOM_SEED)


# ── Per-|H| evaluation ───────────────────────────────────────────────────────

def make_svc120s():
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=120, random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        random_state=RANDOM_SEED)),
    ])


def run_for_h(h_size):
    """
    For a given |H|, select h_size-1 alternative hint indices (+ default=0).
    Label a query positive if any alternative beats the default.
    The "best alternative" per query = argmin runtime in selected set.
    Train SVC-120-S, report model/oracle workload and P90.
    """
    # Always include hint-0 (default) and hint-26 (canonical alternative)
    # Fill remaining slots from shuffled pool
    fixed_alts = [26]
    remaining  = [i for i in all_alt_indices if i != 26]
    rng.shuffle(remaining)
    selected_alts = (fixed_alts + remaining)[: h_size - 1]
    candidate_set = [BENCHMARK_IDX] + selected_alts   # size = h_size

    # hint_l_sub : [n_queries, h_size]
    hint_l_sub = hint_l[:, candidate_set]

    # Best alternative for each query (among non-default slots)
    alt_runtimes   = hint_l_sub[:, 1:]            # [N, h_size-1]
    best_alt_rt, best_alt_local = alt_runtimes.min(dim=1)  # best in alt set
    default_rt     = hint_l_sub[:, 0]

    # Binary label: 1 = at least one alternative beats default
    binary_l = (default_rt > best_alt_rt).float()
    coverage = binary_l.mean().item()

    # Steering choice: if label=1, use best alternative; else default
    # We map back to global hint_l using best_alt_local → selected_alts index
    best_alt_global_idx = torch.tensor(selected_alts, dtype=torch.long)[best_alt_local]

    X_np = X_base.numpy()
    y    = binary_l.numpy()
    benchmark_t = torch.LongTensor([BENCHMARK_IDX])

    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    workloads, p90s, oracle_workloads, oracle_p90s = [], [], [], []
    aurocs, f1s = [], []

    for tr_idx, te_idx in splitter.split(X_np, y):
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        n_pos = max((y_tr == 1).sum(), 1)
        n_neg = max((y_tr == 0).sum(), 1)
        cw    = {0: float(n_pos) / n_neg, 1: float(n_neg) / n_pos}

        clf = make_svc120s()
        clf.named_steps["svc"].set_params(class_weight=cw)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # Model workload: if predicted =1, use best alt; else default
        best_alt_te = best_alt_global_idx[te_idx]
        chosen_rt = torch.where(
            torch.tensor(y_pred, dtype=torch.bool),
            hint_l[te_idx].gather(1, best_alt_te.view(-1, 1)).squeeze(1),
            hint_l[te_idx, BENCHMARK_IDX]
        )
        workloads.append(chosen_rt.sum().item())
        p90s.append(chosen_rt.quantile(0.90).item())

        # Oracle workload: always pick the true best among candidate_set
        oracle_rt = hint_l_sub[te_idx].min(dim=1).values
        oracle_workloads.append(oracle_rt.sum().item())
        oracle_p90s.append(oracle_rt.quantile(0.90).item())

        if len(np.unique(y_te)) > 1:
            aurocs.append(roc_auc_score(y_te, y_prob))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "h_size"               : h_size,
        "coverage"             : coverage,
        "workload_mean"        : np.mean(workloads),
        "workload_std"         : np.std(workloads),
        "p90_mean"             : np.mean(p90s),
        "p90_std"              : np.std(p90s),
        "oracle_workload_mean" : np.mean(oracle_workloads),
        "oracle_workload_std"  : np.std(oracle_workloads),
        "oracle_p90_mean"      : np.mean(oracle_p90s),
        "oracle_p90_std"       : np.std(oracle_p90s),
        "auroc_mean"           : np.mean(aurocs) if aurocs else float("nan"),
        "auroc_std"            : np.std(aurocs)  if aurocs else 0,
        "f1_mean"              : np.mean(f1s)    if f1s    else float("nan"),
        "f1_std"               : np.std(f1s)     if f1s    else 0,
    }


results = []
for h in H_SIZES:
    print(f"\nRunning |H| = {h} ...")
    r = run_for_h(h)
    results.append(r)
    print(f"  Coverage: {r['coverage']:.3f} | "
          f"Model workload: {r['workload_mean']:.1f}±{r['workload_std']:.0f}s | "
          f"Oracle: {r['oracle_workload_mean']:.1f}s | "
          f"AUROC: {r['auroc_mean']:.4f}")

# LLMSteer reference
llmsteer_workload = 2547.7
llmsteer_p90      = 5.7

df_res = pd.DataFrame(results)
df_res.to_csv("results/exp2_config_landscape.csv", index=False)

print("\n" + "=" * 75)
print("CONFIG LANDSCAPE ABLATION RESULTS")
print("=" * 75)
print(f"{'|H|':>5} | {'Coverage':>8} | {'Model Wkld (s)':>16} | {'Oracle Wkld (s)':>17} | {'AUROC':>8}")
print("-" * 75)
for r in results:
    print(f"{r['h_size']:>5} | {r['coverage']:>8.3f} | "
          f"{r['workload_mean']:>8.1f}±{r['workload_std']:>5.0f} | "
          f"{r['oracle_workload_mean']:>9.1f}±{r['oracle_workload_std']:>5.0f} | "
          f"{r['auroc_mean']:>8.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
h_vals = [r["h_size"] for r in results]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel 1: Model workload vs |H|
axes[0].errorbar(h_vals, [r["workload_mean"] for r in results],
                 yerr=[r["workload_std"] for r in results],
                 marker="o", color="#2196F3", linewidth=2, capsize=5,
                 label="AdaSteer-C (model)")
axes[0].errorbar(h_vals, [r["oracle_workload_mean"] for r in results],
                 yerr=[r["oracle_workload_std"] for r in results],
                 marker="s", color="#4CAF50", linewidth=2, capsize=5,
                 linestyle="--", label="Oracle (best-in-set)")
axes[0].axhline(llmsteer_workload, color="grey", linestyle=":", linewidth=1.5,
                label="LLMSteer (published)")
axes[0].set_xlabel("Number of configurations |H|")
axes[0].set_ylabel("Total Workload (s)")
axes[0].set_title("Total Workload vs. |H| (non-monotonic)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(h_vals)

# Panel 2: P90 vs |H|
axes[1].errorbar(h_vals, [r["p90_mean"] for r in results],
                 yerr=[r["p90_std"] for r in results],
                 marker="o", color="#2196F3", linewidth=2, capsize=5,
                 label="AdaSteer-C (model)")
axes[1].errorbar(h_vals, [r["oracle_p90_mean"] for r in results],
                 yerr=[r["oracle_p90_std"] for r in results],
                 marker="s", color="#4CAF50", linewidth=2, capsize=5,
                 linestyle="--", label="Oracle")
axes[1].axhline(llmsteer_p90, color="grey", linestyle=":", linewidth=1.5)
axes[1].set_xlabel("Number of configurations |H|")
axes[1].set_ylabel("P90 Latency (s)")
axes[1].set_title("P90 Latency vs. |H| (oracle monotone ↓)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
axes[1].set_xticks(h_vals)

# Panel 3: Coverage + AUROC vs |H|
ax3 = axes[2]
ax3b = ax3.twinx()
ax3.plot(h_vals, [r["coverage"] for r in results],
         marker="^", color="#FF9800", linewidth=2, label="Supervision coverage")
ax3b.errorbar(h_vals, [r["auroc_mean"] for r in results],
              yerr=[r["auroc_std"] for r in results],
              marker="o", color="#9C27B0", linewidth=2, capsize=5,
              label="AUROC")
ax3.set_xlabel("Number of configurations |H|")
ax3.set_ylabel("Coverage (fraction positive)", color="#FF9800")
ax3b.set_ylabel("AUROC", color="#9C27B0")
ax3.set_title("Coverage ↑ and AUROC recover at |H|=47")
ax3.set_xticks(h_vals)
ax3.grid(alpha=0.3)

fig.suptitle("Experiment 2: Configuration-Space Richness — Coverage and Oracle Improve; Model Performance is Non-Monotonic",
             fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("results/exp2_config_landscape_figure.pdf", bbox_inches="tight")
plt.savefig("results/exp2_config_landscape_figure.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: results/exp2_config_landscape.csv")
print("✅ Saved: results/exp2_config_landscape_figure.{pdf,png}")
