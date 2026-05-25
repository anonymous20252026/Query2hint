"""
Experiment 3b: Multi-Action Steering — Pairwise Condorcet Ranking
==================================================================
Motivation: AdaSteer-C was trained with pairwise execution-preference
supervision ("does hint A run faster than hint B for query Q?").
Experiment 3 forced this into a multiclass classification task (argmin
label) — a mismatch that hurts at large K.

This experiment uses the same candidate sets as Exp 3 and replaces the
multiclass SVC with a Condorcet-vote pairwise ranker:
  - For each pair (i, j) in the K-hint candidate set, train a binary
    LinearSVC on "does hint i beat hint j for this query?"
  - At inference, tally pairwise wins per hint and select the
    Condorcet winner (most wins).

Reports both methods (Multiclass and Pairwise) side by side so the
gain from alignment with the training objective is directly visible.

Outputs:
  results/exp3b_pairwise_ranking.csv
  results/exp3b_pairwise_ranking_figure.pdf / .png
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
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
print("Experiment 3b: Multi-Action Steering — Pairwise Ranking")
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


print("\nLoading data and computing embeddings...")
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

all_alt_indices = list(range(1, n_hints))


# ── Candidate sets: identical sequence to Exp 3 ───────────────────────────────
# Exp 3 uses a single global rng(RANDOM_SEED) and calls rng.shuffle once per K.
# Reproduce that exact sequence here so the comparison is fair.
rng_sets = np.random.RandomState(RANDOM_SEED)
CANDIDATE_SETS = {}
for k in K_SIZES:
    alts_pool = [26] + [i for i in all_alt_indices if i != 26]
    rng_sets.shuffle(alts_pool[1:])
    CANDIDATE_SETS[k] = [BENCHMARK_IDX] + alts_pool[: k - 1]
print(f"\nCandidate sets (same as Exp 3):")
for k, cs in CANDIDATE_SETS.items():
    print(f"  K={k:2d}: {cs}")


# ── Feature preprocessing (shared within a fold) ──────────────────────────────

def preprocess(X_tr, X_te, n_comp):
    sc1 = StandardScaler().fit(X_tr)
    Xtr = sc1.transform(X_tr)
    Xte = sc1.transform(X_te)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit(Xtr)
    Xtr = pca.transform(Xtr)
    Xte = pca.transform(Xte)
    sc2 = StandardScaler().fit(Xtr)
    return sc2.transform(Xtr), sc2.transform(Xte)


# ── Multiclass SVC (mirrors Exp 3 exactly) ────────────────────────────────────

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


# ── Pairwise Condorcet ranker ─────────────────────────────────────────────────

def condorcet_predict(X_tr_f, X_te_f, hint_l_sub_tr, k_size):
    """
    Train one LinearSVC per pair (i, j); label = 1 if hint i is faster.
    Return Condorcet winner (local index in candidate set) for each test query.
    """
    pairs = list(itertools.combinations(range(k_size), 2))
    votes = np.zeros((len(X_te_f), k_size), dtype=np.float32)

    for (li, lj) in pairs:
        y = (hint_l_sub_tr[:, li] < hint_l_sub_tr[:, lj]).numpy().astype(np.float64)
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if n_pos == 0 or n_neg == 0:
            # Degenerate pair — award the vote to the globally faster hint
            winner = li if hint_l_sub_tr[:, li].mean() < hint_l_sub_tr[:, lj].mean() else lj
            votes[:, winner] += 1.0
            continue

        cw = {0: float(n_pos) / n_neg, 1: float(n_neg) / n_pos}
        clf = LinearSVC(max_iter=3000, random_state=RANDOM_SEED, class_weight=cw)
        clf.fit(X_tr_f, y)
        pred = clf.predict(X_te_f)   # 1 → li wins, 0 → lj wins
        votes[:, li] += pred
        votes[:, lj] += (1.0 - pred)

    return votes.argmax(axis=1)


# ── Centroid similarity ranking ───────────────────────────────────────────────

def centroid_predict(X_tr, X_te, class_labels_tr, k_size):
    """
    Zero-extra-training baseline: compute per-hint prototype as the mean
    embedding of training queries for which that hint achieves the lowest
    runtime (i.e., the argmin label). Select the hint whose centroid is
    closest (cosine similarity) to the test query embedding.

    This tests whether AdaSteer-C's representation space already encodes
    hint preferences without any downstream classifier training.
    """
    centroids = np.zeros((k_size, X_tr.shape[1]), dtype=np.float32)
    for local_h in range(k_size):
        idx = np.where(class_labels_tr == local_h)[0]
        if len(idx) == 0:
            # No training query is best served by this hint — use global mean
            centroids[local_h] = X_tr.mean(axis=0)
        else:
            centroids[local_h] = X_tr[idx].mean(axis=0)

    # Cosine similarity: normalise both sides then dot product
    def l2norm(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-8)
        return M / norms

    C = l2norm(centroids)       # [k_size, D]
    Q = l2norm(X_te)            # [n_test, D]
    sims = Q @ C.T              # [n_test, k_size]
    return sims.argmax(axis=1)  # local index of nearest centroid


# ── Per-K evaluation ──────────────────────────────────────────────────────────

def run_for_k(k_size):
    candidate_set = CANDIDATE_SETS[k_size]
    hint_l_sub    = hint_l[:, candidate_set]           # [N, k_size]
    class_labels  = hint_l_sub.argmin(dim=1).numpy()   # multiclass labels

    X_np    = X_base.numpy()
    n_comp  = min(120, k_size * 15)
    sub_mean = hint_l_sub.mean(dim=0)
    best_fixed_in_set = sub_mean.argmin().item()

    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    keys = ["default", "best_fixed", "random", "oracle",
            "mc", "pw", "cen",                   # multiclass / pairwise / centroid
            "default_p90", "best_fixed_p90", "random_p90", "oracle_p90",
            "mc_p90", "pw_p90", "cen_p90",
            "top1_mc", "top1_pw", "top1_cen",
            "regret_mc", "regret_pw", "regret_cen"]
    m = {k: [] for k in keys}

    n_pairs = k_size * (k_size - 1) // 2
    print(f"\n  K={k_size}  candidate_set={candidate_set}  pairs={n_pairs}")

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_np, class_labels)):
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr = class_labels[tr_idx]

        # ── Multiclass SVC ──
        clf_mc = make_multiclass_svc(k_size)
        clf_mc.fit(X_tr, y_tr)
        y_mc = clf_mc.predict(X_te)

        # ── Pairwise Condorcet ──
        X_tr_f, X_te_f = preprocess(X_tr, X_te, n_comp)
        y_pw  = condorcet_predict(X_tr_f, X_te_f, hint_l_sub[tr_idx], k_size)

        # ── Centroid similarity (zero extra training) ──
        y_cen = centroid_predict(X_tr, X_te, y_tr, k_size)

        # ── Baselines ──
        oracle_rt  = hint_l_sub[te_idx].min(dim=1).values
        oracle_lbl = hint_l_sub[te_idx].argmin(dim=1).numpy()
        default_rt = hint_l_sub[te_idx, 0]
        bf_rt      = hint_l_sub[te_idx, best_fixed_in_set]

        rand_rts = []
        for s in RANDOM_SEEDS_FOR_RANDOM_BASELINE:
            rc = np.random.RandomState(s).randint(0, k_size, len(te_idx))
            rand_rts.append(
                hint_l_sub[te_idx].gather(
                    1, torch.tensor(rc, dtype=torch.long).view(-1, 1)
                ).squeeze(1))
        random_rt = torch.stack(rand_rts).mean(dim=0)

        def gather_rt(y_pred):
            return hint_l_sub[te_idx].gather(
                1, torch.tensor(y_pred, dtype=torch.long).view(-1, 1)
            ).squeeze(1)

        mc_rt  = gather_rt(y_mc)
        pw_rt  = gather_rt(y_pw)
        cen_rt = gather_rt(y_cen)

        m["default"].append(default_rt.sum().item())
        m["default_p90"].append(default_rt.quantile(0.90).item())
        m["best_fixed"].append(bf_rt.sum().item())
        m["best_fixed_p90"].append(bf_rt.quantile(0.90).item())
        m["random"].append(random_rt.sum().item())
        m["random_p90"].append(random_rt.quantile(0.90).item())
        m["oracle"].append(oracle_rt.sum().item())
        m["oracle_p90"].append(oracle_rt.quantile(0.90).item())
        m["mc"].append(mc_rt.sum().item())
        m["mc_p90"].append(mc_rt.quantile(0.90).item())
        m["pw"].append(pw_rt.sum().item())
        m["pw_p90"].append(pw_rt.quantile(0.90).item())
        m["cen"].append(cen_rt.sum().item())
        m["cen_p90"].append(cen_rt.quantile(0.90).item())
        m["top1_mc"].append((y_mc  == oracle_lbl).mean())
        m["top1_pw"].append((y_pw  == oracle_lbl).mean())
        m["top1_cen"].append((y_cen == oracle_lbl).mean())
        m["regret_mc"].append((mc_rt  - oracle_rt).mean().item())
        m["regret_pw"].append((pw_rt  - oracle_rt).mean().item())
        m["regret_cen"].append((cen_rt - oracle_rt).mean().item())

        print(f"    fold {fold+1:2d}  MC={mc_rt.sum():.0f}s  "
              f"PW={pw_rt.sum():.0f}s  "
              f"Cen={cen_rt.sum():.0f}s  "
              f"BF={bf_rt.sum():.0f}s  "
              f"Oracle={oracle_rt.sum():.0f}s")

    result = {"k_size": k_size}
    for key, vals in m.items():
        result[f"{key}_mean"] = np.mean(vals)
        result[f"{key}_std"]  = np.std(vals)
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

print("\nRunning experiment (both methods per K)...")
results = []
for k in K_SIZES:
    r = run_for_k(k)
    results.append(r)
    print(f"\n  K={k} summary:")
    print(f"    Default    : {r['default_mean']:.1f}s")
    print(f"    Best Fixed : {r['best_fixed_mean']:.1f}s")
    print(f"    Centroid   : {r['cen_mean']:.1f}±{r['cen_std']:.0f}s  "
          f"Top-1={r['top1_cen_mean']:.3f}  Regret={r['regret_cen_mean']:.2f}s/q")
    print(f"    Multiclass : {r['mc_mean']:.1f}±{r['mc_std']:.0f}s  "
          f"Top-1={r['top1_mc_mean']:.3f}  Regret={r['regret_mc_mean']:.2f}s/q")
    print(f"    Pairwise   : {r['pw_mean']:.1f}±{r['pw_std']:.0f}s  "
          f"Top-1={r['top1_pw_mean']:.3f}  Regret={r['regret_pw_mean']:.2f}s/q")
    print(f"    Oracle     : {r['oracle_mean']:.1f}s")

df_res = pd.DataFrame(results)
df_res.to_csv("results/exp3b_pairwise_ranking.csv", index=False)

# ── Results table ─────────────────────────────────────────────────────────────
k_vals = [r["k_size"] for r in results]
print("\n" + "=" * 90)
print("PAIRWISE RANKING vs. MULTICLASS SVC")
print("=" * 90)
hdr = (f"{'K':>4} | {'Default':>10} | {'BestFixed':>10} | "
       f"{'Multiclass':>14} | {'Pairwise':>14} | {'Oracle':>10} | "
       f"{'Top1-MC':>7} | {'Top1-PW':>7}")
print(hdr)
print("-" * 90)
for r in results:
    print(f"{r['k_size']:>4} | "
          f"{r['default_mean']:>8.1f}   | "
          f"{r['best_fixed_mean']:>8.1f}   | "
          f"{r['mc_mean']:>8.1f}±{r['mc_std']:>4.0f} | "
          f"{r['pw_mean']:>8.1f}±{r['pw_std']:>4.0f} | "
          f"{r['oracle_mean']:>8.1f}   | "
          f"{r['top1_mc_mean']:>7.3f} | "
          f"{r['top1_pw_mean']:>7.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

colors = {
    "default":    "#aaaaaa",
    "best_fixed": "#FF9800",
    "oracle":     "#4CAF50",
    "mc":         "#2196F3",
    "pw":         "#9C27B0",
    "cen":        "#f44336",
    "random":     "#f4a261",
}

# Panel 1: Total workload vs K
ax = axes[0]
ax.errorbar(k_vals, [r["default_mean"]    for r in results],
            yerr=[r["default_std"]    for r in results],
            marker="o", color=colors["default"],    linewidth=2,
            capsize=4, label="Default PostgreSQL")
ax.errorbar(k_vals, [r["best_fixed_mean"] for r in results],
            yerr=[r["best_fixed_std"] for r in results],
            marker="s", color=colors["best_fixed"], linewidth=2,
            capsize=4, label="Best Fixed")
ax.errorbar(k_vals, [r["oracle_mean"]     for r in results],
            yerr=[r["oracle_std"]     for r in results],
            marker="^", color=colors["oracle"],     linewidth=2,
            capsize=4, linestyle="--", label="Oracle best-in-set")
ax.errorbar(k_vals, [r["cen_mean"]        for r in results],
            yerr=[r["cen_std"]        for r in results],
            marker="v", color=colors["cen"],        linewidth=2,
            capsize=4, label="Centroid Similarity (no training)")
ax.errorbar(k_vals, [r["mc_mean"]         for r in results],
            yerr=[r["mc_std"]         for r in results],
            marker="o", color=colors["mc"],         linewidth=2,
            capsize=4, label="Multiclass SVC")
ax.errorbar(k_vals, [r["pw_mean"]         for r in results],
            yerr=[r["pw_std"]         for r in results],
            marker="D", color=colors["pw"],         linewidth=2,
            capsize=4, label="Pairwise Condorcet")
ax.set_xlabel("Candidate set size K")
ax.set_ylabel("Total Workload (s)")
ax.set_title("Total Workload ↓ vs. K")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_xticks(k_vals)

# Panel 2: P90 vs K
ax = axes[1]
for key, label in [("default_p90",    "Default"),
                   ("best_fixed_p90", "Best Fixed"),
                   ("oracle_p90",     "Oracle"),
                   ("cen_p90",        "Centroid Similarity"),
                   ("mc_p90",         "Multiclass SVC"),
                   ("pw_p90",         "Pairwise Condorcet")]:
    mk  = "D" if "pw" in key else ("v" if "cen" in key else ("^" if "oracle" in key else "o"))
    ls  = "--" if "oracle" in key else "-"
    raw = key.replace("_p90", "")
    col = colors.get(raw, "#aaaaaa")
    ax.errorbar(k_vals, [r[f"{key}_mean"] for r in results],
                yerr=[r[f"{key}_std"] for r in results],
                marker=mk, color=col, linewidth=2,
                capsize=4, linestyle=ls, label=label)
ax.set_xlabel("Candidate set size K")
ax.set_ylabel("P90 Latency (s)")
ax.set_title("P90 Latency ↓ vs. K")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_xticks(k_vals)

# Panel 3: Top-1 accuracy comparison
ax = axes[2]
ax.errorbar(k_vals, [r["top1_cen_mean"] for r in results],
            yerr=[r["top1_cen_std"] for r in results],
            marker="v", color=colors["cen"], linewidth=2, capsize=4,
            label="Centroid Similarity")
ax.errorbar(k_vals, [r["top1_mc_mean"] for r in results],
            yerr=[r["top1_mc_std"] for r in results],
            marker="o", color=colors["mc"], linewidth=2, capsize=4,
            label="Multiclass SVC")
ax.errorbar(k_vals, [r["top1_pw_mean"] for r in results],
            yerr=[r["top1_pw_std"] for r in results],
            marker="D", color=colors["pw"], linewidth=2, capsize=4,
            label="Pairwise Condorcet")
ax.set_xlabel("Candidate set size K")
ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Top-1 Accuracy vs. K")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xticks(k_vals)

fig.suptitle(
    "Experiment 3b: Pairwise Condorcet Ranking vs. Multiclass SVC"
    " — Aligning Inference with Execution-Preference Supervision",
    fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("results/exp3b_pairwise_ranking_figure.pdf", bbox_inches="tight")
plt.savefig("results/exp3b_pairwise_ranking_figure.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: results/exp3b_pairwise_ranking.csv")
print("✅ Saved: results/exp3b_pairwise_ranking_figure.{pdf,png}")
