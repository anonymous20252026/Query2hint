"""
Experiment 1b: Encoder Ablation under Pairwise Condorcet Ranking
=================================================================
Tests whether AdaSteer-C's multi-config supervised encoder provides
advantage over off-the-shelf MPNet-Raw under the pairwise Condorcet
ranking protocol (the main AdaSteer inference method).

Uses the SAME candidate sets and SAME fold splits as exp3b so results
are directly comparable.

Encoders tested:
  MPNet-Raw    : sentence-transformers/all-mpnet-base-v2 (no training)
  MPNet-Binary : encoders/encoder_mpnet_binary_supervised
  AdaSteer-C   : encoders/encoder_all-mpnet-base-v2_v1

K sizes: {2, 4, 8, 16}  (same as exp3b)
Protocol: pairwise Condorcet ranking (same as exp3b)
"""

import os, warnings, gc, itertools
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

RANDOM_SEED = 24508
BENCHMARK_IDX = 0
K_FOLDS = 10
TRAIN_SIZE = 0.8
DEVICE = "cuda"
K_SIZES = [2, 4, 8, 16]

ENCODERS = {
    "MPNet-Raw":    "sentence-transformers/all-mpnet-base-v2",
    "MPNet-Binary": "encoders/encoder_mpnet_binary_supervised",
    "AdaSteer-C":   "encoders/encoder_all-mpnet-base-v2_v1",
}

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found.")

print("=" * 65)
print("Experiment 1b: Encoder Ablation under Pairwise Condorcet")
print("=" * 65)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Data ──────────────────────────────────────────────────────────────────────
job = pd.read_csv("data/job.csv", converters={"hint_list": eval, "runtime_list": eval})
ceb = pd.read_csv("data/ceb.csv", converters={"hint_list": eval, "runtime_list": eval})
data = pd.concat([job, ceb]).reset_index(drop=True)
data["mean_runtime"] = data["runtime_list"].apply(np.mean)
data["sql"] = data["sql"].apply(lambda x: x.strip("\n"))
df = data.copy().explode("hint_list").sort_values(["filename", "hint_list"])
df = df.groupby(["filename", "sql"], as_index=False).agg(
    hint_list=("hint_list", list), mean_runtime=("mean_runtime", list))
df = df.reset_index(drop=True)

sqls   = df["sql"].tolist()
hint_l = torch.stack(df["mean_runtime"].apply(torch.Tensor).tolist())
n_hints = len(df["mean_runtime"].iloc[0])
print(f"Queries: {len(df)}, Hint sets: {n_hints}")

all_alt_indices = list(range(1, n_hints))
rng = np.random.RandomState(RANDOM_SEED)


# ── Pairwise Condorcet ─────────────────────────────────────────────────────
def condorcet_predict(X_tr, X_te, hl_sub_tr, k_size):
    pairs = list(itertools.combinations(range(k_size), 2))
    votes = np.zeros((len(X_te), k_size), dtype=np.float32)
    sc1 = StandardScaler().fit(X_tr)
    Xtr1 = sc1.transform(X_tr)
    pca  = PCA(n_components=min(120, Xtr1.shape[1], len(X_tr)-1), random_state=RANDOM_SEED).fit(Xtr1)
    Xtr2 = pca.transform(Xtr1)
    sc2  = StandardScaler().fit(Xtr2)
    Xtr_f = sc2.transform(Xtr2)
    Xte_f = sc2.transform(pca.transform(sc1.transform(X_te)))
    for (li, lj) in pairs:
        y_pair = (hl_sub_tr[:, li] < hl_sub_tr[:, lj]).numpy()
        n1 = int(y_pair.sum()); n0 = int((~y_pair).sum())
        if n1 == 0 or n0 == 0:
            continue
        cw = {0: n1/n0, 1: n0/n1}
        clf = LinearSVC(max_iter=3000, random_state=RANDOM_SEED, class_weight=cw)
        clf.fit(Xtr_f, y_pair.astype(int))
        pred = clf.predict(Xte_f)
        votes[:, li] += pred; votes[:, lj] += (1.0 - pred)
    return votes.argmax(axis=1)


def run_encoder_for_k(embs, k_size, encoder_name):
    # Reproduce same candidate set as exp3b for this k_size
    rng2 = np.random.RandomState(RANDOM_SEED)  # fresh rng same seed
    alts_pool = [26] + [i for i in all_alt_indices if i != 26]
    rng2.shuffle(alts_pool[1:])
    candidate_set = [BENCHMARK_IDX] + alts_pool[:k_size-1]

    hint_l_sub  = hint_l[:, candidate_set]
    class_labels = hint_l_sub.argmin(dim=1).numpy()

    X_np = embs
    spl  = StratifiedShuffleSplit(n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    workloads, oracle_workloads = [], []
    for tr_idx, te_idx in spl.split(X_np, class_labels):
        pred = condorcet_predict(X_np[tr_idx], X_np[te_idx],
                                 hint_l_sub[tr_idx], k_size)
        chosen = hint_l_sub[te_idx].gather(
            1, torch.tensor(pred, dtype=torch.long).view(-1,1)).squeeze(1)
        workloads.append(chosen.sum().item())
        oracle_workloads.append(hint_l_sub[te_idx].min(dim=1).values.sum().item())

    return {
        "encoder": encoder_name,
        "k_size": k_size,
        "workload_mean": np.mean(workloads),
        "workload_std":  np.std(workloads),
        "oracle_mean":   np.mean(oracle_workloads),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
results = []
for enc_name, enc_path in ENCODERS.items():
    print(f"\n=== {enc_name} ===")
    st = SentenceTransformer(enc_path, device=DEVICE)
    st.max_seq_length = 512
    embs = st.encode(sqls, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    del st; gc.collect(); torch.cuda.empty_cache()
    print(f"  Embedding shape: {embs.shape}")

    for k in K_SIZES:
        r = run_encoder_for_k(embs, k, enc_name)
        results.append(r)
        print(f"  K={k}: workload={r['workload_mean']:.1f}±{r['workload_std']:.0f}s  "
              f"oracle={r['oracle_mean']:.1f}s")

df_res = pd.DataFrame(results)
df_res.to_csv("results/exp1b_encoder_ablation.csv", index=False)

# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ENCODER ABLATION — Pairwise Condorcet, 10-fold CV")
print(f"{'Encoder':<16} | {'K=2':>14} | {'K=4':>14} | {'K=8':>14} | {'K=16':>14}")
print("-" * 70)
for enc in ENCODERS:
    row = df_res[df_res["encoder"] == enc].sort_values("k_size")
    vals = [f"{r['workload_mean']:.0f}±{r['workload_std']:.0f}" for _, r in row.iterrows()]
    print(f"{enc:<16} | " + " | ".join(f"{v:>14}" for v in vals))
print(f"LLMSteer (ref)   |           — |           — |        2548 |           —")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"MPNet-Raw": "#f4a261", "MPNet-Binary": "#aaaaaa", "AdaSteer-C": "#2196F3"}
markers = {"MPNet-Raw": "o", "MPNet-Binary": "s", "AdaSteer-C": "^"}
k_vals = K_SIZES
for enc in ENCODERS:
    row = df_res[df_res["encoder"] == enc].sort_values("k_size")
    ax.errorbar(k_vals, row["workload_mean"], yerr=row["workload_std"],
                marker=markers[enc], color=colors[enc], linewidth=2, capsize=4,
                label=enc)
ax.axhline(2547.7, color="grey", linestyle=":", linewidth=1.5, label="LLMSteer (published)")
ax.set_xlabel("Candidate set size K"); ax.set_ylabel("Total Workload (s)")
ax.set_title("Encoder Ablation: Pairwise Condorcet Ranking\n(Lower is better)")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(K_SIZES)
plt.tight_layout()
plt.savefig("results/exp1b_encoder_ablation_figure.pdf", bbox_inches="tight")
plt.savefig("results/exp1b_encoder_ablation_figure.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: results/exp1b_encoder_ablation.csv")
print("✅ Saved: results/exp1b_encoder_ablation_figure.{pdf,png}")
