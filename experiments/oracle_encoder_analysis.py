# ============================================================
# experiments_oracle_analysis.py
#
# PURPOSE:
#   Explain why AdaSteer-O (oracle-initialized, 2557s) underperforms
#   AdaSteer-R (Reptile, 2433s) despite cleaner source supervision.
#
#   Hypotheses tested:
#     H1: Oracle encoder is more source-workload (CEB) specific —
#         lower cosine similarity to JOB query representations from
#         a general encoder.
#     H2: Oracle encoder has higher CEB embedding concentration
#         (lower intra-workload variance) → less generalizable manifold.
#     H3: Oracle per-fold workload has higher variance than Reptile
#         → Oracle is more overfit to specific fold compositions.
#
# USES:
#   - Existing encoders: contrastive, reptile, oracle
#   - data/job.csv, data/ceb.csv (for workload-level analysis)
#   - results/llmsteer_pipeline_*.csv (per-fold workloads)
#
# OUTPUT:
#   results_v3/oracle_analysis.csv         — embedding similarity stats
#   results_v3/oracle_fold_comparison.csv  — per-fold workload comparison
#   results_v3/oracle_analysis_figure.pdf  — PCA visualization
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("results_v3", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CONTRASTIVE_PATH = "encoders/encoder_all-mpnet-base-v2_v1"
REPTILE_PATH     = "encoders/encoder_reptile_mpnet_v4"
ORACLE_PATH      = "adasteer_encoder"
JOB_DATA         = "data/job.csv"
CEB_DATA         = "data/ceb.csv"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("experiments_oracle_analysis.py — Oracle vs Reptile Analysis")
print("=" * 64)
print(f"Device      : {DEVICE}")
print(f"Contrastive : {CONTRASTIVE_PATH}")
print(f"Reptile     : {REPTILE_PATH}")
print(f"Oracle      : {ORACLE_PATH}")
print("=" * 64)

# ── Load queries ──────────────────────────────────────────────────────────────
print("\n[1/5] Loading queries...")
job_df = pd.read_csv(JOB_DATA, converters={"hint_list": eval, "runtime_list": eval})
ceb_df = pd.read_csv(CEB_DATA, converters={"hint_list": eval, "runtime_list": eval})

job_sqls = job_df["sql"].drop_duplicates().tolist()
ceb_sqls = ceb_df["sql"].drop_duplicates().tolist()
all_sqls = list(set(job_sqls + ceb_sqls))

print(f"  JOB queries : {len(job_sqls)}")
print(f"  CEB queries : {len(ceb_sqls)}")
print(f"  Unique total: {len(all_sqls)}")

# ── Encode with all three encoders ───────────────────────────────────────────
print("\n[2/5] Encoding with all three encoders...")
results_emb = {}
for name, path in [("Contrastive", CONTRASTIVE_PATH),
                    ("Reptile",     REPTILE_PATH),
                    ("Oracle",      ORACLE_PATH)]:
    model = SentenceTransformer(path, device=DEVICE)
    embs  = model.encode(all_sqls, batch_size=128, show_progress_bar=False,
                          normalize_embeddings=True)
    results_emb[name] = {sql: embs[i] for i, sql in enumerate(all_sqls)}
    print(f"  {name:12s}: dim={embs.shape[1]}")

# ── H1: Cross-workload embedding similarity ───────────────────────────────────
print("\n[3/5] Testing H1: Workload specificity (CEB vs JOB cosine similarity)...")

analysis_rows = []

for enc_name, emb_dict in results_emb.items():
    job_embs = np.stack([emb_dict[s] for s in job_sqls if s in emb_dict])
    ceb_embs = np.stack([emb_dict[s] for s in ceb_sqls if s in emb_dict])

    # Mean intra-workload similarity
    job_t = torch.tensor(job_embs)
    ceb_t = torch.tensor(ceb_embs)

    # JOB centroid vs CEB centroid distance (lower = more overlap = better transfer)
    job_centroid = job_t.mean(0)
    ceb_centroid = ceb_t.mean(0)
    cross_sim = F.cosine_similarity(job_centroid.unsqueeze(0),
                                     ceb_centroid.unsqueeze(0)).item()

    # Intra-JOB variance (pairwise cosine sim mean)
    n_sample = min(200, len(job_sqls))
    sample_idx = np.random.choice(len(job_sqls), n_sample, replace=False)
    job_sample = job_t[sample_idx]
    sim_matrix = torch.mm(job_sample, job_sample.T)
    # Exclude diagonal
    mask = ~torch.eye(n_sample, dtype=bool)
    intra_job_sim = sim_matrix[mask].mean().item()

    # Intra-CEB variance
    n_sample_ceb = min(200, len(ceb_sqls))
    sample_idx_c = np.random.choice(len(ceb_sqls), n_sample_ceb, replace=False)
    ceb_sample = ceb_t[sample_idx_c]
    sim_matrix_c = torch.mm(ceb_sample, ceb_sample.T)
    mask_c = ~torch.eye(n_sample_ceb, dtype=bool)
    intra_ceb_sim = sim_matrix_c[mask_c].mean().item()

    print(f"  {enc_name:12s}: cross-workload sim={cross_sim:.4f}  "
          f"intra-JOB sim={intra_job_sim:.4f}  intra-CEB sim={intra_ceb_sim:.4f}")

    analysis_rows.append({
        "encoder": enc_name,
        "cross_workload_sim": cross_sim,
        "intra_job_sim": intra_job_sim,
        "intra_ceb_sim": intra_ceb_sim,
    })

# ── H3: Per-fold workload variance from existing results ─────────────────────
print("\n[4/5] Testing H3: Per-fold workload stability...")

fold_comp_rows = []
for enc_name, csv_name in [("Contrastive", "AdaSteer-Contrastive"),
                              ("Reptile",     "AdaSteer-Reptile"),
                              ("Full",        "AdaSteer-Oracle")]:
    csv_path = f"results/llmsteer_pipeline_{csv_name}.csv"
    if not os.path.exists(csv_path):
        print(f"  {enc_name}: {csv_path} not found, skipping")
        continue

    df = pd.read_csv(csv_path)
    # Get best config (lowest workload mean)
    best = df.sort_values("test_model_workload_mean").iloc[0]

    def parse_folds(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return list(val) if hasattr(val, "__iter__") else [val]

    folds = parse_folds(best["test_model_workload"])
    mean_wl = np.mean(folds)
    std_wl  = np.std(folds)
    cv      = std_wl / mean_wl   # coefficient of variation

    print(f"  {enc_name:12s}: best_config={best['estimator'][:20]}  "
          f"wl={mean_wl:.2f}±{std_wl:.2f}s  CV={cv:.3f}")

    for i, fw in enumerate(folds):
        fold_comp_rows.append({
            "encoder": enc_name, "fold": i, "fold_workload": fw,
            "mean_wl": mean_wl, "std_wl": std_wl, "cv": cv,
        })

fold_df = pd.DataFrame(fold_comp_rows)
if len(fold_df) > 0:
    fold_df.to_csv("results_v3/oracle_fold_comparison.csv", index=False)
    print(f"  Saved → results_v3/oracle_fold_comparison.csv")

    # Levene test: do Oracle and Reptile have different variance?
    oracle_folds  = fold_df[fold_df["encoder"] == "Oracle"]["fold_workload"].values
    reptile_folds = fold_df[fold_df["encoder"] == "Reptile"]["fold_workload"].values
    if len(oracle_folds) > 0 and len(reptile_folds) > 0:
        stat, p = stats.levene(oracle_folds, reptile_folds)
        print(f"\n  Levene test (Oracle vs Reptile fold variance): "
              f"stat={stat:.4f}  p={p:.4f}  "
              f"{'Oracle more variable' if oracle_folds.std() > reptile_folds.std() else 'Reptile more variable'}")
        analysis_rows[2]["fold_variance_test_p"] = p
        analysis_rows[2]["fold_std"] = float(oracle_folds.std())
        analysis_rows[1]["fold_variance_test_p"] = p
        analysis_rows[1]["fold_std"] = float(reptile_folds.std())

# ── Save analysis CSV ──────────────────────────────────────────────────────────
analysis_df = pd.DataFrame(analysis_rows)
analysis_df.to_csv("results_v3/oracle_analysis.csv", index=False)
print(f"\n  Saved → results_v3/oracle_analysis.csv")

# ── PCA visualization ─────────────────────────────────────────────────────────
print("\n[5/5] PCA visualization...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {"JOB": "#E64B35", "CEB": "#4DBBD5"}

for ax, (enc_name, emb_dict) in zip(axes, results_emb.items()):
    job_e = np.stack([emb_dict[s] for s in job_sqls[:200] if s in emb_dict])
    ceb_e = np.stack([emb_dict[s] for s in ceb_sqls[:200] if s in emb_dict])

    combined = np.vstack([job_e, ceb_e])
    sc = StandardScaler()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(sc.fit_transform(combined))

    n_job = len(job_e)
    ax.scatter(coords[:n_job, 0], coords[:n_job, 1],
               c=colors["JOB"], alpha=0.4, s=10, label="JOB")
    ax.scatter(coords[n_job:, 0], coords[n_job:, 1],
               c=colors["CEB"], alpha=0.4, s=10, label="CEB")

    ax.set_title(f"AdaSteer-{enc_name[0]}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Embedding Space: JOB vs CEB workloads (PCA, first 200 queries each)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("results_v3/oracle_analysis_figure.pdf", bbox_inches="tight")
plt.savefig("results_v3/oracle_analysis_figure.png", dpi=150, bbox_inches="tight")
print("  Saved → results_v3/oracle_analysis_figure.pdf")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("ANALYSIS SUMMARY")
print("=" * 64)
print("\n  H1: Cross-workload embedding similarity")
print(f"  {'Encoder':12s}  {'Cross-sim':10s}  {'Intra-JOB sim':14s}  {'Intra-CEB sim':14s}")
for row in analysis_rows:
    print(f"  {row['encoder']:12s}  {row['cross_workload_sim']:10.4f}  "
          f"{row['intra_job_sim']:14.4f}  {row['intra_ceb_sim']:14.4f}")

if len(fold_df) > 0 and len(oracle_folds) > 0 and len(reptile_folds) > 0:
    print(f"\n  H3: Fold-level variance")
    print(f"  Oracle  std={oracle_folds.std():.2f}s  CV={oracle_folds.std()/oracle_folds.mean():.3f}")
    print(f"  Reptile std={reptile_folds.std():.2f}s  CV={reptile_folds.std()/reptile_folds.mean():.3f}")

print("\nInterpretation guidance:")
print("  If Oracle cross-workload-sim < Reptile → Oracle more workload-specific (H1 confirmed)")
print("  If Oracle intra-CEB-sim > Reptile     → Oracle more CEB-concentrated (H2 confirmed)")
print("  If Oracle fold CV > Reptile fold CV   → Oracle less stable (H3 confirmed)")
print("=" * 64)
print("Done — results in results_v3/")
