# ============================================================
# experiments_fewshot_nometa.py
#
# PURPOSE:
#   Add a "NoMeta-SVC" ablation baseline to the few-shot experiment.
#   Trains a fresh SVC on K JOB triplet pairs (contrastive embeddings)
#   WITHOUT any meta-learning initialization.
#
#   Uses existing pairwise_results_summary.csv for AdaSteer-C and -R.
#   Only runs the NoMeta-SVC evaluation (fast, no transformer fine-tuning).
#
# OUTPUT:
#   results_v3/fewshot_nometa_summary.csv — K × method AUROC comparison
# ============================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("results_v3", exist_ok=True)

CONTRASTIVE_PATH = "encoders/encoder_all-mpnet-base-v2_v1"
JOB_TRIPLETS     = "stage1_triplets_JOB.csv"
FEW_SHOT_SIZES   = [0, 5, 10, 20, 50]
SEEDS            = [42, 123, 456, 789, 999]
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("experiments_fewshot_nometa.py — NoMeta-SVC baseline")
print("=" * 64)
print(f"Device: {DEVICE}  |  Seeds: {SEEDS}  |  K: {FEW_SHOT_SIZES}")
print("=" * 64)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ── Load JOB triplets ─────────────────────────────────────────────────────────
print("\n[1/4] Loading JOB triplets...")
triplets = pd.read_csv(JOB_TRIPLETS)
print(f"  Triplet columns : {list(triplets.columns)}")
print(f"  Triplet count   : {len(triplets)}")

q_col = "query"    if "query"    in triplets.columns else triplets.columns[0]
p_col = "positive" if "positive" in triplets.columns else triplets.columns[1]
n_col = "negative" if "negative" in triplets.columns else triplets.columns[2]

unique_texts = list(set(
    triplets[q_col].tolist() + triplets[p_col].tolist() + triplets[n_col].tolist()
))
print(f"  Unique texts    : {len(unique_texts)}")

# ── Encode with contrastive encoder ──────────────────────────────────────────
print("\n[2/4] Encoding with Contrastive encoder...")
model = SentenceTransformer(CONTRASTIVE_PATH, device=DEVICE)
model.max_seq_length = 512
embs = model.encode(unique_texts, batch_size=64, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
emb_C = {t: embs[i] for i, t in enumerate(unique_texts)}
print(f"  dim={embs.shape[1]}")

def ranking_accuracy(emb_dict, triplet_df):
    correct = 0; scores_p = []; scores_n = []
    for _, row in triplet_df.iterrows():
        zq = torch.tensor(emb_dict[row[q_col]])
        zp = torch.tensor(emb_dict[row[p_col]])
        zn = torch.tensor(emb_dict[row[n_col]])
        sp = F.cosine_similarity(zq.unsqueeze(0), zp.unsqueeze(0)).item()
        sn = F.cosine_similarity(zq.unsqueeze(0), zn.unsqueeze(0)).item()
        correct += int(sp > sn)
        scores_p.append(sp); scores_n.append(sn)
    acc = correct / len(triplet_df)
    labels = [1]*len(scores_p) + [0]*len(scores_n)
    scores = scores_p + scores_n
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float("nan")
    return acc, auroc

def fewshot_svc_eval(emb_dict, triplet_df, K, seed):
    """NoMeta-SVC: train SVC on K triplet pairs; evaluate on rest."""
    set_seed(seed)
    if K == 0:
        return ranking_accuracy(emb_dict, triplet_df)

    idx = list(range(len(triplet_df)))
    np.random.shuffle(idx)
    train_df = triplet_df.iloc[idx[:K]]
    test_df  = triplet_df.iloc[idx[K:]] if len(idx) > K else triplet_df

    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        zq = emb_dict[row[q_col]]; zp = emb_dict[row[p_col]]; zn = emb_dict[row[n_col]]
        X_train.append(np.concatenate([zq, zp])); y_train.append(1)
        X_train.append(np.concatenate([zq, zn])); y_train.append(0)

    if len(set(y_train)) < 2:
        return ranking_accuracy(emb_dict, triplet_df)

    clf = SVC(kernel="rbf", probability=False, random_state=seed)
    clf.fit(np.array(X_train), np.array(y_train))

    correct = 0; pos_scores = []; neg_scores = []
    for _, row in test_df.iterrows():
        zq = emb_dict[row[q_col]]; zp = emb_dict[row[p_col]]; zn = emb_dict[row[n_col]]
        sp = clf.decision_function(np.concatenate([zq, zp]).reshape(1,-1))[0]
        sn = clf.decision_function(np.concatenate([zq, zn]).reshape(1,-1))[0]
        correct += int(sp > sn)
        pos_scores.append(sp); neg_scores.append(sn)

    acc = correct / len(test_df) if len(test_df) > 0 else float("nan")
    labels = [1]*len(pos_scores) + [0]*len(neg_scores)
    try:
        auroc = roc_auc_score(labels, pos_scores + neg_scores)
    except Exception:
        auroc = float("nan")
    return acc, auroc

# ── Run NoMeta-SVC evaluation ─────────────────────────────────────────────────
print("\n[3/4] Running NoMeta-SVC evaluation...")
records = []
for K in FEW_SHOT_SIZES:
    auroc_N_seeds = []
    for seed in SEEDS:
        _, auroc_N = fewshot_svc_eval(emb_C, triplets, K, seed)
        auroc_N_seeds.append(auroc_N)
        records.append({"K": K, "seed": seed, "auroc_N": auroc_N})
    print(f"  K={K:2d}  NoMeta-SVC={np.mean(auroc_N_seeds):.4f}±{np.std(auroc_N_seeds):.4f}")

raw_df = pd.DataFrame(records)

# ── Merge with existing pairwise results ─────────────────────────────────────
print("\n[4/4] Merging with existing AdaSteer-C/R results...")
pairwise = pd.read_csv("pairwise_results_summary.csv")

summary = raw_df.groupby("K").agg(
    auroc_N_mean=("auroc_N", "mean"),
    auroc_N_std=("auroc_N", "std"),
).reset_index()

merged = summary.merge(
    pairwise[["n_shots","auroc_C_mean","auroc_C_std","auroc_R_mean","auroc_R_std"]],
    left_on="K", right_on="n_shots", how="left"
).drop(columns=["n_shots"])

merged.to_csv("results_v3/fewshot_nometa_summary.csv", index=False)
print(f"  Saved → results_v3/fewshot_nometa_summary.csv")

print("\nAUROC Comparison:")
print(f"{'K':>4}  {'AdaSteer-C':>12}  {'NoMeta-SVC':>12}  {'AdaSteer-R':>12}")
print("-" * 50)
for _, row in merged.iterrows():
    c = f"{row['auroc_C_mean']:.4f}±{row['auroc_C_std']:.4f}" if not pd.isna(row.get('auroc_C_mean')) else "—"
    n = f"{row['auroc_N_mean']:.4f}±{row['auroc_N_std']:.4f}"
    r = f"{row['auroc_R_mean']:.4f}±{row['auroc_R_std']:.4f}" if not pd.isna(row.get('auroc_R_mean')) else "—"
    print(f"{int(row['K']):4d}  {c:>12}  {n:>12}  {r:>12}")
print("=" * 64)
print("Done — results in results_v3/")
