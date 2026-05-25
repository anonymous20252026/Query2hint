# ============================================================
# experiments_fewshot_baseline.py
#
# PURPOSE:
#   Add a "NoMeta" baseline to the few-shot adaptation experiment.
#   Trains a fresh SVC directly on K JOB target queries from scratch
#   (no pre-trained encoder, no meta-initialization) using the same
#   embedding space as AdaSteer-C.
#
#   Comparison:
#     AdaSteer-C  — contrastive encoder, no adaptation (fixed)
#     AdaSteer-R  — Reptile meta-initialized encoder + adaptation
#     NoMeta-SVC  — contrastive encoder embeddings + SVC trained from
#                   scratch on K JOB queries (no meta pre-training)
#
#   This isolates the meta-learning contribution from simple classifier
#   training on target data.
#
# USES:
#   - Existing encoder embeddings (computed fresh here)
#   - data/job.csv, data/ceb.csv
#   - stage1_triplets_JOB.csv (for JOB triplet structure)
#
# OUTPUT:
#   results_v3/fewshot_ablation.csv        — K × method × seed results
#   results_v3/fewshot_ablation_summary.csv — mean ± std per K
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

# ── Config ────────────────────────────────────────────────────────────────────
CONTRASTIVE_PATH = "encoders/encoder_all-mpnet-base-v2_v1"
REPTILE_PATH     = "encoders/encoder_reptile_mpnet_v4"
JOB_TRIPLETS     = "stage1_triplets_JOB.csv"
JOB_DATA         = "data/job.csv"
CEB_DATA         = "data/ceb.csv"
FEW_SHOT_SIZES   = [0, 5, 10, 20, 50]
SEEDS            = [42, 123, 456, 789, 999]
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
print("experiments_fewshot_baseline.py — Few-shot + NoMeta baseline")
print("=" * 64)
print(f"Device         : {DEVICE}")
print(f"Contrastive    : {CONTRASTIVE_PATH}")
print(f"Reptile        : {REPTILE_PATH}")
print(f"Few-shot sizes : {FEW_SHOT_SIZES}")
print(f"Seeds          : {SEEDS}")
print("=" * 64)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Load JOB triplets ─────────────────────────────────────────────────────────
print("\n[1/5] Loading JOB triplets...")
triplets = pd.read_csv(JOB_TRIPLETS)
# Expected columns: query, positive, negative  (or similar)
# Check actual columns
print(f"  Triplet columns : {list(triplets.columns)}")
print(f"  Triplet count   : {len(triplets)}")

# Determine column names dynamically
q_col  = "query"    if "query"    in triplets.columns else triplets.columns[0]
p_col  = "positive" if "positive" in triplets.columns else triplets.columns[1]
n_col  = "negative" if "negative" in triplets.columns else triplets.columns[2]

all_queries  = list(set(triplets[q_col].tolist()))
all_positives = list(set(triplets[p_col].tolist()))
all_negatives = list(set(triplets[n_col].tolist()))

# Unique texts for encoding
unique_texts = list(set(all_queries + all_positives + all_negatives))
print(f"  Unique texts    : {len(unique_texts)}")

# ── Encode all JOB texts ──────────────────────────────────────────────────────
print("\n[2/5] Encoding JOB queries...")

def encode_all(model_path, texts, device=DEVICE):
    model = SentenceTransformer(model_path, device=device)
    embs  = model.encode(texts, batch_size=128, show_progress_bar=False,
                          normalize_embeddings=True)
    return {t: embs[i] for i, t in enumerate(texts)}

print(f"  Encoding with Contrastive ({CONTRASTIVE_PATH})...")
emb_C = encode_all(CONTRASTIVE_PATH, unique_texts)
print(f"  Encoding with Reptile     ({REPTILE_PATH})...")
emb_R = encode_all(REPTILE_PATH, unique_texts)
print("  Done.")

# ── Helper: ranking accuracy and AUROC on triplets ───────────────────────────
def ranking_accuracy(emb_dict, triplet_df, q_col, p_col, n_col):
    correct = 0
    scores_p, scores_n = [], []
    for _, row in triplet_df.iterrows():
        zq = torch.tensor(emb_dict[row[q_col]])
        zp = torch.tensor(emb_dict[row[p_col]])
        zn = torch.tensor(emb_dict[row[n_col]])
        sp = F.cosine_similarity(zq.unsqueeze(0), zp.unsqueeze(0)).item()
        sn = F.cosine_similarity(zq.unsqueeze(0), zn.unsqueeze(0)).item()
        if sp > sn:
            correct += 1
        scores_p.append(sp)
        scores_n.append(sn)
    acc = correct / len(triplet_df)
    # AUROC: positive class = positive pair
    labels = [1] * len(scores_p) + [0] * len(scores_n)
    scores  = scores_p + scores_n
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float("nan")
    return acc, auroc

# ── Zero-shot baselines (K=0, no adaptation) ─────────────────────────────────
print("\n[3/5] Computing zero-shot baselines...")
acc_C0, auroc_C0 = ranking_accuracy(emb_C, triplets, q_col, p_col, n_col)
acc_R0, auroc_R0 = ranking_accuracy(emb_R, triplets, q_col, p_col, n_col)
print(f"  AdaSteer-C  K=0 : RankAcc={acc_C0:.4f}  AUROC={auroc_C0:.4f}")
print(f"  AdaSteer-R  K=0 : RankAcc={acc_R0:.4f}  AUROC={auroc_R0:.4f}")

# ── Few-shot SVC baseline ─────────────────────────────────────────────────────
# For each seed and K:
#   1. Sample K JOB triplets from the pool
#   2. Train SVC on those K triplet embeddings (contrastive encoder)
#      using pair-wise features: [z_anchor, z_pos, z_neg] → binary labels
#   3. Evaluate ranking accuracy on remaining triplets

def fewshot_svc_eval(emb_dict, triplet_df, K, seed, q_col, p_col, n_col):
    """Train SVC on K triplets; evaluate on remaining."""
    set_seed(seed)
    if K == 0:
        # No adaptation — return zero-shot
        return ranking_accuracy(emb_dict, triplet_df, q_col, p_col, n_col)

    idx = list(range(len(triplet_df)))
    np.random.shuffle(idx)
    train_idx = idx[:K]
    test_idx  = idx[K:]

    if len(test_idx) == 0:
        test_idx = idx  # if K >= len, evaluate on all

    train_df = triplet_df.iloc[train_idx]
    test_df  = triplet_df.iloc[test_idx]

    # Build features: concat [z_q, z_p] as positive class, [z_q, z_n] as negative
    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        zq = emb_dict[row[q_col]]
        zp = emb_dict[row[p_col]]
        zn = emb_dict[row[n_col]]
        X_train.append(np.concatenate([zq, zp]))
        X_train.append(np.concatenate([zq, zn]))
        y_train.extend([1, 0])

    if len(set(y_train)) < 2:
        # Can't train if only one class in K samples
        return ranking_accuracy(emb_dict, triplet_df, q_col, p_col, n_col)

    clf = SVC(kernel="rbf", probability=False, random_state=seed)
    clf.fit(np.array(X_train), np.array(y_train))

    # Evaluate on test triplets using pairwise decisions
    correct = 0
    pos_scores, neg_scores = [], []
    for _, row in test_df.iterrows():
        zq = emb_dict[row[q_col]]
        zp = emb_dict[row[p_col]]
        zn = emb_dict[row[n_col]]
        feat_p = np.concatenate([zq, zp]).reshape(1, -1)
        feat_n = np.concatenate([zq, zn]).reshape(1, -1)
        sp = clf.decision_function(feat_p)[0]
        sn = clf.decision_function(feat_n)[0]
        if sp > sn:
            correct += 1
        pos_scores.append(sp)
        neg_scores.append(sn)

    acc = correct / len(test_df) if len(test_df) > 0 else float("nan")
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores  = pos_scores + neg_scores
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float("nan")
    return acc, auroc

# ── Few-shot adaptation for AdaSteer-R (gradient-based, using Reptile encoder) ─
def fewshot_reptile_eval(emb_dict, triplet_df, K, seed, q_col, p_col, n_col,
                          model_path=REPTILE_PATH, steps=50, lr=2e-5):
    """Fine-tune Reptile encoder on K JOB triplets; re-encode; evaluate."""
    set_seed(seed)
    if K == 0:
        return ranking_accuracy(emb_dict, triplet_df, q_col, p_col, n_col)

    idx = list(range(len(triplet_df)))
    np.random.shuffle(idx)
    train_idx = idx[:K]
    test_idx  = idx[K:] if len(idx) > K else idx

    train_df = triplet_df.iloc[train_idx]
    test_df  = triplet_df.iloc[test_idx]

    # Load fresh model for fine-tuning
    from transformers import AutoModel, AutoTokenizer
    import torch.nn as nn

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert_model = AutoModel.from_pretrained(model_path).to(DEVICE)
    optimizer  = torch.optim.Adam(bert_model.parameters(), lr=lr)
    MARGIN     = 0.5

    def encode_with_grad(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(DEVICE)
        out = bert_model(**enc)
        emb = out.last_hidden_state[:, 0]
        return F.normalize(emb, dim=1)

    bert_model.train()
    for _ in range(steps):
        for _, row in train_df.iterrows():
            optimizer.zero_grad()
            zq = encode_with_grad([row[q_col]])
            zp = encode_with_grad([row[p_col]])
            zn = encode_with_grad([row[n_col]])
            loss = F.triplet_margin_loss(zq, zp, zn, margin=MARGIN)
            loss.backward()
            optimizer.step()

    # Re-encode all test texts
    bert_model.eval()
    test_texts = list(set(
        test_df[q_col].tolist() + test_df[p_col].tolist() + test_df[n_col].tolist()
    ))
    with torch.no_grad():
        fine_embs = {}
        for i in range(0, len(test_texts), 64):
            batch = test_texts[i:i+64]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt").to(DEVICE)
            out = bert_model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0], dim=1)
            for t, e in zip(batch, emb.cpu().numpy()):
                fine_embs[t] = e

    return ranking_accuracy(fine_embs, test_df, q_col, p_col, n_col)

# ── Main evaluation loop ───────────────────────────────────────────────────────
print("\n[4/5] Running few-shot evaluation (K × seeds)...")
records = []

for K in FEW_SHOT_SIZES:
    acc_C_seeds, auroc_C_seeds = [], []
    acc_R_seeds, auroc_R_seeds = [], []
    acc_N_seeds, auroc_N_seeds = [], []

    for seed in SEEDS:
        # AdaSteer-C: no adaptation (fixed contrastive encoder)
        acc_C, auroc_C = ranking_accuracy(emb_C, triplets, q_col, p_col, n_col) \
            if K == 0 else ranking_accuracy(emb_C, triplets, q_col, p_col, n_col)

        # NoMeta-SVC: SVC trained on K triplets from scratch (contrastive embs)
        acc_N, auroc_N = fewshot_svc_eval(emb_C, triplets, K, seed,
                                           q_col, p_col, n_col)

        # AdaSteer-R: Reptile fine-tuning on K triplets
        acc_R, auroc_R = fewshot_reptile_eval(emb_R, triplets, K, seed,
                                               q_col, p_col, n_col)

        acc_C_seeds.append(acc_C); auroc_C_seeds.append(auroc_C)
        acc_R_seeds.append(acc_R); auroc_R_seeds.append(auroc_R)
        acc_N_seeds.append(acc_N); auroc_N_seeds.append(auroc_N)

        records.append({
            "K": K, "seed": seed,
            "acc_C": acc_C,   "auroc_C": auroc_C,
            "acc_R": acc_R,   "auroc_R": auroc_R,
            "acc_N": acc_N,   "auroc_N": auroc_N,
            "R_wins_C": int(auroc_R > auroc_C),
            "N_wins_C": int(auroc_N > auroc_C),
        })

    print(f"  K={K:2d}  "
          f"C={np.mean(auroc_C_seeds):.4f}  "
          f"R={np.mean(auroc_R_seeds):.4f}  "
          f"NoMeta={np.mean(auroc_N_seeds):.4f}  "
          f"(R wins: {sum(r>c for r,c in zip(auroc_R_seeds,auroc_C_seeds))}/5)")

raw_df = pd.DataFrame(records)
raw_df.to_csv("results_v3/fewshot_ablation.csv", index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
summary = raw_df.groupby("K").agg(
    acc_C_mean=("acc_C", "mean"),   acc_C_std=("acc_C", "std"),
    auroc_C_mean=("auroc_C", "mean"), auroc_C_std=("auroc_C", "std"),
    acc_R_mean=("acc_R", "mean"),   acc_R_std=("acc_R", "std"),
    auroc_R_mean=("auroc_R", "mean"), auroc_R_std=("auroc_R", "std"),
    acc_N_mean=("acc_N", "mean"),   acc_N_std=("acc_N", "std"),
    auroc_N_mean=("auroc_N", "mean"), auroc_N_std=("auroc_N", "std"),
    R_win_rate=("R_wins_C", "mean"),
    N_win_rate=("N_wins_C", "mean"),
).reset_index()
summary.to_csv("results_v3/fewshot_ablation_summary.csv", index=False)
print(f"\n  Saved → results_v3/fewshot_ablation.csv  ({len(raw_df)} rows)")
print(f"  Saved → results_v3/fewshot_ablation_summary.csv")

print("\n[5/5] Summary table (AUROC)")
print(f"{'K':>4}  {'AdaSteer-C':>12}  {'NoMeta-SVC':>12}  {'AdaSteer-R':>12}  {'R wins':>8}")
print("-" * 56)
for _, row in summary.iterrows():
    print(f"{int(row['K']):4d}  "
          f"{row['auroc_C_mean']:.4f}±{row['auroc_C_std']:.4f}  "
          f"{row['auroc_N_mean']:.4f}±{row['auroc_N_std']:.4f}  "
          f"{row['auroc_R_mean']:.4f}±{row['auroc_R_std']:.4f}  "
          f"{row['R_win_rate']:.0%}")
print("=" * 64)
print("Done — results in results_v3/")
