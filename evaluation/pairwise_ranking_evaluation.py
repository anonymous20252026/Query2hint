# ============================================================
# Adaptsteer — Pairwise Ranking Evaluation (FIXED)
# ============================================================
#
# FIX: model.encode() uses torch.no_grad() internally
#      → gradients don't flow → loss.backward() crashes
#
# SOLUTION: use model's transformer directly during training
#           so gradients flow properly
#
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:
    JOB_PATH             = "stage1_triplets_JOB.csv"

    # ── UPDATED PATHS (v3 models) ─────────────────────────────
    CONTRASTIVE_CEB_PATH = "encoders/encoder_all-mpnet-base-v2_v1"
    REPTILE_PATH         = "encoders/encoder_reptile_mpnet_v4"
    ORACLE_PATH          = "Adaptsteer_encoder"

    # ── Few-shot settings ─────────────────────────────────────
    FEW_SHOT_SIZES       = [0, 5, 10, 20, 50]
    FEW_SHOT_STEPS       = 50     # was 20 → more adaptation steps
    FEW_SHOT_LR          = 2e-5
    FEW_SHOT_SEEDS       = [42, 123, 456, 789, 999]

    # ── Triplet margin — MUST match contrastive training ──────
    TRIPLET_MARGIN       = 0.5    # matches contrastive training

    MAX_SEQ_LENGTH       = 256
    ENCODE_BATCH_SIZE    = 128
    SEED                 = 42
    DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

print("=" * 60)
print("Adaptsteer — Pairwise Ranking Evaluation (FIXED)")
print("=" * 60)
print(f"Device         : {cfg.DEVICE}")
print(f"Contrastive    : {cfg.CONTRASTIVE_CEB_PATH}/")
print(f"Reptile        : {cfg.REPTILE_PATH}/")
print(f"Oracle         : {cfg.ORACLE_PATH}/")
print(f"Few-shot sizes : {cfg.FEW_SHOT_SIZES}")
print(f"Seeds          : {cfg.FEW_SHOT_SEEDS}")
print("=" * 60)


# ============================================================
# SECTION 2: VERIFY MODELS EXIST
# ============================================================

for m in [cfg.CONTRASTIVE_CEB_PATH, cfg.REPTILE_PATH, cfg.ORACLE_PATH]:
    status = "✅" if os.path.isdir(m) else "❌ MISSING"
    print(f"  {status}  {m}/")


# ============================================================
# SECTION 3: LOAD DATA
# ============================================================

print("\nLoading JOB data...")

job         = pd.read_csv(cfg.JOB_PATH)
job_triplets= list(zip(job["anchor"], job["positive"], job["negative"]))
job_queries = job["anchor"].unique().tolist()

print(f"JOB: {len(job)} triplets | {len(job_queries)} queries")
print(f"     {job['positive'].nunique()} unique positive hints")


# ============================================================
# SECTION 4: ENCODE WITH GRADIENTS
# ============================================================
#
# THE KEY FIX IS HERE
#
# model.encode() = inference mode, no gradients
#                  → good for evaluation
#                  → BAD for training/adaptation
#
# encode_with_grad() = training mode, gradients flow
#                    → required for loss.backward()
#
# We use TWO different functions:
#   encode_with_grad() → during adaptation (training)
#   model.encode()     → during evaluation (inference)
#
# ============================================================

def encode_with_grad(model, texts):
    """
    Encode texts WITH gradient tracking.
    Used during adaptation (fine-tuning) so loss.backward() works.

    WHY NOT model.encode():
    → model.encode() calls torch.no_grad() internally
    → tensors have no gradient → loss.backward() crashes

    Uses model.tokenize() + model.forward() directly.
    This is consistent with reptile_CEB_to_JOB_v3.py approach.

    WHY NOT model.encode():
    → model.encode() calls torch.no_grad() internally
    → tensors have no gradient → loss.backward() crashes

    Steps:
      1. tokenize  → input features dict
      2. forward   → token embeddings (with gradients)
      3. mean pool → sentence embedding
      4. normalize → unit vector
    """
    # Step 1: tokenize using model.tokenize() (always available)
    features = model.tokenize(texts)
    features = {k: v.to(cfg.DEVICE) for k, v in features.items()}

    # Step 2: forward through model (keeps computation graph)
    output       = model.forward(features)
    token_embs   = output["token_embeddings"]

    # Step 3: mean pooling weighted by attention mask
    mask         = features["attention_mask"].unsqueeze(-1).float()
    summed       = torch.sum(token_embs * mask, dim=1)
    counts       = torch.clamp(mask.sum(dim=1), min=1e-9)
    sentence_emb = summed / counts

    # Step 4: L2 normalize → unit vector
    return F.normalize(sentence_emb, p=2, dim=1)


def compute_triplet_loss_with_grad(model, anchors, positives, negatives):
    """
    Compute triplet loss WITH gradient tracking.
    Used during few-shot adaptation.

    loss = max(0, d(anchor,positive) - d(anchor,negative) + margin)
    d = cosine distance = 1 - cosine_similarity

    margin from cfg.TRIPLET_MARGIN (must match contrastive training)
    """
    a_emb = encode_with_grad(model, anchors)
    p_emb = encode_with_grad(model, positives)
    n_emb = encode_with_grad(model, negatives)

    d_pos = 1 - F.cosine_similarity(a_emb, p_emb)
    d_neg = 1 - F.cosine_similarity(a_emb, n_emb)

    loss  = F.relu(d_pos - d_neg + cfg.TRIPLET_MARGIN)
    return loss.mean()


# ============================================================
# SECTION 5: EVALUATION FUNCTION (inference — no gradients)
# ============================================================

def pairwise_evaluate(encoder, triplets):
    """
    Pairwise ranking evaluation — INFERENCE mode.

    For each (anchor, positive, negative):
      correct = sim(anchor, positive) > sim(anchor, negative)

    Returns:
      ranking_acc : fraction ranked correctly
      auroc       : AUROC score
      avg_margin  : average sim gap (positive - negative)
    """
    if len(triplets) == 0:
        return {"ranking_acc": 0.0, "auroc": 0.5, "avg_margin": 0.0}

    anchors   = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    # inference → no gradients needed → use model.encode()
    encoder.eval()
    with torch.no_grad():
        a_emb = encoder.encode(anchors,   batch_size=cfg.ENCODE_BATCH_SIZE,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
        p_emb = encoder.encode(positives, batch_size=cfg.ENCODE_BATCH_SIZE,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
        n_emb = encoder.encode(negatives, batch_size=cfg.ENCODE_BATCH_SIZE,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)

    # cosine similarity (dot product of normalized vectors)
    sim_pos     = np.sum(a_emb * p_emb, axis=1)
    sim_neg     = np.sum(a_emb * n_emb, axis=1)
    margins     = sim_pos - sim_neg
    ranking_acc = float((margins > 0).mean())
    avg_margin  = float(margins.mean())

    # AUROC
    scores = np.concatenate([sim_pos, sim_neg])
    labels = np.concatenate([np.ones(len(sim_pos)), np.zeros(len(sim_neg))])
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = 0.5

    return {
        "ranking_acc": round(ranking_acc, 4),
        "auroc"      : round(auroc,       4),
        "avg_margin" : round(avg_margin,  4)
    }


# ============================================================
# SECTION 6: ADAPTATION FUNCTION (training — WITH gradients)
# ============================================================

def adapt_model(model_path, support_triplets, n_steps, lr):
    """
    Load saved model and fine-tune on few-shot support data.

    Uses encode_with_grad() so loss.backward() works correctly.
    Returns adapted model ready for evaluation.
    """
    # load fresh copy
    adapted = SentenceTransformer(model_path)
    adapted.max_seq_length = cfg.MAX_SEQ_LENGTH
    adapted = adapted.to(cfg.DEVICE)

    # zero-shot: no adaptation
    if len(support_triplets) == 0 or n_steps == 0:
        return adapted

    optimizer = torch.optim.Adam(adapted.parameters(), lr=lr)
    rng       = np.random.RandomState(42)

    # ── TRAINING MODE ──────────────────────────────────────
    adapted.train()

    for step in range(n_steps):

        # sample batch from support set
        idx   = rng.choice(
            len(support_triplets),
            size    = min(32, len(support_triplets)),   # was 16 → larger batch
            replace = False
        )
        batch     = [support_triplets[i] for i in idx]
        anchors   = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        negatives = [b[2] for b in batch]

        optimizer.zero_grad()

        # USE encode_with_grad → gradients flow ✅
        loss = compute_triplet_loss_with_grad(
            adapted, anchors, positives, negatives
        )

        loss.backward()   # ✅ works correctly
        optimizer.step()

    return adapted


# ============================================================
# SECTION 7: ZERO-SHOT BASELINE
# ============================================================

print("\n" + "=" * 60)
print("Zero-Shot Baseline (no JOB adaptation)")
print("=" * 60)

for name, path in [
    ("Contrastive (CEB only)", cfg.CONTRASTIVE_CEB_PATH),
    ("Reptile     (CEB only)", cfg.REPTILE_PATH),
    ("Oracle      (CEB+JOB) ", cfg.ORACLE_PATH),
]:
    m      = SentenceTransformer(path)
    m.max_seq_length = cfg.MAX_SEQ_LENGTH
    scores = pairwise_evaluate(m, job_triplets)
    print(f"  {name} | "
          f"Rank Acc={scores['ranking_acc']:.4f} | "
          f"AUROC={scores['auroc']:.4f} | "
          f"Margin={scores['avg_margin']:.4f}")


# ============================================================
# SECTION 8: FEW-SHOT EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("Few-Shot Adaptation Experiment")
print("=" * 60)
print()

all_results = []

for seed in cfg.FEW_SHOT_SEEDS:
    print(f"── Seed {seed} ──────────────────────────────────────────")

    rng_exp  = np.random.RandomState(seed)
    shuffled = job_queries.copy()
    rng_exp.shuffle(shuffled)

    for n_shots in cfg.FEW_SHOT_SIZES:

        # ── Support / evaluation split ───────────────────────
        if n_shots == 0:
            support_triplets = []
            eval_queries     = shuffled
        else:
            support_q        = set(shuffled[:n_shots])
            support_triplets = [t for t in job_triplets if t[0] in support_q]
            eval_queries     = [q for q in shuffled if q not in support_q]

        eval_triplets = [t for t in job_triplets if t[0] in set(eval_queries)]

        if len(eval_triplets) < 10:
            continue

        # ── Model A: Contrastive (CEB only) ──────────────────
        m_A      = adapt_model(
            cfg.CONTRASTIVE_CEB_PATH,
            support_triplets,
            cfg.FEW_SHOT_STEPS,
            cfg.FEW_SHOT_LR
        )
        scores_A = pairwise_evaluate(m_A, eval_triplets)

        # ── Model B: Reptile (CEB only) ───────────────────────
        m_B      = adapt_model(
            cfg.REPTILE_PATH,
            support_triplets,
            cfg.FEW_SHOT_STEPS,
            cfg.FEW_SHOT_LR
        )
        scores_B = pairwise_evaluate(m_B, eval_triplets)

        # ── Oracle (zero-shot only — for reference) ───────────
        oracle_auroc = None
        if n_shots == 0:
            m_C          = SentenceTransformer(cfg.ORACLE_PATH)
            m_C.max_seq_length = cfg.MAX_SEQ_LENGTH
            scores_C     = pairwise_evaluate(m_C, eval_triplets)
            oracle_auroc = scores_C["auroc"]

        # ── Store ─────────────────────────────────────────────
        result = {
            "seed"               : seed,
            "n_shots"            : n_shots,
            "n_eval_triplets"    : len(eval_triplets),
            "n_support_triplets" : len(support_triplets),
            "rank_contrastive"   : scores_A["ranking_acc"],
            "rank_reptile"       : scores_B["ranking_acc"],
            "auroc_contrastive"  : scores_A["auroc"],
            "auroc_reptile"      : scores_B["auroc"],
            "margin_contrastive" : scores_A["avg_margin"],
            "margin_reptile"     : scores_B["avg_margin"],
            "rank_gain"          : round(scores_B["ranking_acc"] - scores_A["ranking_acc"], 4),
            "auroc_gain"         : round(scores_B["auroc"]       - scores_A["auroc"],       4),
            "reptile_wins_rank"  : scores_B["ranking_acc"] > scores_A["ranking_acc"],
            "reptile_wins_auroc" : scores_B["auroc"]       > scores_A["auroc"],
        }
        if oracle_auroc is not None:
            result["auroc_oracle"] = oracle_auroc

        all_results.append(result)

        flag = "✅" if scores_B["auroc"] > scores_A["auroc"] else "❌"
        print(f"  K={n_shots:3d} | "
              f"Rank C={scores_A['ranking_acc']:.4f} R={scores_B['ranking_acc']:.4f} | "
              f"AUROC C={scores_A['auroc']:.4f} R={scores_B['auroc']:.4f} | "
              f"Gain={scores_B['auroc']-scores_A['auroc']:+.4f} {flag}")
    print()


# ============================================================
# SECTION 9: PAPER TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

summary = results_df.groupby("n_shots").agg(
    rank_C_mean  = ("rank_contrastive",  "mean"),
    rank_C_std   = ("rank_contrastive",  "std"),
    rank_R_mean  = ("rank_reptile",      "mean"),
    rank_R_std   = ("rank_reptile",      "std"),
    auroc_C_mean = ("auroc_contrastive", "mean"),
    auroc_C_std  = ("auroc_contrastive", "std"),
    auroc_R_mean = ("auroc_reptile",     "mean"),
    auroc_R_std  = ("auroc_reptile",     "std"),
    rank_gain    = ("rank_gain",         "mean"),
    auroc_gain   = ("auroc_gain",        "mean"),
    win_rate     = ("reptile_wins_auroc","mean")
).reset_index()

oracle_rows      = results_df[results_df["n_shots"] == 0]
oracle_auroc_avg = (oracle_rows["auroc_oracle"].mean()
                    if "auroc_oracle" in oracle_rows.columns else None)

print("=" * 72)
print("PAPER TABLE — Pairwise Ranking (CEB only → JOB few-shot)")
print(f"Averaged over {len(cfg.FEW_SHOT_SEEDS)} seeds: {cfg.FEW_SHOT_SEEDS}")
print("=" * 72)

# ── Ranking Accuracy ─────────────────────────────────────────
print()
print("Ranking Accuracy (higher = better):")
print(f"  {'K':>5} | {'Contrastive':^22} | {'Reptile':^22} | {'Gain':>7} | {'Win%':>5}")
print("  " + "-" * 68)
for _, row in summary.iterrows():
    marker = "✅" if row["rank_gain"] > 0 else "  "
    print(f"  {int(row['n_shots']):>5} | "
          f"{row['rank_C_mean']:.4f} ± {row['rank_C_std']:.4f}      | "
          f"{row['rank_R_mean']:.4f} ± {row['rank_R_std']:.4f}      | "
          f"{row['rank_gain']:>+.4f}  | "
          f"{row['win_rate']*100:>4.0f}% {marker}")

# ── AUROC ─────────────────────────────────────────────────────
print()
print("AUROC (higher = better, 0.5=random, 1.0=perfect):")
print(f"  {'K':>5} | {'Contrastive':^22} | {'Reptile':^22} | {'Gain':>7} | {'Win%':>5}")
print("  " + "-" * 68)
for _, row in summary.iterrows():
    marker = "✅" if row["auroc_gain"] > 0 else "  "
    print(f"  {int(row['n_shots']):>5} | "
          f"{row['auroc_C_mean']:.4f} ± {row['auroc_C_std']:.4f}      | "
          f"{row['auroc_R_mean']:.4f} ± {row['auroc_R_std']:.4f}      | "
          f"{row['auroc_gain']:>+.4f}  | "
          f"{row['win_rate']*100:>4.0f}% {marker}")

if oracle_auroc_avg:
    print("  " + "-" * 68)
    print(f"  {'Oracle':>5} | {oracle_auroc_avg:.4f} ← upper bound (CEB+JOB full training)")

# ── Key finding ───────────────────────────────────────────────
low_shot = summary[summary["n_shots"] <= 10]
low_gain = low_shot["auroc_gain"].mean()

print()
print("─" * 72)
if low_gain > 0:
    print(f"✅ KEY FINDING: Reptile > Contrastive at K≤10")
    print(f"   Average AUROC gain: {low_gain:+.4f}")
    print(f"   → Meta-learning enables faster workload adaptation ✅")
else:
    print(f"⚠️  Reptile gain at K≤10: {low_gain:+.4f}")
    print(f"   → Reptile needs tuning (see recommendations below)")
    print()
    print("Recommendations:")
    print("  1. Reduce INNER_STEPS from 10 → 3-5")
    print("  2. Reduce META_ITERATIONS from 2000 → 500-1000")
    print("  3. Reduce META_LR from 0.1 → 0.05")


# ============================================================
# SECTION 10: SAVE
# ============================================================

results_df.to_csv("pairwise_results_raw.csv",     index=False)
summary.to_csv("pairwise_results_summary.csv",    index=False)
summary.to_csv("learning_curve_pairwise.csv",     index=False)

print()
print("─" * 72)
print("Files saved:")
print("  pairwise_results_raw.csv      ← per-seed raw results")
print("  pairwise_results_summary.csv  ← paper table (averaged)")
print("  learning_curve_pairwise.csv   ← paper figure data")
print()
print("Done ✅")
