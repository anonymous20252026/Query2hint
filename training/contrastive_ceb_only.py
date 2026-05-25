# ============================================================
# AdaSteer — Contrastive Training (CEB Only)
# ============================================================
#
# PURPOSE:
#   Train contrastive encoder on CEB workload ONLY.
#   This is the fair baseline for the meta-learning experiment.
#   The model never sees JOB — so we can fairly test
#   how fast each model adapts to JOB later.
#
# EXPERIMENT DESIGN:
#   ┌─────────────────────────────────────────────────────┐
#   │  contrastive_CEB_only.py   ← YOU ARE HERE           │
#   │  → trains on CEB only                               │
#   │  → saves: encoder_CEB_only/                         │
#   │                                                     │
#   │  reptile_CEB_to_JOB.py     ← NEXT STEP              │
#   │  → starts from encoder_CEB_only/                    │
#   │  → applies Reptile across CEB sub-tasks             │
#   │  → saves: encoder_reptile_CEB/                      │
#   │                                                     │
#   │  EVALUATION:                                        │
#   │  Both models adapt to JOB with K=5,10,20,50 queries │
#   │  Reptile should win at low K → proves meta-learning │
#   └─────────────────────────────────────────────────────┘
#
# NOTE:
#   adasteer_encoder/ (CEB+JOB full model) = ORACLE reference
#   Use that later for TPC-H / TPC-DS generalization tests
#
# ============================================================

# ── INSTALL (run once) ──────────────────────────────────────
# pip install sentence-transformers pandas torch

import os
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Paths ────────────────────────────────────────────────
    CEB_PATH   = "stage1_triplets_CEB.csv"
    JOB_PATH   = "stage1_triplets_JOB.csv"   # loaded but NOT used for training
                                               # kept for reference only
    OUTPUT_DIR = "encoder_CEB_only"           # save here

    # ── Training data ────────────────────────────────────────
    TRAIN_ON   = "CEB"                        # ← KEY: CEB only, not JOB

    # ── Model ────────────────────────────────────────────────
    BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

    # ── Training hyperparameters ─────────────────────────────
    EPOCHS         = 3
    BATCH_SIZE     = 64
    LEARNING_RATE  = 2e-5
    MAX_SEQ_LENGTH = 256
    WARMUP_RATIO   = 0.1
    TRIPLET_MARGIN = 0.2

    # ── Reproducibility ──────────────────────────────────────
    SEED   = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

print("=" * 60)
print("AdaSteer — Contrastive Training (CEB Only)")
print("=" * 60)
print(f"Device      : {cfg.DEVICE}")
print(f"Train on    : {cfg.TRAIN_ON}  ← JOB is HELD OUT for evaluation")
print(f"Base model  : {cfg.BASE_MODEL}")
print(f"Output dir  : {cfg.OUTPUT_DIR}")
print(f"Epochs      : {cfg.EPOCHS}")
print(f"Batch size  : {cfg.BATCH_SIZE}")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD DATA (CEB ONLY)
# ============================================================

print("\nLoading CEB data...")

ceb = pd.read_csv(cfg.CEB_PATH)
job = pd.read_csv(cfg.JOB_PATH)   # load for reference — NOT used in training

print(f"CEB (training)  : {len(ceb)} triplets, {ceb['anchor'].nunique()} unique queries")
print(f"JOB (held out)  : {len(job)} triplets, {job['anchor'].nunique()} unique queries")
print(f"                  ↑ JOB will NOT be used for training")
print()

# sample triplet for sanity check
print("Sample CEB triplet:")
print(f"  anchor  : {ceb.iloc[0]['anchor'][:80]}...")
print(f"  positive: {ceb.iloc[0]['positive']}")
print(f"  negative: {ceb.iloc[0]['negative']}")
print(f"  time_diff: {ceb.iloc[0]['time_diff']:.4f}s")
print()


# ============================================================
# SECTION 3: BUILD TRAINING EXAMPLES
# ============================================================

def build_examples(df):
    """
    Convert dataframe rows into InputExample objects
    for sentence-transformers TripletLoss.

    Each example = (anchor SQL, positive hint, negative hint)
    """
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[
                str(row["anchor"]),    # SQL query
                str(row["positive"]),  # good config
                str(row["negative"])   # bad config
            ]
        ))
    return examples


# build from CEB only
train_examples = build_examples(ceb)

print(f"Training examples built: {len(train_examples)}")
print(f"(Only CEB — JOB completely held out)")


# ============================================================
# SECTION 4: LOAD MODEL
# ============================================================

print(f"\nLoading base model: {cfg.BASE_MODEL}")

model = SentenceTransformer(cfg.BASE_MODEL)
model.max_seq_length = cfg.MAX_SEQ_LENGTH

print(f"Model loaded! Embedding dim: {model.get_sentence_embedding_dimension()}")


# ============================================================
# SECTION 5: DEFINE LOSS + DATALOADER
# ============================================================

triplet_loss = TripletLoss(
    model=model,
    triplet_margin=cfg.TRIPLET_MARGIN
)

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=cfg.BATCH_SIZE
)

steps_per_epoch = len(train_dataloader)
total_steps     = steps_per_epoch * cfg.EPOCHS
warmup_steps    = int(total_steps * cfg.WARMUP_RATIO)

print(f"\nTraining setup:")
print(f"  Triplets (CEB only) : {len(train_examples)}")
print(f"  Steps per epoch     : {steps_per_epoch}")
print(f"  Total steps         : {total_steps}")
print(f"  Warmup steps        : {warmup_steps}")
print(f"  Estimated time      : ~{total_steps / 2.4 / 60:.0f} minutes")


# ============================================================
# SECTION 6: TRAIN
# ============================================================

print("\n" + "=" * 60)
print("Starting Contrastive Training (CEB Only)...")
print("=" * 60)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

model.fit(
    train_objectives = [(train_dataloader, triplet_loss)],
    epochs           = cfg.EPOCHS,
    warmup_steps     = warmup_steps,
    optimizer_params = {"lr": cfg.LEARNING_RATE},
    output_path      = cfg.OUTPUT_DIR,
    show_progress_bar= True,
    save_best_model  = True,
)

print()
print("=" * 60)
print(f"Training complete!")
print(f"Model saved to: {cfg.OUTPUT_DIR}/")
print("=" * 60)


# ============================================================
# SECTION 7: SANITY CHECK
# ============================================================

print("\nRunning sanity check...")

trained_model = SentenceTransformer(cfg.OUTPUT_DIR)
trained_model.max_seq_length = cfg.MAX_SEQ_LENGTH

# use actual CEB-style queries for the check
sql1 = "SELECT COUNT(*) FROM cast_info ci JOIN title t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql2 = "SELECT COUNT(*) FROM cast_info AS ci JOIN title AS t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql3 = "SELECT MIN(name) FROM name n WHERE n.gender = 'f' GROUP BY n.name_pcode_nf"

import torch.nn.functional as F

e1 = trained_model.encode(sql1, convert_to_tensor=True)
e2 = trained_model.encode(sql2, convert_to_tensor=True)
e3 = trained_model.encode(sql3, convert_to_tensor=True)

sim_12 = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
sim_13 = F.cosine_similarity(e1.unsqueeze(0), e3.unsqueeze(0)).item()
gap    = sim_12 - sim_13

print()
print("Sanity check results:")
print(f"  Sim(sql1, sql2) = {sim_12:.4f}  ← should be HIGH  (same query)")
print(f"  Sim(sql1, sql3) = {sim_13:.4f}  ← should be LOW   (different)")
print(f"  Gap             = {gap:.4f}  ← should be > 0.5 (good discrimination)")

if gap > 0.5:
    print("  ✅ Excellent discrimination — model learned well!")
elif gap > 0.2:
    print("  ✅ Good discrimination — model is working")
else:
    print("  ⚠️  Small gap — consider more epochs")


# ============================================================
# SECTION 8: GENERATE + SAVE CEB EMBEDDINGS
# ============================================================

print("\nGenerating embeddings for all CEB queries...")

ceb_queries = ceb["anchor"].unique().tolist()

ceb_embeddings = trained_model.encode(
    ceb_queries,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"CEB embeddings shape: {ceb_embeddings.shape}")

# save with metadata
emb_df = pd.DataFrame({
    "query"      : ceb_queries,
    "source_task": "CEB"
})
emb_cols = pd.DataFrame(
    ceb_embeddings,
    columns=[f"dim_{i}" for i in range(ceb_embeddings.shape[1])]
)
emb_df = pd.concat([emb_df, emb_cols], axis=1)
emb_df.to_csv("embeddings_CEB_only.csv", index=False)

print(f"Embeddings saved to: embeddings_CEB_only.csv")


# ============================================================
# SECTION 9: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONTRASTIVE (CEB ONLY) — COMPLETE")
print("=" * 60)
print()
print("What was produced:")
print(f"  encoder_CEB_only/        ← CEB-only contrastive model")
print(f"  embeddings_CEB_only.csv  ← CEB query embeddings")
print()
print("Experiment map:")
print("  ✅  encoder_CEB_only/     → baseline for meta-learning test")
print("  ✅  adasteer_encoder/     → oracle (CEB+JOB, for TPC-H later)")
print()
print("Next step:")
print("  Run reptile_CEB_to_JOB.py")
print("  → starts from encoder_CEB_only/")
print("  → applies Reptile meta-learning")
print("  → tests few-shot adaptation to JOB")
print("=" * 60)
