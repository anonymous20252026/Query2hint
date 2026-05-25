# ============================================================
# AdaSteer — Stage 2: Contrastive Training
# ============================================================
# What this file does:
#   1. Loads your triplets (anchor SQL, positive hint, negative hint)
#   2. Fine-tunes MPNet using contrastive (triplet) loss
#   3. Saves the trained encoder for later use
#

# ── STEP 0: Install dependencies (run this first in Colab) ──
# !pip install sentence-transformers pandas torch

# ── Imports ─────────────────────────────────────────────────
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import TripletLoss
from sentence_transformers import InputExample
from torch.utils.data import DataLoader as STDataLoader


# ============================================================
# SECTION 1: CONFIGURATION
# Everything you might want to change is here
# ============================================================

class Config:
    # --- Paths ---
    CEB_PATH   = "stage1_triplets_CEB.csv"   # path to your CEB triplets
    JOB_PATH   = "stage1_triplets_JOB.csv"   # path to your JOB triplets
    OUTPUT_DIR = "adasteer_encoder"           # where to save trained model

    # --- Model ---
    BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"  # pretrained model
    # Options:
    # "sentence-transformers/all-mpnet-base-v2"    ← best (paper default)
    # "sentence-transformers/all-MiniLM-L12-v2"   ← smaller, faster
    # "sentence-transformers/all-MiniLM-L6-v2"    ← smallest, fastest

    # --- Training ---
    EPOCHS          = 5          # how many times to go through all data
    BATCH_SIZE      = 32         # how many triplets per update step
    LEARNING_RATE   = 2e-5       # how fast to learn (small = careful)
    MAX_SEQ_LENGTH  = 256        # max tokens per SQL query
    WARMUP_RATIO    = 0.1        # fraction of steps for warmup
    TRAIN_ON        = "both"     # "CEB", "JOB", or "both"

    # --- Triplet Loss ---
    TRIPLET_MARGIN  = 0.5        # how far apart positive/negative should be
    # higher margin = stricter separation

    # --- Reproducibility ---
    SEED = 42

    # --- Device ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

# ── Set random seeds for reproducibility ────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

print("=" * 60)
print("AdaSteer — Contrastive Training")
print("=" * 60)
print(f"Device    : {cfg.DEVICE}")
print(f"Base model: {cfg.BASE_MODEL}")
print(f"Epochs    : {cfg.EPOCHS}")
print(f"Batch size: {cfg.BATCH_SIZE}")
print(f"Train on  : {cfg.TRAIN_ON}")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD DATA
# Load your triplet CSV files
# ============================================================

def load_triplets(ceb_path, job_path, train_on="both"):
    """
    Load triplets from CSV files.

    Each row is:
        anchor   = SQL query (the query we are learning about)
        positive = hint config that runs FAST for this query
        negative = hint config that runs SLOW for this query
        time_diff = how much slower the negative is (seconds)

    Returns a list of tuples: (anchor, positive, negative)
    """

    dfs = []

    if train_on in ("both", "CEB"):
        ceb = pd.read_csv(ceb_path)
        print(f"Loaded CEB: {len(ceb)} triplets, {ceb['anchor'].nunique()} unique queries")
        dfs.append(ceb)

    if train_on in ("both", "JOB"):
        job = pd.read_csv(job_path)
        print(f"Loaded JOB: {len(job)} triplets, {job['anchor'].nunique()} unique queries")
        dfs.append(job)

    # combine into one dataframe
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total triplets: {len(df)}")
    print()

    # convert to list of (anchor, positive, negative) tuples
    triplets = list(zip(df["anchor"], df["positive"], df["negative"]))

    return triplets, df


triplets, df = load_triplets(cfg.CEB_PATH, cfg.JOB_PATH, cfg.TRAIN_ON)


# ── Quick look at what we loaded ────────────────────────────
print("Sample triplet:")
print(f"  anchor  : {triplets[0][0][:80]}...")
print(f"  positive: {triplets[0][1]}")
print(f"  negative: {triplets[0][2]}")
print()


# ============================================================
# SECTION 3: BUILD DATASET
# Convert triplets into a format sentence-transformers understands
# ============================================================

def build_sentence_transformer_examples(triplets):
    """
    sentence-transformers expects InputExample objects.

    InputExample(texts=[anchor, positive, negative])
    → this is exactly our triplet format!
    """
    examples = []
    for anchor, positive, negative in triplets:
        example = InputExample(
            texts=[
                str(anchor),    # the SQL query
                str(positive),  # good hint config
                str(negative)   # bad hint config
            ]
        )
        examples.append(example)
    return examples


train_examples = build_sentence_transformer_examples(triplets)
print(f"Built {len(train_examples)} training examples")


# ============================================================
# SECTION 4: LOAD THE MODEL
# Load pretrained MPNet and prepare it for fine-tuning
# ============================================================

print(f"\nLoading base model: {cfg.BASE_MODEL}")
print("(This may take a minute on first run — downloading weights)")

# SentenceTransformer handles tokenization + encoding for us
model = SentenceTransformer(cfg.BASE_MODEL)

# set max sequence length
# SQL queries can be long — 256 tokens is a good balance
model.max_seq_length = cfg.MAX_SEQ_LENGTH

print(f"Model loaded!")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
print()


# ============================================================
# SECTION 5: DEFINE THE LOSS FUNCTION
# TripletLoss = contrastive loss for (anchor, positive, negative)
# ============================================================

# TripletLoss works like this:
#
#   loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)
#
# Translation:
#   → distance to POSITIVE should be SMALL
#   → distance to NEGATIVE should be LARGE
#   → margin = minimum gap we want between the two
#
# When loss = 0 → model learned it correctly
# When loss > 0 → model still needs to learn

triplet_loss = TripletLoss(
    model=model,
    triplet_margin=cfg.TRIPLET_MARGIN  # default 0.5, higher = stricter
)

print(f"Loss function: TripletLoss (margin={cfg.TRIPLET_MARGIN})")


# ============================================================
# SECTION 6: CREATE DATALOADER
# DataLoader feeds batches of triplets into the model
# ============================================================

train_dataloader = STDataLoader(
    train_examples,
    shuffle=True,          # shuffle so model doesn't memorize order
    batch_size=cfg.BATCH_SIZE
)

# calculate total training steps
steps_per_epoch   = len(train_dataloader)
total_steps       = steps_per_epoch * cfg.EPOCHS
warmup_steps      = int(total_steps * cfg.WARMUP_RATIO)

print(f"\nTraining setup:")
print(f"  Steps per epoch : {steps_per_epoch}")
print(f"  Total steps     : {total_steps}")
print(f"  Warmup steps    : {warmup_steps}")
print()


# ============================================================
# SECTION 7: TRAIN THE MODEL
# This is where fine-tuning actually happens
# ============================================================

print("=" * 60)
print("Starting Contrastive Training...")
print("=" * 60)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# sentence-transformers handles the training loop for us
# internally it does:
#   for each batch:
#     1. encode anchor, positive, negative
#     2. compute triplet loss
#     3. backpropagate
#     4. update weights
#     5. repeat

model.fit(
    train_objectives=[(train_dataloader, triplet_loss)],

    epochs=cfg.EPOCHS,

    warmup_steps=warmup_steps,
    # warmup = start with tiny learning rate, slowly increase
    # prevents the model from "forgetting" what it already knows

    optimizer_params={"lr": cfg.LEARNING_RATE},

    output_path=cfg.OUTPUT_DIR,
    # saves best checkpoint here

    show_progress_bar=True,
    # shows progress during training

    save_best_model=True,
    # saves the model with lowest loss
)

print()
print("=" * 60)
print(f"Training complete!")
print(f"Model saved to: {cfg.OUTPUT_DIR}/")
print("=" * 60)


# ============================================================
# SECTION 8: QUICK TEST — DOES IT WORK?
# Check that the model learned something meaningful
# ============================================================

print("\nRunning quick sanity check...")

# load the saved model
trained_model = SentenceTransformer(cfg.OUTPUT_DIR)

# three SQL queries for testing:
# sql1 and sql2 are semantically equivalent (same query, minor reformat)
# sql3 is structurally different (should be far from sql1)
sql1 = "SELECT COUNT(*) FROM cast_info ci JOIN title t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql2 = "SELECT COUNT(*) FROM cast_info AS ci JOIN title AS t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql3 = "SELECT MIN(name) FROM name n WHERE n.gender = 'f' GROUP BY n.name_pcode_nf"

# encode all three
emb1 = trained_model.encode(sql1, convert_to_tensor=True)
emb2 = trained_model.encode(sql2, convert_to_tensor=True)
emb3 = trained_model.encode(sql3, convert_to_tensor=True)

# compute cosine similarities
from torch.nn.functional import cosine_similarity

sim_12 = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
sim_13 = cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item()
gap    = sim_12 - sim_13

print()
print("Sanity check results:")
print(f"  Sim(sql1, sql2) = {sim_12:.4f}  ← should be HIGH  (same query)")
print(f"  Sim(sql1, sql3) = {sim_13:.4f}  ← should be LOW   (different query)")
print(f"  Gap             = {gap:.4f}  ← should be LARGE (model is discriminating)")
print()

if gap > 0.2:
    print("✅ Model is working well — good discrimination gap!")
elif gap > 0.05:
    print("⚠️  Model is learning but gap is small — try more epochs")
else:
    print("❌ Gap is tiny — model may need more training or check your data")


# ============================================================
# SECTION 9: SAVE EMBEDDINGS FOR ALL QUERIES
# Generate embeddings for every unique SQL query
# This is what you'll use in Stage 3 (Reptile) and evaluation
# ============================================================

print("\nGenerating embeddings for all unique queries...")

# get all unique queries from both workloads
all_queries     = df["anchor"].unique().tolist()
all_tasks       = df.groupby("anchor")["source_task"].first().to_dict()

print(f"Encoding {len(all_queries)} unique queries...")
print("(This may take a few minutes)")

# encode in batches for efficiency
embeddings = trained_model.encode(
    all_queries,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")
# → (num_queries, embedding_dim)
# e.g. (3246, 768) for MPNet


# ── Save embeddings ─────────────────────────────────────────
emb_df = pd.DataFrame({
    "query"      : all_queries,
    "source_task": [all_tasks[q] for q in all_queries],
})

# add embedding dimensions as columns
emb_cols = pd.DataFrame(
    embeddings,
    columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
)

emb_df = pd.concat([emb_df, emb_cols], axis=1)
emb_df.to_csv("query_embeddings.csv", index=False)

print(f"\nEmbeddings saved to: query_embeddings.csv")
print(f"Shape: {emb_df.shape}")
print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("STAGE 2 COMPLETE!")
print("=" * 60)
print()
print("What was produced:")
print(f"  1. Fine-tuned encoder  → {cfg.OUTPUT_DIR}/")
print(f"  2. Query embeddings    → query_embeddings.csv")
print()
print("Next steps:")
print("  Stage 3: Reptile meta-learning on these embeddings")
print("  Stage 4: Train classifier + evaluate on JOB/CEB")
print("=" * 60)
