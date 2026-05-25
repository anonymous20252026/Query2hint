# ============================================================
# AdaSteer — Stage 3: Reptile Meta-Learning
# ============================================================
#
# WHAT THIS FILE DOES:
#
#   Takes the fine-tuned encoder from Stage 2
#   and applies Reptile meta-learning so the model
#   can QUICKLY adapt to new workloads with few queries.
#
# HOW REPTILE WORKS (simple):
#
#   Normal training:  optimize for ONE workload → stuck there
#   Reptile:          optimize across MANY tasks → flexible start
#
#   For each meta-iteration:
#     1. Pick a random task (group of queries)
#     2. Save current weights  θ
#     3. Take K gradient steps on that task  → θ'
#     4. Move θ slightly toward θ'
#     5. Repeat → model learns to adapt fast
#
# KEY EXPERIMENT (proves meta-learning works):
#
#   Train on CEB → adapt to JOB with only 5/10/20 queries
#   Compare: Contrastive model vs Reptile model
#   Expected: Reptile adapts faster with less data
#
# ============================================================

# ── INSTALL (run once in terminal) ──────────────────────────
# pip install sentence-transformers pandas torch scikit-learn

# ── Imports ─────────────────────────────────────────────────
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Paths ────────────────────────────────────────────────
    CEB_PATH          = "stage1_triplets_CEB.csv"
    JOB_PATH          = "stage1_triplets_JOB.csv"
    STAGE2_MODEL_PATH = "adasteer_encoder"       # your Stage 2 output
    OUTPUT_DIR        = "adasteer_reptile"        # where to save this model

    # ── Task Setup ───────────────────────────────────────────
    # We split CEB into N_TASKS sub-tasks for meta-training
    # More tasks = better meta-learning but slower
    N_TASKS           = 20      # number of CEB sub-tasks
    # Each task gets this many triplets per inner loop step
    TASK_BATCH_SIZE   = 32
    # How many gradient steps inside each task (inner loop)
    INNER_STEPS       = 5

    # ── Reptile Hyperparameters ──────────────────────────────
    META_ITERATIONS   = 500     # total reptile updates
    META_LR           = 0.1     # how far to move toward task optimum
                                # (outer learning rate β)
    INNER_LR          = 2e-5   # learning rate inside each task
                                # (same as Stage 2)

    # ── Few-Shot Experiment ──────────────────────────────────
    # Test adaptation with different amounts of JOB data
    FEW_SHOT_SIZES    = [5, 10, 20, 50]   # number of JOB queries
    FEW_SHOT_STEPS    = 20                 # fine-tuning steps per shot
    FEW_SHOT_SEEDS    = [42, 123, 456]    # run multiple times for stability

    # ── Other ────────────────────────────────────────────────
    MAX_SEQ_LENGTH    = 256
    SEED              = 42
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ── Reproducibility ─────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

print("=" * 60)
print("AdaSteer — Stage 3: Reptile Meta-Learning")
print("=" * 60)
print(f"Device         : {cfg.DEVICE}")
print(f"Stage 2 model  : {cfg.STAGE2_MODEL_PATH}")
print(f"Meta iterations: {cfg.META_ITERATIONS}")
print(f"Tasks (CEB)    : {cfg.N_TASKS}")
print(f"Inner steps    : {cfg.INNER_STEPS}")
print(f"Few-shot sizes : {cfg.FEW_SHOT_SIZES}")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD DATA AND BUILD TASKS
# ============================================================

print("\nLoading data...")

ceb = pd.read_csv(cfg.CEB_PATH)
job = pd.read_csv(cfg.JOB_PATH)

print(f"CEB: {len(ceb)} triplets, {ceb['anchor'].nunique()} queries")
print(f"JOB: {len(job)} triplets, {job['anchor'].nunique()} queries")


def build_tasks_from_workload(df, n_tasks, seed=42):
    """
    Split a workload into N tasks for meta-learning.

    Each task = a group of queries with their triplets.
    We split by QUERY (not by triplet) so tasks are independent.

    Example:
        CEB has 3133 unique queries
        n_tasks = 20
        → each task gets ~156 queries with all their triplets
    """
    rng = np.random.RandomState(seed)

    # get all unique queries
    unique_queries = df["anchor"].unique()
    rng.shuffle(unique_queries)

    # split queries into n_tasks groups
    query_groups = np.array_split(unique_queries, n_tasks)

    tasks = []
    for group in query_groups:
        # get all triplets for queries in this group
        task_df = df[df["anchor"].isin(group)].reset_index(drop=True)
        tasks.append(task_df)

    print(f"  Built {n_tasks} tasks")
    print(f"  Avg queries per task  : {np.mean([t['anchor'].nunique() for t in tasks]):.0f}")
    print(f"  Avg triplets per task : {np.mean([len(t) for t in tasks]):.0f}")

    return tasks


print("\nBuilding meta-training tasks from CEB...")
ceb_tasks = build_tasks_from_workload(ceb, cfg.N_TASKS, cfg.SEED)

# JOB is our meta-test workload
# we will use small samples of JOB for few-shot adaptation
print(f"\nJOB reserved for few-shot evaluation")
print(f"  {job['anchor'].nunique()} unique queries available")


# ============================================================
# SECTION 3: DATASET CLASS FOR TRIPLETS
# ============================================================

class TripletDataset(Dataset):
    """
    PyTorch Dataset that holds triplets.

    Each item = (anchor_text, positive_text, negative_text)
    """
    def __init__(self, df):
        self.anchors   = df["anchor"].tolist()
        self.positives = df["positive"].tolist()
        self.negatives = df["negative"].tolist()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return (
            str(self.anchors[idx]),
            str(self.positives[idx]),
            str(self.negatives[idx])
        )


def sample_batch_from_task(task_df, batch_size, rng):
    """
    Sample a random batch of triplets from a task.
    Returns lists of (anchors, positives, negatives).
    """
    indices = rng.choice(len(task_df), size=min(batch_size, len(task_df)), replace=False)
    batch = task_df.iloc[indices]
    return (
        batch["anchor"].tolist(),
        batch["positive"].tolist(),
        batch["negative"].tolist()
    )


# ============================================================
# SECTION 4: LOAD STAGE 2 MODEL
# ============================================================

print(f"\nLoading Stage 2 encoder from: {cfg.STAGE2_MODEL_PATH}")

# load the fine-tuned encoder from Stage 2
model = SentenceTransformer(cfg.STAGE2_MODEL_PATH)
model.max_seq_length = cfg.MAX_SEQ_LENGTH
model = model.to(cfg.DEVICE)

print(f"Encoder loaded!")
print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")


# ============================================================
# SECTION 5: REPTILE CORE FUNCTIONS
# ============================================================

def get_model_weights(model):
    """
    Extract all model parameters as a flat dictionary.
    Used to save and restore weights during Reptile.
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }


def set_model_weights(model, weights):
    """
    Restore model parameters from a saved dictionary.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(weights[name])


def reptile_update(original_weights, adapted_weights, meta_lr):
    """
    Apply Reptile meta-update.

    Formula: θ ← θ + β * (θ' - θ)

    Where:
        θ  = original weights (before inner loop)
        θ' = adapted weights (after inner loop on task)
        β  = meta learning rate

    Intuitively:
        Move original weights SLIGHTLY toward
        where the task-specific training wanted to go.
        Not all the way — just a small step.
    """
    updated = {}
    for name in original_weights:
        # compute the direction toward task optimum
        direction = adapted_weights[name] - original_weights[name]
        # take a small step in that direction
        updated[name] = original_weights[name] + meta_lr * direction
    return updated


def compute_triplet_loss_batch(model, anchors, positives, negatives, margin=0.5):
    """
    Compute triplet loss for a batch of (anchor, positive, negative).

    Triplet loss = max(0, dist(a,p) - dist(a,n) + margin)

    We want:
        dist(anchor, positive) to be SMALL  ← good config close
        dist(anchor, negative) to be LARGE  ← bad config far
        margin = minimum gap between the two
    """
    # encode all three sets of texts
    # normalize=True gives unit vectors (needed for cosine distance)
    a_emb = model.encode(anchors,   convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
    p_emb = model.encode(positives, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
    n_emb = model.encode(negatives, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)

    # cosine distance = 1 - cosine_similarity
    # (we want distance, not similarity)
    dist_pos = 1 - F.cosine_similarity(a_emb, p_emb)  # anchor ↔ positive
    dist_neg = 1 - F.cosine_similarity(a_emb, n_emb)  # anchor ↔ negative

    # triplet loss: penalize when positive is NOT closer than negative
    loss = F.relu(dist_pos - dist_neg + margin)

    return loss.mean()


def inner_loop(model, task_df, inner_steps, inner_lr, batch_size, rng):
    """
    Run K gradient steps on a single task.

    This is the INNER LOOP of Reptile:
        For K steps:
            Sample batch from task
            Compute loss
            Update model weights
        Return adapted weights

    Args:
        model      : current encoder
        task_df    : triplets for this task
        inner_steps: how many gradient steps to take (K)
        inner_lr   : learning rate for inner updates
        batch_size : triplets per step
        rng        : random number generator

    Returns:
        adapted_weights: model weights after task-specific training
    """
    # create optimizer for inner loop
    # we use a fresh optimizer each time
    optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)

    model.train()
    for step in range(inner_steps):
        # sample batch from this task
        anchors, positives, negatives = sample_batch_from_task(
            task_df, batch_size, rng
        )

        # compute loss
        optimizer.zero_grad()
        loss = compute_triplet_loss_batch(model, anchors, positives, negatives)

        # backpropagate
        loss.backward()
        optimizer.step()

    # return the adapted weights after K steps
    return get_model_weights(model)


# ============================================================
# SECTION 6: REPTILE META-TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("Starting Reptile Meta-Training...")
print("=" * 60)

rng = np.random.RandomState(cfg.SEED)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# track training progress
loss_history = []
best_meta_loss = float("inf")

for iteration in range(cfg.META_ITERATIONS):

    # ── Step 1: Save current weights ────────────────────────
    original_weights = get_model_weights(model)

    # ── Step 2: Sample a random task ────────────────────────
    task_idx = rng.randint(0, len(ceb_tasks))
    task_df  = ceb_tasks[task_idx]

    # ── Step 3: Inner loop — adapt to this task ─────────────
    # make a copy of the model for inner loop
    # (so we don't permanently change the model yet)
    inner_model = copy.deepcopy(model)

    adapted_weights = inner_loop(
        model      = inner_model,
        task_df    = task_df,
        inner_steps= cfg.INNER_STEPS,
        inner_lr   = cfg.INNER_LR,
        batch_size = cfg.TASK_BATCH_SIZE,
        rng        = rng
    )

    # ── Step 4: Reptile meta-update ─────────────────────────
    # move original weights slightly toward adapted weights
    new_weights = reptile_update(
        original_weights = original_weights,
        adapted_weights  = adapted_weights,
        meta_lr          = cfg.META_LR
    )

    # ── Step 5: Apply updated weights to model ──────────────
    set_model_weights(model, new_weights)

    # ── Step 6: Track progress ──────────────────────────────
    # compute loss on a small validation batch for monitoring
    if (iteration + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_anchors, val_pos, val_neg = sample_batch_from_task(
                task_df, 64, rng
            )
            val_loss = compute_triplet_loss_batch(
                model, val_anchors, val_pos, val_neg
            ).item()

        loss_history.append(val_loss)

        print(f"  Iter {iteration+1:4d}/{cfg.META_ITERATIONS} | "
              f"Task {task_idx:2d} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Queries in task: {task_df['anchor'].nunique()}")

        # save best model
        if val_loss < best_meta_loss:
            best_meta_loss = val_loss
            model.save(cfg.OUTPUT_DIR)
            print(f"  ✅ Best model saved (loss={best_meta_loss:.4f})")

        model.train()


print("\nMeta-training complete!")
print(f"Best validation loss: {best_meta_loss:.4f}")
print(f"Model saved to: {cfg.OUTPUT_DIR}/")

# load best model for evaluation
meta_model = SentenceTransformer(cfg.OUTPUT_DIR)
meta_model.max_seq_length = cfg.MAX_SEQ_LENGTH


# ============================================================
# SECTION 7: FEW-SHOT ADAPTATION EXPERIMENT
# ============================================================
# THIS IS THE KEY EXPERIMENT FOR YOUR PAPER
#
# We compare:
#   A) Contrastive model (Stage 2) — no meta-learning
#   B) Reptile model    (Stage 3) — with meta-learning
#
# Both adapted to JOB with K queries (K = 5, 10, 20, 50)
#
# Expected result: Reptile adapts faster with less data
# ============================================================

print("\n" + "=" * 60)
print("Few-Shot Adaptation Experiment")
print("JOB workload (never seen during meta-training)")
print("=" * 60)


def encode_queries(encoder, queries, batch_size=64):
    """Encode SQL queries into embeddings."""
    return encoder.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )


def adapt_model(base_model_path, support_triplets, n_steps, inner_lr):
    """
    Fine-tune a model on support triplets (few-shot adaptation).

    This simulates what happens when a new workload arrives:
        1. Start from saved model
        2. Fine-tune on K new examples
        3. Return adapted model

    Args:
        base_model_path : path to the base encoder
        support_triplets: small set of triplets from new workload
        n_steps         : how many fine-tuning steps
        inner_lr        : learning rate

    Returns:
        adapted model
    """
    # load fresh copy of base model
    adapted = SentenceTransformer(base_model_path)
    adapted.max_seq_length = cfg.MAX_SEQ_LENGTH
    adapted = adapted.to(cfg.DEVICE)

    if len(support_triplets) == 0 or n_steps == 0:
        return adapted

    optimizer = torch.optim.Adam(adapted.parameters(), lr=inner_lr)
    adapted.train()

    rng = np.random.RandomState(42)

    for step in range(n_steps):
        # sample from support set
        indices = rng.choice(
            len(support_triplets),
            size=min(32, len(support_triplets)),
            replace=False
        )
        batch = [support_triplets[i] for i in indices]
        anchors   = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        negatives = [b[2] for b in batch]

        optimizer.zero_grad()
        loss = compute_triplet_loss_batch(adapted, anchors, positives, negatives)
        loss.backward()
        optimizer.step()

    return adapted


def evaluate_steering(encoder, test_df):
    """
    Evaluate steering quality using a simple SVM classifier.

    For each query we predict: which hint config is best?
    Then measure F1 score.

    Args:
        encoder : fine-tuned SQL encoder
        test_df : triplets from test workload

    Returns:
        f1 score (higher is better)
    """
    # get unique queries and their best hints
    query_to_best = {}
    for _, row in test_df.iterrows():
        q = row["anchor"]
        if q not in query_to_best:
            query_to_best[q] = row["positive"]

    queries  = list(query_to_best.keys())
    labels   = list(query_to_best.values())

    if len(set(labels)) < 2:
        return 0.0   # only one class, can't evaluate

    # encode queries
    embeddings = encode_queries(encoder, queries)

    # encode labels
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # train/test split (80/20)
    n_train = max(2, int(0.8 * len(queries)))
    X_train, X_test = embeddings[:n_train], embeddings[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    if len(X_test) == 0 or len(set(y_test)) < 2:
        return 0.0

    # train SVM classifier
    clf = SVC(kernel="rbf", C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return f1_score(y_test, y_pred, average="weighted", zero_division=0)


# ── Prepare JOB data ────────────────────────────────────────
job_queries  = job["anchor"].unique().tolist()
job_triplets = list(zip(job["anchor"], job["positive"], job["negative"]))

print(f"\nJOB evaluation data:")
print(f"  Unique queries : {len(job_queries)}")
print(f"  Total triplets : {len(job_triplets)}")


# ── Run the experiment ───────────────────────────────────────
print(f"\nRunning few-shot adaptation experiment...")
print(f"Models compared:")
print(f"  A) Contrastive (Stage 2): {cfg.STAGE2_MODEL_PATH}")
print(f"  B) Reptile     (Stage 3): {cfg.OUTPUT_DIR}")
print()

results = []

for seed in cfg.FEW_SHOT_SEEDS:

    rng_exp = np.random.RandomState(seed)

    # shuffle JOB queries
    shuffled_queries = job_queries.copy()
    rng_exp.shuffle(shuffled_queries)

    for n_shots in cfg.FEW_SHOT_SIZES:

        # ── Select support queries (few-shot) ────────────────
        support_queries = shuffled_queries[:n_shots]

        # get triplets for support queries
        support_triplets = [
            (a, p, n) for a, p, n in job_triplets
            if a in support_queries
        ]

        # get remaining queries for evaluation
        eval_queries  = shuffled_queries[n_shots:]
        eval_df       = job[job["anchor"].isin(eval_queries)]

        if len(eval_df) < 10:
            continue

        # ── Model A: Contrastive (no meta-learning) ──────────
        contrastive_adapted = adapt_model(
            base_model_path  = cfg.STAGE2_MODEL_PATH,
            support_triplets = support_triplets,
            n_steps          = cfg.FEW_SHOT_STEPS,
            inner_lr         = cfg.INNER_LR
        )
        f1_contrastive = evaluate_steering(contrastive_adapted, eval_df)

        # ── Model B: Reptile (with meta-learning) ─────────────
        reptile_adapted = adapt_model(
            base_model_path  = cfg.OUTPUT_DIR,
            support_triplets = support_triplets,
            n_steps          = cfg.FEW_SHOT_STEPS,
            inner_lr         = cfg.INNER_LR
        )
        f1_reptile = evaluate_steering(reptile_adapted, eval_df)

        # ── Model C: Zero-shot (no adaptation at all) ─────────
        # this is the baseline: just use model without any JOB data
        zero_shot = SentenceTransformer(cfg.STAGE2_MODEL_PATH)
        f1_zero   = evaluate_steering(zero_shot, eval_df)

        result = {
            "seed"          : seed,
            "n_shots"       : n_shots,
            "f1_zero_shot"  : round(f1_zero, 4),
            "f1_contrastive": round(f1_contrastive, 4),
            "f1_reptile"    : round(f1_reptile, 4),
            "reptile_wins"  : f1_reptile > f1_contrastive
        }
        results.append(result)

        print(f"  shots={n_shots:2d} | seed={seed} | "
              f"zero-shot={f1_zero:.4f} | "
              f"contrastive={f1_contrastive:.4f} | "
              f"reptile={f1_reptile:.4f} | "
              f"{'✅ reptile wins' if f1_reptile > f1_contrastive else '❌ contrastive wins'}")


# ============================================================
# SECTION 8: SUMMARIZE RESULTS — THIS IS YOUR PAPER TABLE
# ============================================================

print("\n" + "=" * 60)
print("FEW-SHOT RESULTS (averaged over seeds)")
print("This is your key Table for meta-learning claim!")
print("=" * 60)

results_df = pd.DataFrame(results)

summary = results_df.groupby("n_shots").agg(
    zero_shot_mean   = ("f1_zero_shot",   "mean"),
    zero_shot_std    = ("f1_zero_shot",   "std"),
    contrastive_mean = ("f1_contrastive", "mean"),
    contrastive_std  = ("f1_contrastive", "std"),
    reptile_mean     = ("f1_reptile",     "mean"),
    reptile_std      = ("f1_reptile",     "std"),
    reptile_win_rate = ("reptile_wins",   "mean")
).reset_index()

print()
print(f"{'Shots':<8} {'Zero-Shot':<18} {'Contrastive':<18} {'Reptile':<18} {'Reptile Wins'}")
print("-" * 75)
for _, row in summary.iterrows():
    print(f"{int(row['n_shots']):<8} "
          f"{row['zero_shot_mean']:.4f}±{row['zero_shot_std']:.4f}   "
          f"{row['contrastive_mean']:.4f}±{row['contrastive_std']:.4f}   "
          f"{row['reptile_mean']:.4f}±{row['reptile_std']:.4f}   "
          f"{row['reptile_win_rate']*100:.0f}%")

# save results
results_df.to_csv("few_shot_results.csv", index=False)
summary.to_csv("few_shot_summary.csv", index=False)

print()
print("Results saved:")
print("  few_shot_results.csv  ← per-seed results")
print("  few_shot_summary.csv  ← averaged summary (for paper)")


# ============================================================
# SECTION 9: LEARNING CURVE DATA
# ============================================================
# Generate data for the learning curve plot
# X-axis: number of adaptation queries
# Y-axis: F1 score
# This is the key FIGURE for your paper

print("\n" + "=" * 60)
print("Generating learning curve data...")
print("(X: shots, Y: F1 — the key figure for your paper)")
print("=" * 60)

# use more granular shot sizes for smooth curve
curve_shots = [0, 5, 10, 20, 30, 50, 75, 113]
# 0 = zero-shot, 113 = all JOB queries

curve_results = []

for seed in [42, 123]:  # 2 seeds for curve (faster)
    rng_curve = np.random.RandomState(seed)
    shuffled  = job_queries.copy()
    rng_curve.shuffle(shuffled)

    for n_shots in curve_shots:

        if n_shots == 0:
            # zero-shot: no adaptation
            support_triplets = []
        else:
            support_q        = shuffled[:n_shots]
            support_triplets = [
                (a, p, n) for a, p, n in job_triplets if a in support_q
            ]

        eval_q  = shuffled[n_shots:] if n_shots < len(shuffled) else shuffled
        eval_df = job[job["anchor"].isin(eval_q)]

        if len(eval_df) < 5:
            continue

        # contrastive
        c_model  = adapt_model(cfg.STAGE2_MODEL_PATH, support_triplets,
                               cfg.FEW_SHOT_STEPS, cfg.INNER_LR)
        f1_c     = evaluate_steering(c_model, eval_df)

        # reptile
        r_model  = adapt_model(cfg.OUTPUT_DIR, support_triplets,
                               cfg.FEW_SHOT_STEPS, cfg.INNER_LR)
        f1_r     = evaluate_steering(r_model, eval_df)

        curve_results.append({
            "seed"         : seed,
            "n_shots"      : n_shots,
            "f1_contrastive": round(f1_c, 4),
            "f1_reptile"   : round(f1_r, 4)
        })

        print(f"  shots={n_shots:3d} | "
              f"contrastive={f1_c:.4f} | reptile={f1_r:.4f}")

curve_df = pd.DataFrame(curve_results)
curve_df.to_csv("learning_curve_data.csv", index=False)
print("\nLearning curve data saved to: learning_curve_data.csv")


# ============================================================
# SECTION 10: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STAGE 3 COMPLETE!")
print("=" * 60)
print()
print("Files produced:")
print(f"  1. Meta-trained encoder → {cfg.OUTPUT_DIR}/")
print(f"  2. Few-shot results     → few_shot_results.csv")
print(f"  3. Summary table        → few_shot_summary.csv  (for paper)")
print(f"  4. Learning curve data  → learning_curve_data.csv (for paper)")
print()
print("What to check:")
print("  → Reptile F1 > Contrastive F1 at low shots (5, 10)")
print("  → Gap should narrow as shots increase (expected)")
print("  → This proves: meta-learning helps when data is scarce")
print()
print("Next: Stage 4 — Full evaluation + paper plots")
print("=" * 60)
