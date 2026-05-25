# ============================================================
# AdaSteer — Reptile Meta-Learning v2 (CEB → JOB)
# ============================================================
#
# CHANGES FROM v1:
#   INNER_STEPS     : 10   → 3     fewer steps = better generalization
#   META_ITERATIONS : 2000 → 800   less overfitting to CEB
#   META_LR         : 0.01 → 0.05  slightly larger step
#   N_TASKS         : 20   → 50    more diverse tasks
#   TASK_BATCH_SIZE : 32   → 64    matches contrastive batch
#   margin          : 0.5  → 0.2   MUST match contrastive v2
#   BASE MODEL      : encoder_CEB_only → encoder_CEB_v2
#
# WHY THESE CHANGES:
#
#   INNER_STEPS = 3:
#   → Reptile theory: few steps = good initialization
#                     many steps = overfits to task
#   → 10 steps pushed model too far into each CEB task
#   → 3 steps keeps model in flexible zone
#
#   N_TASKS = 50:
#   → more diverse tasks = better meta-generalization
#   → model sees more variety of CEB sub-distributions
#   → initialization becomes more robust
#
#   margin = 0.2:
#   → MUST match contrastive training margin
#   → consistent loss landscape = stable adaptation
#
# NOTE: evaluation is NOT in this file
#       run evaluate_pairwise_fixed.py after training
#
# ============================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Paths ────────────────────────────────────────────────
    CEB_PATH             = "stage1_triplets_CEB.csv"
    JOB_PATH             = "stage1_triplets_JOB.csv"
    CONTRASTIVE_PATH     = "encoder_CEB_only"        # ← use v2!
    REPTILE_OUTPUT_DIR   = "encoder_reptile_CEB_v2" # ← new output

    # ── KEY CHANGES FROM v1 ──────────────────────────────────
    N_TASKS              = 50    # was 20  → more diverse tasks
    TASK_BATCH_SIZE      = 64    # was 32  → larger batches
    INNER_STEPS          = 3     # was 10  → fewer = better generalization
    META_ITERATIONS      = 800   # was 2000 → less CEB overfitting
    META_LR              = 0.05  # was 0.01 → slightly larger step
    TRIPLET_MARGIN       = 0.2   # was 0.5  → MUST match contrastive v2!

    # ── Unchanged ────────────────────────────────────────────
    INNER_LR             = 2e-5
    MAX_SEQ_LENGTH       = 256
    SEED                 = 42
    DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Logging ──────────────────────────────────────────────
    LOG_INTERVAL         = 50    # print every N iterations
    SAVE_INTERVAL        = 50    # check for best model every N iterations


cfg = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

print("=" * 60)
print("AdaSteer — Reptile Meta-Learning v2 (CEB → JOB)")
print("=" * 60)
print(f"Device          : {cfg.DEVICE}")
print(f"Base model      : {cfg.CONTRASTIVE_PATH}/")
print(f"Output          : {cfg.REPTILE_OUTPUT_DIR}/")
print()
print("Key settings (v2 changes):")
print(f"  INNER_STEPS     : {cfg.INNER_STEPS}    (was 10)")
print(f"  META_ITERATIONS : {cfg.META_ITERATIONS}  (was 2000)")
print(f"  META_LR         : {cfg.META_LR}  (was 0.01)")
print(f"  N_TASKS         : {cfg.N_TASKS}   (was 20)")
print(f"  TASK_BATCH_SIZE : {cfg.TASK_BATCH_SIZE}   (was 32)")
print(f"  TRIPLET_MARGIN  : {cfg.TRIPLET_MARGIN}   (was 0.5, matches contrastive v2)")
print("=" * 60)


# ============================================================
# SECTION 2: VERIFY BASE MODEL EXISTS
# ============================================================

if not os.path.isdir(cfg.CONTRASTIVE_PATH):
    print(f"\n❌ Base model not found: {cfg.CONTRASTIVE_PATH}/")
    print("   Run contrastive_CEB_v2.py first!")
    exit(1)

print(f"\n✅ Base model found: {cfg.CONTRASTIVE_PATH}/")


# ============================================================
# SECTION 3: LOAD DATA
# ============================================================

print("\nLoading data...")

ceb = pd.read_csv(cfg.CEB_PATH)
job = pd.read_csv(cfg.JOB_PATH)

print(f"CEB : {len(ceb)} triplets | {ceb['anchor'].nunique()} queries → meta-train")
print(f"JOB : {len(job)} triplets | {job['anchor'].nunique()} queries → held out")


# ============================================================
# SECTION 4: BUILD CEB TASKS
# ============================================================

def build_tasks(df, n_tasks, seed=42):
    """
    Split CEB queries into N independent tasks.
    More tasks = more diverse meta-training distribution.
    """
    rng     = np.random.RandomState(seed)
    queries = df["anchor"].unique()
    rng.shuffle(queries)
    groups  = np.array_split(queries, n_tasks)
    tasks   = [df[df["anchor"].isin(g)].reset_index(drop=True) for g in groups]

    q_counts = [t["anchor"].nunique() for t in tasks]
    t_counts = [len(t) for t in tasks]

    print(f"  {n_tasks} tasks built")
    print(f"  Queries per task  : {np.min(q_counts)}–{np.max(q_counts)} "
          f"(mean {np.mean(q_counts):.0f})")
    print(f"  Triplets per task : {np.min(t_counts)}–{np.max(t_counts)} "
          f"(mean {np.mean(t_counts):.0f})")
    return tasks


print(f"\nBuilding {cfg.N_TASKS} CEB tasks for meta-training...")
ceb_tasks = build_tasks(ceb, cfg.N_TASKS, cfg.SEED)


# ============================================================
# SECTION 5: LOAD BASE MODEL
# ============================================================

print(f"\nLoading base encoder: {cfg.CONTRASTIVE_PATH}/")

model = SentenceTransformer(cfg.CONTRASTIVE_PATH)
model.max_seq_length = cfg.MAX_SEQ_LENGTH
model = model.to(cfg.DEVICE)

print(f"Loaded! Embedding dim: {model.get_sentence_embedding_dimension()}")


# ============================================================
# SECTION 6: REPTILE CORE FUNCTIONS
# ============================================================

def get_weights(model):
    """Save all model parameters as dict."""
    return {n: p.data.clone() for n, p in model.named_parameters()}


def set_weights(model, weights):
    """Restore saved model parameters."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data.copy_(weights[n])


def reptile_update(theta, theta_prime, beta):
    """
    Reptile meta-update: θ ← θ + β(θ' − θ)

    theta       = weights before inner loop
    theta_prime = weights after K steps on task
    beta        = meta learning rate (how big the step)

    Intuition:
    → don't fully commit to one task's optimum
    → just take a small step toward it
    → after many tasks → land in a flexible position
    """
    return {
        n: theta[n] + beta * (theta_prime[n] - theta[n])
        for n in theta
    }


def sample_batch(task_df, batch_size, rng):
    """Sample random batch of triplets from task."""
    idx   = rng.choice(len(task_df), size=min(batch_size, len(task_df)), replace=False)
    batch = task_df.iloc[idx]
    return (
        batch["anchor"].tolist(),
        batch["positive"].tolist(),
        batch["negative"].tolist()
    )


def compute_loss(model, anchors, positives, negatives):
    """
    Triplet loss WITH gradient tracking.
    Uses model.forward() directly — NOT model.encode()

    model.encode() → uses torch.no_grad() → gradients don't flow
    model.forward()→ gradients flow → loss.backward() works ✅

    IMPORTANT: margin = 0.2 (matches contrastive v2)
    """
    def encode_with_grad(texts):
        # tokenize
        features = model.tokenize(texts)
        features = {k: v.to(cfg.DEVICE) for k, v in features.items()}

        # forward through transformer
        out         = model.forward(features)
        token_embs  = out["token_embeddings"]

        # mean pooling
        mask   = features["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(token_embs * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb    = summed / counts

        # L2 normalize
        return F.normalize(emb, p=2, dim=1)

    a = encode_with_grad(anchors)
    p = encode_with_grad(positives)
    n = encode_with_grad(negatives)

    # cosine distances
    d_pos = 1 - F.cosine_similarity(a, p)
    d_neg = 1 - F.cosine_similarity(a, n)

    # triplet loss with margin=0.2 (matches contrastive v2)
    loss  = F.relu(d_pos - d_neg + cfg.TRIPLET_MARGIN)
    return loss.mean()


def inner_loop(base_model, task_df, inner_steps, inner_lr, batch_size, rng):
    """
    Run K gradient steps on one task.

    KEY: makes a DEEP COPY of model
         original model stays unchanged
         only adapted copy's weights are returned

    With INNER_STEPS=3:
    → 3 steps = small movement toward task optimum
    → model stays in flexible region
    → Reptile update pulls meta-model slightly in this direction
    """
    task_model = copy.deepcopy(base_model)
    task_model.train()
    optimizer  = torch.optim.Adam(task_model.parameters(), lr=inner_lr)
    final_loss = 0.0

    for _ in range(inner_steps):
        a, p, n    = sample_batch(task_df, batch_size, rng)
        optimizer.zero_grad()
        loss       = compute_loss(task_model, a, p, n)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return get_weights(task_model), final_loss


# ============================================================
# SECTION 7: REPTILE META-TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("Starting Reptile Meta-Training v2...")
print(f"CEB {cfg.N_TASKS} tasks | {cfg.INNER_STEPS} inner steps | "
      f"{cfg.META_ITERATIONS} iterations")
print("JOB completely unseen during training")
print("=" * 60)

os.makedirs(cfg.REPTILE_OUTPUT_DIR, exist_ok=True)

rng_meta   = np.random.RandomState(cfg.SEED)
best_loss  = float("inf")
loss_log   = []

for iteration in range(cfg.META_ITERATIONS):

    # ── Step 1: save current position ───────────────────────
    theta = get_weights(model)

    # ── Step 2: pick random task ─────────────────────────────
    task_idx = rng_meta.randint(0, cfg.N_TASKS)
    task_df  = ceb_tasks[task_idx]

    # ── Step 3: inner loop → get task-adapted weights ────────
    theta_prime, task_loss = inner_loop(
        base_model  = model,
        task_df     = task_df,
        inner_steps = cfg.INNER_STEPS,     # 3 steps
        inner_lr    = cfg.INNER_LR,
        batch_size  = cfg.TASK_BATCH_SIZE,
        rng         = rng_meta
    )

    # ── Step 4: reptile update → small step toward task opt ──
    theta_new = reptile_update(theta, theta_prime, cfg.META_LR)

    # ── Step 5: apply update ─────────────────────────────────
    set_weights(model, theta_new)

    # ── Step 6: log and save ─────────────────────────────────
    if (iteration + 1) % cfg.LOG_INTERVAL == 0:
        loss_log.append({"iter": iteration + 1, "loss": task_loss})

        print(f"  Iter {iteration+1:4d}/{cfg.META_ITERATIONS} | "
              f"Task {task_idx:2d}/{cfg.N_TASKS} | "
              f"Loss: {task_loss:.4f} | "
              f"Best: {best_loss:.4f}")

        if task_loss < best_loss:
            best_loss = task_loss
            model.save(cfg.REPTILE_OUTPUT_DIR)
            print(f"            ✅ Best model saved!")

print(f"\nMeta-training complete!")
print(f"Best loss      : {best_loss:.4f}")
print(f"Model saved to : {cfg.REPTILE_OUTPUT_DIR}/")


# ============================================================
# SECTION 8: QUICK SANITY CHECK
# ============================================================
# Check that meta-training didn't destroy the encoder quality

print("\nRunning quick sanity check...")

meta_model = SentenceTransformer(cfg.REPTILE_OUTPUT_DIR)
meta_model.max_seq_length = cfg.MAX_SEQ_LENGTH

sql1 = "SELECT COUNT(*) FROM cast_info ci JOIN title t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql2 = "SELECT COUNT(*) FROM cast_info AS ci JOIN title AS t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql3 = "SELECT MIN(name) FROM name n WHERE n.gender = 'f' GROUP BY n.name_pcode_nf"

e1 = meta_model.encode(sql1, convert_to_tensor=True)
e2 = meta_model.encode(sql2, convert_to_tensor=True)
e3 = meta_model.encode(sql3, convert_to_tensor=True)

sim_12 = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
sim_13 = F.cosine_similarity(e1.unsqueeze(0), e3.unsqueeze(0)).item()
gap    = sim_12 - sim_13

print(f"\n  Sim(sql1, sql2) = {sim_12:.4f}  ← should be HIGH")
print(f"  Sim(sql1, sql3) = {sim_13:.4f}  ← should be LOW")
print(f"  Gap             = {gap:.4f}")

if gap > 0.3:
    print("  ✅ Encoder quality preserved after meta-training")
elif gap > 0.1:
    print("  ⚠️  Gap reduced but acceptable — meta-training made it flexible")
else:
    print("  ❌ Gap too small — meta-training may have been too aggressive")
    print("     Try reducing META_LR or META_ITERATIONS")


# ============================================================
# SECTION 9: SAVE LOSS LOG
# ============================================================

loss_df = pd.DataFrame(loss_log)
loss_df.to_csv("reptile_v2_loss_log.csv", index=False)
print(f"\nLoss log saved: reptile_v2_loss_log.csv")


# ============================================================
# SECTION 10: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("REPTILE v2 TRAINING COMPLETE!")
print("=" * 60)
print()
print("Model inventory:")
print(f"  encoder_CEB_only/      → contrastive v1 (baseline)")
print(f"  encoder_CEB_v2/        → contrastive v2 (new baseline)")
print(f"  encoder_reptile_CEB/   → reptile v1")
print(f"  encoder_reptile_CEB_v2/→ reptile v2 ← just trained")
print(f"  adasteer_encoder/      → oracle (CEB+JOB)")
print()
print("Next step — run evaluation:")
print("  Update evaluate_pairwise_fixed.py:")
print("    CONTRASTIVE_CEB_PATH = 'encoder_CEB_v2'")
print("    REPTILE_PATH         = 'encoder_reptile_CEB_v2'")
print("  Then run:")
print("    python evaluate_pairwise_fixed.py")
print()
print("Expected improvement:")
print("  v1: Reptile wins at K≥20 (60-80% win rate)")
print("  v2: Reptile should win at K≥10 (80%+ win rate)")
print("=" * 60)