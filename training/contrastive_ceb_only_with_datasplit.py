# ============================================================
# AdaSteer — Contrastive Training (CEB Only)
# Multi-Model Comparison + CEB Split for Fair Reptile Training
# ============================================================
#
# PURPOSE:
#   Train contrastive encoder on CEB workload ONLY.
#   Tests 5 different base models to find best for SQL tasks.
#   JOB completely held out for fair meta-learning evaluation.
#
# KEY FIX — OPTION A (CEB DATA SPLIT):
#   Problem: Reptile was re-training on same CEB data that
#            contrastive already saw → no new signal → plateau
#
#   Fix:
#     75% CEB → contrastive training  (encoder learns CEB well)
#     25% CEB → held out for Reptile  (genuinely unseen data)
#     JOB     → held out for few-shot evaluation only
#
#   This ensures Reptile sees genuinely new data during
#   meta-training → real gradient signal → proper meta-learning
#
# MODELS TESTED:
#   1. all-mpnet-base-v2          → proven baseline
#   2. codebert-base              → code/SQL aware
#   3. unixcoder-base             → multi-language code model
#   4. codebert-finetuned-l2code  → code-to-language mapping
#   5. grappa_large               → SQL-specific pretraining
#   6. all-MiniLM-L12-v2          → lightweight 12-layer
#
# FOLDER STRUCTURE:
#   encoders/
#     encoder_<model>_v1/        → trained encoder per model
#   results/
#     embeddings_<model>_v1.csv  → CEB embeddings per model
#     model_comparison.csv       → final ranking table
#   data/
#     ceb_contrastive_75.csv     → 75% CEB for contrastive
#     ceb_reptile_25.csv         → 25% CEB for reptile (held out)
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Paths ────────────────────────────────────────────────
    CEB_PATH     = "stage1_triplets_CEB.csv"
    JOB_PATH     = "stage1_triplets_JOB.csv"
    TRAIN_ON     = "CEB"
    ENCODERS_DIR = "encoders"    # all trained models saved here
    RESULTS_DIR  = "results"     # all CSVs saved here
    DATA_DIR     = "data"        # split CEB files saved here

    # ── CEB Split Ratio ──────────────────────────────────────
    # 75% for contrastive training
    # 25% held out for Reptile meta-training (genuinely unseen)
    CEB_TRAIN_RATIO = 0.75

    # ── ALL BASE MODELS WITH PER-MODEL SETTINGS ──────────────
    BASE_MODELS = [
        {
            "name"          : "sentence-transformers/all-mpnet-base-v2",
            "epochs"        : 5,
            "batch_size"    : 32,
            "learning_rate" : 2e-5,
            "max_seq_length": 256,
            "margin"        : 0.5,
            "notes"         : "proven baseline, general English"
        },
        {
            "name"          : "microsoft/codebert-base",
            "epochs"        : 7,
            "batch_size"    : 32,
            "learning_rate" : 2e-5,
            "max_seq_length": 512,
            "margin"        : 0.5,
            "notes"         : "code+SQL aware"
        },
        {
            "name"          : "microsoft/unixcoder-base",
            "epochs"        : 7,
            "batch_size"    : 32,
            "learning_rate" : 2e-5,
            "max_seq_length": 512,
            "margin"        : 0.5,
            "notes"         : "multi-language code model"
        },
        {
            "name"          : "neulab/codebert-base-finetuned-l2code",
            "epochs"        : 7,
            "batch_size"    : 32,
            "learning_rate" : 1e-5,
            "max_seq_length": 512,
            "margin"        : 0.5,
            "notes"         : "code-to-language mapping"
        },
        {
            "name"          : "Salesforce/grappa_large",
            "epochs"        : 5,
            "batch_size"    : 16,
            "learning_rate" : 1e-5,
            "max_seq_length": 512,
            "margin"        : 0.5,
            "notes"         : "SQL-specific pretraining"
        },
        {
            "name"          : "sentence-transformers/all-MiniLM-L12-v2",
            "epochs"        : 5,
            "batch_size"    : 64,
            "learning_rate" : 2e-5,
            "max_seq_length": 256,
            "margin"        : 0.5,
            "notes"         : "lightweight 12-layer, fast inference"
        },
    ]

    # ── Shared settings ──────────────────────────────────────
    WARMUP_RATIO = 0.1
    SEED         = 42
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

# ── Create output directories ────────────────────────────────
os.makedirs(cfg.ENCODERS_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR,  exist_ok=True)
os.makedirs(cfg.DATA_DIR,     exist_ok=True)

print("=" * 60)
print("AdaSteer — Contrastive Training (CEB Only)")
print("Multi-Model Comparison + CEB Split for Reptile")
print("=" * 60)
print(f"Device       : {cfg.DEVICE}")
print(f"Train on     : {cfg.TRAIN_ON}  ← JOB is HELD OUT")
print(f"CEB split    : {int(cfg.CEB_TRAIN_RATIO*100)}% contrastive / "
      f"{int((1-cfg.CEB_TRAIN_RATIO)*100)}% reptile")
print(f"Models       : {len(cfg.BASE_MODELS)} models to train")
print(f"Encoders dir : {cfg.ENCODERS_DIR}/")
print(f"Results dir  : {cfg.RESULTS_DIR}/")
print(f"Data dir     : {cfg.DATA_DIR}/")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD DATA
# ============================================================

print("\nLoading data...")

ceb = pd.read_csv(cfg.CEB_PATH)
job = pd.read_csv(cfg.JOB_PATH)

print(f"CEB (full)     : {len(ceb)} triplets, "
      f"{ceb['anchor'].nunique()} unique queries")
print(f"JOB (held out) : {len(job)} triplets, "
      f"{job['anchor'].nunique()} unique queries")
print(f"               ↑ JOB will NOT be used for training")

print("\nSample CEB triplet:")
print(f"  anchor   : {ceb.iloc[0]['anchor'][:80]}...")
print(f"  positive : {ceb.iloc[0]['positive']}")
print(f"  negative : {ceb.iloc[0]['negative']}")
print(f"  time_diff: {ceb.iloc[0]['time_diff']:.4f}s")


# ============================================================
# SECTION 3: SPLIT CEB INTO CONTRASTIVE (75%) AND REPTILE (25%)
# ============================================================
#
# CRITICAL: Split by QUERY not by triplet
# This ensures no query appears in both splits
# → contrastive model genuinely has not seen reptile queries
# → reptile gets real unseen data → real gradient signal
#
# ============================================================

print("\n" + "=" * 60)
print("CEB Data Split (Option A Fix)")
print("=" * 60)

# get all unique queries and shuffle
all_queries = ceb["anchor"].unique()
rng         = np.random.RandomState(cfg.SEED)
rng.shuffle(all_queries)

# split queries
n_train        = int(len(all_queries) * cfg.CEB_TRAIN_RATIO)
train_queries  = all_queries[:n_train]   # 75% for contrastive
reptile_queries= all_queries[n_train:]   # 25% for reptile

# split triplets based on query split
ceb_train   = ceb[ceb["anchor"].isin(train_queries)].reset_index(drop=True)
ceb_reptile = ceb[ceb["anchor"].isin(reptile_queries)].reset_index(drop=True)

# verify no overlap
train_q_set   = set(ceb_train["anchor"].unique())
reptile_q_set = set(ceb_reptile["anchor"].unique())
overlap       = train_q_set.intersection(reptile_q_set)

print(f"Total CEB queries    : {len(all_queries)}")
print(f"Contrastive queries  : {len(train_queries)} "
      f"({len(train_queries)/len(all_queries)*100:.1f}%)")
print(f"Reptile queries      : {len(reptile_queries)} "
      f"({len(reptile_queries)/len(all_queries)*100:.1f}%)")
print(f"Contrastive triplets : {len(ceb_train)}")
print(f"Reptile triplets     : {len(ceb_reptile)}")
print(f"Query overlap        : {len(overlap)} "
      f"← must be 0 for fair experiment")

if len(overlap) == 0:
    print("✅ No overlap — split is clean and fair!")
else:
    print("❌ ERROR — overlap detected! Check split logic.")
    exit(1)

# save splits to data/ folder
ceb_train_path   = os.path.join(cfg.DATA_DIR, "ceb_contrastive_75.csv")
ceb_reptile_path = os.path.join(cfg.DATA_DIR, "ceb_reptile_25.csv")

ceb_train.to_csv(ceb_train_path,   index=False)
ceb_reptile.to_csv(ceb_reptile_path, index=False)

print(f"\nSaved:")
print(f"  {ceb_train_path}   ← use for contrastive training")
print(f"  {ceb_reptile_path} ← use for reptile meta-training")
print(f"\n  ⚠️  Update reptile_CEB_to_JOB.py:")
print(f"     CEB_PATH = '{ceb_reptile_path}'")


# ============================================================
# SECTION 4: BUILD TRAINING EXAMPLES (75% CEB ONLY)
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
                str(row["anchor"]),
                str(row["positive"]),
                str(row["negative"])
            ]
        ))
    return examples


# use 75% split for contrastive training
train_examples = build_examples(ceb_train)

print(f"\nTraining examples built: {len(train_examples)}")
print(f"(75% CEB only — 25% CEB held for Reptile, JOB completely held out)")


# ============================================================
# SECTION 5: SANITY CHECK FUNCTION
# ============================================================

# fixed SQL queries used consistently across all models
sql1 = "SELECT COUNT(*) FROM cast_info ci JOIN title t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql2 = "SELECT COUNT(*) FROM cast_info AS ci JOIN title AS t ON ci.movie_id = t.id WHERE t.production_year > 2000"
sql3 = "SELECT MIN(name) FROM name n WHERE n.gender = 'f' GROUP BY n.name_pcode_nf"


def run_sanity_check(trained_model):
    """
    Check if model correctly identifies:
      sql1 vs sql2 → HIGH similarity (same query, different aliases)
      sql1 vs sql3 → LOW similarity  (completely different queries)

    Gap = sim(sql1,sql2) - sim(sql1,sql3)
    Higher gap = better discrimination = better model
    """
    e1 = trained_model.encode(sql1, convert_to_tensor=True)
    e2 = trained_model.encode(sql2, convert_to_tensor=True)
    e3 = trained_model.encode(sql3, convert_to_tensor=True)

    sim_12 = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    sim_13 = F.cosine_similarity(e1.unsqueeze(0), e3.unsqueeze(0)).item()
    gap    = sim_12 - sim_13

    print(f"  Sim(sql1, sql2) = {sim_12:.4f}  ← should be HIGH  (same query)")
    print(f"  Sim(sql1, sql3) = {sim_13:.4f}  ← should be LOW   (different)")
    print(f"  Gap             = {gap:.4f}  ← should be > 0.5")

    if gap > 0.5:
        print("  ✅ Excellent discrimination!")
        quality = "excellent"
    elif gap > 0.2:
        print("  ✅ Good discrimination")
        quality = "good"
    else:
        print("  ⚠️  Small gap — model may not have learned well")
        quality = "poor"

    return sim_12, sim_13, gap, quality


# ============================================================
# SECTION 6: TRAIN ALL MODELS
# ============================================================

all_model_results = []

for i, model_cfg in enumerate(cfg.BASE_MODELS):

    base_model_name  = model_cfg["name"]
    model_short_name = base_model_name.split("/")[-1]
    output_dir       = os.path.join(cfg.ENCODERS_DIR,
                                    f"encoder_{model_short_name}_v1")

    print("\n" + "=" * 60)
    print(f"Model {i+1}/{len(cfg.BASE_MODELS)}: {base_model_name}")
    print(f"Notes    : {model_cfg['notes']}")
    print(f"Output   : {output_dir}")
    print(f"Settings : epochs={model_cfg['epochs']} | "
          f"batch={model_cfg['batch_size']} | "
          f"lr={model_cfg['learning_rate']} | "
          f"maxlen={model_cfg['max_seq_length']} | "
          f"margin={model_cfg['margin']}")
    print(f"Training on 75% CEB ({len(ceb_train)} triplets, "
          f"{len(train_queries)} queries)")
    print("=" * 60)

    # ── Check GPU memory before large models ─────────────────
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        print(f"Free GPU memory: {free_mem:.1f} GB")
        if free_mem < 8.0 and "large" in base_model_name.lower():
            print(f"⚠️  Less than 8GB free — reducing batch size to 8")
            model_cfg["batch_size"] = 8

    # ── Load base model ──────────────────────────────────────
    print(f"\nLoading {base_model_name}...")
    try:
        model   = SentenceTransformer(base_model_name)
        model.max_seq_length = model_cfg["max_seq_length"]
        emb_dim = model.get_sentence_embedding_dimension()
        print(f"Model loaded! Embedding dim: {emb_dim}")
    except Exception as e:
        print(f"❌ Failed to load {base_model_name}: {e}")
        print(f"⏭️  Skipping this model...")
        all_model_results.append({
            "model"      : base_model_name,
            "short_name" : model_short_name,
            "output_dir" : "FAILED",
            "emb_dim"    : 0,
            "sim_same"   : 0,
            "sim_diff"   : 0,
            "gap"        : 0,
            "quality"    : "failed",
            "notes"      : model_cfg["notes"],
            "status"     : "load_failed"
        })
        continue

    # ── Setup training ───────────────────────────────────────
    triplet_loss = TripletLoss(
        model=model,
        triplet_margin=model_cfg["margin"]
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=model_cfg["batch_size"]
    )

    steps_per_epoch = len(train_dataloader)
    total_steps     = steps_per_epoch * model_cfg["epochs"]
    warmup_steps    = int(total_steps * cfg.WARMUP_RATIO)

    print(f"\nTraining setup:")
    print(f"  Triplets        : {len(train_examples)} (75% CEB)")
    print(f"  Steps per epoch : {steps_per_epoch}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Warmup steps    : {warmup_steps}")
    print(f"  Estimated time  : ~{total_steps / 2.4 / 60:.0f} minutes")

    # ── Train ────────────────────────────────────────────────
    print(f"\nStarting training...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        model.fit(
            train_objectives = [(train_dataloader, triplet_loss)],
            epochs           = model_cfg["epochs"],
            warmup_steps     = warmup_steps,
            optimizer_params = {"lr": model_cfg["learning_rate"]},
            output_path      = output_dir,
            show_progress_bar= True,
            save_best_model  = True,
        )
        print(f"✅ Training complete! Saved to: {output_dir}/")
        status = "success"
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"⏭️  Skipping to next model...")
        all_model_results.append({
            "model"      : base_model_name,
            "short_name" : model_short_name,
            "output_dir" : output_dir,
            "emb_dim"    : emb_dim,
            "sim_same"   : 0,
            "sim_diff"   : 0,
            "gap"        : 0,
            "quality"    : "failed",
            "notes"      : model_cfg["notes"],
            "status"     : "train_failed"
        })
        del model
        torch.cuda.empty_cache()
        continue

    # ── Sanity check ─────────────────────────────────────────
    print(f"\nSanity check for {model_short_name}:")
    try:
        trained_model = SentenceTransformer(output_dir)
        trained_model.max_seq_length = model_cfg["max_seq_length"]
        sim_12, sim_13, gap, quality = run_sanity_check(trained_model)
    except Exception as e:
        print(f"❌ Sanity check failed: {e}")
        sim_12, sim_13, gap, quality = 0, 0, 0, "failed"
        trained_model = None

    # ── Generate and save embeddings ─────────────────────────
    if trained_model is not None:
        print(f"\nGenerating CEB embeddings for {model_short_name}...")
        try:
            # embed the 75% training queries only
            ceb_queries    = ceb_train["anchor"].unique().tolist()
            ceb_embeddings = trained_model.encode(
                ceb_queries,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"CEB embeddings shape: {ceb_embeddings.shape}")

            emb_df   = pd.DataFrame({
                "query"      : ceb_queries,
                "source_task": "CEB_train_75",
                "model"      : model_short_name
            })
            emb_cols = pd.DataFrame(
                ceb_embeddings,
                columns=[f"dim_{j}" for j in range(ceb_embeddings.shape[1])]
            )
            emb_df = pd.concat([emb_df, emb_cols], axis=1)

            emb_path = os.path.join(cfg.RESULTS_DIR,
                                    f"embeddings_{model_short_name}_v1.csv")
            emb_df.to_csv(emb_path, index=False)
            print(f"Embeddings saved: {emb_path}")

        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")

    # ── Store result ─────────────────────────────────────────
    all_model_results.append({
        "model"      : base_model_name,
        "short_name" : model_short_name,
        "output_dir" : output_dir,
        "emb_dim"    : emb_dim,
        "sim_same"   : round(sim_12, 4),
        "sim_diff"   : round(sim_13, 4),
        "gap"        : round(gap, 4),
        "quality"    : quality,
        "notes"      : model_cfg["notes"],
        "status"     : status
    })

    # ── Free GPU memory before next model ────────────────────
    print(f"\nFreeing GPU memory...")
    del model
    if trained_model is not None:
        del trained_model
    torch.cuda.empty_cache()
    print(f"✅ Done with {model_short_name}\n")


# ============================================================
# SECTION 7: FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON TABLE")
print("Sorted by discrimination gap (higher = better)")
print("=" * 60)

successful = [r for r in all_model_results if r["status"] == "success"]
failed     = [r for r in all_model_results if r["status"] != "success"]

# sort by gap descending
successful.sort(key=lambda x: x["gap"], reverse=True)

print(f"\n{'Rank':<5} | {'Model':<40} | {'Gap':>6} | "
      f"{'Sim Same':>9} | {'Sim Diff':>9} | {'Quality':<10} | Notes")
print("-" * 110)

for rank, r in enumerate(successful, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{medal} {rank:<3} | "
          f"{r['short_name']:<40} | "
          f"{r['gap']:>6.4f} | "
          f"{r['sim_same']:>9.4f} | "
          f"{r['sim_diff']:>9.4f} | "
          f"{r['quality']:<10} | "
          f"{r['notes']}")

if failed:
    print("\nFailed models:")
    for r in failed:
        print(f"  ❌ {r['short_name']:<40} → {r['status']}")

print("-" * 110)

if successful:
    best = successful[0]
    print(f"\n🏆 Best model  : {best['model']}")
    print(f"   Output dir  : {best['output_dir']}/")
    print(f"   Gap score   : {best['gap']:.4f}")
    print(f"   Quality     : {best['quality']}")
    print(f"\n   → Use this in reptile_CEB_to_JOB.py:")
    print(f"     CONTRASTIVE_CEB_PATH = '{best['output_dir']}'")
    print(f"     CEB_PATH             = '{ceb_reptile_path}'")

# ── Save comparison CSV ──────────────────────────────────────
comparison_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
results_df      = pd.DataFrame(all_model_results)
results_df.to_csv(comparison_path, index=False)
print(f"\nFull comparison saved: {comparison_path}")


# ============================================================
# SECTION 8: COMPLETE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ALL MODELS TRAINED — COMPLETE")
print("=" * 60)

print(f"\nData split (Option A fix):")
print(f"  {ceb_train_path}")
print(f"    → {len(ceb_train)} triplets, {len(train_queries)} queries (contrastive)")
print(f"  {ceb_reptile_path}")
print(f"    → {len(ceb_reptile)} triplets, {len(reptile_queries)} queries (reptile)")

print(f"\nFolder structure:")
print(f"  {cfg.ENCODERS_DIR}/")
for r in successful:
    print(f"    {os.path.basename(r['output_dir']):<45} "
          f"gap={r['gap']:.4f} ({r['quality']})")

print(f"\n  {cfg.RESULTS_DIR}/")
print(f"    model_comparison.csv")
for r in successful:
    print(f"    embeddings_{r['short_name']}_v1.csv")

print(f"\n  {cfg.DATA_DIR}/")
print(f"    ceb_contrastive_75.csv  ← used for contrastive training")
print(f"    ceb_reptile_25.csv      ← use for reptile meta-training")

print("\nNext steps:")
print("  1. Check results/model_comparison.csv for best model")
print("  2. Update reptile_CEB_to_JOB.py with these paths:")
if successful:
    print(f"       CONTRASTIVE_CEB_PATH = '{successful[0]['output_dir']}'")
    print(f"       CEB_PATH             = '{ceb_reptile_path}'")
print("  3. Run reptile meta-training")
print("     → model will see genuinely new CEB data")
print("     → real gradient signal → no early plateau")
print("  4. Run few-shot evaluation on JOB")
print("=" * 60)


# # ============================================================
# # AdaSteer — Contrastive Training (CEB Only)
# # Multi-Model Comparison
# # ============================================================
# #
# # PURPOSE:
# #   Train contrastive encoder on CEB workload ONLY.
# #   Tests 5 different base models to find best for SQL tasks.
# #   JOB completely held out for fair meta-learning evaluation.
# #
# # MODELS TESTED:
# #   1. all-mpnet-base-v2          → proven baseline
# #   2. codebert-base              → code/SQL aware
# #   3. unixcoder-base             → multi-language code model
# #   4. codebert-finetuned-l2code  → code-to-language mapping
# #   5. grappa_large               → SQL-specific pretraining
# #
# # FOLDER STRUCTURE:
# #   encoders/
# #     encoder_<model>_v1/        → trained encoder per model
# #   results/
# #     embeddings_<model>_v1.csv  → CEB embeddings per model
# #     model_comparison.csv       → final ranking table
# # ============================================================

# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer, InputExample
# from sentence_transformers.losses import TripletLoss
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import warnings
# warnings.filterwarnings("ignore")


# # ============================================================
# # SECTION 1: CONFIGURATION
# # ============================================================

# class Config:

#     # ── Paths ────────────────────────────────────────────────
#     CEB_PATH     = "stage1_triplets_CEB.csv"
#     JOB_PATH     = "stage1_triplets_JOB.csv"
#     TRAIN_ON     = "CEB"
#     ENCODERS_DIR = "encoders"    # all trained models saved here
#     RESULTS_DIR  = "results"     # all CSVs saved here

#     # ── ALL BASE MODELS WITH PER-MODEL SETTINGS ──────────────
#     BASE_MODELS = [
#         {
#         "name"          : "sentence-transformers/all-mpnet-base-v2",
#         "epochs"        : 5,
#         "batch_size"    : 32,
#         "learning_rate" : 2e-5,
#         "max_seq_length": 256,
#         "margin"        : 0.5,
#         "notes"         : "proven baseline, general English"
#         },
#         {
#             "name"          : "microsoft/codebert-base",
#             "epochs"        : 7,
#             "batch_size"    : 32,
#             "learning_rate" : 2e-5,
#             "max_seq_length": 512,
#             "margin"        : 0.5,
#             "notes"         : "code+SQL aware"
#         },
#         {
#             "name"          : "microsoft/unixcoder-base",
#             "epochs"        : 7,
#             "batch_size"    : 32,
#             "learning_rate" : 2e-5,
#             "max_seq_length": 512,
#             "margin"        : 0.5,
#             "notes"         : "multi-language code model"
#         },
#         {
#             "name"          : "neulab/codebert-base-finetuned-l2code",
#             "epochs"        : 7,
#             "batch_size"    : 32,
#             "learning_rate" : 1e-5,
#             "max_seq_length": 512,
#             "margin"        : 0.5,
#             "notes"         : "code-to-language mapping"
#         },
#         {
#             "name"          : "Salesforce/grappa_large",
#             "epochs"        : 5,
#             "batch_size"    : 16,
#             "learning_rate" : 1e-5,
#             "max_seq_length": 512,
#             "margin"        : 0.5,
#             "notes"         : "SQL-specific pretraining"
#         },
#         {
#             "name"          : "sentence-transformers/all-MiniLM-L12-v2",
#             "epochs"        : 5,
#             "batch_size"    : 64,
#             "learning_rate" : 2e-5,
#             "max_seq_length": 256,
#             "margin"        : 0.5,
#             "notes"         : "lightweight 12-layer, fast inference"
#         },
#     ]

#     # ── Shared settings ──────────────────────────────────────
#     WARMUP_RATIO = 0.1
#     SEED         = 42
#     DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# cfg = Config()


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(cfg.SEED)

# # ── Create output directories ────────────────────────────────
# os.makedirs(cfg.ENCODERS_DIR, exist_ok=True)
# os.makedirs(cfg.RESULTS_DIR,  exist_ok=True)

# print("=" * 60)
# print("AdaSteer — Contrastive Training (CEB Only)")
# print("Multi-Model Comparison")
# print("=" * 60)
# print(f"Device       : {cfg.DEVICE}")
# print(f"Train on     : {cfg.TRAIN_ON}  ← JOB is HELD OUT")
# print(f"Models       : {len(cfg.BASE_MODELS)} models to train")
# print(f"Encoders dir : {cfg.ENCODERS_DIR}/")
# print(f"Results dir  : {cfg.RESULTS_DIR}/")
# print("=" * 60)


# # ============================================================
# # SECTION 2: LOAD DATA
# # ============================================================

# print("\nLoading CEB data...")

# ceb = pd.read_csv(cfg.CEB_PATH)
# job = pd.read_csv(cfg.JOB_PATH)

# print(f"CEB (training) : {len(ceb)} triplets, {ceb['anchor'].nunique()} unique queries")
# print(f"JOB (held out) : {len(job)} triplets, {job['anchor'].nunique()} unique queries")
# print(f"                 ↑ JOB will NOT be used for training")

# print("\nSample CEB triplet:")
# print(f"  anchor   : {ceb.iloc[0]['anchor'][:80]}...")
# print(f"  positive : {ceb.iloc[0]['positive']}")
# print(f"  negative : {ceb.iloc[0]['negative']}")
# print(f"  time_diff: {ceb.iloc[0]['time_diff']:.4f}s")


# # ============================================================
# # SECTION 3: BUILD TRAINING EXAMPLES
# # ============================================================

# def build_examples(df):
#     """
#     Convert dataframe rows into InputExample objects
#     for sentence-transformers TripletLoss.
#     Each example = (anchor SQL, positive hint, negative hint)
#     """
#     examples = []
#     for _, row in df.iterrows():
#         examples.append(InputExample(
#             texts=[
#                 str(row["anchor"]),
#                 str(row["positive"]),
#                 str(row["negative"])
#             ]
#         ))
#     return examples


# train_examples = build_examples(ceb)
# print(f"\nTraining examples built: {len(train_examples)}")
# print(f"(Only CEB — JOB completely held out)")


# # ============================================================
# # SECTION 4: SANITY CHECK FUNCTION
# # ============================================================

# # fixed SQL queries used consistently across all models
# sql1 = "SELECT COUNT(*) FROM cast_info ci JOIN title t ON ci.movie_id = t.id WHERE t.production_year > 2000"
# sql2 = "SELECT COUNT(*) FROM cast_info AS ci JOIN title AS t ON ci.movie_id = t.id WHERE t.production_year > 2000"
# sql3 = "SELECT MIN(name) FROM name n WHERE n.gender = 'f' GROUP BY n.name_pcode_nf"


# def run_sanity_check(trained_model):
#     """
#     Check if model correctly identifies:
#       sql1 vs sql2 → HIGH similarity (same query, different aliases)
#       sql1 vs sql3 → LOW similarity  (completely different queries)

#     Gap = sim(sql1,sql2) - sim(sql1,sql3)
#     Higher gap = better discrimination = better model
#     """
#     e1 = trained_model.encode(sql1, convert_to_tensor=True)
#     e2 = trained_model.encode(sql2, convert_to_tensor=True)
#     e3 = trained_model.encode(sql3, convert_to_tensor=True)

#     sim_12 = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
#     sim_13 = F.cosine_similarity(e1.unsqueeze(0), e3.unsqueeze(0)).item()
#     gap    = sim_12 - sim_13

#     print(f"  Sim(sql1, sql2) = {sim_12:.4f}  ← should be HIGH  (same query)")
#     print(f"  Sim(sql1, sql3) = {sim_13:.4f}  ← should be LOW   (different)")
#     print(f"  Gap             = {gap:.4f}  ← should be > 0.5")

#     if gap > 0.5:
#         print("  ✅ Excellent discrimination!")
#         quality = "excellent"
#     elif gap > 0.2:
#         print("  ✅ Good discrimination")
#         quality = "good"
#     else:
#         print("  ⚠️  Small gap — model may not have learned well")
#         quality = "poor"

#     return sim_12, sim_13, gap, quality


# # ============================================================
# # SECTION 5: TRAIN ALL MODELS
# # ============================================================

# all_model_results = []

# for i, model_cfg in enumerate(cfg.BASE_MODELS):

#     base_model_name  = model_cfg["name"]
#     model_short_name = base_model_name.split("/")[-1]
#     output_dir       = os.path.join(cfg.ENCODERS_DIR,
#                                     f"encoder_{model_short_name}_v1")

#     print("\n" + "=" * 60)
#     print(f"Model {i+1}/{len(cfg.BASE_MODELS)}: {base_model_name}")
#     print(f"Notes    : {model_cfg['notes']}")
#     print(f"Output   : {output_dir}")
#     print(f"Settings : epochs={model_cfg['epochs']} | "
#           f"batch={model_cfg['batch_size']} | "
#           f"lr={model_cfg['learning_rate']} | "
#           f"maxlen={model_cfg['max_seq_length']} | "
#           f"margin={model_cfg['margin']}")
#     print("=" * 60)

#     # ── Check GPU memory before large models ─────────────────
#     if torch.cuda.is_available():
#         free_mem = torch.cuda.mem_get_info()[0] / 1e9
#         print(f"Free GPU memory: {free_mem:.1f} GB")
#         if free_mem < 8.0 and "large" in base_model_name.lower():
#             print(f"⚠️  Less than 8GB free — reducing batch size to 8")
#             model_cfg["batch_size"] = 8

#     # ── Load base model ──────────────────────────────────────
#     print(f"\nLoading {base_model_name}...")
#     try:
#         model   = SentenceTransformer(base_model_name)
#         model.max_seq_length = model_cfg["max_seq_length"]
#         emb_dim = model.get_sentence_embedding_dimension()
#         print(f"Model loaded! Embedding dim: {emb_dim}")
#     except Exception as e:
#         print(f"❌ Failed to load {base_model_name}: {e}")
#         print(f"⏭️  Skipping this model...")
#         all_model_results.append({
#             "model"      : base_model_name,
#             "short_name" : model_short_name,
#             "output_dir" : "FAILED",
#             "emb_dim"    : 0,
#             "sim_same"   : 0,
#             "sim_diff"   : 0,
#             "gap"        : 0,
#             "quality"    : "failed",
#             "notes"      : model_cfg["notes"],
#             "status"     : "load_failed"
#         })
#         continue

#     # ── Setup training ───────────────────────────────────────
#     triplet_loss = TripletLoss(
#         model=model,
#         triplet_margin=model_cfg["margin"]
#     )

#     train_dataloader = DataLoader(
#         train_examples,
#         shuffle=True,
#         batch_size=model_cfg["batch_size"]
#     )

#     steps_per_epoch = len(train_dataloader)
#     total_steps     = steps_per_epoch * model_cfg["epochs"]
#     warmup_steps    = int(total_steps * cfg.WARMUP_RATIO)

#     print(f"\nTraining setup:")
#     print(f"  Triplets        : {len(train_examples)}")
#     print(f"  Steps per epoch : {steps_per_epoch}")
#     print(f"  Total steps     : {total_steps}")
#     print(f"  Warmup steps    : {warmup_steps}")
#     print(f"  Estimated time  : ~{total_steps / 2.4 / 60:.0f} minutes")

#     # ── Train ────────────────────────────────────────────────
#     print(f"\nStarting training...")
#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         model.fit(
#             train_objectives = [(train_dataloader, triplet_loss)],
#             epochs           = model_cfg["epochs"],
#             warmup_steps     = warmup_steps,
#             optimizer_params = {"lr": model_cfg["learning_rate"]},
#             output_path      = output_dir,
#             show_progress_bar= True,
#             save_best_model  = True,
#         )
#         print(f"✅ Training complete! Saved to: {output_dir}/")
#         status = "success"
#     except Exception as e:
#         print(f"❌ Training failed: {e}")
#         print(f"⏭️  Skipping to next model...")
#         all_model_results.append({
#             "model"      : base_model_name,
#             "short_name" : model_short_name,
#             "output_dir" : output_dir,
#             "emb_dim"    : emb_dim,
#             "sim_same"   : 0,
#             "sim_diff"   : 0,
#             "gap"        : 0,
#             "quality"    : "failed",
#             "notes"      : model_cfg["notes"],
#             "status"     : "train_failed"
#         })
#         del model
#         torch.cuda.empty_cache()
#         continue

#     # ── Sanity check ─────────────────────────────────────────
#     print(f"\nSanity check for {model_short_name}:")
#     try:
#         trained_model = SentenceTransformer(output_dir)
#         trained_model.max_seq_length = model_cfg["max_seq_length"]
#         sim_12, sim_13, gap, quality = run_sanity_check(trained_model)
#     except Exception as e:
#         print(f"❌ Sanity check failed: {e}")
#         sim_12, sim_13, gap, quality = 0, 0, 0, "failed"
#         trained_model = None

#     # ── Generate and save embeddings ─────────────────────────
#     if trained_model is not None:
#         print(f"\nGenerating CEB embeddings for {model_short_name}...")
#         try:
#             ceb_queries    = ceb["anchor"].unique().tolist()
#             ceb_embeddings = trained_model.encode(
#                 ceb_queries,
#                 batch_size=64,
#                 show_progress_bar=True,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True
#             )
#             print(f"CEB embeddings shape: {ceb_embeddings.shape}")

#             emb_df   = pd.DataFrame({
#                 "query"      : ceb_queries,
#                 "source_task": "CEB",
#                 "model"      : model_short_name
#             })
#             emb_cols = pd.DataFrame(
#                 ceb_embeddings,
#                 columns=[f"dim_{j}" for j in range(ceb_embeddings.shape[1])]
#             )
#             emb_df = pd.concat([emb_df, emb_cols], axis=1)

#             emb_path = os.path.join(cfg.RESULTS_DIR,
#                                     f"embeddings_{model_short_name}_v1.csv")
#             emb_df.to_csv(emb_path, index=False)
#             print(f"Embeddings saved: {emb_path}")

#         except Exception as e:
#             print(f"❌ Embedding generation failed: {e}")

#     # ── Store result ─────────────────────────────────────────
#     all_model_results.append({
#         "model"      : base_model_name,
#         "short_name" : model_short_name,
#         "output_dir" : output_dir,
#         "emb_dim"    : emb_dim,
#         "sim_same"   : round(sim_12, 4),
#         "sim_diff"   : round(sim_13, 4),
#         "gap"        : round(gap, 4),
#         "quality"    : quality,
#         "notes"      : model_cfg["notes"],
#         "status"     : status
#     })

#     # ── Free GPU memory before next model ────────────────────
#     print(f"\nFreeing GPU memory...")
#     del model
#     if trained_model is not None:
#         del trained_model
#     torch.cuda.empty_cache()
#     print(f"✅ Done with {model_short_name}\n")


# # ============================================================
# # SECTION 6: FINAL COMPARISON TABLE
# # ============================================================

# print("\n" + "=" * 60)
# print("FINAL MODEL COMPARISON TABLE")
# print("Sorted by discrimination gap (higher = better)")
# print("=" * 60)

# successful = [r for r in all_model_results if r["status"] == "success"]
# failed     = [r for r in all_model_results if r["status"] != "success"]

# # sort by gap descending
# successful.sort(key=lambda x: x["gap"], reverse=True)

# print(f"\n{'Rank':<5} | {'Model':<40} | {'Gap':>6} | "
#       f"{'Sim Same':>9} | {'Sim Diff':>9} | {'Quality':<10} | Notes")
# print("-" * 110)

# for rank, r in enumerate(successful, 1):
#     medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
#     print(f"{medal} {rank:<3} | "
#           f"{r['short_name']:<40} | "
#           f"{r['gap']:>6.4f} | "
#           f"{r['sim_same']:>9.4f} | "
#           f"{r['sim_diff']:>9.4f} | "
#           f"{r['quality']:<10} | "
#           f"{r['notes']}")

# if failed:
#     print("\nFailed models:")
#     for r in failed:
#         print(f"  ❌ {r['short_name']:<40} → {r['status']}")

# print("-" * 110)

# if successful:
#     best = successful[0]
#     print(f"\n🏆 Best model  : {best['model']}")
#     print(f"   Output dir  : {best['output_dir']}/")
#     print(f"   Gap score   : {best['gap']:.4f}")
#     print(f"   Quality     : {best['quality']}")
#     print(f"\n   → Use this in reptile_CEB_to_JOB.py:")
#     print(f"     CONTRASTIVE_CEB_PATH = '{best['output_dir']}'")

# # ── Save comparison CSV ──────────────────────────────────────
# comparison_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
# results_df = pd.DataFrame(all_model_results)
# results_df.to_csv(comparison_path, index=False)
# print(f"\nFull comparison saved: {comparison_path}")


# # ============================================================
# # SECTION 7: COMPLETE SUMMARY
# # ============================================================

# print("\n" + "=" * 60)
# print("ALL MODELS TRAINED — COMPLETE")
# print("=" * 60)

# print(f"\nFolder structure:")
# print(f"  {cfg.ENCODERS_DIR}/")
# for r in successful:
#     print(f"    {os.path.basename(r['output_dir']):<45} "
#           f"gap={r['gap']:.4f} ({r['quality']})")

# print(f"\n  {cfg.RESULTS_DIR}/")
# print(f"    model_comparison.csv")
# for r in successful:
#     print(f"    embeddings_{r['short_name']}_v1.csv")

# print("\nNext steps:")
# print("  1. Check results/model_comparison.csv for best model")
# print("  2. Update reptile_CEB_to_JOB.py:")
# if successful:
#     print(f"     CONTRASTIVE_CEB_PATH = '{successful[0]['output_dir']}'")
# print("  3. Run reptile meta-training with best encoder")
# print("  4. Run few-shot evaluation")
# print("=" * 60)



