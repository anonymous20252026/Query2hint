# AdaptSteer — Training Scripts

Scripts for building AdaptSteer-C and AdaptSteer-R encoders from scratch.
Run in the order shown below.

---

## Stage 1 — Generate triplet datasets

```bash
python data_preparation.py      # Reptile meta-learning triplets
python dataset_generation.py    # Stage-1 contrastive triplets
```

Pre-built triplets are already in `data/` — skip this if you use them directly.

---

## Stage 2 — Contrastive encoder (AdaptSteer-C)

| Script | Purpose |
|--------|---------|
| `contrastive_encoder_training.py` | **Main script.** Fine-tunes `all-mpnet-base-v2` with TripletLoss on JOB+CEB triplets. Saves encoder to `finetuned_models/`. |
| `contrastive_ceb_only.py` | Trains on CEB workload only (no JOB). Used to create a clean source encoder for Reptile meta-training. |
| `contrastive_ceb_only_with_datasplit.py` | Improved version: splits CEB 75/25 so Reptile sees genuinely held-out data. Multi-model comparison (MPNet, CodeBERT, …). |

**Quick start — AdaptSteer-C:**

```bash
python training/contrastive_encoder_training.py
```

---

## Stage 3 — Reptile meta-learning (AdaptSteer-R)

| Script | Purpose |
|--------|---------|
| `reptile_meta_training.py` | **Main script.** Applies Reptile across JOB and CEB task episodes starting from the AdaptSteer-C encoder. Saves `encoders/encoder_reptile_mpnet_v4/`. |
| `reptile_ceb_to_job.py` | Cross-workload Reptile: trains on CEB sub-tasks, held-out JOB for evaluation (v2 settings). |
| `reptile_ceb_to_job_final.py` | **Canonical CEB→JOB script** (v3 settings: 3 inner steps, 50 tasks). Produces `encoders/adaptsteer_r/` (the final encoder used in the paper). |

**Quick start — AdaptSteer-R:**

```bash
# Requires AdaptSteer-C encoder at encoders/adaptsteer_c/
python training/reptile_meta_training.py
```

---

## Trained encoder locations

After training, encoders are saved to:

```
encoders/
├── adaptsteer_c/    ← output of contrastive_encoder_training.py
└── adaptsteer_r/    ← output of reptile_ceb_to_job_final.py
```

Pre-trained versions of both are already included in `encoders/`
(weights excluded — train or request from authors).
