import pandas as pd
import torch
import copy
import random
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# --- Configuration ---
BASE_MODEL = "bert-base-uncased"
OUTPUT_PATH = "reptile_BERT"

# Hyperparameters (your updated smooth settings)
META_EPOCHS = 10           # Total Outer Loops
INNER_STEPS = 10           # Steps per task
INNER_LR = 2e-5            # Inner-loop learning rate
META_LR = 0.1              # Reptile outer update size
BATCH_SIZE = 32            # Batch size
MAX_SEQ_LENGTH = 256


def move_features_to_device(features, device):
    """
    sentence-transformers produces features as a list of dicts,
    e.g. [{"input_ids": ..., "attention_mask": ...}, ...]
    This helper moves every tensor inside those dicts onto device.
    """
    for feat in features:
        for k, v in feat.items():
            if torch.is_tensor(v):
                feat[k] = v.to(device)
    return features


def train_reptile_stage1():
    print(f"🦎 [Stage 1] Initializing Reptile Meta-Learning with {BASE_MODEL} (Contrastive)...")

    # --- 1. Load Data ---
    files = {
        "JOB": "stage1_triplets_JOB.csv",
        "CEB": "stage1_triplets_CEB.csv"
    }

    tasks = {}
    for task_name, fname in files.items():
        if os.path.exists(fname):
            print(f"   Loading {task_name} task from {fname}...")
            df = pd.read_csv(fname)

            # Contrastive setup uses (anchor, positive) pairs.
            # The "negative" column is intentionally ignored because the batch provides negatives.
            examples = [
                InputExample(texts=[str(r["anchor"]), str(r["positive"])])
                for _, r in df.iterrows()
            ]
            tasks[task_name] = examples
            print(f"   ✅ {task_name}: {len(examples)} pairs.")
        else:
            print(f"   ❌ Warning: {fname} not found.")

    if not tasks:
        print("❌ No datasets found! Please generate them first.")
        return

    # --- 2. Build the Model ---
    print("🏗️  Constructing Model Architecture...")
    word_embedding_model = models.Transformer(BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    normalize_model = models.Normalize()
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model])

    # Contrastive loss: each other positive in the batch acts as a negative
    loss_model = losses.MultipleNegativesRankingLoss(model=model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_model.to(device)

    print(f"🔥 Training on: {device}")

    # --- 3. Meta-Training Loop ---
    for meta_epoch in range(META_EPOCHS):
        meta_weights = copy.deepcopy(model.state_dict())
        task_names = list(tasks.keys())
        random.shuffle(task_names)

        print(f"\n--- Meta Epoch {meta_epoch + 1}/{META_EPOCHS} ---")

        for task_name in task_names:
            task_data = tasks[task_name]

            # Sample Support Set
            sample_size = BATCH_SIZE * INNER_STEPS
            if len(task_data) < sample_size:
                batch_data = random.choices(task_data, k=sample_size)
            else:
                batch_data = random.sample(task_data, sample_size)

            train_dataloader = DataLoader(
                batch_data,
                shuffle=True,
                batch_size=BATCH_SIZE,
                collate_fn=lambda x: x
            )

            # Inner Loop: Adaptation
            model.load_state_dict(meta_weights)
            inner_optimizer = torch.optim.AdamW(model.parameters(), lr=INNER_LR)

            model.train()
            loss_model.train()

            total_loss = 0.0
            steps = 0

            for batch in train_dataloader:
                if steps >= INNER_STEPS:
                    break

                features, labels = model.smart_batching_collate(batch)
                features = move_features_to_device(features, device)

                # MultipleNegativesRankingLoss does not need labels
                loss_value = loss_model(features, labels=None)
                loss_value.backward()

                inner_optimizer.step()
                inner_optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss_value.item())
                steps += 1

            # Reptile Update
            adapted_weights = model.state_dict()
            for key in meta_weights:
                meta_weights[key] += (META_LR / len(task_names)) * (adapted_weights[key] - meta_weights[key])

            avg_loss = total_loss / max(steps, 1)
            print(f"   [{task_name}] Loss: {avg_loss:.4f}")

        model.load_state_dict(meta_weights)

    # --- 4. Save ---
    print(f"💾 Saving Pre-trained Encoder to {OUTPUT_PATH}...")
    model.save(OUTPUT_PATH)
    print("✅ Stage 1 Complete.")


if __name__ == "__main__":
    train_reptile_stage1()
