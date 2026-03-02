import os
import random
import time
import json
import argparse

import pandas as pd
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses, util


BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

JOB_CSV = "stage1_triplets_JOB.csv"
CEB_CSV = "stage1_triplets_CEB.csv"

DEFAULT_OUTPUT_DIR = "othertests"

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_MAX_SEQ_LENGTH = 256

DEFAULT_TOTAL_STEPS = 6000
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(csv_path: str):
    df = pd.read_csv(csv_path)
    if "anchor" not in df.columns or "positive" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: anchor, positive")

    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)

    return [InputExample(texts=[a, p]) for a, p in zip(df["anchor"], df["positive"])]


def build_balanced_sampler(job_pairs, ceb_pairs, total_steps: int, batch_size: int):
    all_pairs = job_pairs + ceb_pairs

    if len(job_pairs) == 0 or len(ceb_pairs) == 0:
        raise ValueError("JOB or CEB has zero samples. Cannot balance.")

    # This gives 50/50 probability mass across JOB and CEB.
    w_job = 0.5 / len(job_pairs)
    w_ceb = 0.5 / len(ceb_pairs)

    weights = [w_job] * len(job_pairs) + [w_ceb] * len(ceb_pairs)

    # Make dataloader length equal to total_steps.
    # DataLoader length = num_samples / batch_size (with drop_last=True).
    num_samples = total_steps * batch_size

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    return all_pairs, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_csv", type=str, default=JOB_CSV)
    parser.add_argument("--ceb_csv", type=str, default=CEB_CSV)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)

    parser.add_argument("--total_steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()

    if args.total_steps <= 0:
        raise ValueError("--total_steps must be positive")

    if not os.path.exists(args.job_csv):
        raise FileNotFoundError(f"Missing file: {args.job_csv}")

    if not os.path.exists(args.ceb_csv):
        raise FileNotFoundError(f"Missing file: {args.ceb_csv}")

    set_seed(args.seed)

    print(f"Loading pairs from {args.job_csv} and {args.ceb_csv} ...")
    job_pairs = load_pairs(args.job_csv)
    ceb_pairs = load_pairs(args.ceb_csv)

    print(f"JOB pairs: {len(job_pairs)}")
    print(f"CEB pairs: {len(ceb_pairs)}")

    all_pairs, sampler = build_balanced_sampler(
        job_pairs=job_pairs,
        ceb_pairs=ceb_pairs,
        total_steps=args.total_steps,
        batch_size=args.batch_size
    )

    model = SentenceTransformer(args.base_model)
    try:
        model.max_seq_length = args.max_seq_length
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    train_dataloader = DataLoader(
        all_pairs,
        sampler=sampler,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,
        drop_last=True
    )

    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        similarity_fct=util.cos_sim
    )

    warmup_steps = int(args.warmup_ratio * args.total_steps)

    run_name = (
        f"contrastive_mpnet_steps{args.total_steps}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_seed{args.seed}"
        f"_{int(time.time())}"
    )
    output_path = os.path.join(args.output_dir, run_name)
    os.makedirs(output_path, exist_ok=True)

    # Save run config for reproducibility
    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "job_csv": args.job_csv,
                "ceb_csv": args.ceb_csv,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_seq_length": args.max_seq_length,
                "total_steps": args.total_steps,
                "warmup_steps": warmup_steps,
                "warmup_ratio": args.warmup_ratio,
                "seed": args.seed
            },
            f,
            indent=2
        )

    # Strict control: exactly total_steps updates.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        steps_per_epoch=args.total_steps,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=output_path,
        use_amp=True
    )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()



# import os
# import random
# import pandas as pd
# import torch

# from torch.utils.data import DataLoader, WeightedRandomSampler
# from sentence_transformers import SentenceTransformer, InputExample, losses, util

# # ---------------- Configuration ----------------
# BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
# OUTPUT_PATH = "othertests/contrastive_mpnet_stage_2"

# JOB_CSV = "stage1_triplets_JOB.csv"
# CEB_CSV = "stage1_triplets_CEB.csv"

# EPOCHS = 2
# BATCH_SIZE = 32
# LR = 2e-5

# # Steps per epoch controls how many batches you train each epoch.
# # With BATCH_SIZE=32, STEPS_PER_EPOCH=2000 means 64k pairs seen per epoch.
# STEPS_PER_EPOCH = 3000

# MAX_SEQ_LENGTH = 256
# WARMUP_RATIO = 0.1

# SEED = 42
# # ----------------------------------------------


# def set_seed(seed: int):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def load_pairs(csv_path: str):
#     df = pd.read_csv(csv_path)
#     df["anchor"] = df["anchor"].astype(str)
#     df["positive"] = df["positive"].astype(str)

#     pairs = [InputExample(texts=[a, p]) for a, p in zip(df["anchor"], df["positive"])]
#     return pairs


# def main():
#     set_seed(SEED)

#     if not os.path.exists(JOB_CSV) or not os.path.exists(CEB_CSV):
#         raise FileNotFoundError("Missing JOB or CEB csv. Check file paths and working directory.")

#     print(f"Loading pairs from {JOB_CSV} and {CEB_CSV} ...")
#     job_pairs = load_pairs(JOB_CSV)
#     ceb_pairs = load_pairs(CEB_CSV)

#     print(f"JOB pairs: {len(job_pairs)}")
#     print(f"CEB pairs: {len(ceb_pairs)}")

#     all_pairs = job_pairs + ceb_pairs

#     # Balance tasks by sampling probability inversely proportional to dataset size
#     w_job = 1.0 / max(len(job_pairs), 1)
#     w_ceb = 1.0 / max(len(ceb_pairs), 1)
#     weights = [w_job] * len(job_pairs) + [w_ceb] * len(ceb_pairs)

#     num_samples = EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE
#     sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

#     model = SentenceTransformer(BASE_MODEL)
#     try:
#         model.max_seq_length = MAX_SEQ_LENGTH
#     except Exception:
#         pass

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Training on: {device}")

#     train_dataloader = DataLoader(
#         all_pairs,
#         sampler=sampler,
#         batch_size=BATCH_SIZE,
#         collate_fn=lambda x: x,
#         drop_last=True,
#     )

#     train_loss = losses.MultipleNegativesRankingLoss(model, similarity_fct=util.cos_sim)

#     warmup_steps = int(WARMUP_RATIO * STEPS_PER_EPOCH * EPOCHS)

#     model.fit(
#         train_objectives=[(train_dataloader, train_loss)],
#         epochs=EPOCHS,
#         steps_per_epoch=STEPS_PER_EPOCH,
#         warmup_steps=warmup_steps,
#         optimizer_params={"lr": LR},
#         show_progress_bar=True,
#         output_path=OUTPUT_PATH,
#         use_amp=True,  # mixed precision on CUDA
#     )

#     print(f"Saved to: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()
