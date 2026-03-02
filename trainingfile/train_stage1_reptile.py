import argparse
import copy
import os
import random
import re

import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BASE_MODEL = "bert-base-uncased"
OUTPUT_PATH = "reptile_BERT"
DEFAULT_OUTPUT_DIR = "reptile_models"

META_EPOCHS = 20
INNER_STEPS = 10
INNER_LR = 2e-5
META_LR = 0.1
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
TRIPLET_MARGIN = 0.2
SEED = 42

# Safer hard negative mining (CEB only)
ENABLE_SEMIHARD_MINING = True
MINE_EVERY_META_EPOCH = 1
NEG_POOL_SIZE = 8192
NEG_TOPK = 50
ENCODE_BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0

SMALL_MODEL_SUITE = {
    "bert-base": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "paraphrase-minilm-l6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "multi-qa-minilm-l6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "gte-small": "thenlper/gte-small",
    "bge-m3": "BAAI/bge-m3",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-large-en": "BAAI/bge-large-en",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cosine_triplet_metric():
    if not hasattr(losses, "TripletDistanceMetric"):
        raise RuntimeError("TripletDistanceMetric not found in sentence_transformers.losses.")

    tdm = losses.TripletDistanceMetric
    if hasattr(tdm, "COSINE_DISTANCE"):
        return tdm.COSINE_DISTANCE
    if hasattr(tdm, "COSINE"):
        return tdm.COSINE

    available = [x for x in dir(tdm) if x.isupper()]
    raise RuntimeError(f"Could not find a cosine metric. Available: {available}")


def move_features_to_device(features, device):
    for feat in features:
        for k, v in feat.items():
            if torch.is_tensor(v):
                feat[k] = v.to(device)
    return features


def _dedup_preserve_order(items):
    return list(dict.fromkeys(items))


def sanitize_name(model_id: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-")


def short_model_tag(model_id: str):
    low = model_id.lower()
    if "distilbert-base-uncased" in low:
        return "distilbert"
    if "all-minilm-l12-v2" in low:
        return "minilm_l12"
    if "all-minilm-l6-v2" in low:
        return "minilm_l6"
    if "paraphrase-minilm-l6-v2" in low:
        return "paraphrase_minilm_l6"
    if "multi-qa-minilm-l6-cos-v1" in low:
        return "multiqa_minilm_l6"
    if "all-mpnet-base-v2" in low:
        return "mpnet"
    if "gte-small" in low:
        return "gte_small"
    if "bge-m3" in low:
        return "bge_m3"
    if "bge-small-en-v1.5" in low:
        return "bge_small"
    if "bge-large-en" in low:
        return "bge_large"
    if "bert-base-uncased" in low:
        return "bert_base"
    return sanitize_name(model_id.split("/")[-1]).lower()


def make_unique_run_name(output_dir: str, base_name: str):
    candidate = base_name
    v = 2
    while os.path.exists(os.path.join(output_dir, candidate)):
        candidate = f"{base_name}_v{v}"
        v += 1
    return candidate


def resolve_model_ids(args):
    if args.train_all_small_models:
        return list(SMALL_MODEL_SUITE.values())

    if args.models:
        model_ids = []
        tokens = [t.strip() for t in args.models.split(",") if t.strip()]
        for token in tokens:
            model_ids.append(SMALL_MODEL_SUITE.get(token.lower(), token))
        if not model_ids:
            raise ValueError("--models was provided but no valid model ids were found.")
        return model_ids

    return [args.init_model]


def mine_semihard_negatives_cosine(
    model,
    triplet_examples,
    neg_pool_texts,
    pool_size,
    topk,
    margin,
    device,
    encode_batch_size=128,
):
    if not triplet_examples or not neg_pool_texts:
        return triplet_examples

    anchors = [ex.texts[0] for ex in triplet_examples]
    positives = [ex.texts[1] for ex in triplet_examples]
    original_negs = [ex.texts[2] for ex in triplet_examples]

    pool_k = min(pool_size, len(neg_pool_texts))
    pool = random.sample(neg_pool_texts, k=pool_k)

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        anc_emb = model.encode(
            anchors,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=encode_batch_size,
            device=device,
            show_progress_bar=False,
        )
        pos_emb = model.encode(
            positives,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=encode_batch_size,
            device=device,
            show_progress_bar=False,
        )
        pool_emb = model.encode(
            pool,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=encode_batch_size,
            device=device,
            show_progress_bar=False,
        )

        sim_ap = (anc_emb * pos_emb).sum(dim=1)
        sim_pool = anc_emb @ pool_emb.T

        k = min(topk, sim_pool.size(1))
        top_vals, top_idx = torch.topk(sim_pool, k=k, dim=1)

    mined = []
    for i, ex in enumerate(triplet_examples):
        a = anchors[i]
        p = positives[i]
        neg = None

        ap = float(sim_ap[i].item())
        semihard_low = ap - margin
        semihard_high = ap
        chosen_fallback = None

        for val, j in zip(top_vals[i].tolist(), top_idx[i].tolist()):
            cand = pool[j]
            if cand == a or cand == p:
                continue

            if val < semihard_high and val > semihard_low:
                neg = cand
                break

            if chosen_fallback is None and val < semihard_high:
                chosen_fallback = cand

        if neg is None and chosen_fallback is not None:
            neg = chosen_fallback

        if neg is None:
            neg = original_negs[i]

        mined.append(InputExample(texts=[a, p, neg]))

    if model_was_training:
        model.train()

    return mined


def load_triplets(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"anchor", "positive", "negative"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.dropna(subset=["anchor", "positive", "negative"])
    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)
    df["negative"] = df["negative"].astype(str)

    examples = [
        InputExample(texts=[a, p, n])
        for a, p, n in zip(df["anchor"], df["positive"], df["negative"])
    ]
    neg_pool = _dedup_preserve_order(df["negative"].tolist())
    return examples, neg_pool


def load_tasks():
    files = {
        "JOB": "stage1_triplets_JOB.csv",
        "CEB": "stage1_triplets_CEB.csv",
    }

    tasks = {}
    ceb_neg_pool = []

    for task_name, fname in files.items():
        print(f"Loading {task_name} task from {fname}...")
        examples, neg_pool = load_triplets(fname)
        tasks[task_name] = examples
        print(f"  {task_name}: {len(examples)} triplets")
        if task_name == "CEB":
            ceb_neg_pool = neg_pool

    return tasks, ceb_neg_pool


def train_reptile_stage1(
    init_model: str,
    output_path: str,
    tasks,
    ceb_neg_pool,
    meta_epochs: int,
    inner_steps: int,
    inner_lr: float,
    meta_lr: float,
    batch_size: int,
    max_seq_length: int,
    triplet_margin: float,
    seed: int,
    enable_semihard_mining: bool,
    mine_every_meta_epoch: int,
    neg_pool_size: int,
    neg_topk: int,
    encode_batch_size: int,
    grad_clip_norm: float,
):
    set_seed(seed)
    print(f"sentence-transformers version: {sentence_transformers.__version__}")
    print(f"Initializing Reptile from: {init_model}")

    if not tasks:
        raise RuntimeError("No datasets found. Expected stage1_triplets_JOB.csv and stage1_triplets_CEB.csv.")

    model = SentenceTransformer(init_model)
    try:
        model.max_seq_length = max_seq_length
    except Exception:
        pass

    distance_metric = get_cosine_triplet_metric()
    loss_model = losses.TripletLoss(
        model=model,
        distance_metric=distance_metric,
        triplet_margin=triplet_margin,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_model.to(device)
    print(f"Training on: {device}")

    last_task_losses = {}
    for meta_epoch in range(meta_epochs):
        meta_weights = copy.deepcopy(model.state_dict())
        task_names = list(tasks.keys())
        random.shuffle(task_names)

        print(f"\n--- Meta Epoch {meta_epoch + 1}/{meta_epochs} ---")

        for task_name in task_names:
            task_data = tasks[task_name]

            sample_size = batch_size * inner_steps
            if len(task_data) < sample_size:
                batch_data = random.choices(task_data, k=sample_size)
            else:
                batch_data = random.sample(task_data, sample_size)

            model.load_state_dict(meta_weights)

            if (
                enable_semihard_mining
                and task_name == "CEB"
                and ceb_neg_pool
                and (meta_epoch % mine_every_meta_epoch == 0)
            ):
                batch_data = mine_semihard_negatives_cosine(
                    model=model,
                    triplet_examples=batch_data,
                    neg_pool_texts=ceb_neg_pool,
                    pool_size=neg_pool_size,
                    topk=neg_topk,
                    margin=triplet_margin,
                    device=device,
                    encode_batch_size=encode_batch_size,
                )

            train_dataloader = DataLoader(
                batch_data,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=lambda x: x,
            )

            inner_optimizer = torch.optim.AdamW(model.parameters(), lr=inner_lr)

            model.train()
            loss_model.train()

            total_loss = 0.0
            steps = 0

            for batch in train_dataloader:
                if steps >= inner_steps:
                    break

                features, labels = model.smart_batching_collate(batch)
                features = move_features_to_device(features, device)
                if labels is not None:
                    labels = labels.to(device)

                loss_value = loss_model(features, labels)
                loss_value.backward()

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                inner_optimizer.step()
                inner_optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss_value.item())
                steps += 1

            adapted_weights = model.state_dict()
            for key in meta_weights:
                meta_weights[key] += (meta_lr / len(task_names)) * (adapted_weights[key] - meta_weights[key])

            avg_loss = total_loss / max(steps, 1)
            last_task_losses[task_name] = avg_loss
            print(f"   [{task_name}] Loss: {avg_loss:.4f}")

        model.load_state_dict(meta_weights)

    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    print(f"Saved Reptile model to: {output_path}")

    return {
        "last_job_loss": last_task_losses.get("JOB"),
        "last_ceb_loss": last_task_losses.get("CEB"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, default=BASE_MODEL)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model ids or aliases from SMALL_MODEL_SUITE.",
    )
    parser.add_argument(
        "--train_all_small_models",
        action="store_true",
        help="Train/evaluate every model in SMALL_MODEL_SUITE.",
    )
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results_csv", type=str, default="reptile_sweep_results.csv")
    parser.add_argument(
        "--append_results",
        action="store_true",
        help="Append to results_csv if it already exists (instead of overwriting).",
    )
    parser.add_argument("--meta_epochs", type=int, default=META_EPOCHS)
    parser.add_argument("--inner_steps", type=int, default=INNER_STEPS)
    parser.add_argument("--inner_lr", type=float, default=INNER_LR)
    parser.add_argument("--meta_lr", type=float, default=META_LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--triplet_margin", type=float, default=TRIPLET_MARGIN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--disable_semihard_mining", action="store_true")
    parser.add_argument("--mine_every_meta_epoch", type=int, default=MINE_EVERY_META_EPOCH)
    parser.add_argument("--neg_pool_size", type=int, default=NEG_POOL_SIZE)
    parser.add_argument("--neg_topk", type=int, default=NEG_TOPK)
    parser.add_argument("--encode_batch_size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument("--grad_clip_norm", type=float, default=GRAD_CLIP_NORM)
    args = parser.parse_args()

    if args.meta_epochs <= 0:
        raise ValueError("meta_epochs must be positive.")
    if args.inner_steps <= 0:
        raise ValueError("inner_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.inner_lr <= 0:
        raise ValueError("inner_lr must be positive.")
    if args.meta_lr <= 0:
        raise ValueError("meta_lr must be positive.")
    if args.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    if args.mine_every_meta_epoch <= 0:
        raise ValueError("mine_every_meta_epoch must be positive.")
    if args.neg_pool_size <= 0:
        raise ValueError("neg_pool_size must be positive.")
    if args.neg_topk <= 0:
        raise ValueError("neg_topk must be positive.")
    if args.encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive.")

    tasks, ceb_neg_pool = load_tasks()
    model_ids = resolve_model_ids(args)
    sweep_mode = args.train_all_small_models or bool(args.models)

    if sweep_mode:
        os.makedirs(args.output_dir, exist_ok=True)
    elif len(model_ids) > 1:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Models to train with Reptile: {len(model_ids)}")
    all_results = []

    for model_idx, model_id in enumerate(model_ids, start=1):
        if sweep_mode:
            tag = short_model_tag(model_id)
            base_name = f"reptile_{tag}"
            run_name = make_unique_run_name(args.output_dir, base_name)
            output_path = os.path.join(args.output_dir, run_name)
        elif len(model_ids) > 1:
            tag = short_model_tag(model_id)
            base_name = f"reptile_{tag}"
            run_name = make_unique_run_name(args.output_dir, base_name)
            output_path = os.path.join(args.output_dir, run_name)
        else:
            output_path = args.output_path

        print(f"\n=== [{model_idx}/{len(model_ids)}] Reptile training: {model_id} ===")
        metrics = train_reptile_stage1(
            init_model=model_id,
            output_path=output_path,
            tasks=tasks,
            ceb_neg_pool=ceb_neg_pool,
            meta_epochs=args.meta_epochs,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            meta_lr=args.meta_lr,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            triplet_margin=args.triplet_margin,
            seed=args.seed,
            enable_semihard_mining=(not args.disable_semihard_mining),
            mine_every_meta_epoch=args.mine_every_meta_epoch,
            neg_pool_size=args.neg_pool_size,
            neg_topk=args.neg_topk,
            encode_batch_size=args.encode_batch_size,
            grad_clip_norm=args.grad_clip_norm,
        )

        result_row = {
            "model_id": model_id,
            "output_path": output_path,
            "meta_epochs": args.meta_epochs,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "meta_lr": args.meta_lr,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "triplet_margin": args.triplet_margin,
            "seed": args.seed,
            "semihard_mining": not args.disable_semihard_mining,
            "last_job_loss": metrics.get("last_job_loss"),
            "last_ceb_loss": metrics.get("last_ceb_loss"),
        }
        all_results.append(result_row)

        # Free GPU memory before the next model in sweep.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sweep_mode or len(model_ids) > 1:
        results_path = os.path.join(args.output_dir, args.results_csv)
        out_df = pd.DataFrame(all_results)
        if args.append_results and os.path.exists(results_path):
            prev_df = pd.read_csv(results_path)
            out_df = pd.concat([prev_df, out_df], ignore_index=True)
        out_df.to_csv(results_path, index=False)
        print(f"\nReptile sweep summary saved to: {results_path}")


if __name__ == "__main__":
    main()
