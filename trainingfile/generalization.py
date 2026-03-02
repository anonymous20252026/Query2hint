import os
import re
import random
import argparse
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses, util

BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

JOB_CSV = "stage1_triplets_JOB.csv"
CEB_CSV = "stage1_triplets_CEB.csv"

DEFAULT_OUTPUT_DIR = "all_models"

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_TOTAL_STEPS = 12000
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
DEFAULT_MAX_EVAL_PAIRS = 2000
DEFAULT_EVAL_BATCH_SIZE = 256

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
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pairs(csv_path: str):
    df = pd.read_csv(csv_path)
    required_columns = {"anchor", "positive"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.dropna(subset=["anchor", "positive"])
    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)
    return [InputExample(texts=[a, p]) for a, p in zip(df["anchor"], df["positive"])]


def split_pairs(pairs, val_ratio: float, seed: int):
    if len(pairs) == 0 or val_ratio <= 0:
        return pairs, []

    n_val = int(len(pairs) * val_ratio)
    n_val = max(1, min(n_val, len(pairs) - 1))
    shuffled = list(pairs)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def sanitize_name(model_id: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-")


def short_model_tag(model_id: str):
    low = model_id.lower()
    if "bert-base-uncased" in low:
        return "bert_base"
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
    return sanitize_name(model_id.split("/")[-1]).lower()


def make_unique_run_name(output_dir: str, base_name: str):
    """
    Keep short readable names and only add a version suffix if needed.
    Example: finetuned_minilm_l12_ceb, finetuned_minilm_l12_ceb_v2, ...
    """
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

    return [args.base_model]


def evaluate_retrieval_metrics(model, val_pairs, eval_batch_size: int, max_eval_pairs: int, seed: int):
    return evaluate_retrieval_metrics_with_candidates(
        model=model,
        eval_pairs=val_pairs,
        eval_batch_size=eval_batch_size,
        max_eval_pairs=max_eval_pairs,
        seed=seed,
        candidate_pool=None,
    )


def evaluate_retrieval_metrics_with_candidates(
    model,
    eval_pairs,
    eval_batch_size: int,
    max_eval_pairs: int,
    seed: int,
    candidate_pool=None,
):
    if len(eval_pairs) == 0:
        return None

    pairs = list(eval_pairs)
    if max_eval_pairs > 0 and len(pairs) > max_eval_pairs:
        rnd = random.Random(seed)
        idx = rnd.sample(range(len(pairs)), max_eval_pairs)
        pairs = [pairs[i] for i in idx]

    anchors = [ex.texts[0] for ex in pairs]
    positives = [ex.texts[1] for ex in pairs]

    if candidate_pool is None:
        unique_positives = list(dict.fromkeys(positives))
    else:
        # Keep order stable while deduplicating and ensure every target positive exists in pool.
        unique_positives = list(dict.fromkeys(candidate_pool))
        missing = [p for p in positives if p not in set(unique_positives)]
        if missing:
            unique_positives.extend(list(dict.fromkeys(missing)))

    pos_to_idx = {p: i for i, p in enumerate(unique_positives)}
    targets = torch.tensor([pos_to_idx[p] for p in positives], dtype=torch.long)

    q_emb = model.encode(
        anchors,
        batch_size=eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    c_emb = model.encode(
        unique_positives,
        batch_size=eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sims = q_emb @ c_emb.T
    targets = targets.to(sims.device)

    eval_loss = F.cross_entropy(sims, targets).item()
    top1_acc = (sims.argmax(dim=1) == targets).float().mean().item()

    ranked = torch.argsort(sims, dim=1, descending=True)
    target_pos = (ranked == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    mrr = (1.0 / (target_pos.float() + 1.0)).mean().item()

    return {
        "eval_pairs": len(pairs),
        "eval_candidates": len(unique_positives),
        "val_ce_loss": eval_loss,
        "val_top1_acc": top1_acc,
        "val_mrr": mrr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_on", type=str, default="both", choices=["job", "ceb", "both"])
    parser.add_argument(
        "--eval_on",
        type=str,
        default="auto",
        choices=["auto", "job", "ceb", "both"],
        help=(
            "Workloads to evaluate after training. "
            "If eval workload is not in train_on, evaluation is cross-workload (uses full workload pairs)."
        ),
    )
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
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
    parser.add_argument("--total_steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--max_eval_pairs", type=int, default=DEFAULT_MAX_EVAL_PAIRS)
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--eval_candidate_source",
        type=str,
        default="workload",
        choices=["eval_pairs", "workload", "all"],
        help=(
            "Candidate set for retrieval metrics: "
            "eval_pairs=only positives in sampled eval pairs; "
            "workload=all positives from that workload; "
            "all=all positives from all loaded workloads."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--append_results",
        action="store_true",
        help="Append to results_csv if it already exists (instead of overwriting).",
    )
    parser.add_argument("--results_csv", type=str, default="model_sweep_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.lr <= 0:
        raise ValueError("lr must be positive.")
    if args.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    if not (0.0 <= args.warmup_ratio <= 1.0):
        raise ValueError("warmup_ratio must be between 0.0 and 1.0.")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0.0 and <1.0.")
    if args.max_eval_pairs < 0:
        raise ValueError("max_eval_pairs must be >= 0.")
    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    eval_mode = args.train_on if args.eval_on == "auto" else args.eval_on

    need_job = (
        args.train_on in ["job", "both"]
        or eval_mode in ["job", "both"]
        or args.eval_candidate_source == "all"
    )
    need_ceb = (
        args.train_on in ["ceb", "both"]
        or eval_mode in ["ceb", "both"]
        or args.eval_candidate_source == "all"
    )

    if need_job and not os.path.exists(JOB_CSV):
        raise FileNotFoundError(f"Missing {JOB_CSV}")
    if need_ceb and not os.path.exists(CEB_CSV):
        raise FileNotFoundError(f"Missing {CEB_CSV}")

    model_ids = resolve_model_ids(args)

    job_pairs_all = load_pairs(JOB_CSV) if need_job else []
    ceb_pairs_all = load_pairs(CEB_CSV) if need_ceb else []

    print(f"Train mode: {args.train_on}")
    print(f"Eval mode:  {eval_mode}")
    print(f"Eval candidate source: {args.eval_candidate_source}")
    print(f"Models to train: {len(model_ids)}")
    print(f"JOB pairs total: {len(job_pairs_all)}")
    print(f"CEB pairs total: {len(ceb_pairs_all)}")

    if len(job_pairs_all) + len(ceb_pairs_all) == 0:
        raise RuntimeError("No training data loaded.")

    job_train, job_val = split_pairs(job_pairs_all, args.val_ratio, args.seed + 11)
    ceb_train, ceb_val = split_pairs(ceb_pairs_all, args.val_ratio, args.seed + 29)

    train_pairs = job_train + ceb_train
    val_pairs = job_val + ceb_val

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")

    # Build workload-specific evaluation sets.
    eval_workloads = []
    if eval_mode in ["job", "both"]:
        eval_workloads.append("job")
    if eval_mode in ["ceb", "both"]:
        eval_workloads.append("ceb")

    eval_pairs_by_workload = {}
    if "job" in eval_workloads:
        if args.train_on in ["job", "both"]:
            eval_pairs_by_workload["job"] = job_val
        else:
            # Cross-workload: no job training used, evaluate on full JOB set.
            eval_pairs_by_workload["job"] = job_pairs_all
    if "ceb" in eval_workloads:
        if args.train_on in ["ceb", "both"]:
            eval_pairs_by_workload["ceb"] = ceb_val
        else:
            # Cross-workload: no CEB training used, evaluate on full CEB set.
            eval_pairs_by_workload["ceb"] = ceb_pairs_all

    positives_by_workload = {
        "job": [ex.texts[1] for ex in job_pairs_all] if len(job_pairs_all) > 0 else [],
        "ceb": [ex.texts[1] for ex in ceb_pairs_all] if len(ceb_pairs_all) > 0 else [],
    }
    all_positive_pool = positives_by_workload["job"] + positives_by_workload["ceb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    print(f"AMP enabled: {use_amp}")

    num_samples = args.total_steps * args.batch_size
    warmup_steps = int(args.warmup_ratio * args.total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for model_idx, model_id in enumerate(model_ids, start=1):
        print(f"\n=== [{model_idx}/{len(model_ids)}] Training model: {model_id} ===")

        model = SentenceTransformer(model_id)
        try:
            model.max_seq_length = args.max_seq_length
        except Exception:
            pass
        model.to(device)

        if args.train_on == "both":
            w_job = 1.0 / max(len(job_train), 1)
            w_ceb = 1.0 / max(len(ceb_train), 1)
            weights = [w_job] * len(job_train) + [w_ceb] * len(ceb_train)
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        else:
            sampler = RandomSampler(train_pairs, replacement=True, num_samples=num_samples)

        train_dataloader = DataLoader(
            train_pairs,
            sampler=sampler,
            batch_size=args.batch_size,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )

        train_loss = losses.MultipleNegativesRankingLoss(model, similarity_fct=util.cos_sim)

        model_name = short_model_tag(model_id)
        base_run_name = f"finetuned_{model_name}_{args.train_on}"
        run_name = make_unique_run_name(args.output_dir, base_run_name)
        output_path = os.path.join(args.output_dir, run_name)
        os.makedirs(output_path, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=args.total_steps,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
            output_path=output_path,
            use_amp=use_amp,
        )

        per_workload_metrics = {}
        for w in eval_workloads:
            eval_pairs = eval_pairs_by_workload.get(w, [])
            if args.eval_candidate_source == "eval_pairs":
                candidate_pool = None
            elif args.eval_candidate_source == "workload":
                candidate_pool = positives_by_workload.get(w, [])
            else:
                candidate_pool = all_positive_pool

            per_workload_metrics[w] = evaluate_retrieval_metrics_with_candidates(
                model=model,
                eval_pairs=eval_pairs,
                eval_batch_size=args.eval_batch_size,
                max_eval_pairs=args.max_eval_pairs,
                seed=args.seed,
                candidate_pool=candidate_pool,
            )

        # Aggregate metric across selected workloads (weighted by eval pair count).
        valid_metric_items = [(w, m) for w, m in per_workload_metrics.items() if m is not None]
        metrics = None
        if valid_metric_items:
            total_eval_pairs = sum(m["eval_pairs"] for _, m in valid_metric_items)
            if total_eval_pairs > 0:
                metrics = {
                    "eval_pairs": int(total_eval_pairs),
                    "eval_candidates": int(sum(m["eval_candidates"] for _, m in valid_metric_items)),
                    "val_ce_loss": float(
                        sum(m["val_ce_loss"] * m["eval_pairs"] for _, m in valid_metric_items) / total_eval_pairs
                    ),
                    "val_top1_acc": float(
                        sum(m["val_top1_acc"] * m["eval_pairs"] for _, m in valid_metric_items) / total_eval_pairs
                    ),
                    "val_mrr": float(
                        sum(m["val_mrr"] * m["eval_pairs"] for _, m in valid_metric_items) / total_eval_pairs
                    ),
                }

        result_row = {
            "model_id": model_id,
            "output_path": output_path,
            "train_on": args.train_on,
            "eval_on": eval_mode,
            "eval_candidate_source": args.eval_candidate_source,
            "train_pairs": len(train_pairs),
            "val_pairs_total": len(val_pairs),
            "total_steps": args.total_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_seq_length": args.max_seq_length,
            "warmup_steps": warmup_steps,
        }
        if metrics is not None:
            result_row.update(metrics)
            print(
                f"Validation: loss={metrics['val_ce_loss']:.4f}, "
                f"top1={metrics['val_top1_acc']:.4f}, mrr={metrics['val_mrr']:.4f}, "
                f"eval_pairs={metrics['eval_pairs']}"
            )
        else:
            print("Validation skipped (no validation pairs).")

        for workload_name, wm in per_workload_metrics.items():
            if wm is None:
                continue
            result_row[f"{workload_name}_eval_pairs"] = wm["eval_pairs"]
            result_row[f"{workload_name}_eval_candidates"] = wm["eval_candidates"]
            result_row[f"{workload_name}_val_ce_loss"] = wm["val_ce_loss"]
            result_row[f"{workload_name}_val_top1_acc"] = wm["val_top1_acc"]
            result_row[f"{workload_name}_val_mrr"] = wm["val_mrr"]
            print(
                f"  [{workload_name.upper()}] "
                f"loss={wm['val_ce_loss']:.4f}, top1={wm['val_top1_acc']:.4f}, "
                f"mrr={wm['val_mrr']:.4f}, eval_pairs={wm['eval_pairs']}, "
                f"candidates={wm['eval_candidates']}"
            )

        print(f"Saved to: {output_path}")
        all_results.append(result_row)

        # Release model memory before next sweep model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path = os.path.join(args.output_dir, args.results_csv)
    out_df = pd.DataFrame(all_results)
    if args.append_results and os.path.exists(results_path):
        prev_df = pd.read_csv(results_path)
        out_df = pd.concat([prev_df, out_df], ignore_index=True)
    out_df.to_csv(results_path, index=False)
    print(f"\nSweep summary saved to: {results_path}")


if __name__ == "__main__":
    main()
