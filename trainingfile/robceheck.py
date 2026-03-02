import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

CSV_PATH = "stage1_triplets_CEB.csv"

MAX_ROWS = 20000          # None = use full CSV
MAX_QUERIES = 500
NEG_PER_QUERY = 50
MAX_SEQ_LENGTH = 256


def build_baseline_model():
    word_embedding = models.Transformer("bert-base-uncased", max_seq_length=MAX_SEQ_LENGTH)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="mean")
    normalize = models.Normalize()
    return SentenceTransformer(modules=[word_embedding, pooling, normalize])


def build_benchmark(csv_path: str, bench_path: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)
    df["negative"] = df["negative"].astype(str)

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed)

    pos_by_anchor = defaultdict(list)
    neg_by_anchor = defaultdict(list)
    global_neg_pool = df["negative"].tolist()

    for _, r in df.iterrows():
        a = r["anchor"]
        pos_by_anchor[a].append(r["positive"])
        neg_by_anchor[a].append(r["negative"])

    anchors = list(pos_by_anchor.keys())
    anchors.sort()
    random.shuffle(anchors)
    anchors = anchors[: min(MAX_QUERIES, len(anchors))]

    with open(bench_path, "w", encoding="utf-8") as f:
        written = 0
        for a in anchors:
            positives = list(dict.fromkeys(pos_by_anchor[a]))
            negatives_local = list(dict.fromkeys(neg_by_anchor[a]))
            if not positives:
                continue

            p = random.choice(positives)

            negatives = []
            for n in negatives_local:
                if n != p and n != a:
                    negatives.append(n)
                if len(negatives) >= NEG_PER_QUERY:
                    break

            tries = 0
            while len(negatives) < NEG_PER_QUERY and tries < NEG_PER_QUERY * 50:
                cand = random.choice(global_neg_pool)
                if cand != p and cand != a:
                    negatives.append(cand)
                tries += 1

            negatives = negatives[:NEG_PER_QUERY]

            rec = {"anchor": a, "positive": p, "negatives": negatives}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved benchmark to: {bench_path}")
    print(f"Seed: {seed}")
    print(f"Queries written: {written}")
    print(f"Negatives per query: {NEG_PER_QUERY}")


def load_benchmark(bench_path: str):
    items = []
    with open(bench_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def eval_model(model: SentenceTransformer, bench_path: str):
    bench = load_benchmark(bench_path)

    recall1 = []
    recall3 = []
    recall5 = []
    mrr = []

    with torch.no_grad():
        for rec in bench:
            a = rec["anchor"]
            p = rec["positive"]
            negs = rec["negatives"]
            cands = [p] + negs

            q_emb = model.encode([a], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            c_emb = model.encode(cands, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False)

            sims = (q_emb @ c_emb.T).flatten()
            ranks = np.argsort(-sims)
            pos_rank = int(np.where(ranks == 0)[0][0]) + 1

            recall1.append(1 if pos_rank <= 1 else 0)
            recall3.append(1 if pos_rank <= 3 else 0)
            recall5.append(1 if pos_rank <= 5 else 0)
            mrr.append(1.0 / pos_rank)

    return {
        "queries": len(bench),
        "neg_per_query": NEG_PER_QUERY,
        "recall@1": float(np.mean(recall1)),
        "recall@3": float(np.mean(recall3)),
        "recall@5": float(np.mean(recall5)),
        "mrr": float(np.mean(mrr)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--make_benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench_path", type=str, default="fixed_benchmark_job_seed42.jsonl")

    parser.add_argument("--use_reptile", action="store_true")
    parser.add_argument("--model_path", type=str, default="reptile_BERT")

    args = parser.parse_args()

    if args.make_benchmark or (not os.path.exists(args.bench_path)):
        build_benchmark(CSV_PATH, args.bench_path, args.seed)

    if args.use_reptile:
        print(f"Model: {args.model_path}")
        model = SentenceTransformer(args.model_path)
    else:
        print("Model: bert-base-uncased baseline (same architecture)")
        model = build_baseline_model()

    model.eval()
    res = eval_model(model, args.bench_path)

    print("\nResults on fixed benchmark")
    print(f"Benchmark:  {args.bench_path}")
    print(f"Seed:       {args.seed}")
    print(f"Queries:    {res['queries']}")
    print(f"Neg/query:  {res['neg_per_query']}")
    print(f"Recall@1:   {res['recall@1']:.4f}")
    print(f"Recall@3:   {res['recall@3']:.4f}")
    print(f"Recall@5:   {res['recall@5']:.4f}")
    print(f"MRR:        {res['mrr']:.4f}")


if __name__ == "__main__":
    main()
