import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

CSV_PATH = "stage1_triplets_CEB.csv"
BENCH_PATH = "fixed_benchmark_ceB.jsonl"

SEED = 42
MAX_ROWS = 20000          # set None to use full CSV
MAX_QUERIES = 500         # number of anchors to evaluate
NEG_PER_QUERY = 50        # number of negatives per query in the benchmark

MAX_SEQ_LENGTH = 256

def build_baseline_model():
    word_embedding = models.Transformer("bert-base-uncased", max_seq_length=MAX_SEQ_LENGTH)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="mean")
    normalize = models.Normalize()
    return SentenceTransformer(modules=[word_embedding, pooling, normalize])

def build_benchmark():
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(CSV_PATH)
    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)
    df["negative"] = df["negative"].astype(str)

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)

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

    with open(BENCH_PATH, "w", encoding="utf-8") as f:
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

            rec = {
                "anchor": a,
                "positive": p,
                "negatives": negatives
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved benchmark to {BENCH_PATH}")
    print(f"Queries: {len(anchors)}")
    print(f"Negatives per query: {NEG_PER_QUERY}")

def load_benchmark():
    items = []
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def eval_model(model):
    bench = load_benchmark()

    recall1 = []
    recall3 = []
    recall5 = []
    mrr = []

    with torch.no_grad():
        for i, rec in enumerate(bench):
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
    parser.add_argument("--use_reptile", action="store_true")
    parser.add_argument("--model_path", type=str, default="reptile_BERT")
    args = parser.parse_args()

    if args.make_benchmark or (not os.path.exists(BENCH_PATH)):
        build_benchmark()

    if args.use_reptile:
        print(f"Model: trained at {args.model_path}")
        model = SentenceTransformer(args.model_path)
    else:
        print("Model: bert-base-uncased baseline (same architecture)")
        model = build_baseline_model()

    model.eval()
    res = eval_model(model)

    print("\nResults on fixed benchmark")
    print(f"Queries:    {res['queries']}")
    print(f"Neg/query:  {res['neg_per_query']}")
    print(f"Recall@1:   {res['recall@1']:.4f}")
    print(f"Recall@3:   {res['recall@3']:.4f}")
    print(f"Recall@5:   {res['recall@5']:.4f}")
    print(f"MRR:        {res['mrr']:.4f}")

if __name__ == "__main__":
    main()
