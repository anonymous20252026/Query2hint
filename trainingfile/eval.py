# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from collections import defaultdict

# MODEL_PATH = "reptile_BERT"
# CSV_PATH = "stage1_triplets_CEB.csv"   # First try JOB, then change to CEB
# N_EVAL = 500                          # Use all, or set to 10000 for speed

# def main():
#     df = pd.read_csv(CSV_PATH)
#     if N_EVAL:
#         df = df.sample(N_EVAL, random_state=42)

#     model = SentenceTransformer(MODEL_PATH)
#     model.eval()

#     # Group by query (anchor)
#     groups = defaultdict(list)
#     for _, row in df.iterrows():
#         query = str(row["anchor"])
#         pos = str(row["positive"])
#         neg = str(row["negative"])
#         groups[query].append((pos, 1))  # 1 = correct
#         groups[query].append((neg, 0))  # 0 = wrong

#     recall_at_1 = []
#     recall_at_3 = []
#     recall_at_5 = []
#     mrr = []  # Mean Reciprocal Rank

#     with torch.no_grad():
#         for query, candidates in groups.items():
#             # Remove duplicate hints (keep one per text)
#             unique_cands = list({text for text, _ in candidates})
#             labels = [1 if any(text == pos for pos, is_pos in candidates if is_pos) else 0 for text in unique_cands]

#             query_emb = model.encode([query], normalize_embeddings=True, batch_size=1, convert_to_numpy=True)
#             cand_embs = model.encode(unique_cands, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)

#             sims = query_emb @ cand_embs.T  # shape [1, num_cands]
#             sims = sims.flatten()

#             ranks = np.argsort(-sims)  # highest sim first
#             pos_rank = np.where(np.array(labels)[ranks] == 1)[0][0] + 1  # 1-based rank of correct hint

#             recall_at_1.append(1 if pos_rank <= 1 else 0)
#             recall_at_3.append(1 if pos_rank <= 3 else 0)
#             recall_at_5.append(1 if pos_rank <= 5 else 0)
#             mrr.append(1.0 / pos_rank)

#     print(f"Unique queries evaluated: {len(groups)}")
#     print(f"Recall@1: {np.mean(recall_at_1):.4f}")
#     print(f"Recall@3: {np.mean(recall_at_3):.4f}")
#     print(f"Recall@5: {np.mean(recall_at_5):.4f}")
#     print(f"MRR: {np.mean(mrr):.4f}")

# if __name__ == "__main__":
#     main()

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import random

MODEL_PATH = "reptile_BERT"
CSV_PATH = "stage1_triplets_CEB.csv"
N_EVAL_QUERIES = 500           # number of unique anchors to evaluate
NEG_PER_QUERY = 50             # sample this many negatives per query
SEED = 42

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(CSV_PATH)
    df["anchor"] = df["anchor"].astype(str)
    df["positive"] = df["positive"].astype(str)
    df["negative"] = df["negative"].astype(str)

    # Build pools
    pos_by_anchor = defaultdict(list)
    neg_pool = df["negative"].tolist()

    for _, row in df.iterrows():
        pos_by_anchor[row["anchor"]].append(row["positive"])

    anchors = list(pos_by_anchor.keys())
    if N_EVAL_QUERIES and len(anchors) > N_EVAL_QUERIES:
        anchors = random.sample(anchors, N_EVAL_QUERIES)

    model = SentenceTransformer(MODEL_PATH)
    model.eval()

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    mrr = []

    with torch.no_grad():
        for a in anchors:
            # pick one positive for this anchor
            p = random.choice(pos_by_anchor[a])

            # sample negatives; avoid accidentally sampling the positive text
            negs = []
            tries = 0
            while len(negs) < NEG_PER_QUERY and tries < NEG_PER_QUERY * 20:
                cand = random.choice(neg_pool)
                if cand != p:
                    negs.append(cand)
                tries += 1

            # candidate list: 1 positive + K negatives
            cands = [p] + negs
            labels = np.array([1] + [0] * len(negs))

            q_emb = model.encode([a], normalize_embeddings=True, convert_to_numpy=True)
            c_emb = model.encode(cands, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

            sims = (q_emb @ c_emb.T).flatten()
            ranks = np.argsort(-sims)

            # rank of the positive (index 0 in cands)
            pos_rank = int(np.where(ranks == 0)[0][0]) + 1

            recall_at_1.append(1 if pos_rank <= 1 else 0)
            recall_at_3.append(1 if pos_rank <= 3 else 0)
            recall_at_5.append(1 if pos_rank <= 5 else 0)
            mrr.append(1.0 / pos_rank)

    print(f"Unique queries evaluated: {len(anchors)}")
    print(f"Negatives per query: {NEG_PER_QUERY}")
    print(f"Recall@1: {np.mean(recall_at_1):.4f}")
    print(f"Recall@3: {np.mean(recall_at_3):.4f}")
    print(f"Recall@5: {np.mean(recall_at_5):.4f}")
    print(f"MRR: {np.mean(mrr):.4f}")

if __name__ == "__main__":
    main()
