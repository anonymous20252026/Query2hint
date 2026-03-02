import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
import numpy as np
from collections import defaultdict
import random

# -------------------------- CONFIG --------------------------
CSV_PATH = "stage1_triplets_CEB.csv"   # << First try JOB (fast, 113 queries). Then change to CEB for comparison
USE_REPTILE = False                    # << Set to False = plain BERT baseline ("unreptile")
                                       #    Set to True  = your trained reptile_BERT (for direct comparison)
MODEL_PATH = "othertests/contrastive_mpnet_ceb_steps"            # Only used if USE_REPTILE = True
MAX_QUERIES = 500                     # None = all queries, or set 200/500 for speed on CEB
# -----------------------------------------------------------

def main():
    df = pd.read_csv(CSV_PATH)
    
    # Optional: limit total rows for speed
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    # Group by query (anchor)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        query = str(row["anchor"])
        pos = str(row["positive"])
        neg = str(row["negative"])
        groups[query].append((pos, 1))
        groups[query].append((neg, 0))

    # Optional: limit queries for fast testing
    if MAX_QUERIES is not None:
        selected_queries = random.sample(list(groups.keys()), min(MAX_QUERIES, len(groups)))
        groups = {q: groups[q] for q in selected_queries}

    print(f"Unique queries: {len(groups)}")
    print(f"Model: {'reptile_BERT (trained)' if USE_REPTILE else 'bert-base-uncased (untrained baseline)'}")

    # Load or build model
    if USE_REPTILE:
        model = SentenceTransformer(MODEL_PATH)
    else:
        # Exact same architecture as your trained model: BERT + mean pooling + normalize
        word_embedding = models.Transformer("bert-base-uncased", max_seq_length=256)
        pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode='mean')
        normalize = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding, pooling, normalize])

    model.eval()

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    mrr = []

    with torch.no_grad():
        for i, (query, candidates) in enumerate(groups.items()):
            if i % 50 == 0:
                print(f"Processing query {i+1}/{len(groups)}...")

            # Dedup candidates
            unique_cands = list({text for text, _ in candidates})
            labels = [1 if any(text == pos for pos, is_pos in candidates if is_pos) else 0 for text in unique_cands]

            query_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            cand_embs = model.encode(unique_cands, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

            sims = query_emb @ cand_embs.T
            sims = sims.flatten()

            ranks = np.argsort(-sims)
            pos_indices = np.where(np.array(labels)[ranks] == 1)[0]
            if len(pos_indices) == 0:
                continue
            pos_rank = pos_indices[0] + 1  # best positive rank

            recall_at_1.append(1 if pos_rank <= 1 else 0)
            recall_at_3.append(1 if pos_rank <= 3 else 0)
            recall_at_5.append(1 if pos_rank <= 5 else 0)
            mrr.append(1.0 / pos_rank)

    print("\n=== RESULTS ===")
    print(f"Recall@1: {np.mean(recall_at_1):.4f}")
    print(f"Recall@3: {np.mean(recall_at_3):.4f}")
    print(f"Recall@5: {np.mean(recall_at_5):.4f}")
    print(f"MRR:      {np.mean(mrr):.4f}")

if __name__ == "__main__":
    main()