# AdaptSteer — Triplet Datasets

Pre-built contrastive triplets used to train the AdaptSteer-C and AdaptSteer-R encoders.

## Files

| File | Workload | Rows | Size | Description |
|------|----------|------|------|-------------|
| `triplets_JOB.csv` | JOB | 3,320 | 1.6 MB | Join Order Benchmark triplets |
| `triplets_CEB.csv` | CEB | 71,002 | 76 MB | Cardinality Estimation Benchmark triplets |

## Schema

Each row is one contrastive training triplet:

| Column | Type | Description |
|--------|------|-------------|
| `anchor` | string | SQL query text |
| `positive` | string | Hint-configuration label that ran **faster** for this query |
| `negative` | string | Hint-configuration label that ran **slower** for this query |
| `source_task` | string | Workload the triplet came from (`JOB` or `CEB`) |
| `time_diff` | float | Execution-time difference (ms) between positive and negative |

## How these were generated

The triplets were built from raw PostgreSQL execution logs by `data_preparation.py`:

1. For each unique SQL query, all measured hint configurations are ranked by mean execution time.
2. The fastest configuration becomes the **positive** example.
3. The slowest configuration becomes the **negative** example.
4. The SQL text is the shared **anchor**.

Only triplets where `time_diff > 0` are retained.

## Usage

### Encoder training

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv("data/triplets_JOB.csv")

examples = [
    InputExample(texts=[row.anchor, row.positive, row.negative])
    for _, row in df.iterrows()
]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_loader = DataLoader(examples, shuffle=True, batch_size=16)
loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=3,
    output_path="encoders/adaptsteer_c"
)
```

For full training (including Reptile meta-learning for AdaptSteer-R), use `encoder_training.py`.

## Notes

- The CEB file is large (76 MB). If storage is a concern, AdaptSteer-C can be trained on JOB triplets alone, with a small AUROC penalty (~0.3%).
- Both files are required to reproduce the Reptile meta-learning experiment (`fewshot_cross_workload_adaptation.py`).
