# AdaptSteer — Trained Encoders

Pre-trained sentence-transformer encoders used in the paper experiments.
All encoders are HuggingFace-compatible and load directly with `SentenceTransformer`.

---

## Main Encoders (paper Table 2)

| Folder | Paper name | Training method | Base model |
|--------|------------|-----------------|------------|
| `adaptsteer_c/` | **AdaptSteer-C** | Contrastive TripletLoss on multi-config execution preferences | `all-mpnet-base-v2` |
| `adaptsteer_r/` | **AdaptSteer-R** | Same contrastive training + Reptile meta-learning (JOB ↔ CEB episodes) | `all-mpnet-base-v2` |
| `mpnet_binary_supervised/` | MPNet-Binary | TripletLoss on binary-only supervision (hint_0 vs hint_26 only) — ablation baseline | `all-mpnet-base-v2` |

---

## Code-Backbone Encoders (paper Appendix — `code_backbone_selection.py`)

| Folder | Paper name | Base model |
|--------|------------|------------|
| `code_backbones/codebert/` | CodeBERT-FT | `microsoft/codebert-base` |
| `code_backbones/graphcodebert/` | GraphCodeBERT-FT | `microsoft/graphcodebert-base` |
| `code_backbones/codet5_base/` | CodeT5-FT | `Salesforce/codet5-base` |
| `code_backbones/unixcoder/` | UniXCoder-FT | `microsoft/unixcoder-base` |

All four code-backbone encoders were trained with the same contrastive TripletLoss pipeline
used for AdaptSteer-C (see `encoder_training.py`).

---

## Excluded encoders

The following were excluded because they are superseded by the versions above:
- `encoder_reptile_mpnet_v3` → superseded by `adaptsteer_r` (v4)
- `encoder_all-MiniLM-L12-v2_v1` → not used in main paper
- `encoder_codebert-base_v1` → superseded by `code_backbones/codebert`
- `encoder_unixcoder-base_v1` → superseded by `code_backbones/unixcoder`

---

## Loading an encoder

```python
from sentence_transformers import SentenceTransformer

# Load AdaptSteer-R
model = SentenceTransformer("encoders/adaptsteer_r")

# Encode SQL queries
queries = ["SELECT MIN(t.title) FROM title t WHERE t.kind_id = 1"]
embeddings = model.encode(queries)   # shape: (N, 768)
```

## Running binary steering

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

encoder = SentenceTransformer("encoders/adaptsteer_r")

# SVC-120-S pipeline (paper §3.3)
clf = Pipeline([
    ("scaler1", StandardScaler()),
    ("pca",     PCA(n_components=120, random_state=24508)),
    ("scaler2", StandardScaler()),
    ("svc",     SVC(kernel="rbf", probability=True, class_weight="balanced")),
])

# X_train: embeddings of training queries, y_train: binary labels (0=PG default, 1=steer)
X_train = encoder.encode(train_queries)
clf.fit(X_train, y_train)

# Predict for a new query
x_new = encoder.encode(["SELECT ..."])
label = clf.predict(x_new)   # 0 → keep PG default, 1 → apply hint_26
```

For the full end-to-end pipeline see `steering_pipeline.py`.
