# ============================================================
# AdaSteer vs LLMSteer — Pipeline Comparison (FIXED + NoPCA)
# ============================================================
#
# FIXES vs original broken code:
#   [FIX-1] PCA fitted per config (pcs=5/50/120), not once globally
#   [FIX-2] No double StandardScaler — scaling controlled by
#            c["scale"] flag only
#   [FIX-3] class_weight applied via set_params(), not direct assignment
#   [FIX-4] syntaxB/C row-aligned to model_df order
#   [FIX-6] PCA n_components clamped to min(pcs, n_features)
#
# NEW — No-PCA experiment:
#   [NoPCA] use_pca=False configs feed raw (optionally scaled)
#           embeddings directly to the classifier.
#           Key question: is PCA truncation (768→120) killing
#           the signal advantage of SQL-specialized embeddings?
#
# Configs added:
#   SVC_RBF-raw-True  — SVC on full 768-dim scaled embeddings
#   LRG-raw-True      — Logistic Regression on full 768-dim
#   RFC-raw-False     — Random Forest on full 768-dim (no scale needed)
#   GBC-raw-False     — GBC on full 768-dim
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score, f1_score)
from sentence_transformers import SentenceTransformer
import sqlparse

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    JOB_PATH = "./data/job.csv"
    CEB_PATH = "./data/ceb.csv"

    ENCODERS = {
        "AdaSteer-Contrastive": "encoders/encoder_all-mpnet-base-v2_v1",
        "AdaSteer-Reptile"    : "encoders/encoder_reptile_mpnet_v4",
        "AdaSteer-Oracle"     : "adasteer_encoder",
    }

    LLMSTEER_PUBLISHED = {
        "workload_sum": 2547.7,
        "p90"         : 5.7,
        "auroc"       : None,
    }

    RANDOM_SEED  = 24508
    THRESHOLD    = 0.5
    K_FOLDS      = 10
    TRAIN_SIZE   = 0.8

    # ── use_pca=True  → StandardScaler + PCA(pcs) [+ optional post_scale]
    # ── use_pca=False → StandardScaler (if scale=True) or raw
    # ── post_scale=True → second StandardScaler applied after PCA
    #    (paper notation "SVC-120-S", the winning config)
    MODEL_CONFIGS = [
        # ── PCA configs — reproduce compare.py behaviour exactly ──────────────
        # In compare.py ALL configs went through StandardScaler+PCA(768) first.
        # scale=False there → one StandardScaler (pre-PCA only)
        # scale=True there  → two StandardScalers (pre-PCA and post-PCA)
        #
        # Here we map that correctly:
        #   scale=True, post_scale=False  → StandardScaler + PCA(pcs)
        #   scale=True, post_scale=True   → StandardScaler + PCA(pcs) + StandardScaler
        {"name": "SVC-5",     "pcs": 5,   "scale": True, "post_scale": False, "use_pca": True,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "SVC-50",    "pcs": 50,  "scale": True, "post_scale": False, "use_pca": True,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "SVC-120",   "pcs": 120, "scale": True, "post_scale": False, "use_pca": True,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        # SVC-120-S: the winning config — adds second StandardScaler after PCA
        {"name": "SVC-120-S", "pcs": 120, "scale": True, "post_scale": True,  "use_pca": True,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "LRG-120",   "pcs": 120, "scale": True, "post_scale": False, "use_pca": True,
         "estimator": LogisticRegression(random_state=24508, max_iter=2000,
                                         solver="saga", n_jobs=-1)},
        {"name": "GBC-120",   "pcs": 120, "scale": True, "post_scale": False, "use_pca": True,
         "estimator": GradientBoostingClassifier(random_state=24508, n_estimators=100,
                                                  max_depth=3, learning_rate=0.1)},
        # ── No-PCA configs — full 768-dim embedding space ─────────────────────
        {"name": "LinSVC-raw-True",  "pcs": None, "scale": True,  "post_scale": False, "use_pca": False,
         "estimator": CalibratedClassifierCV(
             LinearSVC(random_state=24508, max_iter=3000, C=1.0), cv=3)},
        {"name": "LRG-raw-True",     "pcs": None, "scale": True,  "post_scale": False, "use_pca": False,
         "estimator": LogisticRegression(random_state=24508, max_iter=2000,
                                         solver="saga", n_jobs=-1)},
        {"name": "RFC-raw-False",    "pcs": None, "scale": False, "post_scale": False, "use_pca": False,
         "estimator": RandomForestClassifier(random_state=24508, n_estimators=100,
                                              n_jobs=-1)},
        {"name": "GBC-raw-False",    "pcs": None, "scale": False, "post_scale": False, "use_pca": False,
         "estimator": GradientBoostingClassifier(random_state=24508, n_estimators=100,
                                                  max_depth=3, learning_rate=0.1)},
    ]

    BENCHMARK_IDX = 0
    LONGTAIL_IDX  = 26
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

pca_names   = [c["name"] for c in cfg.MODEL_CONFIGS if     c["use_pca"]]
nopca_names = [c["name"] for c in cfg.MODEL_CONFIGS if not c["use_pca"]]

print("=" * 65)
print("AdaSteer vs LLMSteer — Pipeline Comparison (FIXED + PCA verify)")
print("=" * 65)
print(f"Device        : {cfg.DEVICE}")
print(f"Encoders      : {list(cfg.ENCODERS.keys())}")
print(f"CV folds      : {cfg.K_FOLDS}")
print(f"PCA configs   : {len(pca_names)}  → {pca_names}")
print(f"No-PCA configs: {len(nopca_names)} → {nopca_names}")
print("=" * 65)


# ============================================================
# SECTION 2: LOAD DATA
# ============================================================

print("\nLoading data...")

job_df = pd.read_csv(cfg.JOB_PATH,
                     converters={"hint_list": eval, "runtime_list": eval})
ceb_df = pd.read_csv(cfg.CEB_PATH,
                     converters={"hint_list": eval, "runtime_list": eval})

data = pd.concat([job_df, ceb_df]).reset_index(drop=True)
data["mean_runtime"] = data["runtime_list"].apply(np.mean)
data["sd_runtime"]   = data["runtime_list"].apply(np.std)
data["sql"]          = data["sql"].apply(lambda x: x.strip("\n"))

print(f"Combined data : {len(data)} rows | "
      f"{data['filename'].nunique()} unique queries")


# ============================================================
# SECTION 3: DATA PREPARATION
# ============================================================

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    model_df = data.copy()
    model_df = model_df.drop(columns=["runtime_list", "plan_tree", "sd_runtime"])
    model_df = model_df.explode(column="hint_list")
    model_df = model_df.sort_values(by=["filename", "hint_list"])
    model_df = model_df.groupby(
        by=["filename", "sql"], as_index=False
    ).agg({
        "hint_list"   : lambda x: x.tolist(),
        "mean_runtime": lambda x: x.tolist()
    })
    model_df["opt_l"] = model_df["mean_runtime"].apply(min)
    model_df = model_df.reset_index(drop=True)   # stable row order
    return model_df


def embed_queries(model_df: pd.DataFrame, encoder_path: str) -> tuple:
    """
    Generate embeddings. Rows are aligned to model_df order throughout
    so syntaxB/C tensors index correctly with train_idx/test_idx.
    """
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(encoder_path, device=device)
    st_model.max_seq_length = 512

    sql_texts = model_df["sql"].tolist()   # same order as model_df rows
    print(f"  Embedding {len(sql_texts)} queries "
          f"(dim will be {st_model.get_sentence_embedding_dimension()})...")

    def encode(texts):
        return st_model.encode(
            texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    base_embs = encode(sql_texts)

    syntaxB_sql = [sqlparse.format(s, reindent=True,
                                   use_space_around_operators=True,
                                   indent_tabs=False) for s in sql_texts]
    syntaxC_sql = [sqlparse.format(s, reindent=True,
                                   use_space_around_operators=True,
                                   indent_tabs=True)  for s in sql_texts]

    syntaxB_embs = encode(syntaxB_sql)
    syntaxC_embs = encode(syntaxC_sql)

    del st_model
    torch.cuda.empty_cache()

    model_df = model_df.copy()
    model_df["features"] = list(base_embs)   # each row = 1-D numpy array

    return (
        model_df,
        torch.tensor(syntaxB_embs, dtype=torch.float32),
        torch.tensor(syntaxC_embs, dtype=torch.float32),
    )


def build_tensors(model_df: pd.DataFrame) -> tuple:
    X      = torch.tensor(np.stack(model_df["features"].tolist()),
                          dtype=torch.float32)
    hint_l = torch.stack(model_df["mean_runtime"].apply(torch.Tensor).tolist())
    opt_l  = torch.stack(
        model_df["opt_l"].apply(
            lambda x: torch.Tensor([x]).repeat(hint_l.size(1))
        ).tolist()
    )
    binary_l = (
        hint_l[:, cfg.BENCHMARK_IDX] > hint_l[:, cfg.LONGTAIL_IDX]
    ).float()
    return X, hint_l, opt_l, binary_l


# ============================================================
# SECTION 4: RESULT LIST INIT
# ============================================================

METRIC_KEYS = [
    "model_train_accuracy", "model_test_accuracy",
    "model_train_recall",   "model_test_recall",
    "model_train_precision","model_test_precision",
    "model_train_f1score",  "model_test_f1score",
    "model_train_auroc",    "model_test_auroc",
    "apriori_train_distribution", "apriori_test_distribution",
    "train_model_workload", "test_model_workload",
    "train_opt_workload",   "test_opt_workload",
    "train_benchmark_workload", "test_benchmark_workload",
    "train_apriori_workload","test_apriori_workload",
    "train_model_p90",      "test_model_p90",
    "train_opt_p90",        "test_opt_p90",
    "train_benchmark_p90",  "test_benchmark_p90",
    "train_apriori_p90",    "test_apriori_p90",
    "train_model_median",   "test_model_median",
    "train_opt_median",     "test_opt_median",
    "train_benchmark_median","test_benchmark_median",
    "train_apriori_median", "test_apriori_median",
    "model_spaced_auroc",   "model_tabbed_auroc",
    "model_spaced_workload","model_tabbed_workload",
    "model_spaced_p90",     "model_tabbed_p90",
]

def init_result_lists(model_cfgs):
    for c in model_cfgs:
        for k in METRIC_KEYS:
            c[k] = []
    return model_cfgs


# ============================================================
# SECTION 5: TRAINING LOOP
# ============================================================

def _make_preprocessor(use_pca, scale, pcs, n_feat, random_state,
                       post_scale=False):
    """
    Build and return a fitted-ready preprocessor for one config.

    use_pca=True,  scale=True,  post_scale=False → StandardScaler + PCA(pcs)
    use_pca=True,  scale=True,  post_scale=True  → StandardScaler + PCA(pcs) + StandardScaler
                                                   (paper's "SVC-120-S", S = second scaler)
    use_pca=True,  scale=False                   → PCA(pcs) only
    use_pca=False, scale=True                    → StandardScaler only
    use_pca=False, scale=False                   → identity (None)

    Returns a sklearn object with .fit_transform / .transform,
    or None when no transformation is needed.
    """
    from sklearn.pipeline import Pipeline

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if use_pca:
        n_components = min(pcs, n_feat)
        steps.append(("pca", PCA(n_components=n_components,
                                  random_state=random_state)))
    if use_pca and post_scale:
        steps.append(("post_scaler", StandardScaler()))

    if not steps:
        return None                    # raw unscaled — no-op
    if len(steps) == 1:
        return steps[0][1]             # single transform, no pipeline needed
    return Pipeline(steps)


def run_cv(model_cfgs, X, hint_l, opt_l, targets_l, spaced_X, tabbed_X):
    """
    10-fold stratified CV.

    Per-config preprocessing (FIX-1, FIX-2):
        Each config builds its own preprocessor via _make_preprocessor().
        PCA is fitted with the exact n_components for that config.
        StandardScaler is applied at most once.

    class_weight via set_params() (FIX-3).
    syntaxB/C already row-aligned at embed time (FIX-4).
    """
    benchmark_const = torch.LongTensor([cfg.BENCHMARK_IDX])
    longtail_const  = torch.LongTensor([cfg.LONGTAIL_IDX])

    splitter = StratifiedShuffleSplit(
        n_splits=cfg.K_FOLDS,
        train_size=cfg.TRAIN_SIZE,
        random_state=cfg.RANDOM_SEED,
    )

    X_np   = X.numpy()
    sp_np  = spaced_X.numpy()
    tb_np  = tabbed_X.numpy()
    n_feat = X_np.shape[1]

    for fold_i, (train_idx, test_idx) in enumerate(
        splitter.split(X_np, targets_l)
    ):
        print(f"    Fold {fold_i+1}/{cfg.K_FOLDS}...")

        X_tr_raw = X_np[train_idx]
        X_te_raw = X_np[test_idx]
        sp_raw   = sp_np[test_idx]
        tb_raw   = tb_np[test_idx]
        y_train  = targets_l[train_idx].numpy()
        y_test   = targets_l[test_idx].numpy()

        n_pos = (targets_l[train_idx] == 1).sum().item()
        n_neg = (targets_l[train_idx] == 0).sum().item()
        weights = {0: n_pos / n_neg, 1: n_neg / n_pos}

        apriori_tr = hint_l[train_idx].gather(
            1, torch.where(torch.Tensor(y_train) > cfg.THRESHOLD,
                           longtail_const, benchmark_const).view(-1, 1))
        apriori_te = hint_l[test_idx].gather(
            1, torch.where(torch.Tensor(y_test) > cfg.THRESHOLD,
                           longtail_const, benchmark_const).view(-1, 1))

        for c in model_cfgs:

            # ── per-config preprocessor (FIX-1 + FIX-2) ────
            prep = _make_preprocessor(
                use_pca=c["use_pca"],
                scale=c["scale"],
                pcs=c["pcs"],
                n_feat=n_feat,
                random_state=cfg.RANDOM_SEED,
                post_scale=c.get("post_scale", False),
            )

            if prep is None:
                Xtr, Xte, Xsp, Xtb = (X_tr_raw.copy(), X_te_raw.copy(),
                                        sp_raw.copy(),   tb_raw.copy())
            else:
                Xtr = prep.fit_transform(X_tr_raw)
                Xte = prep.transform(X_te_raw)
                Xsp = prep.transform(sp_raw)
                Xtb = prep.transform(tb_raw)
            # ────────────────────────────────────────────────

            est = deepcopy(c["estimator"])
            try:
                est.set_params(class_weight=weights)   # FIX-3
            except (ValueError, TypeError):
                pass

            est.fit(Xtr, y_train)
            y_tr_pred = est.predict(Xtr)
            y_te_pred = est.predict(Xte)
            y_sp_pred = est.predict(Xsp)
            y_tb_pred = est.predict(Xtb)

            c["model_train_accuracy"].append(accuracy_score(y_train, y_tr_pred))
            c["model_test_accuracy"].append(accuracy_score(y_test,   y_te_pred))
            c["model_train_recall"].append(recall_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_recall"].append(recall_score(y_test,   y_te_pred, zero_division=0))
            c["model_train_precision"].append(precision_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_precision"].append(precision_score(y_test,   y_te_pred, zero_division=0))
            c["model_train_f1score"].append(f1_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_f1score"].append(f1_score(y_test,   y_te_pred, zero_division=0))

            if hasattr(est, "predict_proba"):
                tr_prob = est.predict_proba(Xtr)[:, 1]
                te_prob = est.predict_proba(Xte)[:, 1]
                sp_prob = est.predict_proba(Xsp)[:, 1]
                tb_prob = est.predict_proba(Xtb)[:, 1]
            else:
                tr_prob = est.decision_function(Xtr)
                te_prob = est.decision_function(Xte)
                sp_prob = est.decision_function(Xsp)
                tb_prob = est.decision_function(Xtb)

            c["model_train_auroc"].append(roc_auc_score(y_train, tr_prob))
            c["model_test_auroc"].append(roc_auc_score(y_test,   te_prob))
            c["model_spaced_auroc"].append(roc_auc_score(y_test, sp_prob))
            c["model_tabbed_auroc"].append(roc_auc_score(y_test, tb_prob))
            c["apriori_train_distribution"].append(
                (y_train.shape[0] - y_train.sum()) / y_train.shape[0])
            c["apriori_test_distribution"].append(
                (y_test.shape[0] - y_test.sum()) / y_test.shape[0])

            def get_rt(idx, preds):
                return hint_l[idx].gather(
                    1, torch.where(torch.Tensor(preds) > cfg.THRESHOLD,
                                   longtail_const,
                                   benchmark_const).view(-1, 1))

            m_tr = get_rt(train_idx, y_tr_pred)
            m_te = get_rt(test_idx,  y_te_pred)
            m_sp = get_rt(test_idx,  y_sp_pred)
            m_tb = get_rt(test_idx,  y_tb_pred)

            c["train_model_workload"].append(m_tr.sum().item())
            c["test_model_workload"].append(m_te.sum().item())
            c["train_opt_workload"].append(opt_l[train_idx].mean(dim=1).sum().item())
            c["test_opt_workload"].append(opt_l[test_idx].mean(dim=1).sum().item())
            c["train_benchmark_workload"].append(hint_l[train_idx, 0].sum().item())
            c["test_benchmark_workload"].append(hint_l[test_idx, 0].sum().item())
            c["train_apriori_workload"].append(apriori_tr.sum().item())
            c["test_apriori_workload"].append(apriori_te.sum().item())

            c["train_model_p90"].append(m_tr.quantile(0.90).item())
            c["test_model_p90"].append(m_te.quantile(0.90).item())
            c["train_opt_p90"].append(opt_l[train_idx].mean(dim=1).quantile(0.90).item())
            c["test_opt_p90"].append(opt_l[test_idx].mean(dim=1).quantile(0.90).item())
            c["train_benchmark_p90"].append(hint_l[train_idx, 0].quantile(0.90).item())
            c["test_benchmark_p90"].append(hint_l[test_idx, 0].quantile(0.90).item())
            c["train_apriori_p90"].append(apriori_tr.quantile(0.90).item())
            c["test_apriori_p90"].append(apriori_te.quantile(0.90).item())

            c["train_model_median"].append(m_tr.median().item())
            c["test_model_median"].append(m_te.median().item())
            c["train_opt_median"].append(opt_l[train_idx].mean(dim=1).median().item())
            c["test_opt_median"].append(opt_l[test_idx].mean(dim=1).median().item())
            c["train_benchmark_median"].append(hint_l[train_idx, 0].median().item())
            c["test_benchmark_median"].append(hint_l[test_idx, 0].median().item())
            c["train_apriori_median"].append(apriori_tr.median().item())
            c["test_apriori_median"].append(apriori_te.median().item())

            c["model_spaced_workload"].append(m_sp.sum().item())
            c["model_tabbed_workload"].append(m_tb.sum().item())
            c["model_spaced_p90"].append(m_sp.quantile(0.90).item())
            c["model_tabbed_p90"].append(m_tb.quantile(0.90).item())

    return model_cfgs


# ============================================================
# SECTION 6: SUMMARISE
# ============================================================

def summarize(model_cfgs):
    df = pd.DataFrame(model_cfgs)
    for m in METRIC_KEYS:
        if m in df.columns:
            df[f"{m}_mean"] = df[m].apply(lambda x: np.array(x).mean())
            df[f"{m}_std"]  = df[m].apply(lambda x: np.array(x).std())
    return df


# ============================================================
# SECTION 7: MAIN LOOP
# ============================================================

all_results  = {}
model_df_raw = prepare_features(data)

for encoder_name, encoder_path in cfg.ENCODERS.items():

    print(f"\n{'='*65}")
    print(f"Running : {encoder_name}")
    print(f"Path    : {encoder_path}")
    print(f"{'='*65}")

    if not os.path.isdir(encoder_path):
        print(f"❌ Encoder not found: {encoder_path} — skipping")
        continue

    model_df, syntaxB, syntaxC = embed_queries(
        model_df_raw.copy(), encoder_path
    )

    X, hint_l, opt_l, binary_l = build_tensors(model_df)
    emb_dim = X.shape[1]
    print(f"X shape      : {X.shape}")
    print(f"hint_l shape : {hint_l.shape}")
    print(f"Labels       : {binary_l.mean():.3f} positive rate")
    print(f"Embedding dim: {emb_dim} | "
          f"NoPCA configs will use all {emb_dim} dims")

    model_cfgs = deepcopy(cfg.MODEL_CONFIGS)
    model_cfgs = init_result_lists(model_cfgs)

    print(f"Running {cfg.K_FOLDS}-fold CV "
          f"({len(model_cfgs)} configs = "
          f"{len(pca_names)} PCA + {len(nopca_names)} NoPCA)...")

    model_cfgs = run_cv(
        model_cfgs, X, hint_l, opt_l, binary_l,
        syntaxB, syntaxC
    )

    perf_df = summarize(model_cfgs)
    perf_df["encoder"] = encoder_name
    all_results[encoder_name] = perf_df

    out_path = f"results/verify_{encoder_name.replace(' ', '_')}.csv"
    perf_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")


# ============================================================
# SECTION 8: COMPARISON TABLE
# ============================================================

print("\n" + "=" * 75)
print("COMPARISON TABLE — Best PCA + Best No-PCA per Encoder")
print("(sorted by test workload sum)")
print("=" * 75)

comparison_rows = []

for encoder_name, perf_df in all_results.items():

    # best overall (any config)
    best_all = perf_df.loc[perf_df["test_model_workload_mean"].idxmin()]

    # best no-PCA only
    nopca_df  = perf_df[perf_df["use_pca"] == False]
    best_nopca = (nopca_df.loc[nopca_df["test_model_workload_mean"].idxmin()]
                  if len(nopca_df) else None)

    for tag, row in [("(best-all)",   best_all),
                     ("(best-nopca)", best_nopca)]:
        if row is None:
            continue
        comparison_rows.append({
            "method"      : f"{encoder_name} {tag}",
            "best_model"  : row["name"],
            "workload_sum": round(row["test_model_workload_mean"], 2),
            "workload_std": round(row["test_model_workload_std"],  2),
            "p90"         : round(row["test_model_p90_mean"],      4),
            "p90_std"     : round(row["test_model_p90_std"],       4),
            "auroc"       : round(row["model_test_auroc_mean"],    4),
            "auroc_std"   : round(row["model_test_auroc_std"],     4),
            "f1"          : round(row["model_test_f1score_mean"],  4),
            "use_pca"     : row["use_pca"],
        })

# LLMSteer baseline at top
comparison_rows.insert(0, {
    "method"      : "LLMSteer (OpenAI)",
    "best_model"  : "published",
    "workload_sum": cfg.LLMSTEER_PUBLISHED["workload_sum"],
    "workload_std": 0,
    "p90"         : cfg.LLMSTEER_PUBLISHED["p90"],
    "p90_std"     : 0,
    "auroc"       : cfg.LLMSTEER_PUBLISHED.get("auroc"),
    "auroc_std"   : 0,
    "f1"          : None,
    "use_pca"     : None,
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("results/verify_comparison_summary.csv", index=False)

hdr = f"{'Method':<42} | {'Workload':>13} | {'P90':>8} | {'AUROC':>8} | Best Classifier"
print(hdr)
print("-" * len(hdr))
for _, row in comparison_df.iterrows():
    auroc = f"{row['auroc']:.4f}" if row["auroc"] is not None else "  N/A "
    flag  = " [NoPCA]" if row["use_pca"] is False else ""
    print(f"{row['method']:<42} | "
          f"{row['workload_sum']:>8.2f}±{row['workload_std']:<4.0f} | "
          f"{row['p90']:>8.4f} | "
          f"{auroc:>8} | "
          f"{row['best_model']}{flag}")

print(f"\nSaved: results/verify_comparison_summary.csv")


# ============================================================
# SECTION 9: FIGURE — PCA vs No-PCA side by side
# ============================================================

print("\nGenerating figure...")

# separate PCA and NoPCA rows for each encoder
enc_names  = list(all_results.keys())
colors_pca   = ["#0077ff", "#F44336", "#4CAF50"]
colors_nopca = ["#0055bb", "#C62828", "#2E7D32"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

def bar_group(ax, labels, vals_pca, vals_nopca,
              errs_pca, errs_nopca, ylabel, title, fmt):
    x     = np.arange(len(labels))
    w     = 0.35
    for i, (vp, vn, ep, en, c_p, c_n) in enumerate(
        zip(vals_pca, vals_nopca,
            errs_pca, errs_nopca,
            colors_pca, colors_nopca)
    ):
        ax.bar(x[i] - w/2, vp, w, yerr=ep, color=c_p,
               alpha=0.85, label="PCA" if i == 0 else "",
               error_kw=dict(capsize=3))
        ax.bar(x[i] + w/2, vn, w, yerr=en, color=c_n,
               alpha=0.85, label="No-PCA" if i == 0 else "",
               error_kw=dict(capsize=3))
        offset = max(vp, vn) * 0.02
        ax.text(x[i] - w/2, vp + ep + offset,
                f"{vp:{fmt}}", ha="center", fontsize=7, fontweight="bold")
        ax.text(x[i] + w/2, vn + en + offset,
                f"{vn:{fmt}}", ha="center", fontsize=7, fontweight="bold",
                color="darkblue")

    # LLMSteer baseline line
    if ylabel.startswith("Total"):
        ax.axhline(cfg.LLMSTEER_PUBLISHED["workload_sum"],
                   color="black", ls="--", lw=1.2,
                   label=f"LLMSteer ({cfg.LLMSTEER_PUBLISHED['workload_sum']:.0f}s)")
    elif ylabel.startswith("P90"):
        ax.axhline(cfg.LLMSTEER_PUBLISHED["p90"],
                   color="black", ls="--", lw=1.2,
                   label=f"LLMSteer ({cfg.LLMSTEER_PUBLISHED['p90']:.1f}s)")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("AdaSteer-","") for n in labels],
                       fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_metric(enc, col, use_pca_flag):
    df = all_results[enc]
    sub = df[df["use_pca"] == use_pca_flag]
    if len(sub) == 0:
        return np.nan, np.nan
    best = sub.loc[sub["test_model_workload_mean"].idxmin()]
    return best[f"{col}_mean"], best[f"{col}_std"]


wl_pca   = [get_metric(e, "test_model_workload", True)[0]  for e in enc_names]
wl_npca  = [get_metric(e, "test_model_workload", False)[0] for e in enc_names]
wle_pca  = [get_metric(e, "test_model_workload", True)[1]  for e in enc_names]
wle_npca = [get_metric(e, "test_model_workload", False)[1] for e in enc_names]

p90_pca   = [get_metric(e, "test_model_p90", True)[0]  for e in enc_names]
p90_npca  = [get_metric(e, "test_model_p90", False)[0] for e in enc_names]
p90e_pca  = [get_metric(e, "test_model_p90", True)[1]  for e in enc_names]
p90e_npca = [get_metric(e, "test_model_p90", False)[1] for e in enc_names]

au_pca   = [get_metric(e, "model_test_auroc", True)[0]  for e in enc_names]
au_npca  = [get_metric(e, "model_test_auroc", False)[0] for e in enc_names]
aue_pca  = [get_metric(e, "model_test_auroc", True)[1]  for e in enc_names]
aue_npca = [get_metric(e, "model_test_auroc", False)[1] for e in enc_names]

bar_group(axes[0], enc_names,
          wl_pca,  wl_npca,  wle_pca,  wle_npca,
          "Total Workload Latency (s)", "Workload Sum\n(lower = better)", ".0f")

bar_group(axes[1], enc_names,
          p90_pca, p90_npca, p90e_pca, p90e_npca,
          "P90 Latency (s)", "P90 Tail Latency\n(lower = better)", ".2f")

bar_group(axes[2], enc_names,
          au_pca,  au_npca,  aue_pca,  aue_npca,
          "AUROC", "Classification AUROC\n(higher = better)", ".4f")

fig.suptitle(
    "AdaSteer vs LLMSteer — PCA vs No-PCA Comparison\n"
    "(dark bars = No-PCA; dashed line = LLMSteer baseline)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig("results/verify_comparison_figure.pdf",
            format="pdf", dpi=300, bbox_inches="tight")
plt.savefig("results/verify_comparison_figure.png",
            format="png", dpi=300, bbox_inches="tight")
print("✅ Figures saved: results/verify_comparison_figure.{pdf,png}")


# ============================================================
# SECTION 10: FINAL DIAGNOSIS
# ============================================================

print("\n" + "=" * 65)
print("DIAGNOSIS")
print("=" * 65)

llm_wl = cfg.LLMSTEER_PUBLISHED["workload_sum"]

for enc in enc_names:
    wl_p,  _  = get_metric(enc, "test_model_workload", True)
    wl_np, _  = get_metric(enc, "test_model_workload", False)
    if np.isnan(wl_p) or np.isnan(wl_np):
        continue
    gain_pca  = (1 - wl_p  / llm_wl) * 100
    gain_npca = (1 - wl_np / llm_wl) * 100
    delta     = wl_p - wl_np
    verdict   = ("✅ NoPCA is BETTER — PCA was the bottleneck!"
                 if wl_np < wl_p else
                 "⚠️  PCA still wins — bottleneck is likely model capacity")
    print(f"\n{enc}")
    print(f"  PCA   best: {wl_p:.2f}s  ({gain_pca:+.1f}% vs LLMSteer)")
    print(f"  NoPCA best: {wl_np:.2f}s  ({gain_npca:+.1f}% vs LLMSteer)")
    print(f"  NoPCA gain: {delta:+.2f}s  → {verdict}")

print("\nFiles produced:")
print("  results/verify_comparison_summary.csv")
print("  results/verify_comparison_figure.pdf / .png")
for enc in cfg.ENCODERS:
    print(f"  results/verify_{enc.replace(' ', '_')}.csv")
print("=" * 65) 