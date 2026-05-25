# ============================================================
# Adaptsteer vs LLMSteer — Fair Pipeline Comparison
# ============================================================
#
# PURPOSE:
#   Run the EXACT LLMSteer sklearn pipeline
#   but replace OpenAI text-embedding-3-large
#   with our trained local encoders.
#
#   Everything else is identical:
#   → same data (CEB + JOB combined)
#   → same classifiers (LR, SVC, GBC, RFC)
#   → same PCA components (5, 50, 120)
#   → same 10-fold CV (80/20 split)
#   → same metrics (workload, P90, AUROC)
#
# MODELS COMPARED:
#   LLMSteer    → text-embedding-3-large (OpenAI) [published]
#   Adaptsteer-C  → contrastive mpnet encoder
#   Adaptsteer-R  → reptile mpnet encoder
#   Adaptsteer-F  → full-supervision encoder (CEB+JOB)
#
# OUTPUT:
#   results/llmsteer_comparison_raw.csv
#   results/llmsteer_comparison_summary.csv
#   results/llmsteer_comparison_figure.pdf
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score, f1_score)
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import sqlparse

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

class Config:

    # ── Data paths ───────────────────────────────────────────
    JOB_PATH = "./data/job.csv"
    CEB_PATH = "./data/ceb.csv"

    # ── Encoder paths ────────────────────────────────────────
    ENCODERS = {
        "Adaptsteer-Contrastive": "encoders/encoder_all-mpnet-base-v2_v1",
        "Adaptsteer-Reptile"    : "encoders/encoder_reptile_mpnet_v4",
        "Optimal"     : "Adaptsteer_encoder",
    }

    # ── LLMSteer published results (from paper) ──────────────
    # Update these with actual numbers from their paper/results
    LLMSTEER_PUBLISHED = {
        "workload_sum" : 2547.7,
        "p90"          : 5.7,
        "auroc"        : None,   # add if available from paper
    }

    # ── Training setup (identical to LLMSteer) ───────────────
    RANDOM_SEED  = 24508
    THRESHOLD    = 0.5
    K_FOLDS      = 10
    TRAIN_SIZE   = 0.8

    # ── Classifier configs (identical to LLMSteer notebook) ──
    # Subset of their 30 configs — add more if needed
    MODEL_CONFIGS = [
        {"name": "LRG-5-False",     "pcs": 5,   "scale": False,
         "estimator": LogisticRegression(random_state=24508, max_iter=1000)},
        {"name": "SVC_RBF-5-False", "pcs": 5,   "scale": False,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "GBC-5-False",     "pcs": 5,   "scale": False,
         "estimator": GradientBoostingClassifier(random_state=24508, n_estimators=100)},
        {"name": "RFC-5-False",     "pcs": 5,   "scale": False,
         "estimator": RandomForestClassifier(random_state=24508, n_estimators=100)},
        {"name": "LRG-50-False",    "pcs": 50,  "scale": False,
         "estimator": LogisticRegression(random_state=24508, max_iter=1000)},
        {"name": "SVC_RBF-50-False","pcs": 50,  "scale": False,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "GBC-50-False",    "pcs": 50,  "scale": False,
         "estimator": GradientBoostingClassifier(random_state=24508, n_estimators=100)},
        {"name": "RFC-50-False",    "pcs": 50,  "scale": False,
         "estimator": RandomForestClassifier(random_state=24508, n_estimators=100)},
        {"name": "LRG-120-False",   "pcs": 120, "scale": False,
         "estimator": LogisticRegression(random_state=24508, max_iter=1000)},
        {"name": "SVC_RBF-120-False","pcs": 120,"scale": False,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "GBC-120-False",   "pcs": 120, "scale": False,
         "estimator": GradientBoostingClassifier(random_state=24508, n_estimators=100)},
        {"name": "RFC-120-False",   "pcs": 120, "scale": False,
         "estimator": RandomForestClassifier(random_state=24508, n_estimators=100)},
        {"name": "SVC_RBF-120-True","pcs": 120, "scale": True,
         "estimator": SVC(random_state=24508, kernel="rbf", probability=True)},
        {"name": "RFC-120-True",    "pcs": 120, "scale": True,
         "estimator": RandomForestClassifier(random_state=24508, n_estimators=100)},
    ]

    BENCHMARK_IDX = 0    # hint 0 = PostgreSQL default
    LONGTAIL_IDX  = 26   # hint 26 = longtail config
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

print("=" * 60)
print("Adaptsteer vs LLMSteer — Pipeline Comparison")
print("=" * 60)
print(f"Device   : {cfg.DEVICE}")
print(f"Encoders : {list(cfg.ENCODERS.keys())}")
print(f"CV folds : {cfg.K_FOLDS}")
print(f"Models   : {len(cfg.MODEL_CONFIGS)} classifiers")
print("=" * 60)


# ============================================================
# SECTION 2: LOAD DATA (identical to LLMSteer)
# ============================================================

print("\nLoading data...")

job_df = pd.read_csv(cfg.JOB_PATH,
                     converters={"hint_list": eval, "runtime_list": eval})
ceb_df = pd.read_csv(cfg.CEB_PATH,
                     converters={"hint_list": eval, "runtime_list": eval})

# LLMSteer combines both — we do the same for fair comparison
data = pd.concat([job_df, ceb_df]).reset_index(drop=True)
data["mean_runtime"] = data["runtime_list"].apply(np.mean)
data["sd_runtime"]   = data["runtime_list"].apply(np.std)
data["sql"]          = data["sql"].apply(lambda x: x.strip("\n"))

print(f"Combined data: {len(data)} rows | "
      f"{data['filename'].nunique()} unique queries")


# ============================================================
# SECTION 3: DATA PREPARATION (identical to LLMSteer)
# ============================================================

def prepare_features(data: pd.DataFrame) -> tuple:
    """
    Identical to LLMSteer's prepare_data() except
    embeddings come from local encoder instead of OpenAI.
    """
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

    return model_df


def embed_queries(model_df: pd.DataFrame,
                  encoder_path: str) -> pd.DataFrame:
    """
    Generate embeddings using local encoder.
    Replaces OpenAI API call in LLMSteer.
    """
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model      = SentenceTransformer(encoder_path, device=target_device)
    st_model.max_seq_length = 512

    subset    = model_df[["filename", "sql"]].drop_duplicates()
    filenames = subset["filename"].tolist()
    sql_texts = subset["sql"].tolist()

    print(f"  Embedding {len(sql_texts)} queries...")
    embeddings = st_model.encode(
        sql_texts,
        batch_size           = 64,
        convert_to_numpy     = True,
        normalize_embeddings = True,
        show_progress_bar    = False,
    )

    emb_map = {fname: emb.tolist()
               for fname, emb in zip(filenames, embeddings)}
    model_df = model_df.copy()
    model_df["features"] = model_df["filename"].apply(
        lambda x: emb_map[x]
    )

    # also generate syntax B and C for robustness
    syntaxB_sql = [
        sqlparse.format(sql, reindent=True,
                        use_space_around_operators=True,
                        indent_tabs=False)
        for sql in sql_texts
    ]
    syntaxC_sql = [
        sqlparse.format(sql, reindent=True,
                        use_space_around_operators=True,
                        indent_tabs=True)
        for sql in sql_texts
    ]

    syntaxB_embs = st_model.encode(
        syntaxB_sql, batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    syntaxC_embs = st_model.encode(
        syntaxC_sql, batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    del st_model
    torch.cuda.empty_cache()

    return (
        model_df,
        torch.tensor(syntaxB_embs, dtype=torch.float32),
        torch.tensor(syntaxC_embs, dtype=torch.float32),
    )


def build_tensors(model_df: pd.DataFrame) -> tuple:
    """Build X, hint_l, opt_l, binary_l tensors."""
    df = model_df.drop(columns=["filename", "sql", "hint_list"])

    X      = torch.stack(df["features"].apply(torch.Tensor).tolist())
    hint_l = torch.stack(df["mean_runtime"].apply(torch.Tensor).tolist())
    opt_l  = torch.stack(
        df["opt_l"].apply(
            lambda x: torch.Tensor([x]).repeat(hint_l.size(1))
        ).tolist()
    )
    binary_l = (
        hint_l[:, cfg.BENCHMARK_IDX] > hint_l[:, cfg.LONGTAIL_IDX]
    ).float()

    return X, hint_l, opt_l, binary_l


# ============================================================
# SECTION 4: TRAINING LOOP (identical to LLMSteer notebook)
# ============================================================

def init_result_lists(model_cfgs: list) -> list:
    """Initialize empty result lists for each model config."""
    keys = [
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
    for c in model_cfgs:
        for k in keys:
            c[k] = []
    return model_cfgs


def run_cv(model_cfgs, X, hint_l, opt_l, targets_l,
           spaced_sql, tabbed_sql):
    """
    10-fold CV training loop.
    Identical logic to LLMSteer notebook Cell 18.
    """
    from copy import deepcopy

    pipeline        = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])
    scaler          = StandardScaler()
    benchmark_const = torch.LongTensor([cfg.BENCHMARK_IDX])
    longtail_const  = torch.LongTensor([cfg.LONGTAIL_IDX])
    splitter        = StratifiedShuffleSplit(
        n_splits=cfg.K_FOLDS,
        train_size=cfg.TRAIN_SIZE,
        random_state=cfg.RANDOM_SEED
    )

    for fold_i, (train_idx, test_idx) in enumerate(
        splitter.split(X, targets_l)
    ):
        print(f"    Fold {fold_i+1}/{cfg.K_FOLDS}...")

        X_train = X[train_idx].numpy()
        X_test  = X[test_idx].numpy()
        y_train = targets_l[train_idx].numpy()
        y_test  = targets_l[test_idx].numpy()

        X_train = pipeline.fit_transform(X_train)
        X_test  = pipeline.transform(X_test)
        X_sp    = pipeline.transform(spaced_sql[test_idx].numpy())
        X_tb    = pipeline.transform(tabbed_sql[test_idx].numpy())

        # class weights
        weights = {
            0: float((targets_l[train_idx]==1).sum() /
                     (targets_l[train_idx]==0).sum()),
            1: float((targets_l[train_idx]==0).sum() /
                     (targets_l[train_idx]==1).sum()),
        }

        # baselines
        apriori_tr = hint_l[train_idx].gather(
            1, torch.where(torch.Tensor(y_train) > cfg.THRESHOLD,
                           longtail_const, benchmark_const).view(-1,1))
        apriori_te = hint_l[test_idx].gather(
            1, torch.where(torch.Tensor(y_test) > cfg.THRESHOLD,
                           longtail_const, benchmark_const).view(-1,1))

        for c in model_cfgs:
            PCS = c["pcs"]
            est = deepcopy(c["estimator"])

            if c["scale"]:
                Xtr = scaler.fit_transform(X_train)
                Xte = scaler.transform(X_test)
                Xsp = scaler.transform(X_sp)
                Xtb = scaler.transform(X_tb)
            else:
                Xtr, Xte, Xsp, Xtb = (X_train.copy(), X_test.copy(),
                                        X_sp.copy(), X_tb.copy())

            if hasattr(est, "class_weight"):
                est.class_weight = weights.copy()

            est.fit(Xtr[:, :PCS], y_train)
            y_tr_pred = est.predict(Xtr[:, :PCS])
            y_te_pred = est.predict(Xte[:, :PCS])
            y_sp_pred = est.predict(Xsp[:, :PCS])
            y_tb_pred = est.predict(Xtb[:, :PCS])

            # classification metrics
            c["model_train_accuracy"].append(accuracy_score(y_train, y_tr_pred))
            c["model_test_accuracy"].append(accuracy_score(y_test,  y_te_pred))
            c["model_train_recall"].append(recall_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_recall"].append(recall_score(y_test,  y_te_pred, zero_division=0))
            c["model_train_precision"].append(precision_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_precision"].append(precision_score(y_test,  y_te_pred, zero_division=0))
            c["model_train_f1score"].append(f1_score(y_train, y_tr_pred, zero_division=0))
            c["model_test_f1score"].append(f1_score(y_test,  y_te_pred, zero_division=0))

            if hasattr(est, "predict_proba"):
                tr_prob = est.predict_proba(Xtr[:, :PCS])[:, 1]
                te_prob = est.predict_proba(Xte[:, :PCS])[:, 1]
                sp_prob = est.predict_proba(Xsp[:, :PCS])[:, 1]
                tb_prob = est.predict_proba(Xtb[:, :PCS])[:, 1]
            else:
                tr_prob = est.decision_function(Xtr[:, :PCS])
                te_prob = est.decision_function(Xte[:, :PCS])
                sp_prob = est.decision_function(Xsp[:, :PCS])
                tb_prob = est.decision_function(Xtb[:, :PCS])

            c["model_train_auroc"].append(roc_auc_score(y_train, tr_prob))
            c["model_test_auroc"].append(roc_auc_score(y_test,  te_prob))
            c["model_spaced_auroc"].append(roc_auc_score(y_test, sp_prob))
            c["model_tabbed_auroc"].append(roc_auc_score(y_test, tb_prob))
            c["apriori_train_distribution"].append(
                (y_train.shape[0] - y_train.sum()) / y_train.shape[0])
            c["apriori_test_distribution"].append(
                (y_test.shape[0] - y_test.sum()) / y_test.shape[0])

            # latency metrics
            def get_rt(idx, preds):
                return hint_l[idx].gather(
                    1, torch.where(torch.Tensor(preds) > cfg.THRESHOLD,
                                   longtail_const,
                                   benchmark_const).view(-1,1))

            m_tr = get_rt(train_idx, y_tr_pred)
            m_te = get_rt(test_idx,  y_te_pred)
            m_sp = get_rt(test_idx,  y_sp_pred)
            m_tb = get_rt(test_idx,  y_tb_pred)

            c["train_model_workload"].append(m_tr.sum().item())
            c["test_model_workload"].append(m_te.sum().item())
            c["train_opt_workload"].append(opt_l[train_idx].mean(dim=1).sum().item())
            c["test_opt_workload"].append(opt_l[test_idx].mean(dim=1).sum().item())
            c["train_benchmark_workload"].append(hint_l[train_idx,0].sum().item())
            c["test_benchmark_workload"].append(hint_l[test_idx,0].sum().item())
            c["train_apriori_workload"].append(apriori_tr.sum().item())
            c["test_apriori_workload"].append(apriori_te.sum().item())

            c["train_model_p90"].append(m_tr.quantile(0.90).item())
            c["test_model_p90"].append(m_te.quantile(0.90).item())
            c["train_opt_p90"].append(opt_l[train_idx].mean(dim=1).quantile(0.90).item())
            c["test_opt_p90"].append(opt_l[test_idx].mean(dim=1).quantile(0.90).item())
            c["train_benchmark_p90"].append(hint_l[train_idx,0].quantile(0.90).item())
            c["test_benchmark_p90"].append(hint_l[test_idx,0].quantile(0.90).item())
            c["train_apriori_p90"].append(apriori_tr.quantile(0.90).item())
            c["test_apriori_p90"].append(apriori_te.quantile(0.90).item())

            c["train_model_median"].append(m_tr.median().item())
            c["test_model_median"].append(m_te.median().item())
            c["train_opt_median"].append(opt_l[train_idx].mean(dim=1).median().item())
            c["test_opt_median"].append(opt_l[test_idx].mean(dim=1).median().item())
            c["train_benchmark_median"].append(hint_l[train_idx,0].median().item())
            c["test_benchmark_median"].append(hint_l[test_idx,0].median().item())
            c["train_apriori_median"].append(apriori_tr.median().item())
            c["test_apriori_median"].append(apriori_te.median().item())

            c["model_spaced_workload"].append(m_sp.sum().item())
            c["model_tabbed_workload"].append(m_tb.sum().item())
            c["model_spaced_p90"].append(m_sp.quantile(0.90).item())
            c["model_tabbed_p90"].append(m_tb.quantile(0.90).item())

    return model_cfgs


# ============================================================
# SECTION 5: SUMMARIZE RESULTS
# ============================================================

def summarize(model_cfgs: list) -> pd.DataFrame:
    """Compute mean ± std per metric across CV folds."""
    df = pd.DataFrame(model_cfgs)

    metrics = [
        "model_train_accuracy", "model_test_accuracy",
        "model_train_auroc",    "model_test_auroc",
        "model_train_f1score",  "model_test_f1score",
        "model_train_recall",   "model_test_recall",
        "model_train_precision","model_test_precision",
        "train_model_workload", "test_model_workload",
        "test_benchmark_workload","test_opt_workload",
        "test_apriori_workload",
        "train_model_p90",      "test_model_p90",
        "test_benchmark_p90",   "test_opt_p90",
        "test_apriori_p90",
        "model_spaced_auroc",   "model_tabbed_auroc",
        "model_spaced_workload","model_tabbed_workload",
        "model_spaced_p90",     "model_tabbed_p90",
    ]

    for m in metrics:
        if m in df.columns:
            df[f"{m}_mean"] = df[m].apply(lambda x: np.array(x).mean())
            df[f"{m}_std"]  = df[m].apply(lambda x: np.array(x).std())

    return df


# ============================================================
# SECTION 6: MAIN LOOP — RUN ALL ENCODERS
# ============================================================

all_results  = {}   # encoder_name → summarized DataFrame
model_df_raw = prepare_features(data)

for encoder_name, encoder_path in cfg.ENCODERS.items():

    print(f"\n{'='*60}")
    print(f"Running: {encoder_name}")
    print(f"Path   : {encoder_path}")
    print(f"{'='*60}")

    if not os.path.isdir(encoder_path):
        print(f"❌ Encoder not found: {encoder_path} — skipping")
        continue

    # generate embeddings
    print(f"Generating embeddings...")
    model_df, syntaxB, syntaxC = embed_queries(
        model_df_raw.copy(), encoder_path
    )

    # build tensors
    X, hint_l, opt_l, binary_l = build_tensors(model_df)
    print(f"X shape     : {X.shape}")
    print(f"hint_l shape: {hint_l.shape}")
    print(f"Labels      : {binary_l.mean():.3f} positive rate")

    # init model configs
    from copy import deepcopy
    model_cfgs = deepcopy(cfg.MODEL_CONFIGS)
    model_cfgs = init_result_lists(model_cfgs)

    # run CV
    print(f"Running {cfg.K_FOLDS}-fold CV...")
    model_cfgs = run_cv(
        model_cfgs, X, hint_l, opt_l, binary_l,
        syntaxB, syntaxC
    )

    # summarize
    perf_df = summarize(model_cfgs)
    perf_df["encoder"] = encoder_name
    all_results[encoder_name] = perf_df

    # save per-encoder results
    out_path = f"results/llmsteer_pipeline_{encoder_name.replace(' ', '_')}.csv"
    perf_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")


# ============================================================
# SECTION 7: COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON TABLE — Best Model Per Encoder")
print("(Best = lowest test workload sum)")
print("=" * 60)

comparison_rows = []

for encoder_name, perf_df in all_results.items():
    best = perf_df.loc[perf_df["test_model_workload_mean"].idxmin()]
    comparison_rows.append({
        "method"        : encoder_name,
        "best_model"    : best["name"],
        "workload_sum"  : round(best["test_model_workload_mean"], 2),
        "workload_std"  : round(best["test_model_workload_std"],  2),
        "p90"           : round(best["test_model_p90_mean"],      4),
        "p90_std"       : round(best["test_model_p90_std"],       4),
        "auroc"         : round(best["model_test_auroc_mean"],    4),
        "auroc_std"     : round(best["model_test_auroc_std"],     4),
        "f1"            : round(best["model_test_f1score_mean"],  4),
    })

# add LLMSteer published results
comparison_rows.insert(0, {
    "method"      : "LLMSteer (OpenAI)",
    "best_model"  : "published",
    "workload_sum": cfg.LLMSTEER_PUBLISHED["workload_sum"],
    "workload_std": 0,
    "p90"         : cfg.LLMSTEER_PUBLISHED["p90"],
    "p90_std"     : 0,
    "auroc"       : cfg.LLMSTEER_PUBLISHED.get("auroc", None),
    "auroc_std"   : 0,
    "f1"          : None,
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("results/llmsteer_comparison_summary.csv", index=False)

print(f"\n{'Method':<30} | {'Workload Sum':>14} | {'P90':>8} | {'AUROC':>8} | Best Classifier")
print("-" * 80)
for _, row in comparison_df.iterrows():
    auroc = f"{row['auroc']:.4f}" if row["auroc"] is not None else "N/A"
    print(f"{row['method']:<30} | "
          f"{row['workload_sum']:>10.2f} ± {row['workload_std']:<3.0f} | "
          f"{row['p90']:>8.4f} | "
          f"{auroc:>8} | "
          f"{row['best_model']}")

print(f"\nSaved: results/llmsteer_comparison_summary.csv")


# ============================================================
# SECTION 8: COMPARISON FIGURE
# ============================================================

print("\nGenerating comparison figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods = comparison_df["method"].tolist()
colors  = ["#287D8EFF", "#0077ff", "#F44336", "#4CAF50"]
x       = np.arange(len(methods))

# workload sum
ax1.bar(x, comparison_df["workload_sum"],
        yerr   = comparison_df["workload_std"],
        color  = colors[:len(methods)],
        alpha  = 0.85,
        width  = 0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("Total Workload Latency (s)", fontsize=11)
ax1.set_title("Workload Sum Comparison", fontsize=12)
ax1.grid(axis="y", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.sp