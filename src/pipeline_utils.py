import gc
import json
import os
import torch
import sqlparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Optional, Union, List, Dict

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================
# SECTION 1: ENCODER REGISTRY
# ============================================================
# All trained encoders in one place.
# Change ACTIVE_ENCODER to switch between them.
#
# CONTRASTIVE = fine-tuned with execution-derived supervision
# REPTILE     = contrastive + meta-learning for adaptation
# ORACLE      = trained on CEB+JOB (upper bound reference)
# ============================================================

ENCODER_REGISTRY = {
    "contrastive": "encoders/encoder_all-mpnet-base-v2_v1",
    "reptile"    : "encoders/encoder_reptile_mpnet_v4",
    "oracle"     : "adasteer_encoder",
}
MODEL_REGISTRY = ENCODER_REGISTRY  # alias for backward compatibility

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "❌ CPU only")


# ============================================================
# SECTION 2: MODEL STATE
# Global state for the currently loaded encoder.
# Use set_embedding_model() to switch encoders.
# ============================================================

_embed_model_path = ENCODER_REGISTRY["contrastive"]
_active_encoder_name = "contrastive"
model       = None
tokenizer   = None
st_model    = None


# ============================================================
# SECTION 3: MODEL LOADING
# ============================================================

def _load_local_embedding_model(model_path: str):
    """
    Load a SentenceTransformer encoder from a local path.
    Falls back to AutoModel if SentenceTransformer fails.
    """
    global model, tokenizer, st_model, _embed_model_path
    _embed_model_path = model_path
    model      = None
    tokenizer  = None
    st_model   = None

    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try SentenceTransformer first (preferred)
    try:
        print(f"🔎 Loading encoder: {model_path} → {target_device}")
        st_model = SentenceTransformer(model_path, device=target_device)
        st_model.max_seq_length = 256
        print(f"✅ Loaded as SentenceTransformer"
              f" (dim={st_model.get_sentence_embedding_dimension()})")
        return
    except Exception as e:
        print(f"⚠️  SentenceTransformer failed → falling back to AutoModel. Reason: {e}")

    # Fallback to AutoModel
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        model = model.to("cuda")
    else:
        model = AutoModel.from_pretrained(model_path)
        model = model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✅ Loaded as AutoModel (fallback)")


def set_embedding_model(model_path: str, encoder_name: str = "custom"):
    """
    Switch the active embedding model.

    Args:
        model_path   : path to the encoder folder
        encoder_name : human-readable name for logging
    """
    global _active_encoder_name
    _active_encoder_name = encoder_name
    print(f"\n{'='*50}")
    print(f"Switching encoder → {encoder_name}")
    print(f"Path: {model_path}")
    print(f"{'='*50}")
    _load_local_embedding_model(model_path)


def set_encoder(encoder_name: str):
    """
    Convenience function — switch encoder by name.
    Names: 'contrastive', 'reptile', 'oracle'
    """
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {encoder_name}. "
                         f"Choose from: {list(ENCODER_REGISTRY.keys())}")
    set_embedding_model(ENCODER_REGISTRY[encoder_name], encoder_name)


def get_active_encoder_name() -> str:
    """Return the name of the currently active encoder."""
    return _active_encoder_name

get_active_model_name = get_active_encoder_name  # alias for backward compatibility


# ============================================================
# SECTION 4: EMBEDDING FUNCTIONS
# ============================================================

def local_embed(query: str) -> list:
    """
    Embed a single SQL query into a normalized vector.

    Returns:
        list of floats (embedding vector)
    """
    if st_model is None and (model is None or tokenizer is None):
        _load_local_embedding_model(_embed_model_path)

    if st_model is not None:
        emb = st_model.encode(
            [query],
            convert_to_numpy    = True,
            normalize_embeddings= True,
            show_progress_bar   = False,
        )
        return emb[0].tolist()

    # AutoModel fallback
    inputs = tokenizer(
        query,
        return_tensors = "pt",
        truncation     = True,
        padding        = True
    ).to(model.device)

    with torch.no_grad():
        outputs       = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attn_mask     = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
        pooled        = (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)
        pooled        = F.normalize(pooled, p=2, dim=1)

    return pooled.squeeze().tolist()


def local_embed_batch(queries: list, batch_size: int = 64) -> list:
    """
    Embed a batch of SQL queries into normalized vectors.

    Args:
        queries    : list of SQL strings
        batch_size : encoding batch size

    Returns:
        list of embedding vectors (list of lists)
    """
    if st_model is None and (model is None or tokenizer is None):
        _load_local_embedding_model(_embed_model_path)

    if st_model is not None:
        embeddings = st_model.encode(
            list(queries),
            batch_size          = batch_size,
            convert_to_numpy    = True,
            normalize_embeddings= True,
            show_progress_bar   = False,
        )
        return embeddings.tolist()

    # AutoModel fallback
    all_embeddings = []
    total = len(queries)

    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = queries[start:end]

        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            truncation     = True,
            padding        = True
        ).to(model.device)

        with torch.no_grad():
            outputs       = model(**inputs)
            hidden_states = outputs.last_hidden_state
            attn_mask     = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
            pooled        = (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)
            pooled        = F.normalize(pooled, p=2, dim=1)

        all_embeddings.extend(pooled.detach().cpu().tolist())

        if end % 500 == 0 or end == total:
            print(f"  Embedding: {end}/{total}")

    return all_embeddings


# ============================================================
# SECTION 5: DATA PREPARATION
# ============================================================

def generate_embeddings(
    data    : pd.DataFrame,
    filepath: str,
    filename: str,
    dims    : Optional[int] = None,
    augment : bool = False
) -> pd.DataFrame:
    """
    Generate SQL embeddings for all queries in data.

    If augment=True, also generates embeddings for
    Syntax B (spaced) and Syntax C (tabbed) variants.

    Returns:
        data with 'features' column added
        (and syntaxB/syntaxC tensors if augment=True)
    """
    subset_df  = data[["filename", "sql"]].drop_duplicates()
    filenames  = subset_df["filename"].tolist()
    sql_texts  = subset_df["sql"].tolist()

    print(f"  Encoding {len(sql_texts)} unique SQL queries "
          f"[encoder={get_active_encoder_name()}]...")

    embedded   = local_embed_batch(sql_texts, batch_size=64)
    embeddings = {
        fname: {"embedding": emb}
        for fname, emb in zip(filenames, embedded)
    }
    data["features"] = data["filename"].apply(
        lambda x: embeddings[x]["embedding"]
    )

    if not augment:
        return data

    # Syntax B — extra spaces between tokens
    syntaxB_sql = subset_df["sql"].apply(
        lambda x: sqlparse.format(
            x,
            reindent               = True,
            use_space_around_operators = True,
            indent_tabs            = False
        )
    ).tolist()

    # Syntax C — tab-indented
    syntaxC_sql = subset_df["sql"].apply(
        lambda x: sqlparse.format(
            x,
            reindent               = True,
            use_space_around_operators = True,
            indent_tabs            = True
        )
    ).tolist()

    print(f"  Encoding Syntax B variants...")
    syntaxB_embs = local_embed_batch(syntaxB_sql, batch_size=64)
    print(f"  Encoding Syntax C variants...")
    syntaxC_embs = local_embed_batch(syntaxC_sql, batch_size=64)

    return (
        data,
        torch.tensor(syntaxB_embs),
        torch.tensor(syntaxC_embs)
    )


def prepare_data(
    data       : pd.DataFrame,
    filepath   : str,
    filename   : str,
    binary_model: bool,
    dims       : Optional[int] = None,
    augment    : bool = False,
    return_meta: bool = False,
) -> tuple:
    """
    Prepare features and labels for model training.

    Binary label:
        y=1 if hint[LONGTAIL_IDX] < hint[BENCHMARK_IDX]
          → alternative config is faster → use it
        y=0 otherwise → use default PostgreSQL config

    Returns:
        X         : embedding tensor (n_queries, embed_dim)
        hint_l    : latency matrix (n_queries, n_hints)
        opt_l     : optimal latency tensor
        binary_l  : binary steering labels
        (syntaxB, syntaxC if augment=True)
        (meta_df if return_meta=True)
    """
    BENCHMARK_IDX = 0   # default PostgreSQL configuration
    LONGTAIL_IDX  = 26  # fixed alternative configuration

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

    if augment:
        model_df, syntaxB, syntaxC = generate_embeddings(
            model_df, filepath, filename, dims, augment
        )
    else:
        model_df = generate_embeddings(
            model_df, filepath, filename, dims, augment
        )

    meta_df  = model_df[["filename", "sql"]].copy()
    model_df = model_df.drop(columns=["filename", "sql", "hint_list"])

    X      = torch.stack(
        model_df["features"].apply(lambda x: torch.Tensor(x)).tolist()
    )
    hint_l = torch.stack(
        model_df["mean_runtime"].apply(lambda x: torch.Tensor(x)).tolist()
    )
    opt_l  = torch.stack(
        model_df["opt_l"].apply(
            lambda x: torch.Tensor([x]).repeat(hint_l.size(1))
        ).tolist()
    )

    if binary_model:
        # binary: is the longtail config faster than the benchmark?
        binary_l = (
            hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]
        ).type(torch.float)
    else:
        binary_l = (hint_l == opt_l).type(torch.float)

    if augment:
        if return_meta:
            return X, hint_l, opt_l, binary_l, syntaxB, syntaxC, meta_df
        return X, hint_l, opt_l, binary_l, syntaxB, syntaxC
    else:
        if return_meta:
            return X, hint_l, opt_l, binary_l, meta_df
        return X, hint_l, opt_l, binary_l


# ============================================================
# SECTION 6: DATA LOADING
# ============================================================

def load_data(workload: str = "ceb") -> pd.DataFrame:
    """
    Load execution data for a given workload.

    Args:
        workload: "job", "ceb", or "all"

    Returns:
        DataFrame with columns:
            filename, sql, hint_list, runtime_list,
            mean_runtime, sd_runtime, plan_tree, workload_id
    """
    job_df = pd.read_csv(
        "./data/job.csv",
        converters={
            "hint_list"   : eval,
            "runtime_list": eval,
        }
    )
    job_df["workload_id"] = "job"

    ceb_df = pd.read_csv(
        "./data/ceb.csv",
        converters={
            "hint_list"   : eval,
            "runtime_list": eval,
        }
    )
    ceb_df["workload_id"] = "ceb"

    if workload == "job":
        data = job_df
    elif workload == "ceb":
        data = ceb_df
    elif workload == "all":
        data = pd.concat([job_df, ceb_df], ignore_index=True)
    else:
        raise ValueError(f"Unknown workload: {workload}. "
                         f"Choose: 'job', 'ceb', or 'all'")

    data["mean_runtime"] = data["runtime_list"].apply(np.mean)
    data["sd_runtime"]   = data["runtime_list"].apply(np.std)
    data["sql"]          = data["sql"].apply(lambda x: x.strip("\n"))

    print(f"Loaded workload='{workload}': "
          f"{data['filename'].nunique()} unique queries | "
          f"{len(data)} rows")

    return data


# ============================================================
# SECTION 7: MULTI-ENCODER RUNNER
# ============================================================

def run_with_encoder(
    encoder_name : str,
    data_df      : pd.DataFrame,
    model_cfgs   : list,
    pipeline,
    scaler,
    k            : int   = 10,
    p            : float = 0.8,
    random_seed  : int   = 24508,
    threshold    : float = 0.5,
) -> list:
    """
    Run the full training + evaluation pipeline
    with a specific encoder.

    This function:
    1. Switches to the specified encoder
    2. Generates embeddings for all queries
    3. Runs 10-fold cross validation
    4. Returns results per model config

    Args:
        encoder_name : 'contrastive', 'reptile', or 'oracle'
        data_df      : loaded workload dataframe
        model_cfgs   : list of classifier configs
        pipeline     : sklearn Pipeline (scaler + PCA)
        scaler       : StandardScaler instance
        k            : number of CV folds
        p            : train size fraction
        random_seed  : random seed
        threshold    : decision threshold

    Returns:
        list of model_cfgs with results filled in
    """
    from sklearn.metrics import accuracy_score

    # switch encoder
    set_encoder(encoder_name)

    # generate embeddings with new encoder
    print(f"\n[{encoder_name}] Generating embeddings...")
    original_sql, hint_l, opt_l, targets_l, spaced_sql, tabbed_sql = prepare_data(
        data_df, "./embeddings", f"{encoder_name}_embedding",
        binary_model=True, augment=True
    )

    print(f"[{encoder_name}] Label distribution: "
          f"{int(targets_l.sum())} positive ({targets_l.mean():.1%}) | "
          f"{int((targets_l==0).sum())} negative")

    # reset result storage for each model
    for cfg in model_cfgs:
        for key in [
            "model_train_accuracy", "model_test_accuracy",
            "model_train_recall",   "model_test_recall",
            "model_train_precision","model_test_precision",
            "model_train_f1score",  "model_test_f1score",
            "model_train_auroc",    "model_test_auroc",
            "apriori_train_distribution", "apriori_test_distribution",
            "train_model_workload", "test_model_workload",
            "train_benchmark_workload", "test_benchmark_workload",
            "train_opt_workload",   "test_opt_workload",
            "train_apriori_workload","test_apriori_workload",
            "train_model_p90",      "test_model_p90",
            "train_benchmark_p90",  "test_benchmark_p90",
            "train_opt_p90",        "test_opt_p90",
            "train_apriori_p90",    "test_apriori_p90",
            "train_model_median",   "test_model_median",
            "train_benchmark_median","test_benchmark_median",
            "train_opt_median",     "test_opt_median",
            "train_apriori_median", "test_apriori_median",
            "model_spaced_accuracy","model_spaced_recall",
            "model_spaced_precision","model_spaced_f1score",
            "model_spaced_auroc",   "model_spaced_workload",
            "model_spaced_p90",     "model_spaced_median",
            "model_tabbed_accuracy","model_tabbed_recall",
            "model_tabbed_precision","model_tabbed_f1score",
            "model_tabbed_auroc",   "model_tabbed_workload",
            "model_tabbed_p90",     "model_tabbed_median",
        ]:
            cfg[key] = []

    benchmark_const = torch.LongTensor([0])
    longtail_const  = torch.LongTensor([26])
    splitter        = StratifiedShuffleSplit(
        n_splits=k, train_size=p, random_state=random_seed
    )

    for fold_i, (train_idx, test_idx) in enumerate(
        splitter.split(original_sql, targets_l)
    ):
        print(f"\n[{encoder_name}] Fold {fold_i+1}/{k}")

        X_train = original_sql[train_idx].numpy()
        X_test  = original_sql[test_idx].numpy()
        y_train = targets_l[train_idx].numpy()
        y_test  = targets_l[test_idx].numpy()

        X_train = pipeline.fit_transform(X_train)
        X_test  = pipeline.transform(X_test)

        X_spaced_test = pipeline.transform(spaced_sql[test_idx].numpy())
        X_tabbed_test = pipeline.transform(tabbed_sql[test_idx].numpy())

        weights = {
            0: ((targets_l[train_idx] == 1).sum() /
                (targets_l[train_idx] == 0).sum()).item(),
            1: ((targets_l[train_idx] == 0).sum() /
                (targets_l[train_idx] == 1).sum()).item(),
        }

        # latency baselines
        apriori_train_runtimes = hint_l[train_idx].gather(
            1, torch.where(torch.Tensor(y_train) > threshold,
                           longtail_const, benchmark_const).view(-1, 1)
        )
        apriori_test_runtimes = hint_l[test_idx].gather(
            1, torch.where(torch.Tensor(y_test) > threshold,
                           longtail_const, benchmark_const).view(-1, 1)
        )

        train_opt_workload    = opt_l[train_idx].mean(dim=1).sum()
        train_benchmark_wl    = hint_l[train_idx, 0].sum()
        train_apriori_wl      = apriori_train_runtimes.sum()
        test_opt_workload     = opt_l[test_idx].mean(dim=1).sum()
        test_benchmark_wl     = hint_l[test_idx, 0].sum()
        test_apriori_wl       = apriori_test_runtimes.sum()

        train_opt_p90     = opt_l[train_idx].mean(dim=1).quantile(0.90)
        train_bench_p90   = hint_l[train_idx, 0].quantile(0.90)
        train_apriori_p90 = apriori_train_runtimes.quantile(0.90)
        test_opt_p90      = opt_l[test_idx].mean(dim=1).quantile(0.90)
        test_bench_p90    = hint_l[test_idx, 0].quantile(0.90)
        test_apriori_p90  = apriori_test_runtimes.quantile(0.90)

        train_opt_med    = opt_l[train_idx].mean(dim=1).median()
        train_bench_med  = hint_l[train_idx, 0].median()
        train_apriori_med= apriori_train_runtimes.median()
        test_opt_med     = opt_l[test_idx].mean(dim=1).median()
        test_bench_med   = hint_l[test_idx, 0].median()
        test_apriori_med = apriori_test_runtimes.median()

        for cfg in model_cfgs:
            PCS = cfg["pcs"]

            if cfg["scale"]:
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)
                X_sp = scaler.transform(X_spaced_test)
                X_tb = scaler.transform(X_tabbed_test)
            else:
                X_tr = X_train.copy()
                X_te = X_test.copy()
                X_sp = X_spaced_test.copy()
                X_tb = X_tabbed_test.copy()

            if hasattr(cfg["estimator"], "class_weight"):
                cfg["estimator"].class_weight = weights.copy()

            cfg["estimator"].fit(X_tr[:, :PCS], y_train)
            y_tr_pred = cfg["estimator"].predict(X_tr[:, :PCS])
            y_te_pred = cfg["estimator"].predict(X_te[:, :PCS])
            y_sp_pred = cfg["estimator"].predict(X_sp[:, :PCS])
            y_tb_pred = cfg["estimator"].predict(X_tb[:, :PCS])

            # classification
            cfg["model_train_accuracy"].append(accuracy_score(y_train, y_tr_pred))
            cfg["model_test_accuracy"].append(accuracy_score(y_test,  y_te_pred))
            cfg["model_train_recall"].append(recall_score(y_train, y_tr_pred, zero_division=0))
            cfg["model_test_recall"].append(recall_score(y_test,  y_te_pred, zero_division=0))
            cfg["model_train_precision"].append(precision_score(y_train, y_tr_pred, zero_division=0))
            cfg["model_test_precision"].append(precision_score(y_test,  y_te_pred, zero_division=0))
            cfg["model_train_f1score"].append(f1_score(y_train, y_tr_pred, zero_division=0))
            cfg["model_test_f1score"].append(f1_score(y_test,  y_te_pred, zero_division=0))

            if hasattr(cfg["estimator"], "predict_proba"):
                tr_proba = cfg["estimator"].predict_proba(X_tr[:, :PCS])[:, 1]
                te_proba = cfg["estimator"].predict_proba(X_te[:, :PCS])[:, 1]
                sp_proba = cfg["estimator"].predict_proba(X_sp[:, :PCS])[:, 1]
                tb_proba = cfg["estimator"].predict_proba(X_tb[:, :PCS])[:, 1]
            else:
                tr_proba = cfg["estimator"].decision_function(X_tr[:, :PCS])
                te_proba = cfg["estimator"].decision_function(X_te[:, :PCS])
                sp_proba = cfg["estimator"].decision_function(X_sp[:, :PCS])
                tb_proba = cfg["estimator"].decision_function(X_tb[:, :PCS])

            cfg["model_train_auroc"].append(roc_auc_score(y_train, tr_proba))
            cfg["model_test_auroc"].append(roc_auc_score(y_test,  te_proba))
            cfg["model_spaced_auroc"].append(roc_auc_score(y_test, sp_proba))
            cfg["model_tabbed_auroc"].append(roc_auc_score(y_test, tb_proba))

            cfg["apriori_train_distribution"].append(
                (y_train.shape[0] - y_train.sum()) / y_train.shape[0]
            )
            cfg["apriori_test_distribution"].append(
                (y_test.shape[0] - y_test.sum()) / y_test.shape[0]
            )

            # latency
            def get_runtimes(preds):
                return hint_l[test_idx].gather(
                    1,
                    torch.where(torch.Tensor(preds) > threshold,
                                longtail_const, benchmark_const).view(-1, 1)
                )

            m_te_rt = get_runtimes(y_te_pred)
            m_sp_rt = get_runtimes(y_sp_pred)
            m_tb_rt = get_runtimes(y_tb_pred)

            m_tr_rt = hint_l[train_idx].gather(
                1,
                torch.where(torch.Tensor(y_tr_pred) > threshold,
                            longtail_const, benchmark_const).view(-1, 1)
            )

            cfg["train_model_workload"].append(m_tr_rt.sum().item())
            cfg["test_model_workload"].append(m_te_rt.sum().item())
            cfg["train_benchmark_workload"].append(train_benchmark_wl.item())
            cfg["test_benchmark_workload"].append(test_benchmark_wl.item())
            cfg["train_opt_workload"].append(train_opt_workload.item())
            cfg["test_opt_workload"].append(test_opt_workload.item())
            cfg["train_apriori_workload"].append(train_apriori_wl.item())
            cfg["test_apriori_workload"].append(test_apriori_wl.item())

            cfg["train_model_p90"].append(m_tr_rt.quantile(0.90).item())
            cfg["test_model_p90"].append(m_te_rt.quantile(0.90).item())
            cfg["train_benchmark_p90"].append(train_bench_p90.item())
            cfg["test_benchmark_p90"].append(test_bench_p90.item())
            cfg["train_opt_p90"].append(train_opt_p90.item())
            cfg["test_opt_p90"].append(test_opt_p90.item())
            cfg["train_apriori_p90"].append(train_apriori_p90.item())
            cfg["test_apriori_p90"].append(test_apriori_p90.item())

            cfg["train_model_median"].append(m_tr_rt.median().item())
            cfg["test_model_median"].append(m_te_rt.median().item())
            cfg["train_benchmark_median"].append(train_bench_med.item())
            cfg["test_benchmark_median"].append(test_bench_med.item())
            cfg["train_opt_median"].append(train_opt_med.item())
            cfg["test_opt_median"].append(test_opt_med.item())
            cfg["train_apriori_median"].append(train_apriori_med.item())
            cfg["test_apriori_median"].append(test_apriori_med.item())

            # robustness
            cfg["model_spaced_accuracy"].append(accuracy_score(y_test,  y_sp_pred))
            cfg["model_spaced_recall"].append(recall_score(y_test,  y_sp_pred, zero_division=0))
            cfg["model_spaced_precision"].append(precision_score(y_test, y_sp_pred, zero_division=0))
            cfg["model_spaced_f1score"].append(f1_score(y_test, y_sp_pred, zero_division=0))
            cfg["model_spaced_workload"].append(m_sp_rt.sum().item())
            cfg["model_spaced_p90"].append(m_sp_rt.quantile(0.90).item())
            cfg["model_spaced_median"].append(m_sp_rt.median().item())

            cfg["model_tabbed_accuracy"].append(accuracy_score(y_test,  y_tb_pred))
            cfg["model_tabbed_recall"].append(recall_score(y_test,  y_tb_pred, zero_division=0))
            cfg["model_tabbed_precision"].append(precision_score(y_test, y_tb_pred, zero_division=0))
            cfg["model_tabbed_f1score"].append(f1_score(y_test, y_tb_pred, zero_division=0))
            cfg["model_tabbed_workload"].append(m_tb_rt.sum().item())
            cfg["model_tabbed_p90"].append(m_tb_rt.quantile(0.90).item())
            cfg["model_tabbed_median"].append(m_tb_rt.median().item())

    return model_cfgs


# ============================================================
# SECTION 8: RESULTS SUMMARIZATION
# ============================================================

def summarize_results(model_cfgs: list) -> pd.DataFrame:
    """
    Convert list of model configs with raw results
    into a summary DataFrame with mean ± std per metric.
    """
    df = pd.DataFrame(model_cfgs).drop(columns=["pcs"], errors="ignore")

    scalar_cols = [
        ("model_train_accuracy",    "mean"), ("model_train_accuracy",    "std"),
        ("model_test_accuracy",     "mean"), ("model_test_accuracy",     "std"),
        ("model_train_recall",      "mean"), ("model_train_recall",      "std"),
        ("model_test_recall",       "mean"), ("model_test_recall",       "std"),
        ("model_train_precision",   "mean"), ("model_train_precision",   "std"),
        ("model_test_precision",    "mean"), ("model_test_precision",    "std"),
        ("model_train_f1score",     "mean"), ("model_train_f1score",     "std"),
        ("model_test_f1score",      "mean"), ("model_test_f1score",      "std"),
        ("model_train_auroc",       "mean"), ("model_train_auroc",       "std"),
        ("model_test_auroc",        "mean"), ("model_test_auroc",        "std"),
        ("apriori_train_distribution","mean"),
        ("apriori_test_distribution", "mean"),
        ("train_model_workload",    "mean"), ("train_model_workload",    "std"),
        ("test_model_workload",     "mean"), ("test_model_workload",     "std"),
        ("test_benchmark_workload", "mean"), ("test_benchmark_workload", "std"),
        ("test_opt_workload",       "mean"), ("test_opt_workload",       "std"),
        ("test_apriori_workload",   "mean"), ("test_apriori_workload",   "std"),
        ("train_model_p90",         "mean"), ("train_model_p90",         "std"),
        ("test_model_p90",          "mean"), ("test_model_p90",          "std"),
        ("test_benchmark_p90",      "mean"), ("test_benchmark_p90",      "std"),
        ("test_opt_p90",            "mean"), ("test_opt_p90",            "std"),
        ("test_apriori_p90",        "mean"), ("test_apriori_p90",        "std"),
        ("model_spaced_accuracy",   "mean"), ("model_spaced_f1score",    "mean"),
        ("model_spaced_auroc",      "mean"), ("model_spaced_workload",   "mean"),
        ("model_spaced_p90",        "mean"),
        ("model_tabbed_accuracy",   "mean"), ("model_tabbed_f1score",    "mean"),
        ("model_tabbed_auroc",      "mean"), ("model_tabbed_workload",   "mean"),
        ("model_tabbed_p90",        "mean"),
    ]

    for col, agg in scalar_cols:
        if col in df.columns:
            out_col = f"{col}_{agg}"
            df[out_col] = df[col].apply(
                lambda x: np.array(x).mean() if agg == "mean"
                else np.array(x).std()
            )

    return df


# # ============================================================
# # utils.py — AdaSteer version
# # Replaces OpenAI embeddings with local trained encoders
# # Compatible with CEB and JOB data formats
# # ============================================================

# import os
# import gc
# import torch
# import sqlparse
# import warnings
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
# from typing import Optional, Union, List
# from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer

# warnings.filterwarnings('ignore', category=FutureWarning)


# # ============================================================
# # SECTION 1: DEVICE + MODEL REGISTRY
# # ============================================================

# print("🖥️  Device:", torch.cuda.get_device_name(0)
#       if torch.cuda.is_available() else "❌ CPU only")

# # ── All available encoders ───────────────────────────────────
# MODEL_REGISTRY = {
#     "contrastive" : "encoders/encoder_all-mpnet-base-v2_v1",
#     "reptile"     : "encoders/encoder_reptile_mpnet_v4",
#     "oracle"      : "adasteer_encoder",
# }

# # ── Active model state ───────────────────────────────────────
# _embed_model_path = MODEL_REGISTRY["contrastive"]
# _st_model         = None
# _hf_model         = None
# _hf_tokenizer     = None
# _device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _active_model_name = "contrastive"


# # ============================================================
# # SECTION 2: MODEL LOADING
# # ============================================================

# def _load_model(model_path: str, model_name: str = "unknown"):
#     """
#     Load embedding model from local path.
#     Tries SentenceTransformer first, falls back to AutoModel.
#     Clears previous model from GPU memory before loading new one.
#     """
#     global _st_model, _hf_model, _hf_tokenizer
#     global _embed_model_path, _active_model_name

#     # free previous model from GPU
#     if _st_model is not None:
#         del _st_model
#         _st_model = None
#     if _hf_model is not None:
#         del _hf_model
#         _hf_model = None
#     if _hf_tokenizer is not None:
#         del _hf_tokenizer
#         _hf_tokenizer = None
#     torch.cuda.empty_cache()
#     gc.collect()

#     _embed_model_path  = model_path
#     _active_model_name = model_name
#     target_device      = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"\n{'='*55}")
#     print(f"Loading encoder: [{model_name}]")
#     print(f"Path  : {model_path}")
#     print(f"Device: {target_device}")
#     print(f"{'='*55}")

#     try:
#         _st_model = SentenceTransformer(model_path, device=target_device)
#         dim = _st_model.get_sentence_embedding_dimension()
#         print(f"✅ Loaded as SentenceTransformer (dim={dim})")
#         return
#     except Exception as e:
#         print(f"⚠️  SentenceTransformer failed: {e}")
#         print(f"   Falling back to AutoModel...")

#     try:
#         if torch.cuda.is_available():
#             _hf_model = AutoModel.from_pretrained(
#                 model_path, torch_dtype=torch.float16
#             ).to("cuda")
#         else:
#             _hf_model = AutoModel.from_pretrained(model_path).to("cpu")
#         _hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
#         print(f"✅ Loaded as AutoModel (HuggingFace)")
#     except Exception as e:
#         raise RuntimeError(f"❌ Failed to load model from {model_path}: {e}")


# def set_embedding_model(model_path: str, model_name: str = None):
#     """
#     Switch the active embedding model.

#     Usage:
#         set_embedding_model("encoders/encoder_all-mpnet-base-v2_v1", "contrastive")
#         set_embedding_model("encoders/encoder_reptile_mpnet_v4",      "reptile")
#         set_embedding_model("adasteer_encoder",                        "oracle")

#     Or use registry shortcut:
#         set_embedding_model(MODEL_REGISTRY["reptile"], "reptile")
#     """
#     name = model_name or os.path.basename(model_path)
#     _load_model(model_path, name)


# def get_active_model_name() -> str:
#     """Return name of currently active encoder."""
#     return _active_model_name


# # ============================================================
# # SECTION 3: EMBEDDING FUNCTIONS
# # ============================================================

# def _ensure_model_loaded():
#     """Load default model if none is loaded yet."""
#     if _st_model is None and (_hf_model is None or _hf_tokenizer is None):
#         _load_model(_embed_model_path, _active_model_name)


# def local_embed(query: str) -> list:
#     """
#     Embed a single SQL query string.
#     Returns normalized embedding as Python list.
#     """
#     _ensure_model_loaded()

#     if _st_model is not None:
#         emb = _st_model.encode(
#             [query],
#             convert_to_numpy     = True,
#             normalize_embeddings = True,
#             show_progress_bar    = False,
#         )
#         return emb[0].tolist()

#     # AutoModel fallback
#     inputs = _hf_tokenizer(
#         query,
#         return_tensors = "pt",
#         truncation     = True,
#         padding        = True,
#         max_length     = 512,
#     ).to(_hf_model.device)

#     with torch.no_grad():
#         out            = _hf_model(**inputs)
#         hidden         = out.last_hidden_state
#         mask           = inputs["attention_mask"].unsqueeze(-1).float()
#         pooled         = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#         pooled         = F.normalize(pooled, p=2, dim=1)

#     return pooled.squeeze().cpu().tolist()


# def local_embed_batch(queries, batch_size: int = 64) -> list:
#     """
#     Embed a list of SQL queries efficiently in batches.
#     Returns list of normalized embeddings.
#     """
#     _ensure_model_loaded()
#     queries = list(queries)

#     if _st_model is not None:
#         embeddings = _st_model.encode(
#             queries,
#             batch_size           = batch_size,
#             convert_to_numpy     = True,
#             normalize_embeddings = True,
#             show_progress_bar    = False,
#         )
#         return embeddings.tolist()

#     # AutoModel fallback — manual batching
#     all_embeddings = []
#     total          = len(queries)

#     for start in range(0, total, batch_size):
#         end   = min(start + batch_size, total)
#         batch = queries[start:end]

#         inputs = _hf_tokenizer(
#             batch,
#             return_tensors = "pt",
#             truncation     = True,
#             padding        = True,
#             max_length     = 512,
#         ).to(_hf_model.device)

#         with torch.no_grad():
#             out    = _hf_model(**inputs)
#             hidden = out.last_hidden_state
#             mask   = inputs["attention_mask"].unsqueeze(-1).float()
#             pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
#             pooled = F.normalize(pooled, p=2, dim=1)

#         all_embeddings.extend(pooled.detach().cpu().tolist())

#         if end % 500 == 0 or end == total:
#             print(f"  Embedding progress: {end}/{total}")

#     return all_embeddings


# # ============================================================
# # SECTION 4: DATA LOADING
# # ============================================================

# def load_data(workload: str = "ceb") -> pd.DataFrame:
#     """
#     Load CEB or JOB workload data.

#     Args:
#         workload: "ceb", "job", or "all"

#     Returns:
#         DataFrame with columns:
#             filename, sql, hint_list, runtime_list,
#             plan_tree, mean_runtime, sd_runtime, workload_id
#     """
#     def _load_csv(path, workload_id):
#         df = pd.read_csv(
#             path,
#             converters={
#                 "hint_list"   : eval,
#                 "runtime_list": eval,
#             }
#         )
#         df["workload_id"] = workload_id
#         return df

#     ceb_path = "./data/ceb.csv"
#     job_path = "./data/job.csv"

#     if workload == "ceb":
#         data = _load_csv(ceb_path, "ceb")
#     elif workload == "job":
#         data = _load_csv(job_path, "job")
#     elif workload == "all":
#         ceb = _load_csv(ceb_path, "ceb")
#         job = _load_csv(job_path, "job")
#         data = pd.concat([ceb, job], ignore_index=True)
#     else:
#         raise ValueError(f"Unknown workload '{workload}'. Choose: ceb, job, all")

#     # compute runtime statistics
#     data["mean_runtime"] = data["runtime_list"].apply(np.mean)
#     data["sd_runtime"]   = data["runtime_list"].apply(np.std)

#     # clean SQL strings
#     data["sql"] = data["sql"].apply(lambda x: str(x).strip("\n").strip())

#     print(f"\nLoaded [{workload}] workload:")
#     print(f"  Rows          : {len(data):,}")
#     print(f"  Unique queries: {data['filename'].nunique():,}")
#     print(f"  Hint sets/query: {data.groupby('filename').size().mean():.1f} avg")
#     print(f"  Runtime range : "
#           f"{data['mean_runtime'].min():.3f}s – {data['mean_runtime'].max():.3f}s")

#     return data


# # ============================================================
# # SECTION 5: EMBEDDING GENERATION
# # ============================================================

# def generate_embeddings(
#     data    : pd.DataFrame,
#     filepath: str,
#     filename: str,
#     dims    : Optional[int] = None,
#     augment : bool = False,
# ) -> pd.DataFrame:
#     """
#     Generate SQL embeddings for all unique queries in data.

#     If augment=True:
#         Also generates Syntax B (spaced) and Syntax C (tabbed)
#         embeddings for robustness testing.

#     Returns:
#         data with 'features' column added (embedding per row)
#         If augment=True: also returns syntaxB and syntaxC tensors
#     """
#     subset     = data[["filename", "sql"]].drop_duplicates()
#     filenames  = subset["filename"].tolist()
#     sql_texts  = subset["sql"].tolist()

#     print(f"\n  Generating embeddings [{_active_model_name}]...")
#     print(f"  Unique queries: {len(sql_texts)}")

#     embedded   = local_embed_batch(sql_texts, batch_size=64)
#     emb_map    = {fname: emb for fname, emb in zip(filenames, embedded)}
#     data       = data.copy()
#     data["features"] = data["filename"].apply(lambda x: emb_map[x])

#     if not augment:
#         return data

#     # Syntax B — reindented with spaces
#     print(f"  Generating Syntax B (spaced)...")
#     syntaxB_sql = [
#         sqlparse.format(
#             sql,
#             reindent                  = True,
#             use_space_around_operators= True,
#             indent_tabs               = False
#         )
#         for sql in sql_texts
#     ]
#     syntaxB_embs = local_embed_batch(syntaxB_sql, batch_size=64)

#     # Syntax C — reindented with tabs
#     print(f"  Generating Syntax C (tabbed)...")
#     syntaxC_sql = [
#         sqlparse.format(
#             sql,
#             reindent                  = True,
#             use_space_around_operators= True,
#             indent_tabs               = True
#         )
#         for sql in sql_texts
#     ]
#     syntaxC_embs = local_embed_batch(syntaxC_sql, batch_size=64)

#     return (
#         data,
#         torch.tensor(syntaxB_embs, dtype=torch.float32),
#         torch.tensor(syntaxC_embs, dtype=torch.float32),
#     )


# # ============================================================
# # SECTION 6: DATA PREPARATION
# # ============================================================

# def prepare_data(
#     data        : pd.DataFrame,
#     filepath    : str,
#     filename    : str,
#     binary_model: bool,
#     dims        : Optional[int] = None,
#     augment     : bool = False,
#     return_meta : bool = False,
# ) -> tuple:
#     """
#     Prepare data tensors for classifier training.

#     Pipeline:
#         1. Aggregate hint runtimes per unique SQL query
#         2. Generate SQL embeddings
#         3. Build X (features), hint_l (runtimes), opt_l (optimal),
#            binary_l (labels)

#     binary_model=True  → binary classification
#         y=1 if hint[0] (PostgreSQL default) > hint[26] (longtail)
#         Matches original LLMSteer paper setup

#     binary_model=False → multiclass
#         y = one-hot of optimal hint index per query

#     augment=True → also returns Syntax B and C embeddings
#                    for robustness evaluation

#     return_meta=True → also returns metadata DataFrame
#                        with filename and sql columns
#     """
#     model_df = data.copy()

#     # drop unused columns
#     drop_cols = [c for c in ["runtime_list", "plan_tree", "sd_runtime"]
#                  if c in model_df.columns]
#     model_df  = model_df.drop(columns=drop_cols)

#     # explode hint_list so each row = one hint index
#     model_df  = model_df.explode(column="hint_list")
#     model_df  = model_df.sort_values(by=["filename", "hint_list"])

#     # aggregate back: one row per unique query
#     model_df  = model_df.groupby(
#         ["filename", "sql"], as_index=False
#     ).agg({
#         "hint_list"   : lambda x: x.tolist(),
#         "mean_runtime": lambda x: x.tolist(),
#     })
#     model_df["opt_l"] = model_df["mean_runtime"].apply(min)

#     print(f"\nPreparing data:")
#     print(f"  Unique queries : {len(model_df):,}")
#     print(f"  Encoder        : [{_active_model_name}]")

#     # generate embeddings
#     if augment:
#         model_df, syntaxB, syntaxC = generate_embeddings(
#             model_df, filepath, filename, dims, augment=True
#         )
#     else:
#         model_df = generate_embeddings(
#             model_df, filepath, filename, dims, augment=False
#         )

#     # save metadata before dropping
#     meta_df  = model_df[["filename", "sql"]].copy()
#     model_df = model_df.drop(columns=["filename", "sql", "hint_list"])

#     # build tensors
#     X      = torch.stack(
#         model_df["features"].apply(torch.Tensor).tolist()
#     )
#     hint_l = torch.stack(
#         model_df["mean_runtime"].apply(torch.Tensor).tolist()
#     )
#     opt_l  = torch.stack(
#         model_df["opt_l"].apply(
#             lambda x: torch.Tensor([x]).repeat(hint_l.size(1))
#         ).tolist()
#     )

#     # build labels
#     if binary_model:
#         BENCHMARK_IDX = 0   # hint index 0 = PostgreSQL default (no hint)
#         LONGTAIL_IDX  = 26  # hint index 26 = longtail hint
#         binary_l = (
#             hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]
#         ).float()
#     else:
#         binary_l = (hint_l == opt_l).float()

#     print(f"  Feature shape  : {X.shape}")
#     print(f"  Hint matrix    : {hint_l.shape}")
#     print(f"  Label balance  : {binary_l.mean():.3f} positive rate")

#     if augment:
#         if return_meta:
#             return X, hint_l, opt_l, binary_l, syntaxB, syntaxC, meta_df
#         return X, hint_l, opt_l, binary_l, syntaxB, syntaxC
#     else:
#         if return_meta:
#             return X, hint_l, opt_l, binary_l, meta_df
#         return X, hint_l, opt_l, binary_l
# import gc
# import json
# import os
# import torch
# import sqlparse
# import argparse
# import warnings
# import tiktoken
# import numpy as np
# import pandas as pd
# import torch.nn as nn
# from os import cpu_count
# from copy import deepcopy
# from os.path import exists
# from datetime import datetime
# from dotenv import dotenv_values
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from openai import OpenAI, NotGiven, NOT_GIVEN
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score
# from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
# from sentence_transformers import SentenceTransformer
# import torch
# import torch.nn.functional as F
# import pandas as pd
# from .multiclass import MULTICLASS_RUN_MODES, ERLoss
# from .binary import BINARY_RUN_MODES
# from typing import Optional
# from typing import Union
# from typing import List, Dict

# from transformers import AutoModel, AutoTokenizer
# warnings.filterwarnings('ignore', category=FutureWarning)


# #############Finetuned model loading#############
# print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ CPU only")

# DEFAULT_EMBED_MODEL_PATH = "encoders/encoder_all-mpnet-base-v2_v1"  # ← use v4!
# _embed_model_path = os.environ.get("EMBED_MODEL_PATH", DEFAULT_EMBED_MODEL_PATH)
# model = None
# tokenizer = None
# st_model = None

# # Set device (for tensors and input handling only)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def _load_local_embedding_model(model_path: str):
#     global model, tokenizer, st_model, _embed_model_path
#     _embed_model_path = model_path
#     model = None
#     tokenizer = None
#     st_model = None

#     target_device = "cuda" if torch.cuda.is_available() else "cpu"
#     try:
#         print(f"🔎 Trying SentenceTransformer loader: {model_path} on {target_device}")
#         st_model = SentenceTransformer(model_path, device=target_device)
#         print("✅ Loaded as SentenceTransformer (encode + normalize).")
#         return
#     except Exception as e:
#         print(f"⚠️ SentenceTransformer load failed; falling back to AutoModel. Reason: {e}")

#     if torch.cuda.is_available():
#         print(f"🚀 GPU detected — loading AutoModel on GPU: {model_path}")
#         model = AutoModel.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#         )
#         model = model.to("cuda")
#     else:
#         print(f"⚠️ No GPU detected — loading AutoModel on CPU: {model_path}")
#         model = AutoModel.from_pretrained(model_path)
#         model = model.to("cpu")

#     tokenizer = AutoTokenizer.from_pretrained(model_path)


# def set_embedding_model(model_path: str):
#     _load_local_embedding_model(model_path)


# # Helper to embed SQL query using your quantized model
# def local_embed(query):
#     if st_model is None and (model is None or tokenizer is None):
#         _load_local_embedding_model(_embed_model_path)

#     if st_model is not None:
#         emb = st_model.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         )
#         return emb[0].tolist()

#     inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         hidden_states = outputs.last_hidden_state
#         attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
#         pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
#         pooled = F.normalize(pooled, p=2, dim=1)
#     return pooled.squeeze().tolist()


# def local_embed_batch(queries, batch_size: int = 64):
#     if st_model is None and (model is None or tokenizer is None):
#         _load_local_embedding_model(_embed_model_path)

#     if st_model is not None:
#         embeddings = st_model.encode(
#             list(queries),
#             batch_size=batch_size,
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         )
#         return embeddings.tolist()

#     all_embeddings = []
#     total = len(queries)
#     for start in range(0, total, batch_size):
#         end = min(start + batch_size, total)
#         batch = queries[start:end]
#         inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(model.device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             hidden_states = outputs.last_hidden_state
#             attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
#             pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
#             pooled = F.normalize(pooled, p=2, dim=1)
#         all_embeddings.extend(pooled.detach().cpu().tolist())
#         if end % 500 == 0 or end == total:
#             print(f"Embedding progress: {end}/{total}")

#     return all_embeddings


# # def prepare_data(data: pd.DataFrame, filepath: str, filename: str, binary_model: bool, dims: int|NotGiven = NOT_GIVEN, augment: bool = False) -> tuple:
# def prepare_data(
#     data: pd.DataFrame,
#     filepath: str,
#     filename: str,
#     binary_model: bool,
#     dims: Union[int, NotGiven] = NOT_GIVEN,
#     augment: bool = False,
#     return_meta: bool = False,
# ) -> tuple:
    
#     model_df: pd.DataFrame = data.copy()

#     model_df = model_df.drop(columns=['runtime_list', 'plan_tree', 'sd_runtime'])
#     model_df = model_df.explode(column='hint_list')
#     model_df = model_df.sort_values(by=['filename', 'hint_list'])
#     model_df = model_df.groupby(by=['filename', 'sql'], as_index=False).agg({
#         'hint_list': lambda x: x.tolist(), 
#         'mean_runtime': lambda x: x.tolist()
#     })
#     model_df['opt_l'] = model_df.mean_runtime.apply(min)
    
#     if augment:
#         model_df, syntaxB, syntaxC = generate_embeddings(model_df, filepath, filename, dims, augment)
#     else:
#         model_df = generate_embeddings(model_df, filepath, filename, dims, augment)
#     meta_df = model_df[["filename", "sql"]].copy()
#     model_df = model_df.drop(columns=['filename', 'sql', 'hint_list'])

#     X = torch.stack(model_df.features.apply(lambda x: torch.Tensor(x)).tolist())
#     hint_l = torch.stack(model_df.mean_runtime.apply(lambda x: torch.Tensor(x)).tolist())
#     opt_l = torch.stack(model_df.opt_l.apply(lambda x: torch.Tensor([x]).repeat(hint_l.size(1))).tolist())
#     if binary_model:
#         BENCHMARK_IDX = 0
#         LONGTAIL_IDX = 26
#         binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).type(torch.float)
#     else:
#         binary_l = (hint_l == opt_l).type(torch.float)
    
#     if augment:
#         if return_meta:
#             return X, hint_l, opt_l, binary_l, syntaxB, syntaxC, meta_df
#         return X, hint_l, opt_l, binary_l, syntaxB, syntaxC
#     else:
#         if return_meta:
#             return X, hint_l, opt_l, binary_l, meta_df
#         return X, hint_l, opt_l, binary_l

# def generate_embeddings(data: pd.DataFrame, filepath: str, filename: str, dims: Optional[int] = None, augment: bool = False) -> pd.DataFrame:
#     subset_df = data[['filename','sql']].drop_duplicates()
#     embeddings = {}

#     filenames = subset_df["filename"].tolist()
#     sql_texts = subset_df["sql"].tolist()
#     embedded = local_embed_batch(sql_texts, batch_size=64)

#     for fname, emb in zip(filenames, embedded):
#         embeddings[fname] = {'embedding': emb}

#     data['features'] = data.filename.apply(lambda x: embeddings[x]['embedding'])

#     if not augment:
#         return data
#     else:
#         syntaxB_sql = subset_df.sql.apply(
#             lambda x: sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=False)
#         ).tolist()
#         syntaxC_sql = subset_df.sql.apply(
#             lambda x: sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=True)
#         ).tolist()
#         syntaxB_queries = local_embed_batch(syntaxB_sql, batch_size=64)
#         syntaxC_queries = local_embed_batch(syntaxC_sql, batch_size=64)

#         return data, torch.tensor(syntaxB_queries), torch.tensor(syntaxC_queries)


# def load_data(workload: str = "ceb") -> pd.DataFrame:

#     job_df = pd.read_csv(
#         './data/job.csv',
#         converters={
#             'hint_list': eval,
#             'runtime_list': eval,
#         }
#     )
#     job_df["workload_id"] = "job"

#     ceb_df = pd.read_csv(
#         './data/ceb.csv',
#         converters={
#             'hint_list': eval,
#             'runtime_list': eval,
#         }
#     )
#     ceb_df["workload_id"] = "ceb"

#     if workload == "job":
#         data = job_df
#     elif workload == "ceb":
#         data = ceb_df
#     elif workload == "all":
#         data = pd.concat([job_df, ceb_df], ignore_index=True)
#     else:
#         raise ValueError(f"Unknown workload: {workload}")
    
#     data["mean_runtime"] = data.runtime_list.apply(lambda x: np.mean(x)) # compute mean runtime for each query plan
#     data["sd_runtime"] = data.runtime_list.apply(lambda x: np.std(x)) # compute sd runtime for each query plan
#     data["sql"] = data.sql.apply(lambda x: x.strip('\n'))
#     return data
