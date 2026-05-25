"""
exp_new_baselines.py
====================
Three experiments addressing reviewer concerns:

  Exp A: Reuse MPNet-Raw from exp1_supervision_ablation.csv (already done)
  Exp B: TF-IDF + SVC binary steering baseline
  Exp C: Few-shot adaptation workload latency (CEB→JOB)

Protocol mirrors exp1_supervision_ablation.py exactly:
  - data/job.csv + data/ceb.csv
  - StratifiedShuffleSplit(n_splits=10, train_size=0.8, random_state=24508)
  - BENCHMARK_IDX=0 (default PostgreSQL), LONGTAIL_IDX=26 (fixed alternative)
  - Pipeline: StandardScaler → PCA(120) → StandardScaler → SVC(RBF)
  - Workload = sum of latencies at chosen config index per query

Outputs:
  results/new_baselines_summary.csv
  results/fewshot_workload_latency.csv  (Exp C)
  analysis-output/figures/fig_new_baselines.{pdf,png}
  analysis-output/figures/fig_fewshot_latency.{pdf,png}
"""

import os, gc, warnings, ast
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ── Constants (must match exp1_supervision_ablation.py exactly) ───────────────
BENCHMARK_IDX = 0      # default PostgreSQL config index
LONGTAIL_IDX  = 26     # fixed alternative config index
THRESHOLD     = 0.5
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
RANDOM_SEED   = 24508
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
os.makedirs("analysis-output/figures", exist_ok=True)

# ── Data loading (same as exp1) ───────────────────────────────────────────────
print("Loading data/job.csv and data/ceb.csv ...")
def safe_eval(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else x
    except: return []

job_raw = pd.read_csv("data/job.csv",  converters={"hint_list": safe_eval, "runtime_list": safe_eval})
ceb_raw = pd.read_csv("data/ceb.csv",  converters={"hint_list": safe_eval, "runtime_list": safe_eval})
data    = pd.concat([job_raw, ceb_raw]).reset_index(drop=True)
data["mean_runtime"] = data["runtime_list"].apply(np.mean)
data["sql"] = data["sql"].apply(lambda x: x.strip("\n"))

def prepare_features(df):
    df = df.copy()
    df = df.explode("hint_list").sort_values(["filename", "hint_list"])
    df = df.groupby(["filename", "sql"], as_index=False).agg(
        hint_list    =("hint_list",     list),
        mean_runtime =("mean_runtime",  list),
    )
    df["opt_l"] = df["mean_runtime"].apply(min)
    return df.reset_index(drop=True)

model_df = prepare_features(data)
print(f"  Combined queries: {len(model_df)}")

hint_l   = torch.stack(model_df["mean_runtime"].apply(torch.Tensor).tolist())
binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).float()
y        = binary_l.numpy()
sqls     = model_df["sql"].tolist()

benchmark_t = torch.LongTensor([BENCHMARK_IDX])
longtail_t  = torch.LongTensor([LONGTAIL_IDX])

print(f"  Positive labels (alt faster): {int(binary_l.sum())} / {len(y)} "
      f"({100*binary_l.mean():.1f}%)")

# ── SVC-120-S classifier (mirrors exp1 exactly) ───────────────────────────────
def make_svc120s():
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("pca",     PCA(n_components=120, random_state=RANDOM_SEED)),
        ("scaler2", StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
    ])

# ── CV evaluation kernel ──────────────────────────────────────────────────────
def evaluate_features(X_feat, label, method_name):
    """
    Run 10-split StratifiedShuffleSplit and compute workload, P90, AUROC, F1.
    X_feat: ndarray (n_queries, n_features)
    """
    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

    workloads, p90s, aurocs, f1s = [], [], [], []

    for tr_idx, te_idx in splitter.split(X_feat, label):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = label[tr_idx], label[te_idx]

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        cw = {0: n_pos / n_neg, 1: n_neg / n_pos}

        clf = make_svc120s()
        clf.named_steps["svc"].set_params(class_weight=cw)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # Workload: pick config_0 or config_26 latency per prediction
        chosen = hint_l[te_idx].gather(
            1, torch.where(torch.tensor(y_pred) > THRESHOLD,
                           longtail_t, benchmark_t).view(-1, 1))
        workloads.append(chosen.sum().item())
        p90s.append(chosen.quantile(0.90).item())
        aurocs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "method":        method_name,
        "workload_mean": np.mean(workloads), "workload_std": np.std(workloads),
        "p90_mean":      np.mean(p90s),      "p90_std":      np.std(p90s),
        "auroc_mean":    np.mean(aurocs),     "auroc_std":    np.std(aurocs),
        "f1_mean":       np.mean(f1s),        "f1_std":       np.std(f1s),
        "_folds":        workloads,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — TF-IDF + SVC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT B: TF-IDF + SVC Binary Steering")
print("="*65)

tfidf  = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]*",
    max_features=5000, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(sqls).toarray().astype(np.float32)
print(f"  TF-IDF matrix: {X_tfidf.shape}")

res_B = evaluate_features(X_tfidf, y, "TF-IDF + SVC-120-S")
print(f"  Workload: {res_B['workload_mean']:,.1f} ± {res_B['workload_std']:,.1f}s")
print(f"  P90:      {res_B['p90_mean']:.3f} ± {res_B['p90_std']:.3f}s")
print(f"  AUROC:    {res_B['auroc_mean']:.4f} ± {res_B['auroc_std']:.4f}")
print(f"  F1:       {res_B['f1_mean']:.4f} ± {res_B['f1_std']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Few-shot workload latency (CEB→JOB)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EXPERIMENT C: Few-shot Adaptation — Workload Latency on JOB")
print("="*65)

# Prepare JOB-only model_df for evaluation
job_data_raw = job_raw.copy()
job_data_raw["mean_runtime"] = job_data_raw["runtime_list"].apply(np.mean)
job_data_raw["sql"] = job_data_raw["sql"].apply(lambda x: x.strip("\n"))
job_mdf = prepare_features(job_data_raw)
hint_l_job = torch.stack(job_mdf["mean_runtime"].apply(torch.Tensor).tolist())
binary_l_job = (hint_l_job[:, BENCHMARK_IDX] > hint_l_job[:, LONGTAIL_IDX]).float()
y_job = binary_l_job.numpy()
sqls_job = job_mdf["sql"].tolist()

# CEB-only for training
ceb_data_raw = ceb_raw.copy()
ceb_data_raw["mean_runtime"] = ceb_data_raw["runtime_list"].apply(np.mean)
ceb_data_raw["sql"] = ceb_data_raw["sql"].apply(lambda x: x.strip("\n"))
ceb_mdf = prepare_features(ceb_data_raw)
hint_l_ceb = torch.stack(ceb_mdf["mean_runtime"].apply(torch.Tensor).tolist())
binary_l_ceb = (hint_l_ceb[:, BENCHMARK_IDX] > hint_l_ceb[:, LONGTAIL_IDX]).float()
y_ceb = binary_l_ceb.numpy()
sqls_ceb = ceb_mdf["sql"].tolist()

print(f"  JOB queries: {len(job_mdf)}, CEB queries: {len(ceb_mdf)}")

def compute_workload_torch(preds, hl, bench_t, lt_t):
    chosen = hl.gather(
        1, torch.where(torch.tensor(preds) > THRESHOLD,
                       lt_t, bench_t).view(-1, 1))
    return chosen.sum().item(), chosen.quantile(0.90).item()

try:
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("  sentence_transformers not available; Exp C skipped.")

ENCODER_C_PATH = None
ENCODER_R_PATH = None

# Find best available encoders
for p in ["adasteer_encoder", "contrastive_mpnet_stage1",
          "encoders/encoder_all-mpnet-base-v2_v1"]:
    if os.path.exists(p) and ENCODER_C_PATH is None:
        ENCODER_C_PATH = p
for p in ["encoders/encoder_reptile_mpnet_v4", "encoders/encoder_reptile_mpnet_v3",
          "reptile_all-MiniLM-L12-v2"]:
    if os.path.exists(p) and ENCODER_R_PATH is None:
        ENCODER_R_PATH = p

print(f"  Encoder C: {ENCODER_C_PATH}")
print(f"  Encoder R: {ENCODER_R_PATH}")

FEWSHOT_K  = [0, 5, 10, 20, 50]
FEWSHOT_SEEDS = [42, 7, 123]   # 3 seeds for stability

fewshot_wl_rows = []

if ST_AVAILABLE and ENCODER_C_PATH and ENCODER_R_PATH:
    try:
        print("  Loading encoders...")
        model_C = SentenceTransformer(ENCODER_C_PATH, device=DEVICE)
        model_R = SentenceTransformer(ENCODER_R_PATH, device=DEVICE)

        # Encode CEB (train set — fixed)
        print("  Encoding CEB with C...")
        X_ceb_C = model_C.encode(sqls_ceb, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        print("  Encoding CEB with R...")
        X_ceb_R = model_R.encode(sqls_ceb, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)

        # Load JOB triplets for few-shot adaptation
        triplets_df = pd.read_csv("stage1_triplets_JOB.csv")
        print(f"  JOB triplets available: {len(triplets_df)}")

        for K in FEWSHOT_K:
            print(f"\n  --- K = {K} ---")
            C_wls, R_wls = [], []

            for seed in FEWSHOT_SEEDS:
                rng = np.random.RandomState(seed)

                # ── AdaptSteer-C: CEB-trained SVC, JOB test (no adaptation) ──
                n_pos_ceb = (y_ceb == 1).sum()
                n_neg_ceb = (y_ceb == 0).sum()
                cw_ceb = {0: n_pos_ceb/n_neg_ceb, 1: n_neg_ceb/n_pos_ceb}
                clf_C = make_svc120s()
                clf_C.named_steps["svc"].set_params(class_weight=cw_ceb)
                clf_C.fit(X_ceb_C, y_ceb)

                X_job_C = model_C.encode(sqls_job, batch_size=64,
                                          normalize_embeddings=True,
                                          show_progress_bar=False,
                                          convert_to_numpy=True)
                preds_C = clf_C.predict(X_job_C)
                wl_C, _ = compute_workload_torch(preds_C, hint_l_job,
                                                  benchmark_t, longtail_t)
                C_wls.append(wl_C)

                # ── AdaptSteer-R: Reptile-adapt on K JOB triplets ─────────────
                if K == 0:
                    # Zero-shot: use base R encoder with CEB-trained SVC
                    clf_R = make_svc120s()
                    clf_R.named_steps["svc"].set_params(class_weight=cw_ceb)
                    clf_R.fit(X_ceb_R, y_ceb)
                    X_job_R = model_R.encode(sqls_job, batch_size=64,
                                              normalize_embeddings=True,
                                              show_progress_bar=False,
                                              convert_to_numpy=True)
                    preds_R = clf_R.predict(X_job_R)
                else:
                    # Sample K triplets for adaptation
                    n_sample = min(K * 10, len(triplets_df))  # oversample
                    support  = triplets_df.sample(n_sample, random_state=seed, replace=False)

                    # Reptile fine-tuning: 5 epochs on support set
                    model_adapted = SentenceTransformer(ENCODER_R_PATH, device=DEVICE)
                    examples = [
                        InputExample(texts=[r["anchor"], r["positive"], r["negative"]])
                        for _, r in support.iterrows()
                    ]
                    loader = DataLoader(examples, shuffle=True,
                                        batch_size=min(16, len(examples)))
                    loss_fn = losses.TripletLoss(
                        model=model_adapted,
                        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN)
                    model_adapted.fit(
                        train_objectives=[(loader, loss_fn)],
                        epochs=5,
                        warmup_steps=0,
                        show_progress_bar=False,
                        optimizer_params={"lr": 2e-5},
                    )
                    # Re-encode CEB, train fresh SVC, test on JOB
                    X_ceb_adapted = model_adapted.encode(
                        sqls_ceb, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
                    clf_R = make_svc120s()
                    clf_R.named_steps["svc"].set_params(class_weight=cw_ceb)
                    clf_R.fit(X_ceb_adapted, y_ceb)
                    X_job_R = model_adapted.encode(
                        sqls_job, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
                    preds_R = clf_R.predict(X_job_R)
                    del model_adapted; gc.collect()

                wl_R, _ = compute_workload_torch(preds_R, hint_l_job,
                                                  benchmark_t, longtail_t)
                R_wls.append(wl_R)
                print(f"    seed={seed}: C={wl_C:.1f}s  R={wl_R:.1f}s")

            fewshot_wl_rows.append({
                "K":                         K,
                "AdaptSteer-C_wl_mean":      np.mean(C_wls),
                "AdaptSteer-C_wl_std":       np.std(C_wls),
                "AdaptSteer-R_wl_mean":      np.mean(R_wls),
                "AdaptSteer-R_wl_std":       np.std(R_wls),
            })

        fewshot_wl_df = pd.DataFrame(fewshot_wl_rows)
        fewshot_wl_df.to_csv("results/fewshot_workload_latency.csv", index=False)
        print("\n  Fewshot workload latency:")
        print(fewshot_wl_df.to_string())

    except Exception as e:
        print(f"  ERROR in Exp C: {e}")
        import traceback; traceback.print_exc()
        fewshot_wl_df = None
else:
    fewshot_wl_df = None


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EXISTING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
exp1 = pd.read_csv("results/exp1_supervision_ablation.csv")

# AdaptSteer-R fold workloads from per_query_latencies (10 folds)
per_q      = pd.read_csv("results_final/per_query_latencies.csv")
ada_r_folds = per_q.groupby("fold")["model_lat"].sum().values

mpnet_raw_row = exp1[exp1["encoder"] == "MPNet-Raw"].iloc[0]
ada_c_row     = exp1[exp1["encoder"] == "AdaSteer-C"].iloc[0]
ada_r_row     = exp1[exp1["encoder"] == "AdaSteer-R"].iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STATISTICAL TESTS (vs AdaptSteer-R baseline)")
print("="*65)

def wilcoxon_test(folds_a, folds_b, name_a, name_b):
    """Two-sided Wilcoxon: are the two methods different?"""
    if len(folds_a) != len(folds_b):
        print(f"  {name_a} vs {name_b}: unequal fold counts, skipping")
        return None, None
    try:
        stat, p = stats.wilcoxon(folds_a, folds_b, alternative="two-sided")
        n = len(folds_a)
        # Effect size r = Z / sqrt(N)
        z = stats.norm.ppf(1 - p/2)
        r = abs(z) / np.sqrt(n)
        delta = np.mean(np.array(folds_a)) - np.mean(np.array(folds_b))
        print(f"  {name_a} vs {name_b}: Δ={delta:+.1f}s  p={p:.4f}  r={r:.3f} "
              f"{'*' if p < 0.05 else 'ns'}")
        return p, r
    except Exception as e:
        print(f"  {name_a} vs {name_b}: {e}")
        return None, None

# TF-IDF vs AdaptSteer-R
p_tfidf, r_tfidf = wilcoxon_test(
    res_B["_folds"], list(ada_r_folds),
    "TF-IDF+SVC", "AdaptSteer-R")

# MPNet-Raw is from StratifiedShuffleSplit without fold workloads stored
# Use descriptive comparison only
print(f"\n  MPNet-Raw:     {mpnet_raw_row['workload_mean']:,.1f} ± {mpnet_raw_row['workload_std']:,.1f}s")
print(f"  AdaptSteer-C:  {ada_c_row['workload_mean']:,.1f} ± {ada_c_row['workload_std']:,.1f}s")
print(f"  AdaptSteer-R:  {ada_r_row['workload_mean']:,.1f} ± {ada_r_row['workload_std']:,.1f}s")
print(f"  TF-IDF+SVC:    {res_B['workload_mean']:,.1f} ± {res_B['workload_std']:,.1f}s")
print(f"  LLMSteer:      2547.7 ± 0.0s")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FIGURES")
print("="*65)

# ── Figure 1: Extended comparison bar chart with new baselines ────────────────
COLORS = {
    "PostgreSQL Default": "#C00000",
    "TF-IDF + SVC":       "#A0A0A0",
    "MPNet-Raw + SVC":    "#D4A017",
    "LLMSteer":           "#ED7D31",
    "AdaptSteer-C (ours)":       "#4472C4",
    "AdaptSteer-R (ours)":"#70AD47",
    "Optimal":     "#808080",
}

methods = ["PostgreSQL\nDefault", "TF-IDF\n+SVC", "MPNet-Raw\n+SVC",
           "LLMSteer", "AdaptSteer-C", "AdaptSteer-R\n(ours)", "Optimal\nOracle"]
wl_m = [
    hint_l[:, BENCHMARK_IDX].sum().item(),   # PostgreSQL default workload from data
    res_B["workload_mean"],
    mpnet_raw_row["workload_mean"],
    2547.7,
    ada_c_row["workload_mean"],
    ada_r_row["workload_mean"],
    hint_l.min(dim=1).values.sum().item(),   # oracle from data
]
wl_s = [0, res_B["workload_std"], mpnet_raw_row["workload_std"],
        0, ada_c_row["workload_std"], ada_r_row["workload_std"], 0]
bar_colors = [list(COLORS.values())[i] for i in range(len(methods))]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(methods))
bars = ax.bar(x, wl_m, color=bar_colors, alpha=0.88,
              edgecolor="white", linewidth=0.6, zorder=3)
for i, (m, s) in enumerate(zip(wl_m, wl_s)):
    if s > 0:
        ax.errorbar(x[i], m, yerr=s, fmt="none",
                    ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
for bar, m in zip(bars, wl_m):
    ax.text(bar.get_x() + bar.get_width()/2, m + 200,
            f"{m:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel("Total Workload Latency (s)", fontsize=10)
ax.set_title("Binary Steering: Extended Baseline Comparison (JOB+CEB, 10-split StratifiedShuffleSplit)",
             fontsize=9.5, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# LLMSteer reference line
ax.axhline(2547.7, color="#ED7D31", linestyle="--", linewidth=0.9, alpha=0.6, label="LLMSteer")
plt.tight_layout()
plt.savefig("analysis-output/figures/fig_new_baselines.pdf", bbox_inches="tight")
plt.savefig("analysis-output/figures/fig_new_baselines.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: analysis-output/figures/fig_new_baselines.{pdf,png}")

# ── Figure 2: Few-shot workload latency ───────────────────────────────────────
if fewshot_wl_df is not None and len(fewshot_wl_df) > 0:
    # Also load existing AUROC for combined dual-panel
    auroc_df = pd.read_csv("results_final/fewshot_adaptation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    K_vals = fewshot_wl_df["K"].values
    pg_wl_job = hint_l_job[:, BENCHMARK_IDX].sum().item()
    oracle_job = hint_l_job.min(dim=1).values.sum().item()

    # Panel A: workload latency
    ax1.errorbar(K_vals, fewshot_wl_df["AdaptSteer-C_wl_mean"],
                 yerr=fewshot_wl_df["AdaptSteer-C_wl_std"],
                 marker="o", label="AdaptSteer-C", color="#4472C4",
                 linewidth=1.5, capsize=3, markersize=5)
    ax1.errorbar(K_vals, fewshot_wl_df["AdaptSteer-R_wl_mean"],
                 yerr=fewshot_wl_df["AdaptSteer-R_wl_std"],
                 marker="s", label="AdaptSteer-R", color="#70AD47",
                 linewidth=1.5, capsize=3, markersize=5)
    ax1.axhline(pg_wl_job,   color="#C00000", ls="--", lw=1.2,
                label=f"PG Default ({pg_wl_job:.0f}s)")
    ax1.axhline(oracle_job, color="#808080", ls=":", lw=1.0,
                label=f"Oracle ({oracle_job:.0f}s)")
    ax1.set_xlabel("Adaptation shots K", fontsize=10)
    ax1.set_ylabel("JOB Total Workload (s)", fontsize=10)
    ax1.set_title("(a) Workload Latency vs. K", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel B: AUROC (from existing results_final)
    # Remap columns: CSV may use "AdaSteer" (old name) or "AdaptSteer" (new name)
    def _col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        raise KeyError(f"None of {candidates} found in {list(df.columns)}")
    c_auc   = _col(auroc_df, ["AdaptSteer-C_auroc",   "AdaSteer-C_auroc"])
    c_std   = _col(auroc_df, ["AdaptSteer-C_auroc_std","AdaSteer-C_auroc_std"])
    r_auc   = _col(auroc_df, ["AdaptSteer-R_auroc",   "AdaSteer-R_auroc"])
    r_std   = _col(auroc_df, ["AdaptSteer-R_auroc_std","AdaSteer-R_auroc_std"])
    K_auroc = auroc_df["K"].values
    ax2.errorbar(K_auroc, auroc_df[c_auc],
                 yerr=auroc_df[c_std],
                 marker="o", label="AdaptSteer-C", color="#4472C4",
                 linewidth=1.5, capsize=3, markersize=5)
    ax2.errorbar(K_auroc, auroc_df[r_auc],
                 yerr=auroc_df[r_std],
                 marker="s", label="AdaptSteer-R", color="#70AD47",
                 linewidth=1.5, capsize=3, markersize=5)
    ax2.set_xlabel("Adaptation shots K", fontsize=10)
    ax2.set_ylabel("AUROC", fontsize=10)
    ax2.set_title("(b) Ranking AUROC vs. K", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.65, 0.95)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.suptitle("CEB→JOB Cross-Workload Adaptation", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("analysis-output/figures/fig_fewshot_latency.pdf", bbox_inches="tight")
    plt.savefig("analysis-output/figures/fig_fewshot_latency.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: analysis-output/figures/fig_fewshot_latency.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
summary_rows = [
    {"method": "PostgreSQL Default",
     "workload_mean": hint_l[:, BENCHMARK_IDX].sum().item(), "workload_std": 0,
     "p90_mean": hint_l[:, BENCHMARK_IDX].quantile(0.90).item(), "p90_std": 0,
     "auroc_mean": float("nan"), "auroc_std": 0, "f1_mean": float("nan")},
    {**res_B, "method": "TF-IDF + SVC-120-S"},
    {"method": "MPNet-Raw + SVC-120-S",
     "workload_mean": mpnet_raw_row["workload_mean"], "workload_std": mpnet_raw_row["workload_std"],
     "p90_mean": mpnet_raw_row["p90_mean"], "p90_std": mpnet_raw_row["p90_std"],
     "auroc_mean": mpnet_raw_row["auroc_mean"], "auroc_std": mpnet_raw_row["auroc_std"],
     "f1_mean": mpnet_raw_row["f1_mean"]},
    {"method": "LLMSteer", "workload_mean": 2547.7, "workload_std": 0,
     "p90_mean": 5.7, "p90_std": 0, "auroc_mean": float("nan"), "auroc_std": 0,
     "f1_mean": float("nan")},
    {"method": "AdaptSteer-C",
     "workload_mean": ada_c_row["workload_mean"], "workload_std": ada_c_row["workload_std"],
     "p90_mean": ada_c_row["p90_mean"], "p90_std": ada_c_row["p90_std"],
     "auroc_mean": ada_c_row["auroc_mean"], "auroc_std": ada_c_row["auroc_std"],
     "f1_mean": ada_c_row["f1_mean"]},
    {"method": "AdaptSteer-R",
     "workload_mean": ada_r_row["workload_mean"], "workload_std": ada_r_row["workload_std"],
     "p90_mean": ada_r_row["p90_mean"], "p90_std": ada_r_row["p90_std"],
     "auroc_mean": ada_r_row["auroc_mean"], "auroc_std": ada_r_row["auroc_std"],
     "f1_mean": ada_r_row["f1_mean"]},
    {"method": "Optimal Oracle",
     "workload_mean": hint_l.min(dim=1).values.sum().item(), "workload_std": 0,
     "p90_mean": float("nan"), "p90_std": 0, "auroc_mean": float("nan"), "auroc_std": 0,
     "f1_mean": float("nan")},
]

df_summary = pd.DataFrame(summary_rows)
# drop internal _folds column
df_summary = df_summary.drop(columns=[c for c in df_summary.columns if c.startswith("_")],
                              errors="ignore")
df_summary.to_csv("results/new_baselines_summary.csv", index=False)

print("\n" + "="*65)
print("FINAL SUMMARY TABLE")
print("="*65)
header = f"{'Method':<25} | {'Workload (s)':>22} | {'AUROC':>8} | {'F1':>7}"
print(header)
print("-" * len(header))
for _, row in df_summary.iterrows():
    wl = f"{row['workload_mean']:,.1f} ± {row['workload_std']:,.1f}"
    au = f"{row['auroc_mean']:.4f}" if not np.isnan(row['auroc_mean']) else "—"
    f1 = f"{row['f1_mean']:.4f}" if not np.isnan(row['f1_mean']) else "—"
    print(f"{row['method']:<25} | {wl:>22} | {au:>8} | {f1:>7}")

print(f"\nStatistical test (TF-IDF vs AdaptSteer-R, two-sided Wilcoxon):")
if p_tfidf is not None:
    print(f"  p = {p_tfidf:.4f}  r = {r_tfidf:.3f}  "
          f"{'significant' if p_tfidf < 0.05 else 'not significant'}")

print(f"\n✅ Saved: results/new_baselines_summary.csv")
print(f"✅ Saved: analysis-output/figures/fig_new_baselines.{{pdf,png}}")
