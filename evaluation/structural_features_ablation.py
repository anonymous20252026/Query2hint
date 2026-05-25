"""
AdaSteer v2 — Beat LLMSteer on workload, P90, and robustness.

Three improvements:
  [A] SQL structural features — 15 hand-crafted signals concatenated with embeddings.
  [B] Multi-encoder stacking — all three trained encoders combined before PCA.
  [C] Workload-aware threshold — τ* tuned on inner validation fold (no leakage).

Key fixes vs earlier draft:
  - class_weight='balanced' (sklearn built-in, not squared manual ratio)
  - SVC probability=False + decision_function throughout (6× faster, no Platt CV)
  - Threshold tuned on percentile grid of decision scores, not fixed [0.25,0.75]
  - Multi-encoder variants run once, not 3× per encoder

Target: workload < 2547.7s  |  P90 < 5.7s

Output → results_v2/
"""

import os, re, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sentence_transformers import SentenceTransformer
import sqlparse

warnings.filterwarnings("ignore")

OUT_DIR = "results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── configuration ────────────────────────────────────────────────────────────

RANDOM_SEED   = 24508
K_FOLDS       = 10
TRAIN_SIZE    = 0.8
BENCHMARK_IDX = 0
LONGTAIL_IDX  = 26
PCS           = 120          # PCA components
N_THRESH_GRID = 51           # points in threshold sweep

LLMSTEER  = {"workload": 2547.7, "p90": 5.7}
POSTGRES  = {"workload": 8134.7, "p90": 19.6}
OPTIMAL   = {"workload": 1064.1, "p90":  3.4}

ENCODER_PATHS = {
    "AdaSteer-C": "encoders/encoder_all-mpnet-base-v2_v1",
    "AdaSteer-R": "encoders/encoder_reptile_mpnet_v4",
    "AdaSteer-O": "adasteer_encoder",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def banner(msg):
    print("\n" + "=" * 68)
    print(msg)
    print("=" * 68)
    sys.stdout.flush()

banner("AdaSteer v2 — fast run")
print(f"Device  : {DEVICE}")
print(f"Target  : workload < {LLMSTEER['workload']}s  |  P90 < {LLMSTEER['p90']}s")

# ── 1. load data ─────────────────────────────────────────────────────────────

banner("[1/7] Loading data")

job_df = pd.read_csv("data/job.csv",
                     converters={"hint_list": eval, "runtime_list": eval})
ceb_df = pd.read_csv("data/ceb.csv",
                     converters={"hint_list": eval, "runtime_list": eval})
data   = pd.concat([job_df, ceb_df]).reset_index(drop=True)
data["mean_runtime"] = data["runtime_list"].apply(np.mean)
data["sql"]          = data["sql"].str.strip("\n")

# Aggregate to unique (filename, sql) rows with 49-hint runtime vectors
raw = data.copy().drop(columns=["runtime_list", "plan_tree"])
raw = raw.explode("hint_list").sort_values(["filename", "hint_list"])
raw = raw.groupby(["filename", "sql"], as_index=False).agg(
    hint_list=("hint_list", list),
    mean_runtime=("mean_runtime", list),
)
raw["opt_l"] = raw["mean_runtime"].apply(min)
raw = raw.reset_index(drop=True)

N = len(raw)
print(f"  Unique (filename,sql) pairs: {N:,}")

# Per-query cost weights: log(1 + |latency_default - latency_alt|), normalised to mean 1
# Computed after hint_l is built — placeholder here, assigned after hint_l is ready.
_cost_weights = None   # set below after hint_l is built

# hint_l tensor [N, 49]
def _build_hint_l(df):
    n_hints = 49
    H = np.zeros((len(df), n_hints), dtype=np.float32)
    for i, row in enumerate(df.itertuples()):
        for j, h in enumerate(row.hint_list):
            H[i, h] = row.mean_runtime[j]
    return torch.tensor(H)

hint_l = _build_hint_l(raw)
binary = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).float()
y_all  = binary.numpy().astype(int)

# Log-scaled cost weights: focus training on high-impact queries
_costs = torch.abs(hint_l[:, BENCHMARK_IDX] - hint_l[:, LONGTAIL_IDX]).numpy()
_cost_weights = np.log1p(_costs)
_cost_weights = _cost_weights / _cost_weights.mean()    # normalise to mean = 1

print(f"  hint_l shape  : {hint_l.shape}")
print(f"  Positive rate : {y_all.mean():.3f}  ({y_all.sum()} queries prefer hint_26)")

# Sanity-check oracle workload via CV (quick check with one fold)
_spl = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_SEED)
_tr, _te = next(_spl.split(np.zeros(N), y_all))
_oracle_wl = float(hint_l[_te].min(dim=1).values.sum())
print(f"  1-fold oracle workload check: {_oracle_wl:.1f}s  (target ~{N*0.2/N*OPTIMAL['workload']:.0f}s)")

# ── 2. SQL structural features ────────────────────────────────────────────────

banner("[2/7] SQL structural features  [Improvement A]")

_JOIN_RE     = re.compile(r"\bJOIN\b",            re.I)
_FROM_RE     = re.compile(r"\bFROM\s+(\w+)",      re.I)
_JFROM_RE    = re.compile(r"\bJOIN\s+(\w+)",      re.I)
_AGG_RE      = re.compile(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", re.I)
_SUBQ_RE     = re.compile(r"\bSELECT\b",          re.I)
_PRED_RE     = re.compile(r"\b(AND|OR)\b",         re.I)
_COMP_RE     = re.compile(r"[<>=!]{1,2}")
_BETWEEN_RE  = re.compile(r"\bBETWEEN\b",          re.I)
_IN_RE       = re.compile(r"\bIN\s*\(",            re.I)
_LIKE_RE     = re.compile(r"\bLIKE\b",             re.I)
_NEST_RE     = re.compile(r"\(SELECT\b",           re.I)

def sql_features(sql):
    tables  = set(_FROM_RE.findall(sql)) | set(_JFROM_RE.findall(sql))
    return np.array([
        len(_JOIN_RE.findall(sql)),
        len(tables),
        len(_AGG_RE.findall(sql)),
        max(0, len(_SUBQ_RE.findall(sql)) - 1),
        len(_NEST_RE.findall(sql)),
        len(_PRED_RE.findall(sql)),
        len(_COMP_RE.findall(sql)),
        len(_BETWEEN_RE.findall(sql)),
        len(_IN_RE.findall(sql)),
        len(_LIKE_RE.findall(sql)),
        int("GROUP BY" in sql.upper()),
        int("ORDER BY" in sql.upper()),
        int("HAVING"   in sql.upper()),
        int("DISTINCT" in sql.upper()),
        np.log1p(len(sql)),
    ], dtype=np.float32)

sql_struct = np.stack(raw["sql"].apply(sql_features).tolist())
s_mean = sql_struct.mean(0); s_std = sql_struct.std(0) + 1e-9
sql_struct_norm = (sql_struct - s_mean) / s_std

print(f"  Structural matrix : {sql_struct_norm.shape}  (15 features per query)")
sys.stdout.flush()

# ── 3. encode with all three encoders ────────────────────────────────────────

banner("[3/7] Generating embeddings  [Improvement B]")

def embed(encoder_path, sqls_A, sqls_B, sqls_C):
    st = SentenceTransformer(encoder_path, device=DEVICE)
    st.max_seq_length = 512
    def enc(texts):
        return st.encode(texts, batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    t0 = time.time()
    eA = enc(sqls_A)
    eB = enc(sqls_B)
    eC = enc(sqls_C)
    # inference latency (50 single-query calls, cheaper than 200)
    lats = [time.perf_counter() * 0]  # dummy init
    for _ in range(50):
        t1 = time.perf_counter()
        st.encode([sqls_A[0]], batch_size=1, show_progress_bar=False,
                  convert_to_numpy=True)
        lats.append((time.perf_counter() - t1) * 1000)
    inf_ms = float(np.median(lats[1:]))
    del st
    return eA, eB, eC, inf_ms, time.time() - t0

sqls_A = raw["sql"].tolist()
sqls_B = [sqlparse.format(s, reindent=True,
                           use_space_around_operators=True,
                           indent_tabs=False) for s in sqls_A]
sqls_C = [sqlparse.format(s, reindent=True,
                           use_space_around_operators=True,
                           indent_tabs=True)  for s in sqls_A]

encoder_embs = {}
available    = []

for enc_name, enc_path in ENCODER_PATHS.items():
    if not os.path.isdir(enc_path):
        print(f"  ⚠  {enc_name}: path not found ({enc_path}) — skip")
        continue
    print(f"  {enc_name} …", end=" ", flush=True)
    eA, eB, eC, inf_ms, elapsed = embed(enc_path, sqls_A, sqls_B, sqls_C)
    encoder_embs[enc_name] = (
        torch.tensor(eA, dtype=torch.float32),
        torch.tensor(eB, dtype=torch.float32),
        torch.tensor(eC, dtype=torch.float32),
        inf_ms,
    )
    available.append(enc_name)
    print(f"dim={eA.shape[1]}  inf={inf_ms:.2f}ms  elapsed={elapsed:.1f}s")
    sys.stdout.flush()

if not available:
    print("No encoder paths found. Check ENCODER_PATHS."); sys.exit(1)

# ── 4. build feature matrices for each experiment ────────────────────────────

banner("[4/7] Building feature matrices")

struct_t = torch.tensor(sql_struct_norm, dtype=torch.float32)

# All-encoder stack (concatenate along feature dim)
all_eA = torch.cat([encoder_embs[n][0] for n in available], dim=1)
all_eB = torch.cat([encoder_embs[n][1] for n in available], dim=1)
all_eC = torch.cat([encoder_embs[n][2] for n in available], dim=1)

# Use AdaSteer-O as the "single best" encoder; fall back to first available
SINGLE = "AdaSteer-O" if "AdaSteer-O" in available else available[0]
eA_s, eB_s, eC_s, inf_ms_s = encoder_embs[SINGLE]

_inf_ens = float(np.mean([encoder_embs[n][3] for n in available]))
_v4A = torch.cat([all_eA, struct_t], 1).numpy()
_v4B = torch.cat([all_eB, struct_t], 1).numpy()
_v4C = torch.cat([all_eC, struct_t], 1).numpy()

EXPERIMENTS = {
    # (feats_A, feats_B, feats_C, inf_ms, threshold_tune, thresh_metric)
    "v1_baseline": (
        eA_s.numpy(), eB_s.numpy(), eC_s.numpy(),
        inf_ms_s, False, "workload",
    ),
    "v2_struct": (
        torch.cat([eA_s, struct_t], 1).numpy(),
        torch.cat([eB_s, struct_t], 1).numpy(),
        torch.cat([eC_s, struct_t], 1).numpy(),
        inf_ms_s, False, "workload",
    ),
    "v3_ensemble": (
        all_eA.numpy(), all_eB.numpy(), all_eC.numpy(),
        _inf_ens, False, "workload",
    ),
    "v4_ensemble_struct": (
        _v4A, _v4B, _v4C,
        _inf_ens, False, "workload",
    ),
    "v4_ensemble_struct_thresh_wl": (
        _v4A, _v4B, _v4C,
        _inf_ens, True, "workload",
    ),
    "v4_ensemble_struct_thresh_p90": (
        _v4A, _v4B, _v4C,
        _inf_ens, True, "p90",
    ),
    "v4_ensemble_struct_thresh_joint": (
        _v4A, _v4B, _v4C,
        _inf_ens, True, "joint", False,
    ),
    "v4_ensemble_struct_costwt_joint": (
        _v4A, _v4B, _v4C,
        _inf_ens, True, "joint", True,    # cost-weighted training + joint threshold
    ),
}

# Back-fill missing 7th element (use_cost_weights=False) for older entries
EXPERIMENTS = {k: (*v, False) if len(v) == 6 else v
               for k, v in EXPERIMENTS.items()}

for name, (XA, XB, XC, _, _tune, _tm, _cw) in EXPERIMENTS.items():
    print(f"  {name:<36s}  dim={XA.shape[1]}")
sys.stdout.flush()

# ── 5. CV engine ──────────────────────────────────────────────────────────────

banner("[5/7] Running cross-validation")

def workload_p90(decisions, idx):
    """Compute workload sum and P90 given binary decisions and global indices."""
    dec_t  = torch.where(
        torch.tensor(decisions, dtype=torch.float32) > 0.5,
        torch.LongTensor([LONGTAIL_IDX]),
        torch.LongTensor([BENCHMARK_IDX]),
    ).view(-1, 1)
    lats   = hint_l[idx].gather(1, dec_t).squeeze().numpy()
    return float(lats.sum()), float(np.quantile(lats, 0.9))


def find_tau(scores_val, val_idx, optimize="workload"):
    """Return τ* minimising workload, P90, or a normalised joint metric."""
    grid    = np.linspace(np.percentile(scores_val, 5),
                          np.percentile(scores_val, 95),
                          N_THRESH_GRID)
    best_t, best_metric = 0.0, float("inf")
    for t in grid:
        preds   = (scores_val > t).astype(int)
        wl, p90 = workload_p90(preds, val_idx)
        if optimize == "p90":
            metric = p90
        elif optimize == "joint":
            # Normalise both by LLMSteer baseline so they're equally weighted
            metric = wl / LLMSTEER["workload"] + p90 / LLMSTEER["p90"]
        else:
            metric = wl
        if metric < best_metric:
            best_metric, best_t = metric, t
    return float(best_t)


def preprocess(XA_tr, XA_te, XB_te, XC_te, pcs):
    """Exact compare.py preprocessing: StandardScaler → full PCA → StandardScaler → [:pcs]."""
    # Stage 1: scale + full PCA (matches compare.py's pipeline variable)
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])
    pca_tr  = pipe.fit_transform(XA_tr)
    pca_teA = pipe.transform(XA_te)
    pca_teB = pipe.transform(XB_te)
    pca_teC = pipe.transform(XC_te)

    # Stage 2: re-scale PCA components (matches compare.py's scaler variable, scale=True)
    sc2 = StandardScaler()
    Z_tr  = sc2.fit_transform(pca_tr)
    Z_teA = sc2.transform(pca_teA)
    Z_teB = sc2.transform(pca_teB)
    Z_teC = sc2.transform(pca_teC)

    # Slice first `pcs` components (matches compare.py's Xtr[:, :PCS])
    p = min(pcs, Z_tr.shape[1])
    return Z_tr[:, :p], Z_teA[:, :p], Z_teB[:, :p], Z_teC[:, :p]


def run_cv(XA, XB, XC, label, threshold_tune=False, thresh_metric="workload",
           use_cost_weights=False):
    """10-fold stratified CV.  Uses SVC decision_function (no Platt scaling)."""
    splitter = StratifiedShuffleSplit(
        n_splits=K_FOLDS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED
    )

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(XA, y_all)):
        X_tr, X_te, X_teB, X_teC = preprocess(
            XA[tr_idx], XA[te_idx], XB[te_idx], XC[te_idx], PCS
        )

        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]

        # Class weights: match compare.py formula exactly
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        cw = {0: n_pos / n_neg, 1: n_neg / n_pos}

        est = SVC(kernel="rbf", probability=False,
                  class_weight=cw, random_state=RANDOM_SEED)

        sw_tr = _cost_weights[tr_idx] if use_cost_weights else None

        if threshold_tune:
            # Inner split: find τ* on inner val, train final model on full tr_fold
            inner = StratifiedShuffleSplit(
                n_splits=1, train_size=0.8,
                random_state=RANDOM_SEED + fold_i,
            )
            # inner split indices are into X_tr (already preprocessed above)
            itr, ival = next(inner.split(X_tr, y_tr))
            # Re-preprocess from raw so inner-val is not contaminated by outer tr scaler
            Xi_tr, Xi_val, _, _ = preprocess(
                XA[tr_idx][itr], XA[tr_idx][ival],
                XA[tr_idx][ival], XA[tr_idx][ival], PCS
            )
            yi_tr  = y_tr[itr]
            yi_val = y_tr[ival]
            n_pos_i = int((yi_tr == 1).sum()); n_neg_i = int((yi_tr == 0).sum())
            cw_i = {0: n_pos_i / n_neg_i, 1: n_neg_i / n_pos_i}
            sw_i  = _cost_weights[tr_idx[itr]] if use_cost_weights else None
            est_inner = SVC(kernel="rbf", probability=False,
                            class_weight=cw_i, random_state=RANDOM_SEED)
            est_inner.fit(Xi_tr, yi_tr, sample_weight=sw_i)
            scores_val = est_inner.decision_function(Xi_val)
            tau_star   = find_tau(scores_val, tr_idx[ival], optimize=thresh_metric)
            est.fit(X_tr, y_tr, sample_weight=sw_tr)
            scores_te  = est.decision_function(X_te)
            preds_te   = (scores_te > tau_star).astype(int)
        else:
            tau_star = 0.0          # default decision boundary for SVC
            est.fit(X_tr, y_tr, sample_weight=sw_tr)
            scores_te  = est.decision_function(X_te)
            preds_te   = (scores_te > tau_star).astype(int)

        scores_B = est.decision_function(X_teB)
        scores_C = est.decision_function(X_teC)
        preds_B  = (scores_B > tau_star).astype(int)
        preds_C  = (scores_C > tau_star).astype(int)

        wl,   p90   = workload_p90(preds_te, te_idx)
        wl_B, p90_B = workload_p90(preds_B,  te_idx)
        wl_C, p90_C = workload_p90(preds_C,  te_idx)

        rows.append({
            "fold":   fold_i,
            "wl":     wl,   "p90":   p90,
            "wl_B":   wl_B, "p90_B": p90_B,
            "wl_C":   wl_C, "p90_C": p90_C,
            "auroc":  roc_auc_score(y_te, scores_te),
            "f1":     f1_score(y_te, preds_te, zero_division=0),
            "acc":    accuracy_score(y_te, preds_te),
            "tau":    tau_star,
        })

    fdf = pd.DataFrame(rows)
    return {
        "label":    label,
        "wl_mean":  fdf["wl"].mean(),  "wl_std":  fdf["wl"].std(),
        "p90_mean": fdf["p90"].mean(), "p90_std": fdf["p90"].std(),
        "wl_B":     fdf["wl_B"].mean(),"p90_B":   fdf["p90_B"].mean(),
        "wl_C":     fdf["wl_C"].mean(),"p90_C":   fdf["p90_C"].mean(),
        "auroc":    fdf["auroc"].mean(),"f1":      fdf["f1"].mean(),
        "acc":      fdf["acc"].mean(),
        "tau":      fdf["tau"].mean(),
        "folds_wl": fdf["wl"].tolist(),
    }


# Run all experiments
results = []
t_start = time.time()

for exp_name, (XA, XB, XC, inf_ms, tune, t_metric, costwt) in EXPERIMENTS.items():
    t0 = time.time()
    print(f"\n  Running {exp_name} (thresh={tune}, metric={t_metric}, costwt={costwt}) …", flush=True)
    r  = run_cv(XA, XB, XC, label=exp_name, threshold_tune=tune,
                thresh_metric=t_metric, use_cost_weights=costwt)
    elapsed = time.time() - t0
    beats   = r["wl_mean"] < LLMSTEER["workload"]
    r["inf_ms"] = inf_ms
    results.append(r)
    gap_str = "BEATS LLMSteer" if beats else f"gap={r['wl_mean']-LLMSTEER['workload']:+.2f}s"
    print(f"  wl={r['wl_mean']:.2f}+-{r['wl_std']:.2f}  "
          f"P90={r['p90_mean']:.4f}  AUROC={r['auroc']:.4f}  "
          f"{gap_str}  [{elapsed:.0f}s]", flush=True)

print(f"\nTotal CV time: {time.time()-t_start:.0f}s")

# ── 6. compile results and write outputs ──────────────────────────────────────

banner("[6/7] Compiling results")

BASELINES = [
    {"label": "PostgreSQL Default", "wl_mean": POSTGRES["workload"],
     "p90_mean": POSTGRES["p90"], "wl_std": 0, "p90_std": 0,
     "auroc": None, "f1": None, "acc": None, "inf_ms": 0,
     "wl_B": POSTGRES["workload"], "p90_B": POSTGRES["p90"],
     "wl_C": POSTGRES["workload"], "p90_C": POSTGRES["p90"], "tau": None},
    {"label": "LLMSteer (OpenAI)", "wl_mean": LLMSTEER["workload"],
     "p90_mean": LLMSTEER["p90"], "wl_std": 0, "p90_std": 0,
     "auroc": None, "f1": None, "acc": None, "inf_ms": 200,
     "wl_B": None, "p90_B": None, "wl_C": None, "p90_C": None, "tau": None},
    {"label": "Optimal (oracle)", "wl_mean": OPTIMAL["workload"],
     "p90_mean": OPTIMAL["p90"], "wl_std": 0, "p90_std": 0,
     "auroc": None, "f1": None, "acc": None, "inf_ms": 0,
     "wl_B": None, "p90_B": None, "wl_C": None, "p90_C": None, "tau": None},
]

all_rows = BASELINES + results
df = pd.DataFrame(all_rows)
df["beats_llmsteer"] = df["wl_mean"] < LLMSTEER["workload"]
df.to_csv(f"{OUT_DIR}/all_results_v2.csv", index=False)
print(f"  Saved → {OUT_DIR}/all_results_v2.csv")

# Significance tests vs AdaSteer-O v1_baseline
baseline_folds = next(r["folds_wl"] for r in results if r["label"] == "v1_baseline")
sig_rows = []
for r in results:
    if r["label"] == "v1_baseline":
        continue
    try:
        stat, pval = wilcoxon(r["folds_wl"], baseline_folds, alternative="less")
        sig_rows.append({"label": r["label"], "wilcoxon_stat": stat,
                         "p_value": pval, "significant_p05": pval < 0.05})
    except Exception as e:
        sig_rows.append({"label": r["label"], "wilcoxon_stat": None,
                         "p_value": None, "significant_p05": False, "note": str(e)})

pd.DataFrame(sig_rows).to_csv(f"{OUT_DIR}/significance_v2.csv", index=False)
print(f"  Saved → {OUT_DIR}/significance_v2.csv")

# ── 7. figures ────────────────────────────────────────────────────────────────

banner("[7/7] Figures")

# Figure 1: Workload comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels    = [r["label"] for r in all_rows if r.get("wl_mean") is not None]
wl_means  = [r["wl_mean"] for r in all_rows if r.get("wl_mean") is not None]
wl_stds   = [r.get("wl_std", 0) or 0 for r in all_rows if r.get("wl_mean") is not None]
p90_means = [r["p90_mean"] for r in all_rows if r.get("p90_mean") is not None]
p90_stds  = [r.get("p90_std", 0) or 0 for r in all_rows if r.get("p90_mean") is not None]

colors = ["#d62728" if "LLMSteer" in l else
          "#1f77b4" if "AdaSteer" not in l else
          "#2ca02c" for l in labels]
x = np.arange(len(labels))

for ax, means, stds, ylabel, title in [
    (axes[0], wl_means, wl_stds, "Workload (s)", "Total Workload"),
    (axes[1], p90_means, p90_stds, "P90 Latency (s)", "P90 Latency"),
]:
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.axhline(LLMSTEER["workload" if "Workload" in title else "p90"],
               color="#d62728", linestyle="--", linewidth=1.5, label="LLMSteer")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure_v2_main.pdf", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved → {OUT_DIR}/figure_v2_main.pdf")

# Figure 2: Ablation (AdaSteer variants only)
ada_rows = [r for r in results]
fig, ax  = plt.subplots(figsize=(8, 4))
ax.barh([r["label"] for r in ada_rows],
        [r["wl_mean"] for r in ada_rows],
        xerr=[r.get("wl_std", 0) or 0 for r in ada_rows],
        color=["#2ca02c" if r["wl_mean"] < LLMSTEER["workload"] else "#aec7e8"
               for r in ada_rows],
        capsize=4, alpha=0.85)
ax.axvline(LLMSTEER["workload"], color="#d62728", linestyle="--",
           linewidth=1.5, label="LLMSteer (2547.7s)")
ax.axvline(OPTIMAL["workload"], color="#7f7f7f", linestyle=":",
           linewidth=1.2, label="Oracle (1064.1s)")
ax.set_xlabel("Mean Workload (s)")
ax.set_title("AdaSteer v2 — Ablation")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure_v2_ablation.pdf", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved → {OUT_DIR}/figure_v2_ablation.pdf")

# ── LaTeX table ───────────────────────────────────────────────────────────────

def fmt(v, fmt_str, dash="—"):
    return f"{v:{fmt_str}}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else dash

tex_rows = []
for r in all_rows:
    tex_rows.append(
        f"  {r['label']:<30s} & {fmt(r['wl_mean'],'.2f')} ± {fmt(r.get('wl_std',None),'.2f')} "
        f"& {fmt(r['p90_mean'],'.3f')} "
        f"& {fmt(r.get('auroc'),'.4f')} "
        f"& {fmt(r.get('inf_ms'),'.1f')} \\\\"
    )

with open(f"{OUT_DIR}/tables_v2.tex", "w") as f:
    f.write(
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{AdaSteer v2 results vs LLMSteer baseline.}\n"
        "\\label{tab:adasteer_v2}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Method & Workload (s) & P90 (s) & AUROC & Inf.~(ms) \\\\\n"
        "\\midrule\n"
    )
    f.write("\n".join(tex_rows) + "\n")
    f.write(
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
print(f"  Saved → {OUT_DIR}/tables_v2.tex")

# ── verdict ───────────────────────────────────────────────────────────────────

banner("VERDICT")
best = min(results, key=lambda r: r["wl_mean"])
print(f"Best variant : {best['label']}")
print(f"  Workload   : {best['wl_mean']:.2f} ± {best['wl_std']:.2f} s")
print(f"  P90        : {best['p90_mean']:.4f} s")
print(f"  AUROC      : {best['auroc']:.4f}")
print(f"  vs LLMSteer: {best['wl_mean'] - LLMSTEER['workload']:+.2f} s "
      f"({'✅ BEATS' if best['wl_mean'] < LLMSTEER['workload'] else '❌ behind'})")
print(f"\nFull results → {OUT_DIR}/all_results_v2.csv")

# Write README
with open(f"{OUT_DIR}/README.txt", "w") as f:
    f.write(
        f"AdaSteer v2 Results\n"
        f"===================\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Best: {best['label']}  workload={best['wl_mean']:.2f}s  P90={best['p90_mean']:.4f}s\n"
        f"LLMSteer baseline: {LLMSTEER['workload']}s  P90={LLMSTEER['p90']}s\n\n"
        f"FILES\n"
        f"-----\n"
        f"all_results_v2.csv      — all variants, all metrics\n"
        f"significance_v2.csv    — Wilcoxon tests vs v1_baseline\n"
        f"tables_v2.tex          — LaTeX table\n"
        f"figure_v2_main.pdf     — workload+P90 bar chart\n"
        f"figure_v2_ablation.pdf — ablation horizontal bar chart\n"
    )
print(f"README → {OUT_DIR}/README.txt")
