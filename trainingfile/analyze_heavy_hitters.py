import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.utils import load_data, prepare_data, set_embedding_model

BENCHMARK_IDX = 0
LONGTAIL_IDX = 26


def parse_config_name(cfg: str) -> Tuple[str, int, bool]:
    # Expected format: "{estimator}-pcs{N}-scale{True|False}"
    parts = cfg.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid config name format: {cfg}")
    est = parts[0].lower()
    pcs = int(parts[1].replace("pcs", ""))
    scale = parts[2].replace("scale", "").lower() == "true"
    return est, pcs, scale


def build_estimator(est_name: str, seed: int):
    if est_name == "lr":
        return LogisticRegression(random_state=seed, max_iter=1000)
    if est_name == "svc_rbf":
        return SVC(random_state=seed, kernel="rbf")
    if est_name == "svc_lin":
        return SVC(random_state=seed, kernel="linear")
    raise ValueError(f"Unsupported estimator for heavy-hitter analysis: {est_name}")


def selected_runtime(hint_l: torch.Tensor, idx: np.ndarray, y_pred: np.ndarray, threshold: float):
    benchmark_const = torch.LongTensor([BENCHMARK_IDX])
    longtail_const = torch.LongTensor([LONGTAIL_IDX])
    picks = torch.where(torch.Tensor(y_pred) > threshold, longtail_const, benchmark_const).view(-1, 1)
    return hint_l[idx].gather(1, picks).squeeze(1).numpy()


def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    low = np.quantile(means, alpha / 2.0)
    high = np.quantile(means, 1.0 - alpha / 2.0)
    return float(low), float(high)


def fit_predict(
    x_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    est_name: str,
    pcs: int,
    scale: bool,
    seed: int,
):
    max_valid_pcs = min(len(train_idx), x_np.shape[1])
    effective_pcs = min(pcs, max_valid_pcs)

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("pca", PCA(n_components=effective_pcs)))
    transform = Pipeline(steps)

    x_train = transform.fit_transform(x_np[train_idx])
    x_test = transform.transform(x_np[test_idx])

    est = build_estimator(est_name, seed)
    est.fit(x_train, y_np[train_idx])
    y_pred = est.predict(x_test)
    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["job", "ceb", "all"], required=True)
    parser.add_argument("--model_a_path", required=True)
    parser.add_argument("--model_a_config", required=True)
    parser.add_argument("--model_b_path", required=True)
    parser.add_argument("--model_b_config", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_dir", default="othertests/llmsteer_eval/heavy_hitters")
    parser.add_argument("--prefix", default="compare")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    est_a, pcs_a, scale_a = parse_config_name(args.model_a_config)
    est_b, pcs_b, scale_b = parse_config_name(args.model_b_config)

    data = load_data(args.workload)

    set_embedding_model(args.model_a_path)
    xa, hint_a, _, y_a, meta_a = prepare_data(
        data, "./embeddings", "syntaxA_embedding", True, augment=False, return_meta=True
    )
    set_embedding_model(args.model_b_path)
    xb, hint_b, _, y_b, meta_b = prepare_data(
        data, "./embeddings", "syntaxA_embedding", True, augment=False, return_meta=True
    )

    if not np.array_equal(y_a.numpy(), y_b.numpy()):
        raise ValueError("Label vectors differ between model A and B pipelines.")
    if not torch.equal(hint_a, hint_b):
        raise ValueError("Hint runtime tensors differ between model A and B pipelines.")
    if not meta_a.equals(meta_b):
        raise ValueError("Query metadata ordering differs between model A and B pipelines.")

    y_np = y_a.numpy()
    hint_l = hint_a
    xa_np = xa.numpy()
    xb_np = xb.numpy()

    rows = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=args.n_splits, train_size=args.train_size, random_state=seed)
        for split_id, (train_idx, test_idx) in enumerate(sss.split(xa_np, y_np), start=1):
            yhat_a = fit_predict(xa_np, y_np, train_idx, test_idx, est_a, pcs_a, scale_a, seed)
            yhat_b = fit_predict(xb_np, y_np, train_idx, test_idx, est_b, pcs_b, scale_b, seed)

            rt_a = selected_runtime(hint_l, test_idx, yhat_a, args.threshold)
            rt_b = selected_runtime(hint_l, test_idx, yhat_b, args.threshold)
            delta = rt_a - rt_b

            for local_i, q_idx in enumerate(test_idx):
                rows.append(
                    {
                        "seed": seed,
                        "split": split_id,
                        "query_idx": int(q_idx),
                        "filename": meta_a.iloc[q_idx]["filename"],
                        "sql": meta_a.iloc[q_idx]["sql"],
                        "runtime_a": float(rt_a[local_i]),
                        "runtime_b": float(rt_b[local_i]),
                        "delta_a_minus_b": float(delta[local_i]),
                        "abs_delta": float(abs(delta[local_i])),
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{args.prefix}_per_query.csv"
    df.to_csv(detail_path, index=False)

    total_delta = float(df["delta_a_minus_b"].sum())
    grouped = (
        df.groupby("query_idx", as_index=False)
        .agg(
            filename=("filename", "first"),
            mean_runtime_a=("runtime_a", "mean"),
            mean_runtime_b=("runtime_b", "mean"),
            mean_delta=("delta_a_minus_b", "mean"),
            abs_mean_delta=("delta_a_minus_b", lambda x: float(np.mean(np.abs(x)))),
            count=("delta_a_minus_b", "count"),
        )
        .sort_values("abs_mean_delta", ascending=False)
    )
    grouped_path = out_dir / f"{args.prefix}_heavy_hitters.csv"
    grouped.to_csv(grouped_path, index=False)

    # Top-k heavy hitter table for appendix/paper.
    abs_mass_total = grouped["abs_mean_delta"].sum()
    top_k = grouped.head(min(args.top_k, len(grouped))).copy()
    top_k["rank"] = np.arange(1, len(top_k) + 1)
    top_k["abs_mass_fraction"] = top_k["abs_mean_delta"] / abs_mass_total if abs_mass_total > 0 else 0.0
    top_k["cum_abs_mass_fraction"] = top_k["abs_mass_fraction"].cumsum()
    top_k = top_k[
        [
            "rank",
            "query_idx",
            "filename",
            "mean_runtime_a",
            "mean_runtime_b",
            "mean_delta",
            "abs_mean_delta",
            "abs_mass_fraction",
            "cum_abs_mass_fraction",
            "count",
        ]
    ]
    top_k_csv_path = out_dir / f"{args.prefix}_top{args.top_k}.csv"
    top_k.to_csv(top_k_csv_path, index=False)
    top_k_md_path = out_dir / f"{args.prefix}_top{args.top_k}.md"
    top_k_md_path.write_text(top_k.to_markdown(index=False))

    # Contribution of top-k heavy hitters to overall absolute delta mass.
    abs_mass = grouped["abs_mean_delta"].sum()
    contrib = {}
    for k in [1, 5, 10, 20]:
        topk = grouped.head(min(k, len(grouped)))
        contrib[f"top{k}_abs_mass_fraction"] = float(topk["abs_mean_delta"].sum() / abs_mass) if abs_mass > 0 else 0.0

    ci_low, ci_high = bootstrap_ci(df["delta_a_minus_b"].to_numpy(), args.n_bootstrap, args.alpha, seed=12345)
    summary: Dict[str, object] = {
        "workload": args.workload,
        "model_a_path": args.model_a_path,
        "model_a_config": args.model_a_config,
        "model_b_path": args.model_b_path,
        "model_b_config": args.model_b_config,
        "n_rows": int(len(df)),
        "n_unique_queries": int(df["query_idx"].nunique()),
        "sum_runtime_a": float(df["runtime_a"].sum()),
        "sum_runtime_b": float(df["runtime_b"].sum()),
        "sum_delta_a_minus_b": total_delta,
        "mean_delta_a_minus_b": float(df["delta_a_minus_b"].mean()),
        "bootstrap_ci_mean_delta": [ci_low, ci_high],
        **contrib,
        "top_k_table_csv": str(top_k_csv_path),
        "top_k_table_md": str(top_k_md_path),
    }
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved per-query rows: {detail_path}")
    print(f"Saved heavy-hitters:  {grouped_path}")
    print(f"Saved top-{args.top_k} table: {top_k_csv_path}")
    print(f"Saved top-{args.top_k} markdown: {top_k_md_path}")
    print(f"Saved summary:        {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
