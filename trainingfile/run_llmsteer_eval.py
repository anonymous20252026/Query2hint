import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure repo-root imports (e.g., `models.utils`) work when run as `python trainingfile/...`.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.utils import load_data, prepare_data, set_embedding_model


BENCHMARK_IDX = 0
LONGTAIL_IDX = 26


@dataclass
class EvalConfig:
    name: str
    estimator: object
    pcs: int
    scale: bool


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_bool_list(value: str) -> List[bool]:
    out = []
    for x in parse_csv_list(value):
        lx = x.lower()
        if lx in {"true", "1", "yes"}:
            out.append(True)
        elif lx in {"false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean list item: {x}")
    return out


def build_estimator(name: str, seed: int):
    lname = name.lower()
    if lname == "lr":
        return LogisticRegression(random_state=seed, max_iter=1000)
    if lname == "svc_rbf":
        return SVC(random_state=seed, kernel="rbf", probability=True)
    if lname == "svc_lin":
        return SVC(random_state=seed, kernel="linear", probability=True)
    if lname == "rfc":
        return RandomForestClassifier(random_state=seed, n_estimators=100)
    if lname == "gbc":
        return GradientBoostingClassifier(random_state=seed, n_estimators=100, learning_rate=0.1)
    raise ValueError(f"Unknown estimator: {name}")


def score_continuous(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x)
    return estimator.predict(x)


def evaluate_runtime(hint_l: torch.Tensor, idx: np.ndarray, y_pred: np.ndarray, threshold: float):
    benchmark_const = torch.LongTensor([BENCHMARK_IDX])
    longtail_const = torch.LongTensor([LONGTAIL_IDX])
    picks = torch.where(torch.Tensor(y_pred) > threshold, longtail_const, benchmark_const).view(-1, 1)
    chosen = hint_l[idx].gather(1, picks)
    return {
        "workload_sum": float(chosen.sum().item()),
        "p90": float(chosen.quantile(0.90).item()),
        "median": float(chosen.median().item()),
    }


def run_eval(args):
    seeds = parse_int_list(args.seeds)
    pcs_list = parse_int_list(args.pcs)
    scale_list = parse_bool_list(args.scale_options)
    estimators = parse_csv_list(args.estimators)
    exclude_configs = {x.lower() for x in parse_csv_list(args.exclude_configs)}

    set_embedding_model(args.model_path)
    data = load_data(args.workload)

    if args.augment_eval:
        x_main, hint_l, _, y, x_spaced, x_tabbed = prepare_data(
            data, "./embeddings", "syntaxA_embedding", True, augment=True
        )
    else:
        x_main, hint_l, _, y = prepare_data(data, "./embeddings", "syntaxA_embedding", True, augment=False)
        x_spaced = x_tabbed = None

    y_np = y.numpy()
    all_rows = []
    cfgs_per_split = 0
    for est_name in estimators:
        for pcs in pcs_list:
            for scale in scale_list:
                cfg_name = f"{est_name}-pcs{pcs}-scale{scale}"
                if cfg_name.lower() not in exclude_configs:
                    cfgs_per_split += 1
    total_cfg_count = len(seeds) * args.n_splits * cfgs_per_split
    done_cfg_count = 0
    start_time = time.time()
    print(
        f"Starting eval: seeds={len(seeds)}, splits={args.n_splits}, "
        f"configs_per_split={cfgs_per_split}, total_fits={total_cfg_count}"
    )

    for seed in seeds:
        print(f"\n[Seed {seed}]")
        sss = StratifiedShuffleSplit(n_splits=args.n_splits, train_size=args.train_size, random_state=seed)

        cfgs: List[EvalConfig] = []
        for est_name in estimators:
            for pcs in pcs_list:
                for scale in scale_list:
                    cfg_name = f"{est_name}-pcs{pcs}-scale{scale}"
                    if cfg_name.lower() in exclude_configs:
                        continue
                    cfgs.append(
                        EvalConfig(
                            name=cfg_name,
                            estimator=build_estimator(est_name, seed),
                            pcs=pcs,
                            scale=scale,
                        )
                    )

        x_main_np = x_main.numpy()
        x_spaced_np = x_spaced.numpy() if x_spaced is not None else None
        x_tabbed_np = x_tabbed.numpy() if x_tabbed is not None else None

        for split_id, (train_idx, test_idx) in enumerate(sss.split(x_main_np, y_np), start=1):
            print(f"  Split {split_id}/{args.n_splits}: train={len(train_idx)} test={len(test_idx)}")
            x_train = x_main_np[train_idx]
            x_test = x_main_np[test_idx]
            y_train = y_np[train_idx]
            y_test = y_np[test_idx]

            for cfg in cfgs:
                max_valid_pcs = min(x_train.shape[0], x_train.shape[1])
                if max_valid_pcs < 1:
                    continue
                effective_pcs = min(cfg.pcs, max_valid_pcs)

                steps = []
                if cfg.scale:
                    steps.append(("scaler", StandardScaler()))
                steps.append(("pca", PCA(n_components=effective_pcs)))
                transform = Pipeline(steps)

                x_train_t = transform.fit_transform(x_train)
                x_test_t = transform.transform(x_test)
                cfg.estimator.fit(x_train_t, y_train)

                y_pred = cfg.estimator.predict(x_test_t)
                y_score = score_continuous(cfg.estimator, x_test_t)

                runtime_main = evaluate_runtime(hint_l, test_idx, y_pred, args.threshold)
                row: Dict[str, object] = {
                    "model_path": args.model_path,
                    "workload": args.workload,
                    "seed": seed,
                    "split": split_id,
                    "config_name": cfg.name,
                    "requested_pcs": cfg.pcs,
                    "effective_pcs": effective_pcs,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "auroc": float(roc_auc_score(y_test, y_score)),
                    "test_workload_sum": runtime_main["workload_sum"],
                    "test_p90": runtime_main["p90"],
                    "test_median": runtime_main["median"],
                }

                if args.augment_eval:
                    x_spaced_t = transform.transform(x_spaced_np[test_idx])
                    x_tabbed_t = transform.transform(x_tabbed_np[test_idx])
                    y_spaced = cfg.estimator.predict(x_spaced_t)
                    y_tabbed = cfg.estimator.predict(x_tabbed_t)
                    y_spaced_score = score_continuous(cfg.estimator, x_spaced_t)
                    y_tabbed_score = score_continuous(cfg.estimator, x_tabbed_t)
                    runtime_spaced = evaluate_runtime(hint_l, test_idx, y_spaced, args.threshold)
                    runtime_tabbed = evaluate_runtime(hint_l, test_idx, y_tabbed, args.threshold)
                    row.update(
                        {
                            "spaced_accuracy": float(accuracy_score(y_test, y_spaced)),
                            "spaced_precision": float(precision_score(y_test, y_spaced, zero_division=0)),
                            "spaced_recall": float(recall_score(y_test, y_spaced, zero_division=0)),
                            "spaced_f1": float(f1_score(y_test, y_spaced, zero_division=0)),
                            "spaced_auroc": float(roc_auc_score(y_test, y_spaced_score)),
                            "spaced_workload_sum": runtime_spaced["workload_sum"],
                            "spaced_p90": runtime_spaced["p90"],
                            "spaced_median": runtime_spaced["median"],
                            "tabbed_accuracy": float(accuracy_score(y_test, y_tabbed)),
                            "tabbed_precision": float(precision_score(y_test, y_tabbed, zero_division=0)),
                            "tabbed_recall": float(recall_score(y_test, y_tabbed, zero_division=0)),
                            "tabbed_f1": float(f1_score(y_test, y_tabbed, zero_division=0)),
                            "tabbed_auroc": float(roc_auc_score(y_test, y_tabbed_score)),
                            "tabbed_workload_sum": runtime_tabbed["workload_sum"],
                            "tabbed_p90": runtime_tabbed["p90"],
                            "tabbed_median": runtime_tabbed["median"],
                        }
                    )

                all_rows.append(row)
                done_cfg_count += 1
                if done_cfg_count % 10 == 0 or done_cfg_count == total_cfg_count:
                    elapsed = time.time() - start_time
                    print(
                        f"    Progress {done_cfg_count}/{total_cfg_count} "
                        f"({100.0 * done_cfg_count / total_cfg_count:.1f}%) "
                        f"elapsed={elapsed/60.0:.1f}m last={cfg.name}"
                    )

    detail_df = pd.DataFrame(all_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / args.detail_csv
    detail_df.to_csv(detail_path, index=False)

    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "test_workload_sum",
        "test_p90",
        "test_median",
    ]
    for optional in [
        "spaced_accuracy",
        "spaced_precision",
        "spaced_recall",
        "spaced_f1",
        "spaced_auroc",
        "spaced_workload_sum",
        "spaced_p90",
        "spaced_median",
        "tabbed_accuracy",
        "tabbed_precision",
        "tabbed_recall",
        "tabbed_f1",
        "tabbed_auroc",
        "tabbed_workload_sum",
        "tabbed_p90",
        "tabbed_median",
    ]:
        if optional in detail_df.columns:
            metric_cols.append(optional)

    summary = detail_df.groupby(["model_path", "workload", "config_name"], as_index=False)[metric_cols].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary_path = out_dir / args.summary_csv
    summary.to_csv(summary_path, index=False)

    best_key = "test_workload_sum_mean"
    best_row = summary.sort_values(best_key, ascending=True).head(1).to_dict(orient="records")[0]
    best_path = out_dir / args.best_json
    best_path.write_text(json.dumps(best_row, indent=2))

    print(f"Saved detail results to: {detail_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved best config to:    {best_path}")
    print(f"Best by {best_key}: {best_row['config_name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--workload", type=str, choices=["job", "ceb", "all"], default="ceb")
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--estimators", type=str, default="lr,svc_rbf,svc_lin,rfc,gbc")
    parser.add_argument("--pcs", type=str, default="5,50,120")
    parser.add_argument("--scale_options", type=str, default="false,true")
    parser.add_argument(
        "--exclude_configs",
        type=str,
        default="",
        help=(
            "Comma-separated config names to skip, e.g. "
            "'svc_lin-pcs50-scaletrue,svc_lin-pcs120-scaletrue'"
        ),
    )
    parser.add_argument("--augment_eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default="othertests/llmsteer_eval")
    parser.add_argument("--detail_csv", type=str, default="llmsteer_eval_detail.csv")
    parser.add_argument("--summary_csv", type=str, default="llmsteer_eval_summary.csv")
    parser.add_argument("--best_json", type=str, default="llmsteer_eval_best.json")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
