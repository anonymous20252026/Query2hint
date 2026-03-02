import argparse
from pathlib import Path
from typing import List

import pandas as pd


def find_summary_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/llmsteer_eval_summary.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="othertests/llmsteer_eval")
    parser.add_argument("--metric", type=str, default="test_workload_sum_mean")
    parser.add_argument("--output_dir", type=str, default="othertests/llmsteer_eval")
    parser.add_argument("--all_csv", type=str, default="comparison_all_rows.csv")
    parser.add_argument("--best_csv", type=str, default="comparison_best_by_model_workload.csv")
    args = parser.parse_args()

    root = Path(args.root)
    files = find_summary_files(root)
    if not files:
        raise SystemExit(f"No summary files found under: {root}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["summary_file"] = str(f)
        # folder name helps identify run grouping quickly
        df["run_folder"] = f.parent.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / args.all_csv
    all_df.to_csv(all_path, index=False)

    if args.metric not in all_df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found. Available columns: {list(all_df.columns)}")

    # best config per (model_path, workload)
    group_cols = ["model_path", "workload"]
    best_df = (
        all_df.sort_values(args.metric, ascending=True)
        .groupby(group_cols, as_index=False)
        .first()
    )

    best_path = out_dir / args.best_csv
    best_df.to_csv(best_path, index=False)

    cols = [
        "model_path",
        "workload",
        "config_name",
        "test_workload_sum_mean",
        "test_workload_sum_std",
        "test_p90_mean",
        "test_median_mean",
        "accuracy_mean",
        "f1_mean",
        "auroc_mean",
        "run_folder",
    ]
    present_cols = [c for c in cols if c in best_df.columns]

    print(f"Loaded summaries: {len(files)}")
    print(f"Saved all rows:   {all_path}")
    print(f"Saved best rows:  {best_path}")
    print("\nBest per model/workload:")
    print(best_df[present_cols].to_string(index=False))


if __name__ == "__main__":
    main()
