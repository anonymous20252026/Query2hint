import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_summary_files(root: Path):
    return sorted(root.glob("**/llmsteer_eval_summary.csv"))


def normalize_model_label(model_path: str) -> str:
    base = Path(model_path).name
    mapping = {
        "finetuned_bert_base_both": "FT BERT",
        "finetuned_bert_base_both_v2": "FT BERT",
        "finetuned_distilbert_both": "FT DistilBERT",
        "finetuned_bge_m3_both": "FT BGE-M3",
        "finetuned_bge_large_both": "FT BGE-large-en",
        "finetuned_bge_small_both": "FT BGE-small-en",
        "finetuned_gte_small_both": "FT GTE-small",
        "finetuned_minilm_l6_both": "FT MiniLM",
        "finetuned_minilm_l6_both_v2": "FT MiniLM",
        "finetuned_minilm_l12_both": "FT MiniLM-L12",
        "finetuned_mpnet_both": "FT MPNet",
        "finetuned_multiqa_minilm_l6_both": "FT MultiQA-MiniLM-L6",
        "finetuned_paraphrase_minilm_l6_both": "FT Paraphrase-MiniLM-L6",
        "bert-base-uncased": "BERT-base",
        "distilbert-base-uncased": "DistilBERT",
        "BAAI/bge-m3": "BGE-M3",
        "BAAI/bge-large-en": "BGE-large-en",
        "BAAI/bge-small-en-v1.5": "BGE-small-en",
        "thenlper/gte-small": "GTE-small",
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
        "sentence-transformers/all-MiniLM-L12-v2": "MiniLM-L12",
        "sentence-transformers/all-mpnet-base-v2": "MPNet",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": "MultiQA-MiniLM-L6",
        "sentence-transformers/paraphrase-MiniLM-L6-v2": "Paraphrase-MiniLM-L6",
    }
    return mapping.get(model_path, mapping.get(base, base))


def safe_cross_format_std(row: pd.Series, metric: str):
    cols = [f"{metric}_mean", f"spaced_{metric}_mean", f"tabbed_{metric}_mean"]
    if not all(col in row.index for col in cols):
        return np.nan
    vals = [row[col] for col in cols]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return float(np.std(vals, ddof=0))


def build_table(root: Path, workload: str, metric: str):
    rows = []
    for summary_file in find_summary_files(root):
        df = pd.read_csv(summary_file)
        if "workload" not in df.columns:
            continue
        df = df[df["workload"] == workload].copy()
        if df.empty:
            continue

        best = df.sort_values(metric, ascending=True).iloc[0]
        rows.append(
            {
                "model_label": normalize_model_label(str(best["model_path"])),
                "model_path": best["model_path"],
                "config_name": best["config_name"],
                "accuracy_mean": best["accuracy_mean"],
                "recall_mean": best["recall_mean"],
                "precision_mean": best["precision_mean"],
                "f1_mean": best["f1_mean"],
                "auroc_mean": best["auroc_mean"],
                "p90_latency_std": best["test_p90_std"],
                "median_latency_std": best["test_median_std"],
                "accuracy_std_cross_format": safe_cross_format_std(best, "accuracy"),
                "recall_std_cross_format": safe_cross_format_std(best, "recall"),
                "precision_std_cross_format": safe_cross_format_std(best, "precision"),
                "f1_std_cross_format": safe_cross_format_std(best, "f1"),
                "total_latency_std": best["test_workload_sum_std"],
                "summary_file": str(summary_file),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("model_label").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="othertests/llmsteer_eval")
    parser.add_argument("--workload", type=str, default="all")
    parser.add_argument("--metric", type=str, default="test_workload_sum_mean")
    parser.add_argument("--output_csv", type=str, default="table5_model_comparison.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out_df = build_table(root, args.workload, args.metric)
    output_path = root / args.output_csv
    out_df.to_csv(output_path, index=False)
    print(f"Saved table to: {output_path}")
    if not out_df.empty:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
