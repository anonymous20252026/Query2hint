import os
import re
import sys
from venv import logger

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
# from autosteer.dp_exploration import explore_optimizer_configs
# from connectors import mysql_connector,postgres_connector
# from autosteer.query_span import run_get_query_span
import connectors.connector
from typing import Type
import pandas as pd
from openai import OpenAI
from model_fintune  import tune_all_models
benchmark_pathimdb = "benchmark/queries/job"  # Hardcoded path
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import pandas as pd


# def approx_query_span_and_run(connector, benchmark: str, query: str):
#     run_get_query_span(connector, benchmark, query)
#     connector = connector()
#     # explore_optimizer_configs(connector, f'{benchmark}/{query}')
    
# def natural_sort_key(s):
#     """Helper function for natural sorting."""
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# def extractsql():
#     input_path ='benchmark/job.csv'
#     df=pd.read_csv(input_path)
#     df_sql=df[['sql']]

#     output_path='benchmark/job_sql.csv'
#     df_sql.to_csv(output_path, index=False)

# def removeduplicate():
#     df=pd.read_csv('benchmark/job_sql.csv')
#     df_unique= df.drop_duplicates(subset=['sql'])
#     df_unique.to_csv("benchmark/job_sql.csv", index=False)
#     print(f"✅ Duplicates removed. Unique SQLs saved to: sql_unique.csv")

# def print_gpu_memory():
#     for i in range(torch.cuda.device_count()):
#         props = torch.cuda.get_device_properties(i)
#         total_mem = round(props.total_memory / 1024**2)
#         reserved = round(torch.cuda.memory_reserved(i) / 1024**2)
#         allocated = round(torch.cuda.memory_allocated(i) / 1024**2)
#         free_mem = total_mem - reserved

#         print(f"GPU {i}: {props.name}")
#         print(f"  Total Memory:    {total_mem} MB")
#         print(f"  Reserved Memory: {reserved} MB")
#         print(f"  Allocated Memory:{allocated} MB")
#         print(f"  Free Memory:     {free_mem} MB\n")


def _first_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HFEncoder:
    def __init__(self, model_name):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        return embeddings if convert_to_tensor else embeddings.numpy()


class T5Encoder:
    def __init__(self, model_name):
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        return embeddings if convert_to_tensor else embeddings.numpy()


def _load_st_encoder(candidates):
    model_ref = _first_existing_path(*candidates)
    if model_ref is None:
        model_ref = candidates[-1]
    return SentenceTransformer(model_ref)


def _parse_finetuned_family(dir_name):
    if not dir_name.startswith("finetuned_"):
        return None, None

    core = dir_name[len("finetuned_"):]
    version = 1
    m = re.search(r"_v(\d+)$", core)
    if m:
        version = int(m.group(1))
        core = re.sub(r"_v\d+$", "", core)
    core = re.sub(r"_(job|ceb|both)$", "", core)
    return core, version


def run_embedding_quality_analysis():


    sql1 = (
        "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price "
        "FROM store_sales "
        "WHERE ss_store_sk = 5 "
        "AND ss_sold_date_sk BETWEEN 2451119 AND 2451149;"
    )
    sql2 = (
        "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price "
        "FROM store_sales "
        "WHERE ss_sold_date_sk BETWEEN 2451119 AND 2451149 "
        "AND ss_store_sk = 5;"
    )
    sql3 = (
        "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price "
        "FROM store_sales "
        "WHERE ss_sold_date_sk BETWEEN 2451119 AND 2451149 "
        "AND ss_store_sk IN ("
        "SELECT s_store_sk "
        "FROM store "
        "WHERE s_state = 'CA'"
        ");"
    )


    # sql1 = (
    #     "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price "
    #     "FROM store_sales "
    #     "WHERE ss_store_sk = 5 "
    #     "AND ss_sold_date_sk BETWEEN 2451119 AND 2451149;"
    # )
    # sql2 = (
    #     "select\n"
    #     "    ss_customer_sk ,\n"
    #     "    ss_item_sk     ,\n"
    #     "    ss_quantity    ,\n"
    #     "    ss_sales_price\n"
    #     "from\n"
    #     "    store_sales\n"
    #     "where\n"
    #     "    ( ss_store_sk = 5 )\n"
    #     "    and ( ss_sold_date_sk >= 2451119 and ss_sold_date_sk <= 2451149 );"
    # )
    # sql3 = (
    #     "SELECT\n"
    #     "    c.c_customer_id,\n"
    #     "    c.c_first_name,\n"
    #     "    c.c_last_name,\n"
    #     "    SUM(ss.ss_net_paid) AS total_paid\n"
    #     "FROM store_sales AS ss\n"
    #     "JOIN customer AS c\n"
    #     "  ON ss.ss_customer_sk = c.c_customer_sk\n"
    #     "WHERE ss.ss_sold_date_sk BETWEEN 2451119 AND 2451149\n"
    #     "GROUP BY\n"
    #     "    c.c_customer_id,\n"
    #     "    c.c_first_name,\n"
    #     "    c.c_last_name;"
    # )

    base_model_ids = {
        "bert_base": "bert-base-uncased",
        "bge_large": "BAAI/bge-large-en",
        "bge_m3": "BAAI/bge-m3",
        "bge_small": "BAAI/bge-small-en-v1.5",
        "gte_small": "thenlper/gte-small",
        "minilm_l12": "sentence-transformers/all-MiniLM-L12-v2",
        "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "multiqa_minilm_l6": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "paraphrase_minilm_l6": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    }

    model_roots = ["all_models", "stage1_added_models"]
    selected_finetuned = {}
    for root in model_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            full_path = os.path.join(root, name)
            if os.path.isdir(full_path) and name.startswith("finetuned_"):
                family, version = _parse_finetuned_family(name)
                if not family:
                    continue
                prev = selected_finetuned.get(family)
                if prev is None or version > prev["version"]:
                    selected_finetuned[family] = {
                        "version": version,
                        "path": full_path,
                        "dir_name": name,
                    }

    results = []

    for family in sorted(selected_finetuned):
        if family not in base_model_ids:
            print(f"Skipped {family}: no base-model mapping defined")
            continue

        base_model_id = base_model_ids[family]
        finetuned_info = selected_finetuned[family]
        finetuned_path = finetuned_info["path"]
        finetuned_label = finetuned_info["dir_name"]

        try:
            base_model = SentenceTransformer(base_model_id)
            emb1 = base_model.encode(sql1, convert_to_tensor=True)
            emb2 = base_model.encode(sql2, convert_to_tensor=True)
            emb3 = base_model.encode(sql3, convert_to_tensor=True)
            results.append(
                {
                    "family": family,
                    "variant": "base",
                    "model": family,
                    "model_ref": base_model_id,
                    "sim_sql1_sql2": util.cos_sim(emb1, emb2).item(),
                    "sim_sql1_sql3": util.cos_sim(emb1, emb3).item(),
                }
            )
        except Exception as exc:
            print(f"Skipped base {family}: {exc}")
            continue

        try:
            finetuned_model = SentenceTransformer(finetuned_path)
            emb1 = finetuned_model.encode(sql1, convert_to_tensor=True)
            emb2 = finetuned_model.encode(sql2, convert_to_tensor=True)
            emb3 = finetuned_model.encode(sql3, convert_to_tensor=True)
            results.append(
                {
                    "family": family,
                    "variant": "fine_tuned",
                    "model": finetuned_label,
                    "model_ref": finetuned_path,
                    "sim_sql1_sql2": util.cos_sim(emb1, emb2).item(),
                    "sim_sql1_sql3": util.cos_sim(emb1, emb3).item(),
                }
            )
        except Exception as exc:
            print(f"Skipped fine-tuned {finetuned_label}: {exc}")

    out_df = pd.DataFrame(results).sort_values(["family", "variant"]).reset_index(drop=True)
    out_csv = "embedding_quality_base_vs_finetuned.csv"
    out_df.to_csv(out_csv, index=False)

    print("\nEmbedding quality analysis (base vs fine-tuned)")
    print(f"{'Model':40s} {'Variant':12s} {'Sim(sql1,sql2)':>14s} {'Sim(sql1,sql3)':>14s}")
    for row in out_df.itertuples(index=False):
        print(f"{row.model:40s} {row.variant:12s} {row.sim_sql1_sql2:14.4f} {row.sim_sql1_sql3:14.4f}")
    print(f"\nSaved similarity results to: {out_csv}")

    return out_df


if __name__ == '__main__':
    # ConnectorType = mysql_connector.MySqlConnector
    # ConnectorType = postgres_connector.PostgresConnector
    # connector=ConnectorType()
    # storage.TESTED_DATABASE = ConnectorType.get_name()
   
   
    # if not os.path.isdir(benchmark_pathimdb):
    #     logger.fatal('Cannot access the benchmark directory containing the sql files with path=%s', benchmark_pathimdb)
    #     sys.exit(1)

    # storage.BENCHMARK_ID = storage.register_benchmark(benchmark_pathimdb)
    # # Check which DB is connected
    # db_result = connector.execute("SELECT DATABASE();")
    # print("Connected to DB:", db_result.result[0][0])
    print('\n')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('\t Query Span Approximation')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    # approx_query_span_and_run(ConnectorType, benchmark_path, '1.sql')
    # logger.info('LLMSteer Data Set Generation')
    # queries = sorted(list(filter(lambda q: q.endswith('.sql'), os.listdir(benchmark_pathimdb))),key=natural_sort_key)
    # logger.info('Found the following SQL files: %s', queries)
    # approx_query_span_and_run(ConnectorType, benchmark_pathimdb, "1.sql")
    # for query in queries:
    #     logger.info('run Q%s...', query)
    #     approx_query_span_and_run(ConnectorType, benchmark_pathimdb, query)
    
    # ============================Finetune and Quantization of the model===================================================
    # extractsql()
    # removeduplicate()
    # tunebert()
    # quantizebert()
    # print_gpu_memory()
    # tune_bge()
    # tuneallmodels()
    # fine_tune_bge_m3()
    # fine_tune_bge_large_en()

    # tune_all_models()
    run_embedding_quality_analysis()
