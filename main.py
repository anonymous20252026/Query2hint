import os
import re
import sys
from venv import logger

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from autosteer.dp_exploration import explore_optimizer_configs
from connectors import mysql_connector,postgres_connector
from autosteer.query_span import run_get_query_span
import connectors.connector
from typing import Type
import pandas as pd
import storage
from openai import OpenAI
from modelprep.modeltuning  import fine_tune_bge_large_en, fine_tune_bge_m3, fine_tune_tinyllama_dpo, tuneallmodels, tunebert,quantizebert,embed_sql,tune_bge
benchmark_path = "benchmark/queries/tpch"
benchmark_pathimdb = "benchmark/queries/job"  # Hardcoded path
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import pandas as pd


def approx_query_span_and_run(connector, benchmark: str, query: str):
    run_get_query_span(connector, benchmark, query)
    connector = connector()
    # explore_optimizer_configs(connector, f'{benchmark}/{query}')
    
def natural_sort_key(s):
    """Helper function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extractsql():
    input_path ='benchmark/job.csv'
    df=pd.read_csv(input_path)
    df_sql=df[['sql']]

    output_path='benchmark/job_sql.csv'
    df_sql.to_csv(output_path, index=False)

def removeduplicate():
    df=pd.read_csv('benchmark/job_sql.csv')
    df_unique= df.drop_duplicates(subset=['sql'])
    df_unique.to_csv("benchmark/job_sql.csv", index=False)
    print(f"✅ Duplicates removed. Unique SQLs saved to: sql_unique.csv")

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = round(props.total_memory / 1024**2)
        reserved = round(torch.cuda.memory_reserved(i) / 1024**2)
        allocated = round(torch.cuda.memory_allocated(i) / 1024**2)
        free_mem = total_mem - reserved

        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory:    {total_mem} MB")
        print(f"  Reserved Memory: {reserved} MB")
        print(f"  Allocated Memory:{allocated} MB")
        print(f"  Free Memory:     {free_mem} MB\n")


if __name__ == '__main__':
    ConnectorType = mysql_connector.MySqlConnector
    ConnectorType = postgres_connector.PostgresConnector
    connector=ConnectorType()
    storage.TESTED_DATABASE = ConnectorType.get_name()
   
   
    if not os.path.isdir(benchmark_pathimdb):
        logger.fatal('Cannot access the benchmark directory containing the sql files with path=%s', benchmark_pathimdb)
        sys.exit(1)

    storage.BENCHMARK_ID = storage.register_benchmark(benchmark_pathimdb)
    # Check which DB is connected
    db_result = connector.execute("SELECT DATABASE();")
    print("Connected to DB:", db_result.result[0][0])
    print('\n')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('\t Query Span Approximation')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    # approx_query_span_and_run(ConnectorType, benchmark_path, '1.sql')
    logger.info('LLMSteer Data Set Generation')
    queries = sorted(list(filter(lambda q: q.endswith('.sql'), os.listdir(benchmark_pathimdb))),key=natural_sort_key)
    logger.info('Found the following SQL files: %s', queries)
    approx_query_span_and_run(ConnectorType, benchmark_pathimdb, "1.sql")
    for query in queries:
        logger.info('run Q%s...', query)
        approx_query_span_and_run(ConnectorType, benchmark_pathimdb, query)
    
    # ============================Finetune and Quantization of the model===================================================
    extractsql()
    removeduplicate()
    tunebert()
    quantizebert()
    print_gpu_memory()
    tune_bge()
    tuneallmodels()
    fine_tune_bge_m3()
    fine_tune_bge_large_en()
    
# Load normal BERT model
normal_model = SentenceTransformer("bert-base-uncased")  # or any base model

# Load finetuned BAAI/bge-m3 model
Baii_model = SentenceTransformer("tuned_models/bge-m3-sql-contrastive")  # or any base model

# Load  BAAI/bge-m3 model
Baii_Nmodel = SentenceTransformer("BAAI/bge-m3")  # or any base model

# Load finetuned BAAI/bge-m3 model
Bge_largemodel_finetune = SentenceTransformer("tuned_models/bge-large-en-sql-contrastive")  # or any base model

# Load  BAAI/bge-m3 model
Bge_largemodel = SentenceTransformer("BAAI/bge-large-en")

# Load your fine-tuned model
finetuned_model = SentenceTransformer("modelprep/finaltune/bert-sql-contrastive")
# OpenAI Model
opeaimodel = SentenceTransformer('BAAI/bge-large-en') 
# DistilBert-base-uncased
Distillbert_model = SentenceTransformer("models_finetuned/distilbert_sql_contrastive")  # or any base model

# MinilmBERT
minilm_model = SentenceTransformer("models_finetuned/minilm_sql_contrastive")  # or any base model

normal_minilm_model = SentenceTransformer("all-MiniLM-L6-v2")  # or any base model
normal_distilbert_model = SentenceTransformer("distilbert-base-uncased")  # or any base model

# Same queries
sql1 = "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price FROM store_sales WHERE ss_store_sk = 5 AND ss_sold_date_sk BETWEEN 2451119 AND 2451149;"
sql2 = "SELECT ss_customer_sk, ss_item_sk, ss_quantity, ss_sales_price FROM store_sales WHERE ss_store_sk = 5 AND ss_sold_date_sk BETWEEN 2451119 AND 2451149;"
sql3 = "SELECT c_customer_id, c_first_name, c_last_name, SUM(ss_net_paid) AS total_paid FROM store_sales JOIN customer ON ss_customer_sk = c_customer_sk WHERE ss_sold_date_sk BETWEEN 2451119 AND 2451149 GROUP BY c_customer_id, c_first_name, c_last_name;"

#Encode using both models
emb1_normal = normal_model.encode(sql1, convert_to_tensor=True)
emb2_normal = normal_model.encode(sql2, convert_to_tensor=True)
emb3_normal = normal_model.encode(sql3, convert_to_tensor=True)

emb1_finetuned = finetuned_model.encode(sql1, convert_to_tensor=True)
emb2_finetuned = finetuned_model.encode(sql2, convert_to_tensor=True)
emb3_finetuned = finetuned_model.encode(sql3, convert_to_tensor=True)

#Get embeddings
emb1_openai = opeaimodel.encode(sql1)
emb2_openai = opeaimodel.encode(sql2)
emb3_openai = opeaimodel.encode(sql3)

# Finetune DiltillBert
emb1_Dbert = Distillbert_model.encode(sql1)
emb2_Dbert = Distillbert_model.encode(sql2)
emb3_Dbert = Distillbert_model.encode(sql3)

# normal MinilmBERT
emb1_Nminilm = normal_minilm_model.encode(sql1)
emb2_Nminilm = normal_minilm_model.encode(sql2)
emb3_Nminilm = normal_minilm_model.encode(sql3)

#Normal DistillBert
emb1_NDbert = normal_distilbert_model.encode(sql1)
emb2_NDbert = normal_distilbert_model.encode(sql2)
emb3_NDbert = normal_distilbert_model.encode(sql3)

#Finetuned minilmBERT
emb1_minilm = minilm_model.encode(sql1)
emb2_minilm = minilm_model.encode(sql2)
emb3_minilm = minilm_model.encode(sql3)

#bai bge m3 model
emb1_Baii = Baii_model.encode(sql1)
emb2_Baii = Baii_model.encode(sql2)
emb3_Baii = Baii_model.encode(sql3)

#bai bge m3 model
emb1_NBaii = Baii_Nmodel.encode(sql1)
emb2_NBaii = Baii_Nmodel.encode(sql2)
emb3_NBaii = Baii_Nmodel.encode(sql3)


emb1_LNBaii = Bge_largemodel.encode(sql1)
emb2_LNBaii = Bge_largemodel.encode(sql2)
emb3_LNBaii = Bge_largemodel.encode(sql3)

emb1_FLNBaii = Bge_largemodel_finetune.encode(sql1)
emb2_FLNBaii = Bge_largemodel_finetune.encode(sql2)
emb3_FLNBaii = Bge_largemodel_finetune.encode(sql3)

# Compare similarities
print("Normal BERT - Similarity sql1 vs sql2:", util.cos_sim(emb1_normal, emb2_normal).item())
print("Normal BERT - Similarity sql1 vs sql3:", util.cos_sim(emb1_normal, emb3_normal).item())

print("Fine-tuned BERT - Similarity sql1 vs sql2:", util.cos_sim(emb1_finetuned, emb2_finetuned).item())
print("Fine-tuned BERT - Similarity sql1 vs sql3:", util.cos_sim(emb1_finetuned, emb3_finetuned).item())

# # Compare
# print("OpenAI - Similarity sql1 vs sql2:", util.cos_sim(emb1_openai, emb2_openai).item())
# print("OpenAI - Similarity sql1 vs sql3:", util.cos_sim(emb1_openai, emb3_openai).item())

# DestillBert
print("NormalDistillbert - Similarity sql1 vs sql2:", util.cos_sim(emb1_NDbert, emb2_NDbert).item())
print("NormalDistillbert - Similarity sql1 vs sql3:", util.cos_sim(emb1_NDbert, emb3_NDbert).item())

print("FintunedDistillbert - Similarity sql1 vs sql2:", util.cos_sim(emb1_Dbert, emb2_Dbert).item())
print("FintunedDistillbert - Similarity sql1 vs sql3:", util.cos_sim(emb1_Dbert, emb3_Dbert).item())


# MinilmBERT
print("NormalMinilm - Similarity sql1 vs sql2:", util.cos_sim(emb1_Nminilm, emb2_Nminilm).item())
print("NormalMinilm - Similarity sql1 vs sql3:", util.cos_sim(emb1_Nminilm, emb3_Nminilm).item())

print("FintunedMinilm - Similarity sql1 vs sql2:", util.cos_sim(emb1_minilm, emb2_minilm).item())
print("FintunedMinilm - Similarity sql1 vs sql3:", util.cos_sim(emb1_minilm, emb3_minilm).item())


# Baii BGE M3
print("Baii BGE M3 - Similarity sql1 vs sql2:", util.cos_sim(emb1_Baii, emb2_Baii).item())
print("Baii BGE M3 - Similarity sql1 vs sql3:", util.cos_sim(emb1_Baii, emb3_Baii).item())

print("Baii Normal BGE M3 - Similarity sql1 vs sql2:", util.cos_sim(emb1_NBaii, emb2_NBaii).item())
print("Baii Normal BGE M3 - Similarity sql1 vs sql3:", util.cos_sim(emb1_NBaii, emb3_NBaii).item())

print("Baii BGE Large - Similarity sql1 vs sql2:", util.cos_sim(emb1_LNBaii, emb2_LNBaii).item())
print("Baii BGE Large - Similarity sql1 vs sql3:", util.cos_sim(emb1_LNBaii, emb3_LNBaii).item())

print("Baii BGE Large Finetuned - Similarity sql1 vs sql2:", util.cos_sim(emb1_FLNBaii, emb2_FLNBaii).item())
print("Baii BGE Large Finetuned - Similarity sql1 vs sql3:", util.cos_sim(emb1_FLNBaii, emb3_FLNBaii).item())