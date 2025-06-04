import gc
import json
import torch
import sqlparse
import argparse
import warnings
import tiktoken
import numpy as np
import pandas as pd
import torch.nn as nn
from os import cpu_count
from copy import deepcopy
from os.path import exists
from datetime import datetime
from dotenv import dotenv_values
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from openai import OpenAI, NotGiven, NOT_GIVEN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import pandas as pd
from .multiclass import MULTICLASS_RUN_MODES, ERLoss
from .binary import BINARY_RUN_MODES
from typing import Optional
from typing import Union
from typing import List, Dict

warnings.filterwarnings('ignore', category=FutureWarning)

print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ CPU only")

if torch.cuda.is_available():
    print("🚀 GPU detected — loading quantized model...")
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        "modelprep/finaltune/bert-sql-contrastive",
        # quantization_config=quant_config,
        device_map="auto"  # Automatically placed on GPU
    )
else:
    print("⚠️ No GPU detected — loading full-precision model on CPU...")
    model = AutoModel.from_pretrained("modelprep/finaltune/bert-sql-contrastive")
    model = model.to("cpu")  # ✅ Only needed on CPU

tokenizer = AutoTokenizer.from_pretrained("modelprep/finaltune/bert-sql-contrastive")

# Set device (for tensors and input handling only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to embed SQL query using your quantized model
def local_embed(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    return pooled.squeeze().tolist()


# def prepare_data(data: pd.DataFrame, filepath: str, filename: str, binary_model: bool, dims: int|NotGiven = NOT_GIVEN, augment: bool = False) -> tuple:
def prepare_data(data: pd.DataFrame, filepath: str, filename: str, binary_model: bool, dims: Union[int, NotGiven] = NOT_GIVEN, augment: bool = False) -> tuple:
    
    model_df: pd.DataFrame = data.copy()

    model_df = model_df.drop(columns=['runtime_list', 'plan_tree', 'sd_runtime'])
    model_df = model_df.explode(column='hint_list')
    model_df = model_df.sort_values(by=['filename', 'hint_list'])
    model_df = model_df.groupby(by=['filename', 'sql'], as_index=False).agg({
        'hint_list': lambda x: x.tolist(), 
        'mean_runtime': lambda x: x.tolist()
    })
    model_df['opt_l'] = model_df.mean_runtime.apply(min)
    
    if augment:
        model_df, syntaxB, syntaxC = generate_embeddings(model_df, filepath, filename, dims, augment)
    else:
        model_df = generate_embeddings(model_df, filepath, filename, dims, augment)
    model_df = model_df.drop(columns=['filename', 'sql', 'hint_list'])

    X = torch.stack(model_df.features.apply(lambda x: torch.Tensor(x)).tolist())
    hint_l = torch.stack(model_df.mean_runtime.apply(lambda x: torch.Tensor(x)).tolist())
    opt_l = torch.stack(model_df.opt_l.apply(lambda x: torch.Tensor([x]).repeat(hint_l.size(1))).tolist())
    if binary_model:
        BENCHMARK_IDX = 0
        LONGTAIL_IDX = 26
        binary_l = (hint_l[:, BENCHMARK_IDX] > hint_l[:, LONGTAIL_IDX]).type(torch.float)
    else:
        binary_l = (hint_l == opt_l).type(torch.float)
    
    if augment:
        return X, hint_l, opt_l, binary_l, syntaxB, syntaxC
    else:
        return X, hint_l, opt_l, binary_l

def generate_embeddings(data: pd.DataFrame, filepath: str, filename: str, dims: Optional[int] = None, augment: bool = False) -> pd.DataFrame:
    subset_df = data[['filename','sql']].drop_duplicates()
    embeddings = {}

    for i, row in subset_df.iterrows():
        sql = row['sql']
        fname = row['filename']
        emb = local_embed(sql)
        embeddings[fname] = {'embedding': emb}

    data['features'] = data.filename.apply(lambda x: embeddings[x]['embedding'])

    if not augment:
        return data
    else:
        syntaxB_queries = subset_df.sql.apply(lambda x: local_embed(
            sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=False)
        ))
        syntaxC_queries = subset_df.sql.apply(lambda x: local_embed(
            sqlparse.format(x, reindent=True, use_space_around_operators=True, indent_tabs=True)
        ))

        return data, torch.tensor(syntaxB_queries.tolist()), torch.tensor(syntaxC_queries.tolist())


def load_data() -> pd.DataFrame:

    job_df = pd.read_csv(
        './data/job.csv',
        converters={
            'hint_list': eval,
            'runtime_list': eval,
        }
    )
    ceb_df = pd.read_csv(
        './data/ceb.csv',
        converters={
            'hint_list': eval,
            'runtime_list': eval,
        }
    )
    data = pd.concat([job_df, ceb_df]).reset_index(drop=True)
    
    data["mean_runtime"] = data.runtime_list.apply(lambda x: np.mean(x)) # compute mean runtime for each query plan
    data["sd_runtime"] = data.runtime_list.apply(lambda x: np.std(x)) # compute sd runtime for each query plan
    data["sql"] = data.sql.apply(lambda x: x.strip('\n'))
    return data



    parser = argparse.ArgumentParser(
        prog='TrainModels',
        description='Main driver module for training deep learning models on the task of database hint steering.'
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Indicator for training binary classification or multiclass classification task.',
        required=False,
    )
    parser.add_argument(
        '--run-mode',
        action='store',
        choices=['EROnly', 'BCEOnly', 'LNOnly', 'NLNOnly', 'ALL',],
        help='The run mode determines the set of models that are evaluated.',
        required=True,
    )
    parser.add_argument(
        '--processes',
        action='store',
        help=
            'The number of CPU cores to use during training. '
            'Default will use N-1 available cores. '
            'Argument above N or equal to 0 will use N-1 CPU cores. '
            'Argument of -1 or will use all available cores.',
        required=False,
        type=int,
        default=cpu_count()-1,
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='To employ multiple CPU cores for parallel training.',
        required=False,
    )
    return parser