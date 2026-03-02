import datetime
import os
import torch

from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers import BitsAndBytesConfig
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from huggingface_hub import login
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
import random
import math


def tune_all_models(
    pairs_csv="benchmark/finetune_pairs.csv",
    output_root="finetuned_models",
    seed=42,
    epochs=3,
    lr_small=2e-5,     # for MiniLM
    lr_base=2e-5,      # for BERT/DistilBERT/BGE-m3
    lr_large=1e-5,     # for BGE-large
    max_seq_length=512
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    df = (
        pd.read_csv(pairs_csv)
        .dropna(subset=["sql", "hint_a", "hint_b"])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    assert len(df) > 0, "No rows found in pairs_csv."

    train_examples = [
        InputExample(
            texts=[f"{row['sql']} [HINT] {row['hint_a']}",
                   f"{row['sql']} [HINT] {row['hint_b']}"]
        )
        for _, row in df.iterrows()
    ]

    dev_examples = []
    if len(df) >= 100:
        dev_n = max(200, int(0.1 * len(df)))
        dev_df = df.sample(n=min(dev_n, len(df)), random_state=seed+1).reset_index(drop=True)
        for _, row in dev_df.iterrows():
            dev_examples.append(InputExample(
                texts=[f"{row['sql']} [HINT] {row['hint_a']}",
                       f"{row['sql']} [HINT] {row['hint_b']}"],
                label=1.0
            ))
        neg_df = dev_df.sample(frac=1.0, random_state=seed+2).reset_index(drop=True)
        for (_, row), (_, row_neg) in zip(dev_df.iterrows(), neg_df.iterrows()):
            if row['sql'] == row_neg['sql']:
                continue
            dev_examples.append(InputExample(
                texts=[f"{row['sql']} [HINT] {row['hint_a']}",
                       f"{row_neg['sql']} [HINT] {row_neg['hint_b']}"],
                label=0.0
            ))

    model_list = {
        "bert-base":      "bert-base-uncased",
        "distilbert":     "distilbert-base-uncased",
        "minilm":         "sentence-transformers/all-MiniLM-L6-v2",
        "bge-m3":         "BAAI/bge-m3",
        "bge-large-en":   "BAAI/bge-large-en"
    }

    per_model_hparams = {
        "bert-base":     {"batch_size": 32, "lr": lr_base},
        "distilbert":    {"batch_size": 48, "lr": lr_base},
        "minilm":        {"batch_size": 64, "lr": lr_small},
        "bge-m3":        {"batch_size": 32, "lr": lr_base},
        "bge-large-en":  {"batch_size": 16, "lr": lr_large},
    }

    def build_st_model(model_id: str) -> SentenceTransformer:
        hf_backbones = {"bert-base-uncased", "distilbert-base-uncased"}
        if model_id in hf_backbones:
            word = models.Transformer(model_id, max_seq_length=max_seq_length)
            pool = models.Pooling(
                word.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            return SentenceTransformer(modules=[word, pool])
        m = SentenceTransformer(model_id)
        if hasattr(m, "_first_module") and hasattr(m._first_module(), "max_seq_length"):
            m._first_module().max_seq_length = max_seq_length
        return m

    os.makedirs(output_root, exist_ok=True)

    for name, model_id in model_list.items():
        print(f"\n========== Training: {name} ({model_id}) ==========")
        try:
            model = build_st_model(model_id).to(device)

            batch_size = per_model_hparams[name]["batch_size"]
            lr = per_model_hparams[name]["lr"]

            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
                drop_last=True,
                pin_memory=torch.cuda.is_available()
            )
            loss_fn = losses.MultipleNegativesRankingLoss(model)

            evaluator = None
            if dev_examples:
                evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                    dev_examples, name=f"dev-{name}"
                )

            steps_per_epoch = math.ceil(len(train_examples) / batch_size)
            total_steps = steps_per_epoch * epochs
            warmup_steps = max(100, int(0.1 * total_steps))

            model.fit(
                train_objectives=[(train_dataloader, loss_fn)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": lr},
                show_progress_bar=True,
                use_amp=torch.cuda.is_available(),
                evaluator=evaluator,
                evaluation_steps=max(steps_per_epoch // 2, 50) if evaluator else 0,
                save_best_model=True if evaluator else False
            )

            save_dir = os.path.join(output_root, f"{name}_sql_contrastive")
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir)
            print(f"Saved: {save_dir}")

        except Exception as e:
            print(f"Failed {name}: {e}")

    print("\nAll fine-tuning runs finished.")


def embed_sql(query):
    # 1. Load quantized model with proper config
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModel.from_pretrained(
        "modelprep/quantizemodel/hf_bert_sql_contrastive",
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("modelprep/quantizemodel/hf_bert_sql_contrastive")

    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
        masked = last_hidden * attention_mask
        summed = masked.sum(dim=1)
        counted = attention_mask.sum(dim=1)
        mean_pooled = summed / counted

    return F.normalize(mean_pooled, p=2, dim=1).squeeze().cpu().numpy()
