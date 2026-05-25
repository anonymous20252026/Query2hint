---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8000
- loss:TripletLoss
base_model: microsoft/graphcodebert-base
widget:
- source_sentence: SELECT t.title, n.name, cn.name, COUNT(*) FROM title as t, movie_keyword
    as mk, keyword as k, movie_companies as mc, company_name as cn, company_type as
    ct, kind_type as kt, cast_info as ci, name as n, role_type as rt WHERE t.id =
    mk.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id
    AND ci.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND k.id = mk.keyword_id
    AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND kt.id = t.kind_id
    AND ci.person_id = n.id AND ci.role_id = rt.id AND (t.title ILIKE '%200%') AND
    (n.surname_pcode ILIKE '%y%') AND (cn.name ILIKE '%news%') AND (kt.kind IN ('episode','movie','tv
    movie','tv series','video game','video movie')) AND (rt.role IN ('cinematographer','miscellaneous
    crew','producer')) GROUP BY t.title, n.name, cn.name ORDER BY COUNT(*) DESC;
  sentences:
  - Hint_6 Hint_20 Hint_34 Hint_48
  - Hint_3 Hint_17 Hint_31 Hint_45
  - Hint_1 Hint_15 Hint_29 Hint_43
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as
    it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci,
    role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id
    AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id
    = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id
    AND mi2.info_type_id = it2.id AND (it1.id in ('105')) AND (it2.id in ('8')) AND
    t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info
    in ('$1,000,000','$2,000','$500')) AND (mi2.info in ('USA')) AND (kt.kind in ('tv
    movie','tv series','video game')) AND (rt.role in ('producer')) AND (n.gender
    in ('m')) AND (t.production_year <= 2015) AND (t.production_year >= 1975) AND
    (k.keyword IN ('anal-sex','character-name-in-title','family-relationships','father-son-relationship','female-nudity','flashback','friendship','hardcore','homosexual','husband-wife-relationship','interview','lesbian-sex','nudity','oral-sex','sex'));
  sentences:
  - Hint_46 Hint_47
  - Hint_6 Hint_20 Hint_34 Hint_48
  - Hint_11 Hint_12
- source_sentence: 'SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type
    as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as
    mii1, movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info
    as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id =
    ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id
    = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id
    AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id =
    kt.id AND (kt.kind IN (''tv movie'',''video movie'')) AND (t.production_year <=
    2015) AND (t.production_year >= 1925) AND (mi1.info IN (''OFM:35 mm'',''PFM:Video'',''RAT:1.33
    : 1'')) AND (it1.id IN (''7'')) AND it3.id = ''100'' AND it4.id = ''101'' AND
    (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii2.info::float <= 7.0) AND
    (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 3.0 <= mii2.info::float) AND
    (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 0.0 <= mii1.info::float) AND
    (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii1.info::float <= 10000.0)
    AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id
    AND n.id = pi1.person_id AND n.id = an.person_id AND ci.person_id = an.person_id
    AND an.person_id = pi1.person_id AND rt.id = ci.role_id AND (n.gender in (''f'',''m''))
    AND (n.name_pcode_nf in (''A4163'',''A4253'',''A5362'',''C6231'',''C6235'',''E4213'',''F6521'',''F6525'',''K6235'',''R2632'',''S3152'')
    OR n.name_pcode_nf IS NULL) AND (ci.note in (''(archive footage)'') OR ci.note
    IS NULL) AND (rt.role in (''actor'',''actress'')) AND (it5.id in (''31''));'
  sentences:
  - Hint_2 Hint_16 Hint_30 Hint_44
  - Hint_34
  - Hint_16
- source_sentence: 'SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type
    as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as
    mii1, movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info
    as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id =
    ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id
    = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id
    AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id =
    kt.id AND (kt.kind IN (''episode'',''movie'')) AND (t.production_year <= 2015)
    AND (t.production_year >= 1925) AND (mi1.info IN (''CAM:Panavision Cameras and
    Lenses'',''LAB:Technicolor'',''MET:'',''OFM:16 mm'',''PCS:Digital Intermediate'',''PCS:Super
    35'',''PFM:16 mm'',''RAT:1.66 : 1'')) AND (it1.id IN (''12'',''17'',''7'')) AND
    it3.id = ''100'' AND it4.id = ''101'' AND (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$''
    AND mii2.info::float <= 5.0) AND (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$''
    AND 2.0 <= mii2.info::float) AND (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$''
    AND 0.0 <= mii1.info::float) AND (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$''
    AND mii1.info::float <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id
    AND it5.id = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id
    AND ci.person_id = an.person_id AND an.person_id = pi1.person_id AND rt.id = ci.role_id
    AND (n.gender in (''f'',''m'')) AND (n.name_pcode_nf in (''A5164'',''B3625'',''B6352'',''C6232'',''C6436'',''D5241'',''F4126'',''F6536'',''G6342'',''L2524'',''M232'',''P4526'',''R5431'',''S5265'',''T5236''))
    AND (ci.note in (''(voice)'') OR ci.note IS NULL) AND (rt.role in (''actor'',''actress''))
    AND (it5.id in (''37''));'
  sentences:
  - Hint_27
  - Hint_29
  - Hint_34
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as
    mi1, info_type as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type
    as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND it1.id = '3' AND it2.id = '2' AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.role_id = rt.id AND mi1.info IN ('Animation','Documentary','Horror','Music','Sci-Fi','Thriller')
    AND mi2.info IN ('Color') AND kt.kind IN ('episode','movie','video movie') AND
    rt.role IN ('director','production designer') AND n.gender IN ('f') AND t.production_year
    <= 2010 AND 2000 < t.production_year;
  sentences:
  - Hint_34
  - Hint_2 Hint_16 Hint_30 Hint_44
  - Hint_9
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on microsoft/graphcodebert-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base) <!-- at revision 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND it1.id = '3' AND it2.id = '2' AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info IN ('Animation','Documentary','Horror','Music','Sci-Fi','Thriller') AND mi2.info IN ('Color') AND kt.kind IN ('episode','movie','video movie') AND rt.role IN ('director','production designer') AND n.gender IN ('f') AND t.production_year <= 2010 AND 2000 < t.production_year;",
    'Hint_2 Hint_16 Hint_30 Hint_44',
    'Hint_9',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 8,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                            | sentence_1                                                                       | sentence_2                                                                       |
  |:--------|:--------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                                | string                                                                           | string                                                                           |
  | details | <ul><li>min: 158 tokens</li><li>mean: 384.38 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 13.5 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 9.25 tokens</li><li>max: 34 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_1                                  | sentence_2                                   |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:---------------------------------------------|
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k, movie_companies as mc, company_type as ct, company_name as cn WHERE t.id = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND t.id = mk.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (it1.id IN ('3')) AND (mi1.info in ('Adventure','Fantasy','Music','Musical','Mystery','Sci-Fi','War')) AND (kt.kind in ('movie')) AND (rt.role in ('actress','cinematographer','miscellaneous crew')) AND (n.gender in ('f') OR n.gender IS NULL) AND (n.surname_pcode in ('L','R2','S53') OR n.surname_pcode IS NULL) AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (cn.name in ('Metro-Goldwyn-Mayer (MGM)','Paramount Pictures','Warner Home Video')) AND (ct.kind in ('distributors'));</code> | <code>Hint_43</code>                        | <code>Hint_1</code>                          |
  | <code>SELECT n.gender, rt.role, cn.name, COUNT(*) FROM title as t, movie_companies as mc, company_name as cn, company_type as ct, kind_type as kt, cast_info as ci, name as n, role_type as rt, movie_info as mi1, info_type as it WHERE t.id = mc.movie_id AND t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND kt.id = t.kind_id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info_type_id = it.id AND (kt.kind ILIKE '%m%') AND (rt.role IN ('actor','cinematographer','costume designer','director','editor','miscellaneous crew','producer','production designer','writer')) AND (t.production_year <= 1990) AND (t.production_year >= 1945) AND (it.id IN ('2')) AND (mi1.info ILIKE '%col%') AND (cn.name ILIKE '%broa%') GROUP BY n.gender, rt.role, cn.name ORDER BY COUNT(*) DESC;</code>                                                                                                                           | <code>Hint_1 Hint_15</code>                 | <code>Hint_23</code>                         |
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND it1.id = '8' AND it2.id = '4' AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info IN ('Belgium','Canada','China','Czechoslovakia','France','India','Iran','Japan','Romania','Soviet Union','Sweden') AND mi2.info IN ('Czech','Dutch','English','French','Georgian','Mandarin','Persian','Romanian','Swedish','Tamil') AND kt.kind IN ('episode','movie','tv movie') AND rt.role IN ('cinematographer','writer') AND n.gender IN ('f') AND t.production_year <= 1990 AND 1950 < t.production_year;</code>                                                                                                                                                              | <code>Hint_2 Hint_16 Hint_30 Hint_44</code> | <code>Hint_32 Hint_33 Hint_46 Hint_47</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 1.0   | 500  | 4.3515        |


### Framework Versions
- Python: 3.10.19
- Sentence Transformers: 3.0.1
- Transformers: 4.44.2
- PyTorch: 2.6.0+cu124
- Accelerate: 0.33.0
- Datasets: 3.0.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification}, 
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->