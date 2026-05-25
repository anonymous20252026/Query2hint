---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8000
- loss:TripletLoss
base_model: microsoft/codebert-base
widget:
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as
    it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci,
    role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id
    AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id
    = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id
    AND mi2.info_type_id = it2.id AND (it1.id in ('6')) AND (it2.id in ('7')) AND
    t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info
    in ('Dolby','Mono','Stereo')) AND (mi2.info in ('LAB:Consolidated Film Industries
    (CFI), Hollywood (CA), USA','OFM:Video','PCS:(anamorphic)','PCS:CinemaScope','PCS:Panavision','PCS:Spherical','PCS:Techniscope','RAT:4:3'))
    AND (kt.kind in ('episode','movie','tv movie')) AND (rt.role in ('composer'))
    AND (n.gender IS NULL) AND (t.production_year <= 1990) AND (t.production_year
    >= 1950);
  sentences:
  - Hint_37
  - Hint_6 Hint_20 Hint_34 Hint_48
  - Hint_3 Hint_17 Hint_31 Hint_45
- source_sentence: SELECT n.name, mi1.info, MIN(t.production_year), MAX(t.production_year)
    FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info
    as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id
    AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND
    ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND (it1.id IN ('3','4','6'))
    AND (mi1.info IN ('Bulgarian','Catalan','Czech','Dolby Stereo','Malayalam','Oriya','Serbo-Croatian','Sonics-DDP','Tamil','Ultra
    Stereo','Vietnamese','Yiddish')) AND (n.name ILIKE '%min%') AND (kt.kind IN ('episode','movie','tv
    movie')) AND (rt.role IN ('director','editor','miscellaneous crew','producer','production
    designer')) GROUP BY mi1.info, n.name;
  sentences:
  - Hint_14
  - Hint_6 Hint_20 Hint_34 Hint_48
  - Hint_42
- source_sentence: SELECT n.name, mi1.info, MIN(t.production_year), MAX(t.production_year)
    FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info
    as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id
    AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND
    ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND (it1.id IN ('2','5')) AND
    (mi1.info IN ('Australia:MA15+','Brazil:12','Finland:K-13','Iceland:14','Italy:VM14','Japan:G','Netherlands:12','Netherlands:6','Norway:11','Philippines:PG-13','Philippines:R-13','Singapore:PG13','South
    Korea:12','South Korea:15','Sweden:Btl','UK:X','USA:TV-MA')) AND (n.name ILIKE
    '%ren%') AND (kt.kind IN ('episode','movie','video movie')) AND (rt.role IN ('actress','costume
    designer','director','miscellaneous crew','producer')) GROUP BY mi1.info, n.name;
  sentences:
  - Hint_22
  - Hint_7 Hint_8 Hint_11 Hint_12 Hint_21 Hint_22 Hint_25 Hint_26
  - Hint_39 Hint_40
- source_sentence: SELECT n.gender, rt.role, cn.name, COUNT(*) FROM title as t, movie_companies
    as mc, company_name as cn, company_type as ct, kind_type as kt, cast_info as ci,
    name as n, role_type as rt, movie_info as mi1, info_type as it1, person_info as
    pi, info_type as it2 WHERE t.id = mc.movie_id AND t.id = ci.movie_id AND t.id
    = mi1.movie_id AND mi1.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND
    cn.id = mc.company_id AND ct.id = mc.company_type_id AND kt.id = t.kind_id AND
    ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info_type_id = it1.id AND n.id
    = pi.person_id AND pi.info_type_id = it2.id AND ci.person_id = pi.person_id AND
    (kt.kind IN ('episode','tv movie','tv series','video game','video movie')) AND
    (rt.role IN ('actor','actress','cinematographer','composer','costume designer','guest','miscellaneous
    crew','producer','production designer','writer')) AND (t.production_year <= 1990)
    AND (t.production_year >= 1945) AND (it1.id IN ('8')) AND (mi1.info ILIKE '%us%')
    AND (pi.info ILIKE '%lo%') AND (it2.id IN ('39')) GROUP BY n.gender, rt.role,
    cn.name ORDER BY COUNT(*) DESC;
  sentences:
  - Hint_0 Hint_1 Hint_14 Hint_15
  - Hint_37
  - Hint_13
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as
    it1, movie_info as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword
    as mk, keyword as k, movie_companies as mc, company_type as ct, company_name as
    cn WHERE t.id = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND
    t.id = mk.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND
    k.id = mk.keyword_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.role_id = rt.id AND (it1.id IN ('18')) AND (mi1.info in ('CBS Studio
    50, New York City, New York, USA','Desilu Studios - 9336 W. Washington Blvd.,
    Culver City, California, USA','New York City, New York, USA','Paramount Studios
    - 5555 Melrose Avenue, Hollywood, Los Angeles, California, USA','Stage 9, 20th
    Century Fox Studios - 10201 Pico Blvd., Century City, Los Angeles, California,
    USA','Warner Brothers Burbank Studios - 4000 Warner Boulevard, Burbank, California,
    USA')) AND (kt.kind in ('episode')) AND (rt.role in ('actress','miscellaneous
    crew','producer','production designer')) AND (n.gender in ('f') OR n.gender IS
    NULL) AND (n.surname_pcode in ('B634','D12','D5','F65','H62','J525','M45','P62','R2','R3','T46','T521','W425'))
    AND (t.production_year <= 2015) AND (t.production_year >= 1925) AND (cn.name in
    ('20th Century Fox Television','American Broadcasting Company (ABC)','British
    Broadcasting Corporation (BBC)','Columbia Broadcasting System (CBS)','Warner Bros.
    Television','Warner Home Video')) AND (ct.kind in ('distributors','production
    companies'));
  sentences:
  - Hint_3 Hint_17 Hint_31 Hint_45
  - Hint_6 Hint_20 Hint_34 Hint_48
  - Hint_1 Hint_15
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on microsoft/codebert-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) <!-- at revision 3b0952feddeffad0063f274080e3c23d75e7eb39 -->
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
    "SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k, movie_companies as mc, company_type as ct, company_name as cn WHERE t.id = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND t.id = mk.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (it1.id IN ('18')) AND (mi1.info in ('CBS Studio 50, New York City, New York, USA','Desilu Studios - 9336 W. Washington Blvd., Culver City, California, USA','New York City, New York, USA','Paramount Studios - 5555 Melrose Avenue, Hollywood, Los Angeles, California, USA','Stage 9, 20th Century Fox Studios - 10201 Pico Blvd., Century City, Los Angeles, California, USA','Warner Brothers Burbank Studios - 4000 Warner Boulevard, Burbank, California, USA')) AND (kt.kind in ('episode')) AND (rt.role in ('actress','miscellaneous crew','producer','production designer')) AND (n.gender in ('f') OR n.gender IS NULL) AND (n.surname_pcode in ('B634','D12','D5','F65','H62','J525','M45','P62','R2','R3','T46','T521','W425')) AND (t.production_year <= 2015) AND (t.production_year >= 1925) AND (cn.name in ('20th Century Fox Television','American Broadcasting Company (ABC)','British Broadcasting Corporation (BBC)','Columbia Broadcasting System (CBS)','Warner Bros. Television','Warner Home Video')) AND (ct.kind in ('distributors','production companies'));",
    'Hint_6 Hint_20 Hint_34 Hint_48',
    'Hint_1 Hint_15',
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
  |         | sentence_0                                                                            | sentence_1                                                                        | sentence_2                                                                       |
  |:--------|:--------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                                | string                                                                            | string                                                                           |
  | details | <ul><li>min: 157 tokens</li><li>mean: 384.83 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 13.84 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 9.19 tokens</li><li>max: 34 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                  | sentence_2                                  |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:--------------------------------------------|
  | <code>SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, movie_keyword as mk, keyword as k WHERE t.id = mi1.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND t.id = mk.movie_id AND mii2.movie_id = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mk.movie_id = mi1.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('episode','movie')) AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (mi1.info IN ('OFM:35 mm','PCS:Spherical','PFM:35 mm','RAT:1.33 : 1','RAT:1.37 : 1')) AND (it1.id IN ('7')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float <= 9.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 4.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float <= 1000.0);</code> | <code>Hint_11 Hint_12</code>                | <code>Hint_7 Hint_8</code>                  |
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND (it1.id in ('7')) AND (it2.id in ('3')) AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('CAM:Canon 7D','OFM:Video','PCS:Panavision','PCS:Spherical','PCS:Super 35','PFM:Video','RAT:1.37 : 1','RAT:2.35 : 1')) AND (mi2.info in ('Action','Fantasy','History','Music','Short','Sport','Thriller','War','Western')) AND (kt.kind in ('tv movie','tv series','video game')) AND (rt.role in ('miscellaneous crew','writer')) AND (n.gender IS NULL) AND (t.production_year <= 2015) AND (t.production_year >= 1925);</code>                                                                                                                                               | <code>Hint_2 Hint_16 Hint_30 Hint_44</code> | <code>Hint_6 Hint_20 Hint_34 Hint_48</code> |
  | <code>SELECT mi1.info, pi.info, COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info as ci, role_type as rt, name as n, info_type as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id AND pi.info_type_id = it2.id AND (it1.id IN ('18')) AND (it2.id IN ('21')) AND (mi1.info ILIKE '%us%') AND (pi.info ILIKE '%1950%') AND (kt.kind IN ('episode','movie','tv mini series','tv movie','tv series','video game','video movie')) AND (rt.role IN ('costume designer','director','editor','production designer')) GROUP BY mi1.info, pi.info;</code>                                                                                                                                                                                                                                                                                                                                                                               | <code>Hint_2 Hint_16</code>                 | <code>Hint_41</code>                        |
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
| 1.0   | 500  | 4.3078        |


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