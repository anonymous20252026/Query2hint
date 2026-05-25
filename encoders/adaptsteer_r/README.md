---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:53393
- loss:TripletLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as
    kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1,
    movie_info_idx as mii2, movie_keyword as mk, keyword as k, aka_name as an, name
    as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE
    t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id =
    mii2.movie_id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id
    = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id
    = kt.id AND (kt.kind IN ('episode','movie')) AND (t.production_year <= 2015) AND
    (t.production_year >= 1925) AND (mi1.info IN ('100','16','18','22','23','24','45','75','80','85','92','95','USA:30'))
    AND (it1.id IN ('1')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$'
    AND mii2.info::float <= 8.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND
    0.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0
    <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float
    <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id
    = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND rt.id
    = ci.role_id AND (n.gender in ('f','m')) AND (n.name_pcode_nf in ('A5351','A5354','C6416','C6432','F6325','I5263','J5325','J62','M6243','O4162','R2635','S3162'))
    AND (ci.note IS NULL) AND (rt.role in ('actor','actress')) AND (it5.id in ('25'));
  sentences:
  - Hint_0 Hint_1 Hint_14 Hint_15
  - Hint_22
  - Hint_9
- source_sentence: SELECT mi1.info, n.name, COUNT(*) FROM title as t, kind_type as
    kt, movie_info as mi1, info_type as it1, cast_info as ci, role_type as rt, name
    as n, info_type as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id =
    mi1.movie_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id
    AND pi.info_type_id = it2.id AND (it1.id IN ('5','6')) AND (it2.id IN ('17'))
    AND (mi1.info IN ('Australia:G','Germany:18','Netherlands:16','Singapore:M18','UK:15','UK:18','USA:T'))
    AND (n.name ILIKE '%gri%') AND (kt.kind IN ('tv series','video game','video movie'))
    AND (rt.role IN ('director','editor','miscellaneous crew','producer','production
    designer')) AND (t.production_year <= 2015) AND (t.production_year >= 1975) GROUP
    BY mi1.info, n.name;
  sentences:
  - Hint_1 Hint_15
  - Hint_30 Hint_44
  - Hint_27
- source_sentence: SELECT n.name, mi1.info, MIN(t.production_year), MAX(t.production_year)
    FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info
    as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id
    AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND
    ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND (it1.id IN ('3','5','6'))
    AND (mi1.info IN ('Brazil:10','Canada:14','Film-Noir','France:-16','Germany:Not
    Rated','Iceland:LH','India:A','India:U','India:UA','Ireland:15','Italy:VM18','Japan:R-15','Malaysia:18','News','Norway:12','Taiwan:GP','USA:TV-MA'))
    AND (n.name ILIKE '%thomas%') AND (kt.kind IN ('episode','movie','tv movie'))
    AND (rt.role IN ('actor','cinematographer','composer','costume designer')) GROUP
    BY mi1.info, n.name;
  sentences:
  - Hint_2 Hint_16 Hint_30 Hint_44
  - Hint_28 Hint_29
  - Hint_13 Hint_41
- source_sentence: 'SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type
    as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as
    mii1, movie_info_idx as mii2, movie_keyword as mk, keyword as k WHERE t.id = mi1.movie_id
    AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND t.id = mk.movie_id AND mii2.movie_id
    = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mk.movie_id = mi1.movie_id
    AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id =
    it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN (''episode'',''movie''))
    AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (mi1.info
    IN (''OFM:35 mm'',''PCS:Spherical'',''PFM:35 mm'',''RAT:1.33 : 1'',''RAT:1.37
    : 1'')) AND (it1.id IN (''7'')) AND it3.id = ''100'' AND it4.id = ''101'' AND
    (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii2.info::float <= 9.0) AND
    (mii2.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 4.0 <= mii2.info::float) AND
    (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 0.0 <= mii1.info::float) AND
    (mii1.info ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii1.info::float <= 1000.0);'
  sentences:
  - Hint_11 Hint_12
  - Hint_39 Hint_40
  - Hint_22
- source_sentence: SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as
    kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1,
    movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info
    as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id =
    ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id
    = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id
    AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id =
    kt.id AND (kt.kind IN ('movie')) AND (t.production_year <= 1975) AND (t.production_year
    >= 1875) AND (mi1.info IN ('English','French','German','Italian')) AND (it1.id
    IN ('103','4')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$'
    AND mii2.info::float <= 11.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND
    7.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0
    <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float
    <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id
    = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND ci.person_id
    = an.person_id AND an.person_id = pi1.person_id AND rt.id = ci.role_id AND (n.gender
    in ('m')) AND (n.name_pcode_nf in ('A4163','A4253','A5362','C6421','C6423','C6424','E6523','F6525','R1631','R1632'))
    AND (ci.note in ('(uncredited)') OR ci.note IS NULL) AND (rt.role in ('actor'))
    AND (it5.id in ('34'));
  sentences:
  - Hint_3
  - Hint_1
  - Hint_1 Hint_15 Hint_29 Hint_43
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 256 tokens
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: MPNetModel 
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
    "SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('movie')) AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (mi1.info IN ('English','French','German','Italian')) AND (it1.id IN ('103','4')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND mii2.info::float <= 11.0) AND (mii2.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND 7.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND 0.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND mii1.info::float <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND ci.person_id = an.person_id AND an.person_id = pi1.person_id AND rt.id = ci.role_id AND (n.gender in ('m')) AND (n.name_pcode_nf in ('A4163','A4253','A5362','C6421','C6423','C6424','E6523','F6525','R1631','R1632')) AND (ci.note in ('(uncredited)') OR ci.note IS NULL) AND (rt.role in ('actor')) AND (it5.id in ('34'));",
    'Hint_1',
    'Hint_3',
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


* Size: 53,393 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                            | sentence_1                                                                        | sentence_2                                                                       |
  |:--------|:--------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                                | string                                                                            | string                                                                           |
  | details | <ul><li>min: 251 tokens</li><li>mean: 255.99 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 10.69 tokens</li><li>max: 26 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 7.37 tokens</li><li>max: 26 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | sentence_1                                 | sentence_2                                 |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|:-------------------------------------------|
  | <code>SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('tv movie','video movie')) AND (t.production_year <= 2015) AND (t.production_year >= 1925) AND (mi1.info IN ('Canada','France','Germany','UK','USA')) AND (it1.id IN ('8','9')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float <= 8.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND ci.person_id = an.person_id AND an.person_id = pi1.person_id AND rt.id = ci.role_id AND (n.gender in ('m') OR n.gender IS NULL) AND (n.name_pcode_nf in ('B6563','D1316','F6523','J5252','P3625','R1635','R2631','S3151','W4125') OR n.name_pcode_nf IS NULL) AND (ci.note IS NULL) AND (rt.role in ('actor','director')) AND (it5.id in ('34'));</code> | <code>Hint_8</code>                        | <code>Hint_20</code>                       |
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k, movie_companies as mc, company_type as ct, company_name as cn WHERE t.id = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND t.id = mk.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (it1.id IN ('18')) AND (mi1.info in ('20th Century Fox Studios - 10201 Pico Blvd., Century City, Los Angeles, California, USA','Metro-Goldwyn-Mayer Studios - 10202 W. Washington Blvd., Culver City, California, USA','Stage 9, 20th Century Fox Studios - 10201 Pico Blvd., Century City, Los Angeles, California, USA')) AND (kt.kind in ('episode')) AND (rt.role in ('actress')) AND (n.gender in ('f')) AND (n.name_pcode_nf in ('A4252')) AND (t.production_year <= 2015) AND (t.production_year >= 1925) AND (cn.name in ('20th Century Fox Television','Granada Television','National Broadcasting Company (NBC)')) AND (ct.kind in ('distributors','production companies'));</code>                                                                                                                                                                                                                                                                                                                                                                                                          | <code>Hint_42</code>                       | <code>Hint_44</code>                       |
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND it1.id = '3' AND it2.id = '2' AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info IN ('Action','Adventure','Animation','Documentary','Family','Music','Romance','Sci-Fi','Short','Thriller') AND mi2.info IN ('Color') AND kt.kind IN ('tv movie') AND rt.role IN ('director') AND n.gender IN ('m') AND t.production_year <= 1997 AND 1981 < t.production_year;</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>Hint_0 Hint_1 Hint_14 Hint_15</code> | <code>Hint_4 Hint_5 Hint_18 Hint_19</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 0.5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
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
- `num_train_epochs`: 5
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
- `fp16`: False
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2996 | 500  | 0.2762        |
| 0.5992 | 1000 | 0.1769        |
| 0.8987 | 1500 | 0.1525        |
| 1.1983 | 2000 | 0.1488        |
| 1.4979 | 2500 | 0.1414        |
| 1.7975 | 3000 | 0.1311        |
| 2.0971 | 3500 | 0.1255        |
| 2.3966 | 4000 | 0.1198        |
| 2.6962 | 4500 | 0.1124        |
| 2.9958 | 5000 | 0.1045        |
| 3.2954 | 5500 | 0.1029        |
| 3.5950 | 6000 | 0.0948        |
| 3.8945 | 6500 | 0.0915        |
| 4.1941 | 7000 | 0.0889        |
| 4.4937 | 7500 | 0.085         |
| 4.7933 | 8000 | 0.0823        |


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