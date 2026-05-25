# AdaptSteer

AdaptSteer: Workload-Adaptive Optimizer Steering via
Execution-Derived Preference Supervision

 # Abstract
Database query optimizers often select suboptimal execution plans,
and effective steering requires identifying configurations that im-
prove query and workload performance without modifying DBMS
internals. Existing external steering systems can reason over mul-
tiple optimizer configurations, but many rely on online feedback,
exploration, or engine-specific steering mechanisms. More recent
LLM-based SQL steering approaches avoid deeper optimizer integra-
tion, but often depend on narrow supervision and general-purpose
embeddings that are costly, not explicitly aligned with execution
behavior, and difficult to adapt under workload shifts.
We present AdaptSteer, a novel workload-adaptive optimizer-
steering framework that learns from execution-derived preference
supervision. AdaptSteer converts latencies measured across diverse
query–configuration pairs into preference-based contrastive train-
ing signals for a compact SQL encoder, without invasive DBMS mod-
ifications or external APIs. The resulting representation supports
two lightweight decision heads: a calibrated classifier for safety-first
binary steering and a pairwise Condorcet ranker that decomposes
𝐾-way configuration selection into binary execution-preference
comparisons. We further apply Reptile-based meta-learning to the
trained encoder, enabling few-shot adaptation to unseen workloads.
Evaluations on the JOB and CEB benchmarks show that Adapt-
Steer reduces total workload latency by 70.1% relative to the Post-
greSQL default and achieves 4.4% lower total workload latency than
LLMSteer under the same binary protocol, without relying on exter-
nal API calls. Beyond binary steering, AdaptSteer improves multi-
action selection through pairwise preference ranking and benefits
from multi-configuration supervision as the action space grows.
Under CEB→JOB adaptation, AdaptSteer’s meta-learned variant
improves AUROC by 4.2 percentage points over the contrastive-
only baseline with only 20 target-workload adaptation queries.
Together, AdaptSteer unifies execution-derived preference supervi-
sion, binary and multi-action steering, and few-shot adaptation in
a compact, deployable optimizer-steering framework.
## Architecture
<img width="647" height="186" alt="image" src="https://github.com/user-attachments/assets/993bde1f-7a43-4f7b-8a3b-6ec7f238cf9b" />

---

## Repository Structure

```
AdaptSteer/
├── README.md
├── requirements.txt
├── Adaptsteer.ipynb                             Full pipeline walkthrough notebook
│
├── data_preparation.py                          Generate Reptile meta-learning triplets
├── dataset_generation.py                        Generate Stage-1 contrastive triplets
├── encoder_training.py                          Fine-tune encoders (TripletLoss / Reptile)
├── steering_pipeline.py                         AdaptSteer inference pipeline
│
├── src/                                         Core model definitions and shared utilities
│   ├── binary_classifier.py                     PyTorch binary MLP decision heads
│   ├── multiclass_classifier.py                 PyTorch multiclass MLP decision heads
│   └── pipeline_utils.py                        Shared SVC pipeline, encoder loading, metrics
│
├── training/                                    Encoder training scripts
│   ├── README.md                                Training order and quick-start commands
│   ├── contrastive_encoder_training.py          Stage 2: TripletLoss fine-tuning (main)
│   ├── contrastive_ceb_only.py                  Train contrastive encoder on CEB only
│   ├── contrastive_ceb_only_with_datasplit.py   Improved: CEB 75/25 split + multi-model
│   ├── reptile_meta_training.py                 Stage 3: Reptile meta-learning (main)
│   ├── reptile_ceb_to_job.py                    Cross-workload Reptile: CEB → JOB
│   └── reptile_ceb_to_job_final.py              Final Reptile script (produces AdaptSteer-R)
│
├── evaluation/                                  Evaluation and comparison scripts
│   ├── pipeline_comparison.py                   AdaptSteer vs LLMSteer (main comparison)
│   ├── pipeline_comparison_nopca.py             Extended: no-PCA and fixed-pipeline variants
│   ├── optimized_final_results.py               Threshold tuning + soft-voting ensemble
│   ├── structural_features_ablation.py          v2: SQL structural features + multi-encoder
│   └── pairwise_ranking_evaluation.py           Pairwise Condorcet ranking evaluation
│
├── experiments/                                 Paper experiments (§4.x)
│   ├── supervision_source_ablation.py           §4.2 — Does supervision signal drive the gain?
│   ├── encoder_quality_ablation.py              §4.2 — Encoder comparison under Condorcet
│   ├── config_landscape_ablation.py             §4.3 — Does a richer hint space help?
│   ├── multiaction_steering_multiclass.py       §4.4 — K-class multiclass SVC steering
│   ├── multiaction_steering_condorcet.py        §4.4 — K-class Condorcet pairwise ranker
│   ├── significance_wilcoxon_tests.py           §4.2 — Wilcoxon tests vs LLMSteer
│   ├── inference_latency.py                     §4.6 — Per-query CPU/GPU latency
│   ├── fewshot_cross_workload_adaptation.py     §4.5 — CEB→JOB few-shot transfer
│   ├── fewshot_nometa_ablation.py               §4.5 — Ablation: remove meta-learning
│   ├── label_efficiency_e2e.py                  §4.5 — Workload vs labeled-example budget K
│   ├── oracle_encoder_analysis.py               Appendix — Oracle vs Reptile analysis
│   ├── additional_baselines.py                  Appendix — TF-IDF and extra baselines
│   └── code_backbone_selection.py               Appendix — Code-specialized backbone
│
├── figures/                                     Figure generation
│   ├── generate_paper_figures.py                Main comparison and learning-curve figures
│   ├── generate_robustness_figures.py           SQL syntax-robustness figures
│   ├── learning_curve_publication.py            Publication-quality few-shot learning curve
│   ├── learning_curve_static.py                 Learning curve with hardcoded summary data
│   └── workload_comparison_figure.py            Workload and P90 bar chart (paper figure)
│
├── data/                                        Pre-built training datasets
│   ├── triplets_JOB.csv                         JOB contrastive triplets (3,320 rows)
│   ├── triplets_CEB.csv                         CEB contrastive triplets (71,002 rows)
│   └── README.md
│
└── encoders/                                    Trained encoder configs and tokenizers
    ├── README.md                                Loading instructions and usage examples
    ├── adaptsteer_c/                            AdaptSteer-C (contrastive)
    ├── adaptsteer_r/                            AdaptSteer-R (Reptile meta-learned)
    ├── mpnet_binary_supervised/                 MPNet-Binary (ablation)
    └── code_backbones/                          Code-specialized encoders (appendix)
        ├── codebert/
        ├── graphcodebert/
        ├── codet5_base/
        └── unixcoder/
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. GPU recommended for encoder training; CPU sufficient for evaluation.

---

## Reproducing Paper Results

```bash
# Table 2: main binary steering (supervision source ablation)
python experiments/supervision_source_ablation.py

# Table 3: multi-action steering with Condorcet ranker
python experiments/multiaction_steering_condorcet.py

# Figure 4: few-shot cross-workload adaptation
python experiments/fewshot_cross_workload_adaptation.py
```

To retrain encoders from scratch, follow `training/README.md`.

---

## Key Results (JOB + CEB, binary steering)

| Method | Workload (s) ↓ | AUROC ↑ |
|--------|---------------|---------|
| PostgreSQL Default | 8,135 | — |
| LLMSteer | 2,548 | — |
| **AdaptSteer-R** | **2,435 ± 320** | **0.820 ± 0.012** |
| AdaptSteer-C | 2,493 ± 287 | 0.810 ± 0.011 |

---

## Citation

```bibtex
@inproceedings{adaptsteer2025,
  title     = {AdaptSteer: Workload-Adaptive PostgreSQL Query Optimizer Steering
               via Execution-Derived Preference Supervision},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM International Conference on Information
               and Knowledge Management (CIKM)},
  year      = {2025}
}
```

