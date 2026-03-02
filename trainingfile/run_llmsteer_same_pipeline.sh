#!/usr/bin/env bash
set -euo pipefail

# Runs the LLMSteer-style evaluation pipeline on one or more embedding models.
# Usage:
#   bash trainingfile/run_llmsteer_same_pipeline.sh <model_path1> <model_path2> ...
# Example:
#   bash trainingfile/run_llmsteer_same_pipeline.sh \
#     contrastive_mpnet_stage1 \
#     reptile_all-MiniLM-L12-v2 \
#     othertests/all-MiniLM-L12-v2

if [[ $# -eq 0 ]]; then
  MODELS=(
    "contrastive_mpnet_stage1"
    "reptile_all-MiniLM-L12-v2"
    "othertests/all-MiniLM-L12-v2"
  )
else
  MODELS=("$@")
fi

WORKLOADS=("ceb" "job")
SEEDS="24508"
N_SPLITS=10
TRAIN_SIZE=0.8
THRESHOLD=0.5
ESTIMATORS="lr,svc_rbf,svc_lin,rfc,gbc"
PCS="5,50,120"
SCALE_OPTIONS="false,true"
# Matches notebook comments that disabled:
#   SVC_LIN-50-True, SVC_LIN-120-True
EXCLUDE_CONFIGS="svc_lin-pcs50-scaletrue,svc_lin-pcs120-scaletrue"
OUT_ROOT="othertests/llmsteer_eval"

mkdir -p "${OUT_ROOT}"

for model_path in "${MODELS[@]}"; do
  model_tag="$(echo "${model_path}" | tr '/ ' '__')"
  for workload in "${WORKLOADS[@]}"; do
    run_dir="${OUT_ROOT}/${model_tag}_${workload}_samepipe"
    echo
    echo "=== Running model=${model_path} workload=${workload} ==="
    python trainingfile/run_llmsteer_eval.py \
      --model_path "${model_path}" \
      --workload "${workload}" \
      --seeds "${SEEDS}" \
      --n_splits "${N_SPLITS}" \
      --train_size "${TRAIN_SIZE}" \
      --threshold "${THRESHOLD}" \
      --estimators "${ESTIMATORS}" \
      --pcs "${PCS}" \
      --scale_options "${SCALE_OPTIONS}" \
      --exclude_configs "${EXCLUDE_CONFIGS}" \
      --augment_eval \
      --output_dir "${run_dir}"
  done
done

echo
echo "=== Merging summaries ==="
python trainingfile/merge_llmsteer_summaries.py \
  --root "${OUT_ROOT}" \
  --output_dir "${OUT_ROOT}" \
  --all_csv "comparison_all_rows.csv" \
  --best_csv "comparison_best_by_model_workload.csv"

echo
echo "Done. Best-per-model table:"
echo "  ${OUT_ROOT}/comparison_best_by_model_workload.csv"
