#!/bin/bash
set -euo pipefail

# Configuration (override via environment variables before calling this script)
EXP_DIR="${EXP_DIR:-/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot}"            # Directory containing checkpoint-* subfolders
DATA_FILE="${DATA_FILE:-/home/work/tmp_hyeonbin/pred-sent/data/csqa/test.json}"      # Evaluation dataset file
RESULT_DIR="${RESULT_DIR:-${EXP_DIR}/eval_results}"  # Where to store per-checkpoint jsonl results
CSV_PATH="${CSV_PATH:-${RESULT_DIR}/qwen_3_0.6b_csqa_cot_accuracy.csv}"   # Aggregated CSV output
TP="${TP:-1}"                                        # vLLM tensor parallel size
BATCH_SIZE="${BATCH_SIZE:-64}"                       # Batch size for eval batching
TEMP="${TEMP:-0}"                                  # Sampling temperature

EVAL_PY="/home/work/tmp_hyeonbin/pred-sent/src/sft/eval.py"

mkdir -p "${RESULT_DIR}"

# Initialize CSV with header
echo "checkpoint,accuracy" > "${CSV_PATH}"

# Collect checkpoints
mapfile -t CKPTS < <(ls -d "${EXP_DIR}"/checkpoint-* "${EXP_DIR}"/best-checkpoint-* 2>/dev/null | sort -V || true)

if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "No checkpoints found under ${EXP_DIR}."
  exit 0
fi

for CKPT in "${CKPTS[@]}"; do
  NAME="$(basename "${CKPT}")"
  OUT_JSONL="${RESULT_DIR}/${NAME}.jsonl"
  echo "Evaluating ${CKPT}..."
  CUDA_VISIBLE_DEVICES=2 python "${EVAL_PY}" \
    --model "${CKPT}" \
    --data_file "${DATA_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TP}" \
    --temp "${TEMP}" \
    --result_file "${OUT_JSONL}" \
    --output_csv "${CSV_PATH}"
done

echo "Aggregated results written to: ${CSV_PATH}"