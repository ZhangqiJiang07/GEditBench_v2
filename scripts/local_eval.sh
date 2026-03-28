#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}"
fi


BENCH=${1:-editscore}
LORA_PATH="${REPO_ROOT}/lora_weights"
BASEMODEL_PATH="/path/to/Qwen/Qwen3-VL-8B-Instruct"


LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1 python ${SRC_ROOT}/inference/run_eval.py \
  --lora-model-path ${LORA_PATH} \
  --base-model-path ${BASEMODEL_PATH} \
  --use-flash-attn \
  --bmk ${BENCH} \
  --bmk-config ${REPO_ROOT}/configs/datasets/bmk.json \
  --max-retries 3 \
  --max-new-tokens 1024 \
  --temperature 0.7 \
  --num-beams 1 \
  --image-min-pixels $((256*32*32)) \
  --image-max-pixels $((1280*32*32)) \
  --prompt-id vlm/assessment/visual_consistency/pairwise \
  --prompt-version v1 \
  --seed 42 \
  --max-num-seqs 1 \
  --num-pass 1 \
  --logger-level INFO \
  --do-sample \
  --use-vllm \
  --merged-model-cache-dir ~/.cache/geditv2/merged_models \
  --save-details
