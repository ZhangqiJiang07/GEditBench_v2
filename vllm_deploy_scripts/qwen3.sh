#!/bin/bash

# Qwen3-4B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 \
  --port 25111 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp

# Qwen3-8B
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 \
  --port 25111 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 40000 \
  --distributed-executor-backend mp