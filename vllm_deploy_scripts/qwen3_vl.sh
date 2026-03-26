# !/bin/bash

# H100 * 8; H200 * 4
# Qwen3-VL-235B-A22B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-VL-235B-A22B-Instruct \
  --served-model-name Qwen3-VL-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --enable-expert-parallel \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --quantization fp8 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp


# Qwen3-VL-32B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp

# Qwen3-VL-30B-A3B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp

# Qwen3-VL-8B-Instruct CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp

# Qwen3-VL-4B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp


# Qwen3-VL-8B-Instruct with LoRA
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/lora_models_cache/0302_qwen3_vl_8b_train_run_114644_checkpoint_20000 \
  --served-model-name Qwen3-VL-8B-Instruct_lora \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp