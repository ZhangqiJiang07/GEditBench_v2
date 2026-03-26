# !/bin/bash
# Qwen2.5-VL-72B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen2.5-VL-72B-Instruct \
  --served-model-name Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max_num_seqs 32 \
  --max-model-len 48000

# Qwen2.5-VL-7B-Instruct
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-zoo/Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max_num_seqs 32 \
  --max-model-len 48000