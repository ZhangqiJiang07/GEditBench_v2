# !/bin/bash
# H100 * 4
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/jfs/model-zoo/zai-org/GLM-4.5V \
  --served-model-name GLM4.5-V \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max_num_seqs 32 \
  --max-model-len 48000