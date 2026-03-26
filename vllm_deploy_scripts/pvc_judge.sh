# Qwen3-VL-8B-Instruct with LoRA
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/pvc_judge/ \
  --served-model-name PVC-Judge \
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