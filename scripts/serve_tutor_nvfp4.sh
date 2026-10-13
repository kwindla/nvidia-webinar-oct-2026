#!/usr/bin/env bash
# Serve the ATC tutor round 5 NVFP4 build (daily-co/atc-tutor-nano-r5-nvfp4) on a local
# Blackwell GPU with vLLM 0.31.0: OpenAI-compatible at http://127.0.0.1:8083/v1, model
# "gb-phone" (alias "atc-tutor"). Foreground; Ctrl+C stops and removes the container.
#
# Same settings as the Modal bench: BF16 KV cache (the export's hf_quant_config would
# otherwise pick FP8), fp32 Mamba state, thinking off. The memory cap leaves room for the
# fine-tuned ASR server sharing the GPU; set GPU_MEM_UTIL to change it. The image's CUDA
# forward-compat libcuda is masked because GeForce parts reject compat mode (error 804).
set -euo pipefail

MODEL_DIR="${TUTOR_MODEL_DIR:-$HOME/local-models/atc-tutor-r5-nvfp4}"
PORT="${PORT:-8083}"
[[ -f "$MODEL_DIR/config.json" ]] || { echo "no checkpoint at $MODEL_DIR" >&2; exit 1; }

exec docker run --rm --name atc-tutor-nvfp4 \
  --gpus all --ipc=host \
  --tmpfs /usr/local/cuda/compat \
  -p "127.0.0.1:$PORT:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0 \
  --model /model \
  --served-model-name gb-phone atc-tutor \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --host 0.0.0.0 --port 8000 "$@"
