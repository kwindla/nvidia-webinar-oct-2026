#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CPU preserves GPU capacity for the fine-tune and the user-managed local LLM.
exec taskset -c "${BASE_ASR_CPUS:-0-7}" .tools/nemo-speech/nemo-speech-0.2.0-linux-x86_64-cuda/bin/nemo-speech serve \
  --asr-model models/base/base-f16.gguf --device cpu \
  --host 127.0.0.1 --port 8082 \
  --asr.streaming.rnnt_right_context 3 \
  --asr.endpointing.enable=false "$@"
