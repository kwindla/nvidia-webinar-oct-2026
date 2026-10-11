#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .tools/nemo-speech/nemo-speech-0.2.0-linux-x86_64-cuda/bin/nemo-speech serve \
  --asr-model models/atc02/atc02-f16.gguf \
  --host 127.0.0.1 --port 8081 \
  --asr.streaming.rnnt_right_context 3 \
  --asr.endpointing.enable=false "$@"
