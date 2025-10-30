#!/usr/bin/env bash
set -euo pipefail

TP_VAL=${TP:-1}
PP_VAL=${PP:-1}
WORLD_VAL=${WORLD:-1}

echo "TP=${TP_VAL} PP=${PP_VAL} WORLD=${WORLD_VAL}"
echo "Models dir: /models"
nvidia-smi || true

python3 /opt/trtllm/build.py "$@"
