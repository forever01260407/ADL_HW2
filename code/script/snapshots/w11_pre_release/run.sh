#!/usr/bin/env bash
# Reproduce W08 training only. Does not select/promote a model or run a Judge.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
task_python="${HW2_PYTHON:-python3}"
exec "$task_python" code/script/train.py \
  --config code/script/configs/w08_train.json \
  --output artifacts/training/w08_epoch1 "$@"
