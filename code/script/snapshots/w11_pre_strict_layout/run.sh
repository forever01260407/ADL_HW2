#!/usr/bin/env bash
# Reproduce both training stages from raw BGE-base using frozen Train-only negatives.
# Writes separate artifacts/reproduced/ weights; never overwrites the submitted H5.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
task_python="${HW2_PYTHON:-python3}"
exec "$task_python" code/script/train_release.py "$@"
