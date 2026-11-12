#!/usr/bin/env bash
# Downloads/verifies model assets only; never installs packages or trains.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
task_python="${HW2_PYTHON:-python3}"
exec "$task_python" code/script/download_release.py "$@"
