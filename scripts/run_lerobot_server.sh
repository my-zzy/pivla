#!/usr/bin/env bash
# Run offline training in the active server environment (conda activate lerobot).
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
training_python="${LEROBOT_PYTHON:-python}"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_lerobot_server.sh scripts/m4_train.py --config configs/train.yaml [--probe]" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_root/airvla_isaac/source/airvla_isaac${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd -- "$project_root"
exec "$training_python" -u -B "$@"
