#!/usr/bin/env bash
# Local, offline dataset work using the existing environment with protected installs.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
local_python=/home/user/miniconda3/envs/lerobot/bin/python
if [[ ! -x "$local_python" || ! -d /home/user/lerobot/src ]]; then
  echo 'This launcher requires the local desktop LeRobot installation.' >&2
  echo 'On the server: conda activate lerobot' >&2
  echo 'Then use: LC_ALL=C.UTF-8 bash scripts/run_lerobot_server.sh <the same arguments>' >&2
  exit 2
fi
runtime_root="$project_root/artifacts/M3/runtime/lerobot"
mkdir -p "$runtime_root"/{cache,config,data,huggingface,matplotlib}
exec bwrap --die-with-parent --ro-bind / / --dev-bind /dev /dev --bind /proc /proc \
  --bind "$project_root" "$project_root" --tmpfs /tmp --tmpfs /var/tmp \
  --chdir "$project_root" --setenv PYTHONDONTWRITEBYTECODE 1 \
  --setenv PYTHONPATH "$project_root/airvla_isaac/source/airvla_isaac:/home/user/lerobot/src" \
  --setenv XDG_CACHE_HOME "$runtime_root/cache" --setenv XDG_CONFIG_HOME "$runtime_root/config" \
  --setenv XDG_DATA_HOME "$runtime_root/data" --setenv HF_HOME "$runtime_root/huggingface" \
  --setenv HF_HUB_OFFLINE 1 --setenv HF_DATASETS_OFFLINE 1 \
  --setenv MPLCONFIGDIR "$runtime_root/matplotlib" \
  "$local_python" -u -B "$@"
