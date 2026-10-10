#!/usr/bin/env bash
# Run existing Isaac Python with read-only host mounts and project-local runtime data.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -eq 0 ]; then
    echo 'Usage: bash scripts/run_isaac.sh path/to/script.py [arguments]' >&2
    exit 2
fi
runtime_tag="$(basename -- "$1" .py)"
runtime_root="$project_root/artifacts/M0/runtime/$runtime_tag"
isaac_python=/home/user/miniconda3/envs/isaaclab/bin/python
lab_root=/home/user/IsaacLab
project_pythonpath="$project_root/airvla_isaac/source/airvla_isaac"
for lab_package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    project_pythonpath="$project_pythonpath:$lab_root/source/$lab_package"
done
mkdir -p "$runtime_root"/{cache,config,data,cuda,warp,kit,logs}
if ! command -v bwrap >/dev/null; then
    echo 'bwrap is required to protect shared installations; no fallback with writable host paths.' >&2
    exit 1
fi
kit_options="--portable-root $runtime_root/kit --/app/settings/persistent=false --/log/file=$runtime_root/logs/kit.log"
exec bwrap --die-with-parent --ro-bind / / --dev-bind /dev /dev --bind /proc /proc \
    --bind "$project_root" "$project_root" --tmpfs /tmp --tmpfs /var/tmp \
    --chdir "$project_root" \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    --setenv PYTHONPATH "$project_pythonpath" \
    --setenv XDG_CACHE_HOME "$runtime_root/cache" \
    --setenv XDG_CONFIG_HOME "$runtime_root/config" \
    --setenv XDG_DATA_HOME "$runtime_root/data" \
    --setenv CUDA_CACHE_PATH "$runtime_root/cuda" \
    --setenv WARP_CACHE_PATH "$runtime_root/warp" \
    "$isaac_python" -B "$@" --headless --kit_args "$kit_options"
