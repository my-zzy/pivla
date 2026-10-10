#!/usr/bin/env bash
# Run existing Isaac Python with read-only host mounts and project-local runtime data.
# Headless by default; --gui opens the desktop simulator window.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -eq 0 ]; then
    echo 'Usage: bash scripts/run_isaac.sh path/to/script.py [--gui] [arguments]' >&2
    exit 2
fi
runtime_tag="$(basename -- "$1" .py)"
script_args=()
headless_args=(--headless)
desktop_args=()
gui=false
for arg in "$@"; do
    if [[ "$arg" == --gui ]]; then
        gui=true
        headless_args=()
    else
        script_args+=("$arg")
    fi
done
if "$gui"; then
    for arg in "${script_args[@]}"; do
        if [[ "$arg" == --headless ]]; then
            echo 'Choose either --gui or --headless.' >&2
            exit 2
        fi
    done
    desktop_args+=(--setenv HEADLESS 0 --setenv LIVESTREAM 0)
    # Preserve the desktop X11 socket inside the otherwise private /tmp.
    if [[ -d /tmp/.X11-unix ]]; then
        desktop_args+=(--ro-bind /tmp/.X11-unix /tmp/.X11-unix)
    fi
fi
milestone="${AIRVLA_MILESTONE:-M0}"
if [[ ! "$milestone" =~ ^M[0-8]$ ]]; then
    echo 'AIRVLA_MILESTONE must be M0 through M8.' >&2
    exit 2
fi
if [[ "$runtime_tag" == m1_* ]]; then milestone=M1; fi
runtime_root="$project_root/artifacts/$milestone/runtime/$runtime_tag"
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
# A desktop application may permit namespaces while an ordinary terminal does
# not. Check this before starting Sim and explain the host-policy prerequisite.
if ! bwrap --ro-bind / / /bin/true 2>"$runtime_root/logs/bwrap-check.log"; then
    cat "$runtime_root/logs/bwrap-check.log" >&2
    cat >&2 <<EOF

The read-only simulator sandbox could not start. Isaac Sim has not been launched.
On Ubuntu 24.04, "setting up uid map: Permission denied" can mean AppArmor
requires a user-namespace permission profile for /usr/bin/bwrap.

See: $project_root/docs/launcher-troubleshooting.md
Prepared profile: $project_root/configs/apparmor/pivla-bwrap

The launcher will not fall back to running with writable shared installations.
EOF
    exit 1
fi
kit_options="--portable-root $runtime_root/kit --/app/settings/persistent=false --/log/file=$runtime_root/logs/kit.log"
exec bwrap --die-with-parent --ro-bind / / --dev-bind /dev /dev --bind /proc /proc \
    --bind "$project_root" "$project_root" --tmpfs /tmp --tmpfs /var/tmp \
    "${desktop_args[@]}" \
    --chdir "$project_root" \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    --setenv PYTHONPATH "$project_pythonpath" \
    --setenv XDG_CACHE_HOME "$runtime_root/cache" \
    --setenv XDG_CONFIG_HOME "$runtime_root/config" \
    --setenv XDG_DATA_HOME "$runtime_root/data" \
    --setenv CUDA_CACHE_PATH "$runtime_root/cuda" \
    --setenv WARP_CACHE_PATH "$runtime_root/warp" \
    "$isaac_python" -u -B "${script_args[@]}" "${headless_args[@]}" --kit_args "$kit_options"
