#!/usr/bin/env bash
# Recreate the pinned Python environment on the Azure VM.
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch=/mnt/ai-swarm-scratch
venv=/home/azureuser/ai-swarm/.venv

[[ "$(id -un)" == azureuser ]] || { printf 'Run as azureuser on the VM.\n' >&2; exit 1; }
mountpoint -q "$scratch" || { printf 'Scratch disk is not mounted.\n' >&2; exit 1; }
[[ -f "$project_dir/requirements.lock.txt" ]] || {
  printf 'Missing requirements.lock.txt.\n' >&2; exit 1;
}

python3 -m venv "$venv"
export PIP_CACHE_DIR="$scratch/pip-cache"
"$venv/bin/python" -m pip install --disable-pip-version-check \
  'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$venv/bin/python" -m pip install --disable-pip-version-check \
  -r "$project_dir/requirements.lock.txt"
"$venv/bin/python" -m pip check
"$venv/bin/python" -I -c 'import laya, torch; print("laya", laya.__version__, "torch", torch.__version__)'
