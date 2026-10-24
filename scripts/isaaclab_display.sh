#!/usr/bin/env bash
set -eo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_root="${ISAACLAB_ROOT:-/home/ubuntu22/IsaacLab}"
if [[ ! -x "$isaac_root/isaaclab.sh" || ! -x "$isaac_root/_isaac_sim/python.sh" ]]; then
    echo "Isaac Lab / Isaac Sim binary environment not found at $isaac_root" >&2
    echo "Set ISAACLAB_ROOT to the installed Isaac Lab directory." >&2
    exit 1
fi
run_dir="$workspace/log/isaaclab/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
cd "$workspace"
# Always use Isaac Sim's Python 3.12, independent of ROS or an active Conda env.
exec env -u CONDA_PREFIX -u VIRTUAL_ENV -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONPATH="$isaac_root/source/isaaclab" \
    "$isaac_root/_isaac_sim/python.sh" "$workspace/isaaclab/scripts/display.py" \
    --run-dir "$run_dir" "--kit_args=--/log/file=$run_dir/kit.log" "$@"
