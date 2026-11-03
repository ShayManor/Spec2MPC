#!/usr/bin/env bash
# Run a command in the LIBERO env with the repo on the path: envs/libero/run.sh python -m ...
here="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$here/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$here/config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
exec "${UV:-$HOME/.local/bin/uv}" run --project "$here" "$@"
