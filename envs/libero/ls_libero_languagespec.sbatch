#!/bin/bash
#SBATCH -J ls_libero_languagespec
#SBATCH -p gpu
#SBATCH -A scholar
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -o /scratch/scholar/manors/languagespec/libero/logs/%x-%j.out
# LIBERO Phase 1 smoke: the slow tests (one LIBERO task through the stub executor) with EGL rendering.
set -u
L=/scratch/scholar/manors/languagespec/libero
date -Is; hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd $L/code
export MENAGERIE_PATH=$L/assets/mujoco_menagerie MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$L/corevenv/bin/python -m pytest -q -p no:cacheprovider -m slow tests/v2/test_bench_libero.py -rA
echo "exit $?"; date -Is
