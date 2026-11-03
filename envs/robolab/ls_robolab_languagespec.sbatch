#!/bin/bash
#SBATCH -J ls_robolab_languagespec
#SBATCH -p gpu
#SBATCH -A scholar
#SBATCH --gres=gpu:1
#SBATCH --constraint=A40
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /scratch/scholar/manors/languagespec/robolab/logs/%x-%j.out
# RoboLab Phase 1 smoke on one A40: one RoboLab-120 task through the stub executor server.
set -u
L=/scratch/scholar/manors/languagespec/robolab
date -Is; hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
bash $L/code/envs/robolab/run_smoke.sh ${TASK:-BananaInBowlTask}
echo "exit $?"; date -Is
