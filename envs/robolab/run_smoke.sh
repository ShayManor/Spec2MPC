#!/bin/bash
# RoboLab smoke inside the Isaac Lab 2.2.0 container (Ubuntu 22.04; the isaacsim 5.0 wheels need glibc 2.35,
# Scholar has 2.34, so Isaac Sim runs from nvcr.io/nvidia/isaac-lab:2.2.0 under Apptainer).
# usage: run_smoke.sh [task]   env: L (work dir), PORT
set -u
L=${L:-/scratch/scholar/manors/languagespec/robolab}
TASK=${1:-BananaInBowlTask}
PORT=${PORT:-8765}
SIF=$L/isaaclab_2.2.0.sif
mkdir -p $L/home $L/kit/cache $L/kit/data $L/kit/logs $L/results $L/pyuser
BINDS="-B $L -B $L/kit/cache:/isaac-sim/kit/cache -B $L/kit/data:/isaac-sim/kit/data -B $L/kit/logs:/isaac-sim/kit/logs"
# keep the container's Vulkan ICD (bare libGLX_nvidia.so.0, found in /.singularity.d/libs); the host's ICD points
# at /usr/lib64, which does not exist in the container
apptainer exec --nv --cleanenv --home $L/home $BINDS \
  --env OMNI_KIT_ACCEPT_EULA=Y,ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,PYTHONUSERBASE=$L/pyuser \
  --env PYTHONPATH=$L/code:$L/RoboLab,TASK=$TASK,PORT=$PORT,L=$L,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  $SIF bash -c '
set -u
PY=/workspace/isaaclab/_isaac_sim/python.sh
$PY -c "import flask, json_numpy, tyro, h5py" 2>/dev/null || \
  $PY -m pip install --user -q flask json_numpy msgpack-numpy tyro h5py imageio python-dotenv pyzmq websockets "setuptools<81"
$PY -c "import torch; print(\"torch cuda\", torch.cuda.is_available(), torch.cuda.device_count())"
$PY $L/code/envs/robolab/stub_executor.py $PORT > $L/logs/stub_executor.log 2>&1 &
SRV=$!
for i in $(seq 30); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
cd $L/RoboLab
$PY $L/code/envs/robolab/smoke.py --task $TASK --headless --executor-url http://127.0.0.1:$PORT \
  --results-dir $L/results
RC=$?
kill $SRV
exit $RC
'
