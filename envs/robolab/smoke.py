"""Phase 1 RoboLab smoke: one RoboLab-120 task end to end through the executor server with a stub executor.

Runs inside the Isaac Lab 2.2.0 container (see run_smoke.sh). Perception and the MPC are stubs here: the scene
holds only the camera image and the controller holds the arm. Prints one line `RESULT {json}`.
usage: python smoke.py --task BananaInBowlTask --executor-url http://127.0.0.1:8765 --headless
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--executor-url", default="http://127.0.0.1:8765")
parser.add_argument("--executor-name", default="stub")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--results-dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from spec2mpc.bench.client import HoldController, run_episode  # noqa: E402
from spec2mpc.bench.robolab import BENCH, GRIPPER_OPEN_CTRL, RoboLabEnv, register  # noqa: E402
from spec2mpc.core import RobotSpec, Scene  # noqa: E402
from spec2mpc.orchestrator import RecordedAgent  # noqa: E402

# stub replica spec (the real one comes from spec2mpc.robots.v2.load_robot("arm", "panda"))
PANDA = RobotSpec("arm", "panda", "franka_emika_panda/panda.xml", {}, ["ee", "gripper"],
                  [f"actuator{i}" for i in range(1, 9)], {}, {"min_clearance_m": 0.02, "max_speed_frac": 0.5,
                                                          "max_force_frac": 0.5}, {}, {})
seen = {}


def perceive(obs, robot):
    seen.update(rgb=list(obs.rgb.shape), depth=None if obs.depth is None else list(obs.depth.shape),
                depth_range=None if obs.depth is None else [float(np.nanmin(obs.depth)), float(np.nanmax(obs.depth))],
                K=None if obs.K is None else np.asarray(obs.K).round(1).tolist(),
                cam_pose=None if obs.cam_pose is None else np.asarray(obs.cam_pose).round(3).tolist(),
                qpos=np.asarray(obs.qpos).round(3).tolist())
    return Scene([], obs.rgb, obs.t_s)


try:
    register([args.task])
    from robolab.core.environments.factory import get_envs

    task_env = get_envs(task=args.task)[0]
    env = RoboLabEnv(task_env, PANDA, device=args.device or "cuda:0", seed=args.seed)
    out = run_episode(env, args.executor_url, args.executor_name, env.instruction, bench=BENCH, seed=args.seed,
                      perceive=perceive, agent=RecordedAgent([env.instruction]),
                      make_controller=lambda r: HoldController(r, env.control_dt_s, GRIPPER_OPEN_CTRL),
                      results_dir=args.results_dir)
    print("OBS " + json.dumps(seen), flush=True)
    print("RESULT " + json.dumps({"task": task_env, **out}), flush=True)
    env.close()
    app.close()
except Exception:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)  # app.close() exits 0, which would hide the failure
