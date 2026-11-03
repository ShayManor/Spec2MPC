"""LIBERO worker: one benchmark episode per process, JSON lines on stdin/stdout (held-out evaluation only).

Runs inside envs/libero (robosuite 1.4.0). Requests: {"cmd": "open", "suite", "index", "img"},
{"cmd": "reset", "seed"}, {"cmd": "step", "action": [7 + 1]}, {"cmd": "result"}, {"cmd": "close"}.
Replies carry raw robosuite quantities; spec2mpc/bench/libero.py maps them to the Menagerie replica.
Only the success flag leaves this process; BDDL goal predicates stay here.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np

# library prints go to stderr so stdout carries only replies
_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils import camera_utils as CU  # noqa: E402

CAMERA = "agentview"
N_WAIT = 10  # settle steps after reset (objects drop onto the table), standard in LIBERO evals


def _b64(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"b64": base64.b64encode(a.tobytes()).decode(), "dtype": str(a.dtype), "shape": list(a.shape)}


class Worker:
    def __init__(self) -> None:
        self.env = None

    def open(self, suite: str, index: int, img: int) -> dict:
        s = benchmark.get_benchmark_dict()[suite]()
        task = s.get_task(index)
        self.init_states = s.get_task_init_states(index)  # [n_init, n_state]
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.img = img
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img, camera_widths=img,
                                      camera_names=[CAMERA], camera_depths=True, controller="JOINT_POSITION")
        ctl = self.env.env.robots[0].controller
        return {"name": task.name, "language": task.language, "n_init": len(self.init_states),
                "dq_max": np.asarray(ctl.output_max, dtype=float).tolist(),
                "control_freq": float(self.env.env.control_freq)}

    def _obs(self, o: dict) -> dict:
        sim = self.env.env.sim
        depth = CU.get_real_depth_map(sim, o[f"{CAMERA}_depth"])[::-1, :, 0].astype(np.float32)  # [H, W] m
        rgb = o[f"{CAMERA}_image"][::-1].astype(np.uint8)  # [H, W, 3], flipped to top-left origin
        K = CU.get_camera_intrinsic_matrix(sim, CAMERA, self.img, self.img)  # [3, 3]
        T_wc = CU.get_camera_extrinsic_matrix(sim, CAMERA)  # [4, 4] camera in world, OpenCV axes
        bid = sim.model.body_name2id("robot0_link0")
        T_wb = np.eye(4)
        T_wb[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
        T_wb[:3, 3] = sim.data.body_xpos[bid]
        T_bc = np.linalg.inv(T_wb) @ T_wc  # [4, 4] camera in robot base
        return {"rgb": _b64(rgb), "depth": _b64(depth), "K": K.tolist(), "T_bc": T_bc.tolist(),
                "joint_pos": o["robot0_joint_pos"].tolist(), "joint_vel": o["robot0_joint_vel"].tolist(),
                "gripper_qpos": o["robot0_gripper_qpos"].tolist(), "gripper_qvel": o["robot0_gripper_qvel"].tolist(),
                "success": self.success}

    def reset(self, seed: int) -> dict:
        self.env.seed(seed)
        self.env.reset()
        o = self.env.set_init_state(self.init_states[seed % len(self.init_states)])
        self.success, self.steps = False, 0
        hold = np.zeros(7 + 1)
        hold[-1] = -1.0  # open gripper
        for _ in range(N_WAIT):
            o, _, _, _ = self.env.step(hold)
        return self._obs(o)

    def step(self, action: list[float]) -> dict:
        o, _, done, _ = self.env.step(np.asarray(action, dtype=float))
        self.steps += 1
        self.success = self.success or bool(self.env.check_success())
        return {**self._obs(o), "done": bool(done) or self.success}

    def result(self) -> dict:
        return {"success": self.success, "steps": self.steps}

    def close(self) -> dict:
        if self.env is not None:
            self.env.close()
        return {}


def main() -> None:
    w = Worker()
    for line in sys.stdin:
        req = json.loads(line)
        cmd = req.pop("cmd")
        try:
            rep = getattr(w, cmd)(**req)
        except Exception as e:  # report and keep serving; the client decides
            rep = {"error": f"{type(e).__name__}: {e}"}
        _OUT.write(json.dumps(rep) + "\n")
        if cmd == "close":
            break


if __name__ == "__main__":
    main()
