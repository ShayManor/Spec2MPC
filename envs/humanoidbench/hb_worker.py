"""HumanoidBench env worker: JSON lines on stdin/stdout, run inside envs/humanoidbench.

Evaluation only. Success thresholds stay in this process: the caller gets success and
score, never the threshold. Success is the HumanoidBench rule: episode return >= success_bar.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

_OUT = sys.stdout
sys.stdout = sys.stderr  # HumanoidBench prints; keep stdout for the protocol

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import humanoid_bench  # noqa: E402,F401  (registers <robot>-<task>-v0)
from humanoid_bench.env import TASKS  # noqa: E402

_BODIES = ["pelvis", "torso_link"]
_SITES = ["head", "left_foot", "right_foot", "left_hand", "right_hand"]


class Worker:
    def __init__(self) -> None:
        self.env = None
        self.render = False
        self.renderer = None

    # ---- helpers ----

    def _base(self):
        return self.env.unwrapped

    def _frames(self) -> dict[str, list[float]]:
        # world-frame poses [7] xyz + quat wxyz of the robot frames the scene needs
        m, d = self._base().model, self._base().data
        out = {}
        for b in _BODIES:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if i >= 0:
                out[b] = d.xpos[i].tolist() + d.xquat[i].tolist()
        for s in _SITES:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
            if i >= 0:
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, d.site_xmat[i])
                out[s] = d.site_xpos[i].tolist() + q.tolist()
        return out

    def _obs(self) -> dict:
        d = self._base().data
        o = {"qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "t_s": float(d.time) - self.t0,
             "frames": self._frames(), "steps": self.steps, "return": self.ret, "done": self.done}
        if self.render:
            o |= self._camera()
        return o

    def _camera(self) -> dict:
        # task camera: rgb [H, W, 3] uint8, depth [H, W] float32 m, K [3, 3], cam_pose [7] in pelvis frame, OpenCV axes
        m, d, r = self._base().model, self._base().data, self.renderer
        r.update_scene(d, camera=self.cam)
        img = np.ascontiguousarray(r.render(), dtype=np.uint8)
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cam)
        dep = np.ascontiguousarray(r.render(), dtype=np.float32)
        r.disable_depth_rendering()
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam)
        f = 0.5 * r.height / np.tan(0.5 * np.deg2rad(m.cam_fovy[c]))
        K = [[f, 0.0, 0.5 * r.width], [0.0, f, 0.5 * r.height], [0.0, 0.0, 1.0]]
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        Rb = d.xmat[b].reshape(3, 3)
        Rc = d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # OpenGL -> OpenCV axes
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Rb.T @ Rc).reshape(-1))
        pose = (Rb.T @ (d.cam_xpos[c] - d.xpos[b])).tolist() + q.tolist()
        enc = lambda a: {"b64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}
        return {"rgb": enc(img), "depth": enc(dep), "K": K, "cam_pose": pose}

    def _hold(self) -> list[float]:
        # position targets equal to the current joint angles, one per actuator
        m, d = self._base().model, self._base().data
        return [float(d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]) for a in range(m.nu)]

    # ---- commands ----

    def info(self, req: dict) -> dict:
        return {"tasks": sorted(TASKS)}

    def make(self, req: dict) -> dict:
        if self.env is not None:
            self.close(req)
        self.task, self.robot = req["task"], req.get("robot", "h1hand")
        self.render = bool(req.get("render", False))
        self.env = gym.make(f"{self.robot}-{self.task}-v0")
        b, m = self._base(), self._base().model
        self.max_steps = int(self.env.spec.max_episode_steps)
        if self.render:
            self.cam = type(b.task).camera_name
            self.renderer = mujoco.Renderer(m, int(req.get("height", 256)), int(req.get("width", 256)))
        jnt = [m.actuator_trnid[a, 0] for a in range(m.nu)]
        return {"actuators": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)],
                "joints": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jnt],
                "qposadr": [int(m.jnt_qposadr[j]) for j in jnt], "dofadr": [int(m.jnt_dofadr[j]) for j in jnt],
                "kp": m.actuator_gainprm[:, 0].tolist(), "kv": (-m.actuator_biasprm[:, 2]).tolist(),
                "ctrl_low": b.action_low.tolist(), "ctrl_high": b.action_high.tolist(),
                "control_dt": float(b.dt), "max_steps": self.max_steps,
                "control": "pos" if self.robot not in ("g1", "digit") else "torque"}

    def reset(self, req: dict) -> dict:
        self.env.reset(seed=int(req["seed"]))
        self.t0 = float(self._base().data.time)
        self.steps, self.ret, self.done, self.terminated = 0, 0.0, False, False
        return {**self._obs(), "hold": self._hold()}

    def step(self, req: dict) -> dict:
        if self.done:
            raise RuntimeError("step after episode end")
        b = self._base()
        u = np.clip(np.asarray(req["ctrl"], dtype=np.float64), b.action_low, b.action_high)  # [nu]
        a = 2 * (u - b.action_low) / (b.action_high - b.action_low) - 1  # [nu] in [-1, 1]
        _, r, term, trunc, _ = self.env.step(a.astype(np.float32))
        self.steps += 1
        self.ret += float(r)
        self.terminated = bool(term)
        self.done = bool(term or trunc or self.steps >= self.max_steps)
        return self._obs()

    def result(self, req: dict) -> dict:
        bar = float(type(self._base().task).success_bar)
        return {"success": bool(self.ret >= bar), "score": float(np.clip(self.ret / bar, 0.0, 1.0)),
                "info": {"task": self.task, "robot": self.robot, "return": self.ret,
                         "steps": self.steps, "fell": self.terminated, "done": self.done}}

    def close(self, req: dict) -> dict:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.env is not None:
            self.env.close()
            self.env = None
        return {}


def main() -> None:
    w = Worker()
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        t = time.time()
        try:
            out = {"ok": True, **getattr(w, req["cmd"])(req)}
        except Exception as e:  # report and keep serving
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["wall_s"] = time.time() - t
        _OUT.write(json.dumps(out) + "\n")
        _OUT.flush()
        if req.get("cmd") == "close":
            break


if __name__ == "__main__":
    main()
