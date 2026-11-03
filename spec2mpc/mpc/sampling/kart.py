"""The v1 kart MPCC behind the SamplingMPC interface (configs/v2/robots/kart.yaml: solver: kart_mpcc).

The MPCC, its weights (v1 adapter on a theta, default theta unless given) and its warm start are the v1 code;
this wrapper only maps the MuJoCo kart state to the v1 state, the program stage to a v1 reference path, the
perceived scene to a v1 clearance grid, and the v1 rate inputs to the MuJoCo actuators.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from spec2mpc.core import Program, RobotSpec, Scene
from spec2mpc.mpc.adapter import adapt
from spec2mpc.mpc.mpcc import MPCC, MPCCState
from spec2mpc.mpc.theta import ThetaBatch, default_theta
from spec2mpc.robots.kart import RuntimeConfig, load_fixed_kart
from spec2mpc.sim.occupancy import GridSpec, cell_centers
from spec2mpc.sim.paths import RefPath
from spec2mpc.sim.scenes import _resample
from spec2mpc.mpc.sampling.stages import Until, resolve_frames

ROOT = Path(__file__).parent.parent.parent.parent
Status = Literal["running", "done", "timeout", "invalid_reference", "floor_clip"]
_FREE_M = 50.0  # clearance reported where the scene has no obstacle geometry


def v1_configs() -> tuple:
    # -> (mpc_cfg, kart_cfg, sim_cfg): the v1 configs the wrapper reads
    return (OmegaConf.load(ROOT / "configs/mpc/kart.yaml"), OmegaConf.load(ROOT / "configs/robots/kart.yaml"),
            OmegaConf.load(ROOT / "configs/sim/default.yaml"))


def ref_path(points: np.ndarray, closed: bool, half_width_m: float, ds_m: float, fine_ds_m: float) -> RefPath:
    # points [N, 2+] -> RefPath [1] resampled at uniform arc length ds_m (v1 scene convention)
    pts = np.asarray(points, np.float64)[:, :2]
    if len(pts) == 2:  # the v1 spline needs three knots
        pts = np.stack([pts[0], 0.5 * (pts[0] + pts[1]), pts[1]])
    seg = np.linalg.norm(np.diff(np.concatenate([pts, pts[:1]]) if closed else pts, axis=0), axis=1)
    xy, s, hd, cu, length = _resample(pts, closed, float(seg.sum()), ds_m, fine_ds_m)
    f = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)  # noqa: E731
    return RefPath(xy=f(xy), s=f(s), heading=f(hd), curvature=f(cu),
                   length_m=torch.tensor([float(s[-1]) + (ds_m if closed else 0.0)], dtype=torch.float32),
                   closed=torch.tensor([closed]), half_width_m=torch.tensor([half_width_m], dtype=torch.float32))


class KartMPCC:
    def __init__(self, robot: RobotSpec, horizon_steps: int | None = None, n_samples: int | None = None,
                 method: str = "ps", *, theta: ThetaBatch | None = None, tire_model: str = "linear", **_):
        # horizon_steps, n_samples, method belong to the sampling MPC; the v1 horizon is configs/mpc/kart.yaml
        self.robot = robot
        self.mpc_cfg, self.kart_cfg, self.sim_cfg = v1_configs()
        rate = float(robot.runtime.get("control_rate_hz", 20.0))
        self.runtime = RuntimeConfig(control_rate_hz=rate, solver_iters=int(robot.runtime.get("solver_iters", 10)),
                                     perception_rate_hz=float(robot.perception.get("rate_hz", rate)))
        self.v1 = load_fixed_kart(self.kart_cfg, "evc", self.runtime)
        self.p = self.v1.params
        self.mpc = MPCC(self.mpc_cfg, tire_model)
        self.theta = theta if theta is not None else default_theta(1, self.mpc_cfg)
        self.w = adapt(self.theta, self.p, self.mpc_cfg)
        self.spec = GridSpec.from_cfg(self.sim_cfg.grid)
        self.dt = 1.0 / rate
        self.method, self.n_samples, self.H = "v1_mpcc", 0, self.mpc.N
        from spec2mpc.robots.v2 import build_model
        self.m = build_model(robot)
        self.d = mujoco.MjData(self.m)
        self.frames = resolve_frames(self.m, robot)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self._q_root, self._v_root = self.m.jnt_qposadr[jid("root")], self.m.jnt_dofadr[jid("root")]
        self._q_steer = self.m.jnt_qposadr[jid("steer_fl")]
        wheel = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "wheel_rl")
        self._wheel_r = float(self.m.geom_size[self.m.body_geomadr[wheel], 0])  # cylinder radius, m
        self.program: Program | None = None
        self.stage_index, self.status = 0, "running"
        self.info: dict = {"floor_clips": 0, "errors": []}
        self.t_s = 0.0
        self.arm_dofs = np.zeros(0, np.int32)

    # ---- the v1 controller (spec2mpc.sim.rollout Controller signature) ----

    def v1_controller(self, path: RefPath, B: int = 1):
        # -> Controller for run_episodes_with_stats: same solve, weights, warm start, and sdf age as the v1 loop
        state = MPCCState.init(B, self.mpc.N, path.xy.device)
        w = adapt(self.theta.index(torch.zeros(B, dtype=torch.long)), self.p.index(torch.zeros(B, dtype=torch.long)),
                  self.mpc_cfg)
        p = self.p.index(torch.zeros(B, dtype=torch.long))
        mem = {"sdf": None, "age": 0, "first": True}

        def ctrl(x_hat, sdf, origin, t_s):
            if sdf is not mem["sdf"]:
                mem["sdf"], mem["age"] = sdf, 0
            if mem["first"]:  # v1 seeds the warm-start progress at the start state's projection
                from spec2mpc.sim.paths import project_global
                state.s = project_global(path, x_hat[:, None, :2])[:, 0].clone()
                mem["first"] = False
            res = self.mpc.solve(x_hat, state, w, p, path, sdf, origin, mem["age"] * self.dt, self.spec,
                                 self.runtime.solver_iters)
            mem["age"] += 1
            return res.u0, res.ok
        return ctrl

    # ---- program interface ----

    def set_program(self, program: Program, scene: Scene) -> None:
        self.program, self.stage_index, self.status, self.t_s = program, 0, "running", 0.0
        self.info = {"floor_clips": 0, "errors": []}
        try:
            from spec2mpc.program.validate import validate  # Task 1
            errs = validate(program, self.robot, scene)
        except ImportError:
            errs = []
        if errs:
            self.info["errors"] = errs
            self.status = "invalid_reference"
        self._ready = False
        self._F, self._delta = 0.0, None

    def _x(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        # MuJoCo kart state -> v1 state [NX] (X, Y, psi, vx, vy, omega, delta, F)
        q = qpos[self._q_root:self._q_root + 7]
        w, x, y, z = q[3:7]
        psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = qvel[self._v_root:self._v_root + 6]
        c, s = math.cos(psi), math.sin(psi)
        delta = float(qpos[self._q_steer]) if self._delta is None else self._delta
        return np.array([q[0], q[1], psi, c * v[0] + s * v[1], -s * v[0] + c * v[1], v[5], delta, self._F])

    def _enter_stage(self, x: np.ndarray, scene: Scene) -> None:
        st = self.program.stages[self.stage_index]
        self._until = Until(st.until, self.frames, [])
        self._t_stage0 = self.t_s
        term = next(t for t in st.terms if t.type in ("follow_path", "pose_to"))
        if term.type == "follow_path":
            pts = np.asarray(scene.get(term.args[1]).points, np.float64)
            seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            closed = bool(np.linalg.norm(pts[-1, :2] - pts[0, :2]) < 3.0 * np.median(seg))
        else:  # go-to: a straight path from the kart to the target
            pts = np.stack([x[:2], np.asarray(scene.get(term.args[1]).pose[:2], np.float64)])
            closed = False
        self._path_ids = [term.args[1]]
        sc = self.sim_cfg
        self.path = ref_path(pts, closed, float(self.sim_cfg.scene.lap.half_width_m[0]), float(sc.ds_m),
                             float(sc.scene.spline_fine_ds_m))
        self.state = MPCCState.init(1, self.mpc.N, "cpu")
        from spec2mpc.sim.paths import project_global
        self.state.s = project_global(self.path, torch.as_tensor(x[None, None, :2], dtype=torch.float32))[:, 0]
        self._sdf_t, self._age = None, 0
        self._ready = True

    def _sdf(self, x: np.ndarray, scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
        # perceived obstacle geometry -> clearance grid [1, F, n, n] centered on the kart (static over the frames)
        origin = torch.as_tensor(x[None, :2], dtype=torch.float32)
        cells = cell_centers(self.spec, origin)[0].numpy().reshape(-1, 2)  # [n*n, 2]
        d = np.full(len(cells), _FREE_M)
        for e in scene.entities:
            if e.kind in ("robot_frame", "path", "region"):
                continue
            if e.points is not None and len(e.points) > 1:
                pts = np.asarray(e.points, np.float64)[:, :2]
                pts = pts[np.linalg.norm(pts - x[:2], axis=1) < self.spec.size_m]
                if len(pts):
                    d = np.minimum(d, np.min(np.linalg.norm(cells[:, None] - pts[None], axis=-1), axis=1))
            elif e.extent is not None:
                r = float(np.max(np.asarray(e.extent)[:2]))
                d = np.minimum(d, np.linalg.norm(cells - np.asarray(e.pose[:2]), axis=1) - r)
        n, F = self.spec.n, len(self.spec.frame_offsets_s)
        sdf = torch.as_tensor(d.reshape(n, n), dtype=torch.float32)[None, None].expand(1, F, n, n).contiguous()
        return sdf, origin

    def stop_control(self) -> np.ndarray:
        # hold the current steer, zero drive torque
        return np.array([float(self.d.qpos[self._q_steer]), 0.0, 0.0])

    def step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray:
        self.d.qpos[:], self.d.qvel[:] = qpos, qvel
        try:
            return self._step(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64), scene)
        finally:
            self.t_s += self.dt

    def _step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray:
        if self.program is None or self.status in ("done", "timeout", "invalid_reference"):
            return self.stop_control()
        x = self._x(qpos, qvel)
        try:
            if not self._ready:
                self._enter_stage(x, scene)
            for i in self._path_ids + self._until.entity_ids(scene):
                scene.get(i)
            mujoco.mj_kinematics(self.m, self.d)
            fired = self._until.fired(self.m, self.d, scene, self.t_s - self._t_stage0)
        except KeyError:  # a referenced entity was dropped: stop safely
            self.status = "invalid_reference"
            return self.stop_control()
        if fired:
            if self.stage_index == len(self.program.stages) - 1:
                self.status = "timeout" if self._until.term.type == "timeout" else "done"
                return self.stop_control()
            self.stage_index += 1
            self._ready = False
            return self._step(qpos, qvel, scene)
        if self._sdf_t != scene.t_s:
            self._grid, self._sdf_t, self._age = self._sdf(x, scene), scene.t_s, 0
        sdf, origin = self._grid
        res = self.mpc.solve(torch.as_tensor(x[None], dtype=torch.float32), self.state, self.w, self.p, self.path,
                             sdf, origin, self._age * self.dt, self.spec, self.runtime.solver_iters)
        self._age += 1
        rates = res.u0[0].numpy().astype(np.float64)  # (delta_rate, F_rate)
        p = self.p
        self._delta = float(np.clip(x[6] + rates[0] * self.dt, -float(p.max_steer_rad), float(p.max_steer_rad)))
        self._F = float(np.clip(x[7] + rates[1] * self.dt, -float(p.max_brake_force_N), float(p.max_drive_force_N)))
        tq = 0.5 * self._F * self._wheel_r  # per rear wheel, N m
        lo, hi = self.m.actuator_ctrlrange[1]
        return np.array([self._delta, np.clip(tq, lo, hi), np.clip(tq, lo, hi)])
