"""Universal sampling MPC on MJX: predictive sampling (default), MPPI, or CEM over spline-parameterized controls.

Each control step: shift the nominal spline, sample perturbed splines plus the nominal and a hold spline, roll
all of them out in the robot's own MuJoCo model (MJX, batched), score them with the active stage's terms plus
hard-limit penalties, and return the first control of the chosen spline after an execution-time speed filter
on the CPU model. Objects are targets and clearance geometry only; the planning model is the robot alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from mujoco import mjx

from spec2mpc.core import Program, RobotSpec, Scene, load_family_config
from spec2mpc.mpc.sampling.adapter import clip_term, physical_limits, stage_limits
from spec2mpc.mpc.sampling.costs import BUILDERS, NON_COST_TERMS, Ctx, entity_ids
from spec2mpc.mpc.sampling.stages import (ARM_LIKE, GROUND_FAMILIES, Frames, Until, foot_frames, gripper_actuators,
                                          hold_ctrl, resolve_frames, root_body)

Status = Literal["running", "done", "timeout", "invalid_reference", "floor_clip"]
CFG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "v2" / "mpc" / "sampling.yaml"
TERMINAL = ("done", "timeout", "invalid_reference")


def load_cfg(family: str) -> dict:
    raw = yaml.safe_load(CFG_PATH.read_text())
    cfg = {k: v for k, v in raw.items() if k != "families"}
    cfg.update(raw.get("families", {}).get(family, {}))
    return cfg


def with_floor(robot: RobotSpec) -> mujoco.MjModel:
    # the robot model with its randomized params (Task 3 apply_params, as build_model does) on a floor plane
    from spec2mpc.robots.v2 import apply_params
    spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
    return apply_params(spec.compile(), robot.params)


def planning_model(robot: RobotSpec) -> mujoco.MjModel:
    # ground families plan on a floor; arms plan without contacts; legged robots keep only foot and floor contacts
    if robot.family in GROUND_FAMILIES:
        m = with_floor(robot)
    else:
        from spec2mpc.robots.v2 import build_model
        m = build_model(robot)
    cfg = load_cfg(robot.family)
    m.opt.iterations = min(int(m.opt.iterations), int(cfg["plan_iterations"]))  # MJX constraint solver budget
    m.opt.ls_iterations = min(int(m.opt.ls_iterations), int(cfg["plan_ls_iterations"]))
    if robot.family == "arm":
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if robot.family in GROUND_FAMILIES - {"quadruped", "humanoid"}:
        # wheeled robots: robot geoms touch only the floor (no self-contact, and MJX lacks mesh-cylinder pairs)
        for g in range(m.ngeom):
            floor = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
            m.geom_contype[g], m.geom_conaffinity[g] = (1, 1) if floor else (0, 1)
    if robot.family in ("quadruped", "humanoid"):
        fr = resolve_frames(m, robot)
        keep_geoms = {i for k, i, n in zip(fr.kinds, fr.ids, fr.names) if k == "geom" and "foot" in n}
        keep_bodies = {i for k, i, n in zip(fr.kinds, fr.ids, fr.names) if k == "body" and "foot" in n}
        for g in range(m.ngeom):
            floor = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
            if not (floor or g in keep_geoms or m.geom_bodyid[g] in keep_bodies):
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m


def _interp_matrix(t_query: np.ndarray, t_knots: np.ndarray) -> np.ndarray:
    # linear interpolation weights [Q, K], clamped at both ends
    W = np.zeros((len(t_query), len(t_knots)))
    for q, t in enumerate(np.clip(t_query, t_knots[0], t_knots[-1])):
        k = min(int(np.searchsorted(t_knots, t, side="right")) - 1, len(t_knots) - 2)
        a = (t - t_knots[k]) / max(t_knots[k + 1] - t_knots[k], 1e-12)
        W[q, k], W[q, k + 1] = 1.0 - a, a
    return W


def _actuated_joints(m: mujoco.MjModel, i: int) -> list[int]:
    # joints moved by actuator i: its joint, or the joints a fixed tendon couples (tendon-driven grippers)
    tid = int(m.actuator_trnid[i, 0])
    if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_TENDON):
        a = m.tendon_adr[tid]
        return [int(m.wrap_objid[k]) for k in range(a, a + m.tendon_num[tid])
                if int(m.wrap_type[k]) == int(mujoco.mjtWrap.mjWRAP_JOINT)]
    return [tid] if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT) else []


class SamplingMPC:
    def __init__(self, robot: RobotSpec, horizon_steps: int | None = None, n_samples: int | None = None,
                 method: Literal["ps", "mppi", "cem"] = "ps", *, model: mujoco.MjModel | None = None,
                 seed: int = 0):
        self.cfg = load_cfg(robot.family)
        n_samples = n_samples or int(robot.runtime.get("n_samples", 64))
        self.robot, self.method, self.n_samples = robot, method, max(int(n_samples), 3)
        self.m = model if model is not None else planning_model(robot)
        self.d = mujoco.MjData(self.m)
        self.mx = mjx.put_model(self.m)
        rate = float(robot.runtime.get("control_rate_hz", self.cfg["control_rate_hz"]))
        self.n_sub = max(1, round(1.0 / (rate * self.m.opt.timestep)))
        self.dt = self.n_sub * self.m.opt.timestep  # control period, s
        self.H = int(horizon_steps or max(2, round(robot.runtime.get("horizon_s", self.cfg["horizon_s"]) / self.dt)))
        self.K = max(2, min(int(self.cfg["n_knots"]), self.H))
        tk = np.linspace(0.0, (self.H - 1) * self.dt, self.K)
        self._tk = tk
        self.B = _interp_matrix(np.arange(self.H) * self.dt, tk)  # [H, K]
        self.S = _interp_matrix(tk + self.dt, tk)  # [K, K] knot shift by one control period
        lim = self.m.actuator_ctrllimited.astype(bool)
        self.lo = np.where(lim, self.m.actuator_ctrlrange[:, 0], -1.0)
        self.hi = np.where(lim, self.m.actuator_ctrlrange[:, 1], 1.0)
        self.frames: Frames = resolve_frames(self.m, robot)
        self.feet = foot_frames(robot.frames)
        self.gripper = gripper_actuators(self.m, robot.actuators) if robot.family in ARM_LIKE else []
        self.root = root_body(self.m)
        self.phys = physical_limits(robot, self.cfg["nominal_limits"])
        # joints the speed limit covers: actuated hinge and slide joints, not wheels (the base speed covers them)
        act_j = sorted({j for i in range(self.m.nu) for j in _actuated_joints(self.m, i)})
        self.arm_dofs = np.array([self.m.jnt_dofadr[j] for j in act_j
                                  if int(self.m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                                 int(mujoco.mjtJoint.mjJNT_SLIDE))
                                  and "wheel" not in (self.m.joint(j).name or "").lower()], np.int32)
        if self.gripper:  # finger joints are not arm joints for the speed limit
            g = {int(self.m.jnt_dofadr[j]) for i in self.gripper for j in _actuated_joints(self.m, i)}
            self.arm_dofs = np.array([i for i in self.arm_dofs if i not in g], np.int32)
        bias = self.m.actuator_biasprm
        self._pos_act = ((self.m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE) & (bias[:, 1] < 0)
                         & (self.m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))  # [nu] joint position servos
        self.key = jax.random.PRNGKey(seed)
        self._plans: dict[Any, Any] = {}
        self.program: Program | None = None
        self.stage_index, self.status = 0, "running"
        self.info: dict = {"floor_clips": 0, "errors": []}
        self.t_s = 0.0
        self._nominal: np.ndarray | None = None
        self._stop: np.ndarray | None = None

    # ---- program and stages ----

    def set_program(self, program: Program, scene: Scene) -> None:
        self.program, self.stage_index, self.status, self.t_s = program, 0, "running", 0.0
        self.info = {"floor_clips": 0, "errors": []}
        self._nominal, self._stop = None, None
        try:
            from spec2mpc.program.validate import validate  # Task 1
            errs = validate(program, self.robot, scene)
        except ImportError:
            errs = []
        if errs:
            self.info["errors"] = errs
            self.status = "invalid_reference"
            return
        self._stage_ready = False

    def _enter_stage(self, scene: Scene) -> None:
        st = self.program.stages[self.stage_index]
        mujoco.mj_kinematics(self.m, self.d)
        fpos, fmat = self.frames.pos(self.d), self.frames.mat(self.d)
        lims, clipped = stage_limits(st, self.robot)
        terms = []
        for t in st.terms:
            t, c = clip_term(t, self.robot)
            clipped |= c
            terms.append(t)
        if clipped:
            self.info["floor_clips"] += 1
            self.status = "floor_clip"
        cfg = dict(self.cfg, vmax=self.phys["base_speed_mps"])
        ctx = Ctx(list(self.robot.frames), scene, fpos, fmat, cfg, self.dt, self.feet,
                  self.robot.family in GROUND_FAMILIES, float(self.d.xpos[self.root][2]))
        self._fns, self._srcs = [], []
        for t in terms:
            if t.type in NON_COST_TERMS:
                continue
            fn, srcs = BUILDERS[t.type](t, ctx)
            self._fns.append(fn)
            self._srcs.append(srcs)
        self._until = Until(st.until, self.frames, self.gripper)
        self._t_stage0 = self.t_s
        # fixed actuators: a gripper term pins the gripper actuators open or closed
        self._fixed = np.zeros(self.m.nu, bool)
        self._fixed_val = np.zeros(self.m.nu)
        for t in terms:
            if t.type == "gripper" and self.gripper:
                v = self.hi if t.args[0] == "open" else self.lo
                if t.args[0] == "hold":
                    v = np.asarray(self.d.ctrl, np.float64)
                self._fixed[self.gripper] = True
                self._fixed_val[self.gripper] = v[self.gripper]
        # hard limits as absolute values
        mg = self.cfg["safety_margin"]
        speed = lims.get("speed", 1.0)
        self._lim = {"joint_speed": speed * self.phys["joint_speed_radps"],
                     "base_speed": speed * self.phys["base_speed_mps"],
                     "tilt": lims.get("tilt", 10.0) * self.phys["tilt_rad"] if "tilt" in lims else 10.0,
                     "force": lims.get("force", lims.get("effort", 1.0))}
        self._lim_plan = {k: np.float32(v * mg) for k, v in self._lim.items()}
        # motor actuators: the force fraction narrows the control range directly
        frac = min(lims.get("force", 1.0), lims.get("effort", 1.0))
        motor = self.m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE
        self._lo = np.where(motor, self.lo * frac, self.lo)
        self._hi = np.where(motor, self.hi * frac, self.hi)
        key = (st.until.type, tuple(str(t) for t in terms), tuple(sorted(e.id for e in scene.entities)))
        if key not in self._plans:
            self._plans[key] = self._build_plan(self._fns)
        self._plan = self._plans[key]
        self._stage_ready = True

    def _referenced_ids(self, scene: Scene) -> list[str]:
        ids = [i for srcs in self._srcs for s in srcs for i in entity_ids(s)]
        return ids + self._until.entity_ids(scene)

    # ---- planning ----

    def _features(self, d) -> dict:
        return {"fpos": self.frames.pos(d), "fmat": self.frames.mat(d), "qpos": d.qpos, "qvel": d.qvel,
                "ctrl": d.ctrl, "force": d.actuator_force, "base_pos": d.xpos[self.root],
                "base_mat": d.xmat[self.root].reshape(3, 3), "com": d.subtree_com[self.root], "t": d.time}

    def _build_plan(self, fns: list):
        mx, n_sub, B = self.mx, self.n_sub, jnp.asarray(self.B, jnp.float32)
        dx0 = mjx.make_data(mx)
        w = self.cfg["weights"]
        arm = self.robot.family in ARM_LIKE
        ground = self.robot.family not in ("arm",)
        arm_dofs = jnp.asarray(self.arm_dofs)
        dt = self.dt
        method, N = self.method, self.n_samples
        temp, iters = float(self.cfg["temperature"]), int(self.cfg["cem_iters"])
        n_elite = max(2, int(self.cfg["cem_elite_frac"] * N))
        feats = self._features

        def rollout(d, U):  # U [H, nu] -> features [H, ...]
            def ctrl_step(d, u):
                d = d.replace(ctrl=u)
                d = jax.lax.fori_loop(0, n_sub, lambda i, d: mjx.step(mx, d), d)
                return d, feats(d)
            return jax.lax.scan(ctrl_step, d, U)[1]

        pos_act = jnp.asarray(self._pos_act)

        def cost(knots, d, f0, dyn, lim, fixed, fixed_val, hold):
            U = jnp.where(fixed, fixed_val, B @ knots)  # [H, nu]
            f = rollout(d, U)
            feat = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b], 0), f0, f)  # [H+1, ...]
            c = sum(jnp.sum(fn(feat, dd)) for fn, dd in zip(fns, dyn)) if fns else 0.0
            c = c + w["effort"] * jnp.sum(jnp.where(pos_act, U - hold, U) ** 2)  # position actuators: vs hold
            viol = 0.0
            if arm and arm_dofs.shape[0]:
                viol += jnp.sum(jnp.maximum(jnp.abs(feat["qvel"][1:, arm_dofs]) - lim["joint_speed"], 0.0))
            if ground:
                sp = jnp.sqrt(jnp.sum(jnp.diff(feat["base_pos"], axis=0) ** 2, -1) + 1e-12) / dt
                viol += jnp.sum(jnp.maximum(sp - lim["base_speed"], 0.0))
                tilt = jnp.arccos(jnp.clip(feat["base_mat"][1:, 2, 2], -1.0, 1.0))
                viol += jnp.sum(jnp.maximum(tilt - lim["tilt"], 0.0))
            return c + w["violation"] * viol

        def plan(qpos, qvel, t, nominal, hold, key, dyn, lim, fixed, fixed_val, lo, hi, sigma):
            d = mjx.forward(mx, dx0.replace(qpos=qpos, qvel=qvel, time=t))
            f0 = feats(d)
            score = jax.vmap(lambda k: cost(k, d, f0, dyn, lim, fixed, fixed_val, hold))

            def sample(key, mean, std, n):
                eps = jax.random.normal(key, (n,) + mean.shape)
                return jnp.clip(mean + std * eps, lo, hi)

            if method == "cem":
                def body(i, carry):
                    mean, std, key, best, best_c = carry
                    key, k = jax.random.split(key)
                    cand = jnp.concatenate([mean[None], sample(k, mean, std, N - 1)], 0)
                    c = score(cand)
                    idx = jnp.argsort(c)[:n_elite]
                    el = cand[idx]
                    better = c[idx[0]] < best_c
                    return (el.mean(0), el.std(0) + 1e-3 * sigma, key, jnp.where(better, el[0], best),
                            jnp.minimum(best_c, c[idx[0]]))
                init = (nominal, jnp.broadcast_to(sigma, nominal.shape), key, nominal, jnp.inf)
                _, _, _, best, best_c = jax.lax.fori_loop(0, iters, body, init)
                return best, best_c
            cand = jnp.concatenate([nominal[None], jnp.broadcast_to(hold, nominal.shape)[None],
                                    sample(key, nominal, sigma, N - 2)], 0)  # [N, K, nu]
            c = score(cand)
            if method == "mppi":
                cmin = jnp.min(c)
                wts = jax.nn.softmax(-(c - cmin) / (temp * (jnp.std(c) + 1e-9)))
                mean = jnp.einsum("n,nku->ku", wts, cand)
                return mean, cmin
            i = jnp.argmin(c)
            return cand[i], c[i]

        return jax.jit(plan)

    # ---- control ----

    def stop_control(self) -> np.ndarray:
        # latched at the first stop: servos keep the targets of that moment, so the robot comes to rest
        if self._stop is None:
            mujoco.mj_forward(self.m, self.d)
            self._stop = hold_ctrl(self.m, self.d)
        return self._stop.copy()

    def _speeds(self, d: mujoco.MjData) -> tuple[float, float]:
        # (max joint speed over arm dofs, base linear speed) on the CPU model
        js = float(np.max(np.abs(d.qvel[self.arm_dofs]))) if self.arm_dofs.size else 0.0
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.root, v, 0)
        return js, float(np.linalg.norm(v[3:]))

    def _ok(self, qpos: np.ndarray, qvel: np.ndarray, u: np.ndarray) -> bool:
        d = mujoco.MjData(self.m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos, qvel, u
        arm, ground = self.robot.family in ARM_LIKE, self.robot.family != "arm"
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, d)
            js, bs = self._speeds(d)
            if (arm and js > self._lim["joint_speed"]) or (ground and bs > self._lim["base_speed"]):
                return False
        return True

    def _filter(self, qpos: np.ndarray, qvel: np.ndarray, u: np.ndarray, hold: np.ndarray) -> np.ndarray:
        # largest step from hold toward u that keeps the speed limit over the next control period
        if self._ok(qpos, qvel, u):
            return u
        lo, hi = 0.0, 1.0
        for _ in range(int(self.cfg["filter_iters"])):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self._ok(qpos, qvel, hold + mid * (u - hold)) else (lo, mid)
        return hold + lo * (u - hold)

    def step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray:
        # one control step: check stage transitions, plan, return the control [nu] for this control period
        self.d.qpos[:], self.d.qvel[:] = qpos, qvel
        self.d.time = self.t_s
        try:
            return self._step(np.asarray(qpos), np.asarray(qvel), scene)
        finally:
            self.t_s += self.dt

    def _step(self, qpos: np.ndarray, qvel: np.ndarray, scene: Scene) -> np.ndarray:
        if self.program is None or self.status in TERMINAL:
            return self.stop_control()
        for _ in range(len(self.program.stages) + 1):
            if not self._stage_ready:
                try:
                    self._enter_stage(scene)
                except KeyError:
                    self.status = "invalid_reference"
                    return self.stop_control()
            try:
                for i in self._referenced_ids(scene):
                    scene.get(i)
                mujoco.mj_forward(self.m, self.d)
                fired = self._until.fired(self.m, self.d, scene, self.t_s - self._t_stage0)
            except KeyError:  # perception dropped a referenced entity: stop safely
                self.status = "invalid_reference"
                return self.stop_control()
            if not fired:
                break
            last = self.stage_index == len(self.program.stages) - 1
            if last:
                self.status = "timeout" if self._until.term.type == "timeout" else "done"
                return self.stop_control()
            self.stage_index += 1
            self._stage_ready = False
            self._nominal = None
        mujoco.mj_forward(self.m, self.d)
        hold = hold_ctrl(self.m, self.d)
        hold = np.where(self._fixed, self._fixed_val, hold)
        if self._nominal is None:
            self._nominal = np.broadcast_to(hold, (self.K, self.m.nu)).copy()
        else:
            self._nominal = self.S @ self._nominal
        dyn = [[s.dyn(scene) for s in srcs] for srcs in self._srcs]
        self.key, k = jax.random.split(self.key)
        sigma = self.cfg["noise_frac"] * 0.5 * (self._hi - self._lo)
        # position servos: the target moves by about the speed limit times servo_lead_s per sample, so samples
        # stay near the speed limit and improvements accumulate through the warm-started nominal
        sigma = np.where(self._pos_act, np.minimum(sigma, self._lim["joint_speed"] * float(self.cfg["servo_lead_s"])),
                         sigma)
        f32 = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
        best, c = self._plan(f32(qpos), f32(qvel), f32(self.t_s), f32(self._nominal), f32(hold), k,
                             jax.tree_util.tree_map(f32, dyn), self._lim_plan, jnp.asarray(self._fixed),
                             f32(self._fixed_val), f32(self._lo), f32(self._hi), f32(sigma))
        self._nominal = np.asarray(best, np.float64)
        self.info["cost"] = float(c)
        u = np.where(self._fixed, self._fixed_val, self.B[0] @ self._nominal)
        u = np.clip(u, self._lo, self._hi)
        return self._filter(qpos, qvel, u, hold)


def make_solver(robot: RobotSpec, **kw) -> SamplingMPC | Any:
    # configs/v2/robots/<family>.yaml: solver selects sampling or the v1 kart MPCC
    solver = load_family_config(robot.family).get("solver", "sampling")
    if solver == "kart_mpcc":
        from spec2mpc.mpc.sampling.kart import KartMPCC as K
        return K(robot, **kw)
    return SamplingMPC(robot, **kw)

