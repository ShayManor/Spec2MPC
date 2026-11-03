"""MuJoCo SimEnv for the generated suite: Task 5 world + Task 3 robot, executed by the family solver.

The plant is the composed world (objects have dynamics there); the MPC plans on the robot alone. The episode
trace follows the format at the top of spec2mpc/rubric/v2/predicates.py.
"""
from __future__ import annotations

import time
from typing import Callable

import mujoco
import numpy as np

from spec2mpc.core import Entity, Episode, Program, RobotSpec, Scene, TaskPublic
from spec2mpc.mpc.sampling.solver import TERMINAL, load_cfg, make_solver

# default scene kind per family when the task's goal does not name one
DEFAULT_KIND = {"arm": "tabletop", "mobile_manipulator": "floor_obstacles", "mobile_base": "floor_obstacles",
                "rover2d": "course", "kart": "track", "quadruped": "terrain", "humanoid": "floor_obstacles",
                "quadrotor": "airspace"}
SIM_SOURCES = ("robocasa", "metaworld", "robosuite")  # source cards that need their own simulator
_OPT_FIELDS = ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations", "noslip_iterations")


def scene_kind(task: TaskPublic) -> str:
    if task.goal:
        try:
            from spec2mpc.tasks.goals import GOALS  # Task 5
            g = GOALS[task.goal] if isinstance(GOALS, dict) else next(x for x in GOALS if x.name == task.goal)
            return g.scene_kind(task.scene_seed)
        except (ImportError, KeyError, StopIteration, AttributeError):
            pass
    return DEFAULT_KIND[task.family]


class _Map:
    # index maps from the robot-only model into the composed world model
    def __init__(self, rm: mujoco.MjModel, m: mujoco.MjModel):
        qadr, dadr, bodies = [], [], []
        for b in range(1, rm.nbody):
            cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rm.body(b).name)
            if cb < 0:
                raise KeyError(f"robot body {rm.body(b).name!r} missing from the composed model")
            bodies.append(cb)
            for k in range(rm.body_jntnum[b]):
                jr, jc = rm.body_jntadr[b] + k, m.body_jntadr[cb] + k
                nq = {0: 7, 1: 4}.get(int(rm.jnt_type[jr]), 1)
                nv = {0: 6, 1: 3}.get(int(rm.jnt_type[jr]), 1)
                qadr += [(rm.jnt_qposadr[jr] + i, m.jnt_qposadr[jc] + i) for i in range(nq)]
                dadr += [(rm.jnt_dofadr[jr] + i, m.jnt_dofadr[jc] + i) for i in range(nv)]
        self.q_r, self.q_c = (np.array(x, np.int64) for x in zip(*qadr)) if qadr else (np.zeros(0, np.int64),) * 2
        self.v_r, self.v_c = (np.array(x, np.int64) for x in zip(*dadr)) if dadr else (np.zeros(0, np.int64),) * 2
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, rm.actuator(i).name)
                             for i in range(rm.nu)], np.int64)
        self.bodies = set(bodies)
        self.nq_r, self.nv_r = rm.nq, rm.nv


def compose(world_xml: str, robot: RobotSpec) -> tuple[mujoco.MjModel, mujoco.MjModel, _Map]:
    # -> (world model with the robot at the origin, robot-only model, index maps); robot params applied to the
    # robot's bodies, geoms, and dofs only (Task 3 apply_params semantics)
    from spec2mpc.robots.v2 import build_model
    rm = build_model(robot)
    w = mujoco.MjSpec.from_string(world_xml)
    c = mujoco.MjSpec.from_file(robot.mjcf_path)
    w.attach(c, frame=w.worldbody.add_frame(pos=[0, 0, 0]), prefix="")
    m = w.compile()
    for f in _OPT_FIELDS:
        setattr(m.opt, f, getattr(rm.opt, f))
    mp = _Map(rm, m)
    p = robot.params
    for b in mp.bodies:
        m.body_mass[b] *= p.get("mass_scale", 1.0)
        m.body_inertia[b] *= p.get("mass_scale", 1.0)
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            m.geom_friction[g, 0] *= p.get("friction_scale", 1.0)
    m.dof_damping[mp.v_c] *= p.get("damping_scale", 1.0)
    return m, rm, mp


class MjSimEnv:
    def __init__(self, robot: RobotSpec, kind: str | None = None, *, mpc_kw: dict | None = None, seed: int = 0):
        self.robot, self.kind, self.mpc_kw = robot, kind, dict(mpc_kw or {})
        self.ecfg = load_cfg(robot.family)["env"]
        self.mpc = None
        self.rng = np.random.default_rng(seed)
        self.task_id = ""

    # ---- setup ----

    def reset(self, task: TaskPublic, seed: int) -> Scene:
        self.robot, self.task_id = task.robot, task.task_id
        xml, self.gt0 = self.world_of(task)
        return self.reset_world(xml, self.gt0, seed)

    def world_of(self, task: TaskPublic) -> tuple[str, Scene]:
        # (MJCF world, ground-truth Scene): source cards (task id "<source>/<task>/<seed>") from their source
        # module, generated cards from the Task 5 scene generator
        import importlib
        from spec2mpc.tasks.cards import SOURCE_MODULES  # Task 5b
        src = task.task_id.split("/", 1)[0] if "/" in task.task_id else "generated"
        if src in SIM_SOURCES:
            raise NotImplementedError(f"{src} cards run in their own simulator env (envs/robocasa), not MjSimEnv")
        if src in SOURCE_MODULES:
            return importlib.import_module(f"spec2mpc.tasks.sources.{SOURCE_MODULES[src]}").scene(task)
        from spec2mpc.tasks.scenes import make_scene  # Task 5
        return make_scene(self.kind or scene_kind(task), self.robot, task.scene_seed)

    def reset_world(self, world_xml: str, gt: Scene, seed: int) -> Scene:
        # also used directly by tests and by callers that build their own world
        self.gt0 = gt
        self.m, self.rm, self.map = compose(world_xml, self.robot)
        self.d = mujoco.MjData(self.m)
        if self.rm.nkey:
            self.d.qpos[self.map.q_c] = self.rm.key_qpos[0][self.map.q_r]
        mujoco.mj_forward(self.m, self.d)
        self.rng = np.random.default_rng(seed)
        if self.mpc is None:
            self.mpc = make_solver(self.robot, **self.mpc_kw)
        self.n_sub = max(1, round(self.mpc.dt / self.m.opt.timestep))
        rate = float(self.robot.perception.get("rate_hz", 1.0 / self.mpc.dt))
        self.perc_every = max(1, round(1.0 / (rate * self.mpc.dt)))
        self._last_seen: dict[str, tuple[float, Entity]] = {}
        self._static = [e for e in gt.entities if e.kind in ("path", "region")]
        self._ids = [e.id for e in gt.entities]
        self._body = {e.id: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, e.id) for e in gt.entities}
        self._joint = {e.id: j for e in gt.entities
                       if (j := mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{e.id}_joint")) >= 0}
        self._scene = self.observe()
        return self._scene

    # ---- observation ----

    def qpos(self) -> np.ndarray:
        # robot joint positions in the robot model's layout [nq_robot]
        q = np.zeros(self.map.nq_r)
        q[self.map.q_r] = self.d.qpos[self.map.q_c]
        return q

    def qvel(self) -> np.ndarray:
        v = np.zeros(self.map.nv_r)
        v[self.map.v_r] = self.d.qvel[self.map.v_c]
        return v

    def robot_frames(self) -> list[Entity]:
        from spec2mpc.robots.v2 import frame_pose, frame_ref
        return [Entity(f, f, "robot_frame", frame_pose(self.m, self.d, *frame_ref(self.robot, f)), None, None, None)
                for f in self.robot.frames]

    def observe(self) -> Scene:
        # perceived scene: ground truth plus profile noise (Task 7), registry robot frames, the static paths and
        # regions of the world, and a short track hold so a one-frame miss does not drop an entity
        from spec2mpc.perception.v2.ground_truth import perceive_gt
        t = float(self.d.time)
        seen = perceive_gt(self.m, self.d, self.robot, self.robot.perception, self.rng)
        for e in seen.entities:
            if e.kind != "robot_frame":
                self._last_seen[e.id] = (t, e)
        hold = float(self.ecfg["track_hold_s"])
        objs = [e for ts, e in self._last_seen.values() if t - ts <= hold + 1e-9]
        ids = {e.id for e in objs}
        return Scene(self.robot_frames() + objs + [e for e in self._static if e.id not in ids], None, t)

    # ---- execution ----

    def _apply(self, u: np.ndarray) -> None:
        self.d.ctrl[self.map.act] = u
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)

    def run_program(self, program: Program, max_s: float) -> Episode:
        t0 = time.perf_counter()
        scene = self.observe()
        self.mpc.set_program(program, scene)
        log, k = self._log_init(), 0
        t_end = self.d.time + max_s
        status = "running"
        while True:
            if k % self.perc_every == 0:
                scene = self.observe()
            u = self.mpc.step(self.qpos(), self.qvel(), scene)
            if self.mpc.status in TERMINAL:
                status = self.mpc.status
                if status == "invalid_reference":  # stop command for one period so the robot is left at rest
                    self._apply(u)
                    self._log(log)
                break
            self._apply(u)
            self._log(log)
            k += 1
            if self.d.time >= t_end - 1e-9:
                status = "timeout"
                break
        self._scene = self.observe()
        info = getattr(self.mpc, "info", {})
        return Episode(self.task_id, "", [program], self._log_done(log, info), status, time.perf_counter() - t0)

    def run_actions(self, policy: Callable[[Scene], np.ndarray], max_s: float) -> Episode:
        t0 = time.perf_counter()
        log, k = self._log_init(), 0
        t_end = self.d.time + max_s
        scene = self.observe()
        while self.d.time < t_end - 1e-9:
            if k % self.perc_every == 0:
                scene = self.observe()
            self._apply(np.asarray(policy(scene), np.float64))
            self._log(log)
            k += 1
        self._scene = self.observe()
        return Episode(self.task_id, "", [], self._log_done(log, {}), "done", time.perf_counter() - t0)

    # ---- trace (spec2mpc/rubric/v2/predicates.py format) ----

    def _log_init(self) -> dict:
        return {"t_s": [], "pose": {i: [] for i in self._ids + list(self.robot.frames)}, "open": {i: [] for i in self._joint},
                "contacts": [], "h": [], "base": [], "energy": [], "e": 0.0}

    def _entity_of_body(self, b: int) -> str:
        if b in self.map.bodies:
            return f"robot:{self.m.body(b).name}"
        while b > 0:
            name = self.m.body(b).name
            if name in self._body:
                return name
            b = int(self.m.body_parentid[b])
        return "floor" if "floor" in self._body else "world"

    def _log(self, log: dict) -> None:
        m, d = self.m, self.d
        log["t_s"].append(float(d.time))
        frames = {e.id: e.pose for e in self.robot_frames()}
        static = {e.id: e.pose for e in self.gt0.entities}
        for i in log["pose"]:
            if i in frames:
                log["pose"][i].append(frames[i])
            elif self._body.get(i, -1) >= 0:
                b = self._body[i]
                log["pose"][i].append(np.r_[d.xpos[b], d.xquat[b]])
            else:
                log["pose"][i].append(static[i])
        for i, j in self._joint.items():
            lo, hi = m.jnt_range[j]
            log["open"][i].append(float(np.clip((d.qpos[m.jnt_qposadr[j]] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)))
        cs, f6 = [], np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            mujoco.mj_contactForce(m, d, c, f6)
            a, b = self._entity_of_body(int(m.geom_bodyid[con.geom1])), self._entity_of_body(int(m.geom_bodyid[con.geom2]))
            cs.append((a, b, float(abs(f6[0]))))
        log["contacts"].append(cs)
        base = frames.get("base", next(iter(frames.values())) if frames else np.r_[0, 0, 0, 1.0, 0, 0, 0])
        log["base"].append(base)
        log["h"].append(float(base[2]))
        dt = self.n_sub * m.opt.timestep
        log["e"] += float(np.sum(np.abs(d.actuator_force[self.map.act] * d.actuator_velocity[self.map.act]))) * dt
        log["energy"].append(log["e"])

    def _log_done(self, log: dict, info: dict) -> dict:
        T = len(log["t_s"])
        tr: dict = {"t_s": np.asarray(log["t_s"], np.float64)}
        for i, ps in log["pose"].items():
            tr[f"entity_pose/{i}"] = np.asarray(ps, np.float64).reshape(T, 7)
        for e in self.gt0.entities:
            if e.extent is not None:
                tr[f"entity_extent/{e.id}"] = np.asarray(e.extent, np.float64)
            if e.kind == "region" and e.points is not None:
                tr[f"entity_points/{e.id}"] = np.asarray(e.points, np.float64)
        for i, v in log["open"].items():
            tr[f"open_frac/{i}"] = np.asarray(v, np.float64)
        contacts = np.empty(T, dtype=object)
        for k, c in enumerate(log["contacts"]):
            contacts[k] = c
        tr["contacts"] = contacts
        tr["robot_base_height"] = np.asarray(log["h"], np.float64)
        tr["robot_base_pose"] = np.asarray(log["base"], np.float64).reshape(T, 7)
        tr["energy_J"] = np.asarray(log["energy"], np.float64)
        tr["floor_clips"] = np.full(T, float(info.get("floor_clips", 0)))
        return tr


def make_env(task: TaskPublic, seed: int = 0, **kw) -> MjSimEnv:
    # eval harness factory (scripts/v2/eval_suite.py --env-factory spec2mpc.mpc.sampling.env:make_env)
    return MjSimEnv(task.robot, kw.pop("kind", None), seed=seed, **kw)
