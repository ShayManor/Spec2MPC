"""Task 2: universal sampling MPC. Fast tests run on CPU MJX with tiny sample counts; GPU tests are slow."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from spec2mpc.core import Entity, Program, RobotSpec, Scene, Stage, TaskPublic, Term
from spec2mpc.mpc.sampling import SamplingMPC, clip_to_floors, make_solver
from spec2mpc.mpc.sampling.env import MjSimEnv, make_env

IDQ = np.array([1.0, 0, 0, 0])
TIMING = Path(__file__).parent.parent.parent / "data" / "v2" / "mpc_timing.json"
FAST = dict(horizon_steps=8, n_samples=12)


def _robot(family: str, name: str) -> RobotSpec:
    try:
        from spec2mpc.robots.v2 import load_robot  # Task 3
        r = load_robot(family, name, None)
    except Exception as e:  # menagerie missing on this machine
        pytest.skip(f"robot {family}/{name} unavailable: {e}")
    if not os.path.exists(r.mjcf_path):
        pytest.skip(f"{r.mjcf_path} missing")
    return r


def _ent(id: str, pos, kind: str = "object", extent=None) -> Entity:
    return Entity(id, id, kind, np.r_[np.asarray(pos, float), IDQ], None if extent is None else np.asarray(extent),
                  None, None)


def _goto(frame: str, target: str, until: Term) -> Program:
    return Program([Stage([Term("pose_to", [frame, target], {})], until)])


class Plant:
    # the robot model alone as the plant (same model the MPC plans on)
    def __init__(self, mpc: SamplingMPC):
        self.mpc = mpc
        self.m = mpc.m
        self.d = mujoco.MjData(self.m)
        if self.m.nkey:
            self.d.qpos[:] = self.m.key_qpos[0]
        mujoco.mj_forward(self.m, self.d)

    def frame(self, j: int) -> np.ndarray:
        mujoco.mj_kinematics(self.m, self.d)
        return self.mpc.frames.pos(self.d)[j].copy()

    def run(self, scene, n: int, stop_on_terminal: bool = True, every=None) -> dict:
        speeds = []
        for k in range(n):
            if every is not None:
                scene = every(k, scene)
            u = self.mpc.step(self.d.qpos.copy(), self.d.qvel.copy(), scene)
            if stop_on_terminal and self.mpc.status in ("done", "timeout", "invalid_reference"):
                break
            self.d.ctrl[:] = u
            for _ in range(self.mpc.n_sub):
                mujoco.mj_step(self.m, self.d)
                if self.mpc.arm_dofs.size:
                    speeds.append(np.abs(self.d.qvel[self.mpc.arm_dofs]).max())
        return {"steps": k + 1, "max_joint_speed": max(speeds) if speeds else 0.0, "u": u}


@pytest.fixture(scope="module")
def panda() -> RobotSpec:
    return _robot("arm", "panda")


def _arm_setup(robot: RobotSpec, offset, **kw):
    mpc = SamplingMPC(robot, seed=0, **{**FAST, **kw})
    plant = Plant(mpc)
    ee = plant.frame(robot.frames.index("ee"))
    scene = Scene([_ent("target", ee + np.asarray(offset))], None, 0.0)
    return mpc, plant, scene


def test_arm_reaches_target(panda):
    mpc, plant, scene = _arm_setup(panda, [0.15, 0.2, -0.15], n_samples=24)
    assert abs(np.linalg.norm([0.15, 0.2, -0.15]) - 0.29) < 0.01
    mpc.set_program(_goto("ee", "target", Term("timeout", [], {"t_s": 100.0})), scene)
    plant.run(scene, 200)
    assert np.linalg.norm(plant.frame(1) - scene.get("target").pose[:3]) < 0.02


def test_floor_clip(panda):
    # limit(speed, 1.0) on arm (max_speed_frac 0.5): executed joint speed stays under half the joint speed limit
    assert clip_to_floors({"speed": 1.0}, panda)["speed"] == 0.5
    mpc, plant, scene = _arm_setup(panda, [0.3, 0.3, -0.2])
    prog = Program([Stage([Term("pose_to", ["ee", "target"], {"w": 10.0}), Term("limit", ["speed"], {"value": 1.0})],
                          Term("timeout", [], {"t_s": 100.0}))])
    mpc.set_program(prog, scene)
    out = plant.run(scene, 40)
    lim = mpc.phys["joint_speed_radps"]
    assert mpc.info["floor_clips"] == 1
    assert out["max_joint_speed"] <= 0.5 * lim + 1e-3
    assert out["max_joint_speed"] > 0.2 * lim  # the limit binds, the arm is not just standing still


def test_stage_transition_on_until(panda):
    mpc, plant, scene = _arm_setup(panda, [0.0, 0.1, 0.0])
    scene = Scene(scene.entities + [_ent("target2", scene.get("target").pose[:3] + [0, 0, -0.1])], None, 0.0)
    prog = Program([Stage([Term("pose_to", ["ee", "target"], {})], Term("dist_below", ["ee", "target"], {"d": 0.05})),
                    Stage([Term("pose_to", ["ee", "target2"], {})], Term("timeout", [], {"t_s": 100.0}))])
    mpc.set_program(prog, scene)
    seen = []
    for _ in range(80):
        plant.run(scene, 1)
        seen.append(mpc.stage_index)
        if mpc.stage_index == 1:
            break
    assert seen[0] == 0 and seen[-1] == 1


def test_timeout_reports(panda):
    # unreachable target (5 m away) with until timeout(2.0): ends with timeout at t <= 2.0 + dt
    mpc, plant, scene = _arm_setup(panda, [5.0, 0.0, 0.0])
    mpc.set_program(_goto("ee", "target", Term("timeout", [], {"t_s": 2.0})), scene)
    plant.run(scene, 1000)
    assert mpc.status == "timeout"
    assert mpc.t_s - mpc.dt <= 2.0 + mpc.dt


def test_dropped_entity_stops_safely(panda):
    mpc, plant, scene = _arm_setup(panda, [0.2, 0.2, 0.0])
    mpc.set_program(_goto("ee", "target", Term("timeout", [], {"t_s": 100.0})), scene)
    plant.run(scene, 10)
    q_before = plant.d.qpos.copy()
    empty = Scene([], None, 0.1)
    out = plant.run(empty, 1, stop_on_terminal=False)
    assert mpc.status == "invalid_reference"
    # zero velocity command: position actuators hold the current joint positions
    act_j = [mpc.m.jnt_qposadr[mpc.m.actuator_trnid[i, 0]] for i in range(7)]
    assert np.allclose(out["u"][:7], q_before[act_j], atol=1e-6)
    plant.run(empty, 100, stop_on_terminal=False)
    assert np.abs(plant.d.qvel[mpc.arm_dofs]).max() < 0.05


def test_invalid_program_never_moves(panda):
    mpc, plant, scene = _arm_setup(panda, [0.2, 0.0, 0.0])
    mpc.set_program(_goto("ee", "nope", Term("timeout", [], {"t_s": 1.0})), scene)
    assert mpc.status == "invalid_reference"


def test_methods_run(panda):
    for method in ("mppi", "cem"):
        mpc, plant, scene = _arm_setup(panda, [0.1, 0.0, 0.0], method=method)
        mpc.set_program(_goto("ee", "target", Term("timeout", [], {"t_s": 100.0})), scene)
        d0 = np.linalg.norm(plant.frame(1) - scene.get("target").pose[:3])
        plant.run(scene, 30)
        assert np.linalg.norm(plant.frame(1) - scene.get("target").pose[:3]) < d0


def test_make_solver_selects_by_family(panda):
    assert isinstance(make_solver(panda, **FAST), SamplingMPC)


# ---- SimEnv ----

def _tabletop_env(panda, **kw):
    from spec2mpc.tasks.scenes import make_scene  # Task 5
    task = TaskPublic("t0", "arm", panda, 0, "touch the block", None)
    env = make_env(task, 0, mpc_kw=FAST)
    scene = env.reset(task, 0)
    return env, scene


def test_env_observe_and_qpos(panda):
    env, scene = _tabletop_env(panda)
    ids = {e.id for e in scene.entities}
    assert {"base", "ee"} <= ids
    assert any(e.kind == "object" for e in scene.entities)
    q = env.qpos()
    assert q.shape == (env.rm.nq,) and np.all(np.isfinite(q))
    # the scene follows the robot: move the robot, observe again, the ee entity moves
    ee0 = scene.get("ee").pose[:3].copy()
    env.d.qpos[env.map.q_c[0]] += 0.5
    mujoco.mj_forward(env.m, env.d)
    assert np.linalg.norm(env.observe().get("ee").pose[:3] - ee0) > 0.05
    assert not np.allclose(env.qpos(), q)


def test_env_run_program_trace_scores(panda):
    env, scene = _tabletop_env(panda)
    obj = next(e for e in scene.entities if e.kind == "object")
    prog = Program([Stage([Term("pose_to", ["ee", obj.id], {"z_offset_m": 0.15})],
                          Term("timeout", [], {"t_s": 0.3}))])
    ep = env.run_program(prog, 5.0)
    assert ep.status == "timeout"
    tr = ep.trace
    T = len(tr["t_s"])
    assert T > 0 and tr[f"entity_pose/{obj.id}"].shape == (T, 7) and tr["entity_pose/ee"].shape == (T, 7)
    assert tr["robot_base_height"].shape == (T,) and len(tr["contacts"]) == T
    try:
        from spec2mpc.rubric.v2 import score  # Task 4
    except ImportError:
        return
    from spec2mpc.core import TaskTest
    s = score(TaskTest("t0", [{"pred": "not_fallen", "args": ["robot"], "min_height_m": -1.0}], [], 5.0), tr)
    assert 0.0 <= s["score"] <= 1.0


def test_env_max_s_bounds_episode(panda):
    env, scene = _tabletop_env(panda)
    obj = next(e for e in scene.entities if e.kind == "object")
    ep = env.run_program(_goto("ee", obj.id, Term("timeout", [], {"t_s": 100.0})), 0.2)
    assert ep.status == "timeout" and ep.trace["t_s"][-1] <= 0.2 + env.mpc.dt + 1e-9


CARDS = Path(__file__).parent.parent.parent / "data" / "v2" / "cards" / "train"


@pytest.mark.parametrize("source", ["barn", "tracks", "commonroad", "playground"])
def test_env_source_card_scene(source):
    # source cards build their world from the source module (Task 5b), not the scene generator
    from spec2mpc.core.io import load_json
    from spec2mpc.tasks.cards import source_available
    if not source_available(source):
        pytest.skip(f"{source} assets missing on this machine")
    paths = sorted(CARDS.glob(f"{source}__*.json"))
    if not paths:
        pytest.skip(f"no {source} train cards on this machine")
    pub = load_json(paths[0]).public_view()
    pub = dataclasses.replace(pub, robot=_robot(pub.robot.family, pub.robot.name))
    env = make_env(pub, 0, mpc_kw=FAST)
    scene = env.reset(pub, 0)
    gt_ids = {e.id for e in env.gt0.entities}
    assert "base" in {e.id for e in scene.entities} and gt_ids
    assert {e.id for e in scene.entities if e.kind in ("path", "region")} <= gt_ids
    assert env.qpos().shape == (env.rm.nq,)


def test_env_sim_source_not_implemented(panda):
    task = TaskPublic("robocasa/PickPlace/0", "arm", panda, 0, "put the cup away", None)
    with pytest.raises(NotImplementedError):
        make_env(task, 0, mpc_kw=FAST).reset(task, 0)


# ---- slow: GPU go-to per family, timing, kart ----

GOTO = {"arm": ("panda", "ee"), "mobile_manipulator": ("tiago", "base"), "mobile_base": ("diffdrive", "base"),
        "rover2d": ("rover2d", "base"), "kart": ("kart", "base"), "quadruped": ("go2", "base"),
        "humanoid": ("h1", "base"), "quadrotor": ("skydio_x2", "base")}


def _goto_offset(family: str) -> np.ndarray:
    if family == "arm":
        return np.array([0.0, 0.0, -0.3]) * 0  # arm go-to uses a reachable 0.3 m target below
    return np.array([1.0, 0.0, 0.0])


@pytest.mark.slow
@pytest.mark.parametrize("family", list(GOTO))
def test_go_to_each_family(family):
    name, frame = GOTO[family]
    robot = _robot(family, name)
    if family == "kart":  # the v1 MPCC plans on its own bicycle model; the plant is the MuJoCo kart on a floor
        return _kart_go_to(robot, name)
    mpc = make_solver(robot)
    plant = Plant(mpc)
    j = robot.frames.index(frame)
    start = plant.frame(j)
    off = np.array([0.15, 0.2, -0.15]) if family == "arm" else np.array([1.0, 0.0, 0.0])
    goal = start + off
    scene = Scene([_ent("goal", goal, "region")], None, 0.0)
    if family in ("quadruped", "humanoid"):
        terms = [Term("pose_to", [frame, "goal"], {"w_pos": 1.0}), Term("balance", [], {}),
                 Term("gait", ["trot" if family == "quadruped" else "walk"], {})]
    else:
        terms = [Term("pose_to", [frame, "goal"], {})]
    tol = 0.02 if family == "arm" else 0.15
    mpc.set_program(Program([Stage(terms, Term("dist_below", [frame, "goal"], {"d": tol}))]), scene)
    h0 = plant.frame(robot.frames.index("base"))[2]
    n = int(round(10.0 / mpc.dt))
    t0 = time.perf_counter()
    steps, fell = 0, False
    floor_h = {"quadruped": 0.5, "humanoid": 0.6}.get(family)
    for _ in range(n):
        plant.run(scene, 1)
        steps += 1
        if floor_h is not None and plant.frame(robot.frames.index("base"))[2] < floor_h * h0:
            fell = True
            break
        if mpc.status == "done":
            break
    wall = (time.perf_counter() - t0) / steps
    dist = float(np.linalg.norm(plant.frame(j) - goal))
    _record_timing(family, name, mpc, wall, mpc.status == "done" and not fell, dist, steps * mpc.dt)
    assert not fell, f"{family} fell"
    assert mpc.status == "done", f"{family}: {dist:.3f} m from goal after {steps * mpc.dt:.1f} s"


FLAT = ('<mujoco model="flat"><worldbody><body name="floor"><geom name="floor_g0" type="plane" size="100 100 0.1"/>'
        '</body></worldbody></mujoco>')


def _kart_go_to(robot: RobotSpec, name: str) -> None:
    env = MjSimEnv(robot)
    start = env.reset_world(FLAT, Scene([], None, 0.0), 0).get("base").pose[:3]
    goal = _ent("goal", start + [1.0, 0.0, 0.0], "region")
    env.reset_world(FLAT, Scene([goal], None, 0.0), 0)
    prog = Program([Stage([Term("pose_to", ["base", "goal"], {})], Term("dist_below", ["base", "goal"], {"d": 0.15}))])
    t0 = time.perf_counter()
    ep = env.run_program(prog, 10.0)
    steps = max(len(ep.trace["t_s"]), 1)
    dist = float(np.linalg.norm(ep.trace["entity_pose/base"][-1, :3] - goal.pose[:3]))
    _record_timing("kart", name, env.mpc, (time.perf_counter() - t0) / steps, ep.status == "done", dist,
                   steps * env.mpc.dt)
    assert ep.status == "done", f"kart: {dist:.3f} m from goal"


def _record_timing(family: str, name: str, mpc: SamplingMPC, wall_s: float, ok: bool, dist: float, t: float) -> None:
    import jax
    TIMING.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    data[family] = {"robot": name, "wall_s_per_control_step": wall_s, "control_dt_s": mpc.dt,
                    "horizon_steps": mpc.H, "n_samples": mpc.n_samples, "method": mpc.method,
                    "device": str(jax.devices()[0]), "go_to_success": ok, "final_dist_m": dist, "sim_s": t}
    TIMING.write_text(json.dumps(data, indent=1, sort_keys=True))


KART_LAP = Path(__file__).parent / "fixtures" / "kart_v1_lap.pt"


def _kart_lap(controller_from=None):
    # the first 4 s of a v1 lap (EVC kart, 60 m track, 20 Hz so one control period is one MPC stage) on CPU
    import torch
    from omegaconf import OmegaConf
    from spec2mpc.mpc.theta import default_theta
    from spec2mpc.robots.kart import RuntimeConfig, load_fixed_kart
    from spec2mpc.sim.conditions import sample_conditions
    from spec2mpc.sim.rollout import run_episodes_with_stats
    from spec2mpc.sim.scenes import generate_scenes
    sim_cfg = OmegaConf.load("configs/sim/default.yaml")
    sim_cfg.scene.lap.length_m, sim_cfg.scene.lap.half_width_m = [60.0, 60.0], [3.0, 3.5]
    sim_cfg.scene.n_laps, sim_cfg.grid.res_m = 1, 0.25
    sim_cfg.episode.timeout_cap_lap_s = 4.0
    mpc_cfg = OmegaConf.load("configs/mpc/kart.yaml")
    rt = RuntimeConfig(control_rate_hz=20.0, solver_iters=10, perception_rate_hz=10.0)
    robot = load_fixed_kart(OmegaConf.load("configs/robots/kart.yaml"), "evc", rt)
    # exact localization: v1 seeds its warm-start progress from the true state, the controller sees only x_hat
    z = torch.zeros(1)
    robot = dataclasses.replace(robot, perception=dataclasses.replace(robot.perception, loc_noise_std_m=z,
                                                                       loc_bias_m=z))
    scene = generate_scenes(1, sim_cfg, torch.Generator().manual_seed(3), "lap")
    cond = sample_conditions(1, sim_cfg, torch.Generator().manual_seed(4))
    ctrl = controller_from(scene.path) if controller_from else None
    traj, _ = run_episodes_with_stats(robot, scene, default_theta(1, mpc_cfg), cond, sim_cfg, mpc_cfg,
                                      torch.Generator().manual_seed(5), controller=ctrl)
    return torch.cat([traj.x[0], traj.u[0]], dim=1)  # [T, NX + NU]


@pytest.mark.slow
def test_kart_wrapper_matches_v1():
    import torch
    torch.manual_seed(0)
    kart = _robot("kart", "kart")
    solver = make_solver(kart)
    assert type(solver).__name__ == "KartMPCC"
    ref = _kart_lap()
    if not KART_LAP.exists():
        KART_LAP.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ref, KART_LAP)
    stored = torch.load(KART_LAP)
    got = _kart_lap(solver.v1_controller)
    assert got.shape == stored.shape and ref.shape == stored.shape
    assert float((got - stored).abs().max()) < 1e-6
    assert float((ref - stored).abs().max()) < 1e-6

