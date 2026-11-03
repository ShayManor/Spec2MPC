from __future__ import annotations

import re

import mujoco
import numpy as np

from spec2mpc.core.types import Entity, RobotSpec, Scene

N_POINTS = 256  # surface points per entity


def _label(m: mujoco.MjModel, name: str) -> str:
    # optional <custom><text name="label:<body>" data="..."/></custom>, else "red_mug_0" -> "red mug"
    for i in range(m.ntext):
        if m.text(i).name == f"label:{name}":
            return m.text(i).data
    return re.sub(r"_\d+$", "", name).replace("_", " ")


def _frame_pose(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> np.ndarray | None:
    # [7] world pose of a robot frame: site first, then body
    q = np.zeros(4)
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
        s = d.site(name)
        mujoco.mju_mat2Quat(q, s.xmat)
        return np.concatenate([s.xpos, q])
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
        b = d.body(name)
        return np.concatenate([b.xpos, b.xquat])
    return None


class _G:  # geom type ids as plain ints (pybind enums do not compare inside tuples)
    mjGEOM_SPHERE, mjGEOM_ELLIPSOID, mjGEOM_BOX = (int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                                   int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_BOX))
    mjGEOM_CYLINDER, mjGEOM_CAPSULE, mjGEOM_MESH = (int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                                                    int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_MESH))


def _surface_local(m: mujoco.MjModel, g: int, n: int, rng: np.random.Generator) -> np.ndarray:
    # [n, 3] points on geom g's surface in the geom frame; empty for planes and height fields
    t, s = int(m.geom_type[g]), m.geom_size[g]
    G = _G
    if t in (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return u * (s[:1] if t == G.mjGEOM_SPHERE else s)
    if t == G.mjGEOM_BOX:
        u = rng.uniform(-1, 1, size=(n, 3))
        area = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]])
        k = rng.choice(3, size=n, p=area / area.sum())
        u[np.arange(n), k] = rng.choice([-1.0, 1.0], size=n)
        return u * s
    if t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        r, h = s[0], s[1]
        side, cap = 4 * np.pi * r * h, (2 * np.pi * r * r if t == G.mjGEOM_CYLINDER else 4 * np.pi * r * r)
        on_side = rng.uniform(size=n) < side / (side + cap)
        a = rng.uniform(0, 2 * np.pi, size=n)
        out = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-h, h, size=n)], axis=1)
        m_cap = ~on_side
        if t == G.mjGEOM_CYLINDER:
            rad = r * np.sqrt(rng.uniform(size=n))
            out[m_cap] = np.stack([rad * np.cos(a), rad * np.sin(a), rng.choice([-h, h], size=n)], axis=1)[m_cap]
        else:
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            u[:, 2] = np.abs(u[:, 2]) * rng.choice([-1.0, 1.0], size=n)
            out[m_cap] = (u * r + np.sign(u[:, 2:3]) * np.array([0, 0, h]))[m_cap]
        return out
    if t == G.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v[rng.choice(len(v), size=min(n, len(v)), replace=False)]
    return np.zeros((0, 3))


def _points(m: mujoco.MjModel, d: mujoco.MjData, bodies: list[int], n: int) -> np.ndarray:
    # [N, 3] world-frame surface points of the given bodies; deterministic (seeded by geom id)
    geoms = [g for b in bodies for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]
    out = []
    for g in geoms:
        loc = _surface_local(m, g, max(16, n // len(geoms)), np.random.default_rng(g))
        out.append(loc @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(out) if out else np.zeros((0, 3))


def _robot_roots(m: mujoco.MjModel, robot: RobotSpec) -> set[int]:
    # top-level bodies whose subtree holds an actuated joint or a named robot frame
    bodies = set()
    for a in range(m.nu):
        tid, tt = m.actuator_trnid[a, 0], int(m.actuator_trntype[a])
        if tt in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            bodies.add(m.jnt_bodyid[tid])
        elif tt == int(mujoco.mjtTrn.mjTRN_SITE):
            bodies.add(m.site_bodyid[tid])
        elif tt == int(mujoco.mjtTrn.mjTRN_BODY):
            bodies.add(tid)
    for f in robot.frames:
        if (i := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)) >= 0:
            bodies.add(m.site_bodyid[i])
        elif (i := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f)) >= 0:
            bodies.add(i)
    return {int(m.body_rootid[b]) for b in bodies if b > 0}


def _scene_entities(m: mujoco.MjModel, d: mujoco.MjData, robot: RobotSpec, n: int) -> list[Entity]:
    # noise-free objects (free-joint roots) and fixtures (static roots plus their articulated parts)
    robot_roots = _robot_roots(m, robot)
    owner = np.zeros(m.nbody, dtype=int)  # body -> entity body (its root, or nearest articulated ancestor)
    for b in range(1, m.nbody):
        root = m.body_rootid[b]
        owner[b] = b if (b == root or (m.body_jntnum[b] > 0 and m.body_jntadr[root] < 0)) else owner[m.body_parentid[b]]
    out = []
    for e in sorted(set(owner[1:].tolist())):
        root = int(m.body_rootid[e])
        if root in robot_roots:
            continue
        pts = _points(m, d, [b for b in range(1, m.nbody) if owner[b] == e], n)
        if len(pts) == 0:
            continue
        free = m.body_jntnum[root] > 0 and m.jnt_type[m.body_jntadr[root]] == mujoco.mjtJoint.mjJNT_FREE
        pose = np.concatenate([d.xpos[e], d.xquat[e]])
        R = d.xmat[e].reshape(3, 3)
        extent = np.abs((pts - d.xpos[e]) @ R).max(axis=0)  # half-sizes of the pose-centered box that bounds it
        name = m.body(e).name or f"body{e}"
        out.append(Entity(name, _label(m, name), "object" if free else "fixture", pose, extent, pts,
                          None if e == root else (m.body(root).name or f"body{root}")))
    return out


def perceive_gt(model: mujoco.MjModel, data: mujoco.MjData, robot: RobotSpec, profile: dict,
                rng: np.random.Generator, n_points: int = N_POINTS) -> Scene:
    """Simulator ground truth with noise from the perception profile (v1 rule).

    Profile keys (all optional): range_m, fov_rad (sensed from robot frame "base", else the first frame),
    loc_noise_std_m (per entity), loc_bias_m (one offset per call), rot_noise_std_rad, p_miss.
    Robot frames are proprioceptive and exact. Latency is the caller's job (call on a delayed state).
    """
    frames = [(f, p) for f in robot.frames if (p := _frame_pose(model, data, f)) is not None]
    ents = _scene_entities(model, data, robot, n_points)
    sensor = dict(frames).get("base", frames[0][1] if frames else np.array([0, 0, 0, 1.0, 0, 0, 0]))
    R_s = np.zeros(9)
    mujoco.mju_quat2Mat(R_s, sensor[3:])
    heading = R_s.reshape(3, 3)[:2, 0]

    u = rng.normal(size=3)
    bias = u / np.linalg.norm(u) * float(profile.get("loc_bias_m", 0.0))
    std, rot_std = float(profile.get("loc_noise_std_m", 0.0)), float(profile.get("rot_noise_std_rad", 0.0))
    rng_m, fov, p_miss = float(profile.get("range_m", np.inf)), float(profile.get("fov_rad", 2 * np.pi)), float(profile.get("p_miss", 0.0))

    out = [Entity(f, f, "robot_frame", p, None, None, None) for f, p in frames]
    for e in ents:  # fixed draw order per entity keeps results seeded
        dp = bias + rng.normal(size=3) * std
        rv = rng.normal(size=3) * rot_std
        miss = rng.uniform() < p_miss
        pos = e.pose[:3] + dp
        q = e.pose[3:].copy()
        if rot_std > 0:
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, rv / (np.linalg.norm(rv) + 1e-12), np.linalg.norm(rv))
            mujoco.mju_mulQuat(q, dq, e.pose[3:])
        rel = pos - sensor[:3]
        ang = np.arctan2(heading[0] * rel[1] - heading[1] * rel[0], heading @ rel[:2])
        if miss or np.linalg.norm(rel) > rng_m or (fov < 2 * np.pi and abs(ang) > fov / 2):
            continue
        out.append(Entity(e.id, e.label, e.kind, np.concatenate([pos, q]), e.extent, e.points + dp, e.parent))
    return Scene(out, None, float(data.time))
