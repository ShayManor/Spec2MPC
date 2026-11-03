from __future__ import annotations

import zlib

import mujoco
import numpy as np

from spec2mpc.core.types import Entity, Scene

# Antipodal-sampling grasp proposer on the object point cloud (parallel-jaw gripper).
# Stand-in for an off-the-shelf grasp network; same output format, replaceable later.

MAX_PTS = 512
K_NN = 12


def _normals(p: np.ndarray) -> np.ndarray:
    # [N, 3] outward unit normals by local PCA over the K nearest neighbours
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)  # [N, N]
    nn = np.argsort(d2, axis=1)[:, :K_NN]  # [N, K]
    q = p[nn] - p[nn].mean(axis=1, keepdims=True)  # [N, K, 3]
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))  # ascending eigenvalues
    n = v[:, :, 0]
    flip = ((p - p.mean(axis=0)) * n).sum(-1) < 0
    n[flip] *= -1
    return n


def _rot(y: np.ndarray) -> np.ndarray:
    # [3, 3] gripper frame: y = closing axis, z = approach (as close to top-down as possible)
    z = np.array([0, 0, -1.0]) - y * (-y[2])
    if np.linalg.norm(z) < 1e-3:
        z = np.array([1.0, 0, 0]) - y * y[0]
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)


def grasps_for(e: Entity, n: int = 8, max_width_m: float = 0.08, min_width_m: float = 0.005,
               cone_rad: float = 0.35, tol_m: float = 0.006) -> list[Entity]:
    """Up to n antipodal parallel-jaw grasps on e.points; [] when no pair fits the gripper."""
    rng = np.random.default_rng(zlib.crc32(e.id.encode()))
    p = e.points
    if p is None or len(p) < 2 * K_NN:
        return []
    if len(p) > MAX_PTS:
        p = p[rng.choice(len(p), MAX_PTS, replace=False)]
    nrm = _normals(p)
    cands = []  # (quality, i, j)
    for i in rng.permutation(len(p))[:256]:
        a = -nrm[i]  # into the object
        v = p - p[i]  # [N, 3]
        t = v @ a
        perp = np.linalg.norm(v - t[:, None] * a, axis=1)
        ok = (t > min_width_m) & (t <= max_width_m) & (perp < tol_m)
        if not ok.any():
            continue
        js = np.nonzero(ok)[0]
        u = v[js] / np.linalg.norm(v[js], axis=1, keepdims=True)
        q = np.minimum((-nrm[i] * u).sum(-1), (nrm[js] * u).sum(-1))  # both contacts inside the friction cone
        k = int(np.argmax(q))
        if q[k] > np.cos(cone_rad):
            cands.append((float(q[k]), int(i), int(js[k])))
    cands.sort(key=lambda c: -c[0])
    out: list[Entity] = []
    chosen: list[tuple[np.ndarray, np.ndarray]] = []
    for _, i, j in cands:
        c, y = 0.5 * (p[i] + p[j]), (p[j] - p[i]) / np.linalg.norm(p[j] - p[i])
        if any(np.linalg.norm(c - c2) < 0.01 and abs(y @ y2) > 0.95 for c2, y2 in chosen):
            continue  # near-duplicate
        chosen.append((c, y))
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, _rot(y).reshape(-1))
        out.append(Entity(f"{e.id}/grasp_{len(out)}", "grasp", "grasp", np.concatenate([c, quat]),
                          None, np.stack([p[i], p[j]]), e.id))
        if len(out) == n:
            break
    return out


def add_grasps(scene: Scene, n: int = 8, max_width_m: float = 0.08) -> Scene:
    """Scene plus up to n grasp entities (kind "grasp", parent = object id) per graspable object.

    Graspable = kind "object" with a point cloud and an antipodal pair narrower than the gripper.
    Earlier grasps of those objects are replaced. Grasp points are the two finger contacts [2, 3].
    """
    objs = [e for e in scene.entities if e.kind == "object"]
    ids = {e.id for e in objs}
    keep = [e for e in scene.entities if not (e.kind == "grasp" and e.parent in ids)]
    new = [g for e in objs for g in grasps_for(e, n, max_width_m)]
    return Scene(keep + new, scene.image, scene.t_s)
