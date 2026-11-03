"""Procedural scenes for the generated suite: an MJCF world (no robot) plus its ground-truth Scene.

make_scene(kind, robot, seed) is deterministic in (kind, robot, seed). The robot sits at the world origin facing +x;
the SimEnv attaches the robot model to this world. Conventions the SimEnv relies on:
  - every entity with geometry is a MuJoCo body named by the entity id; its geoms are named "<id>_g<k>", so
    contacts are reported by body name (= entity id); the ground is the static body "floor"
  - articulated entities (drawer_*, door_*) move on joint "<id>_joint"; open_frac/<id> = (q - lo) / (hi - lo) with
    hi = fully open, so a drawer built open has range [-L, 0]
  - regions and paths have no collision geometry; ground regions carry an xy polygon in Entity.points, air regions
    are boxes (pose + extent); paths carry their waypoints in Entity.points
  - extents are box half-sizes in the entity frame; cylinders use [r, r, h]
Only primitive geoms are used, so no benchmark asset can enter a generated scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spec2mpc.core import Entity, RobotSpec, Scene

SCENE_KINDS = ("tabletop", "shelf", "kitchen", "drawer_cabinet", "floor_obstacles", "terrain", "course", "track",
               "road", "airspace")
COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.2, 0.7, 0.25), "blue": (0.2, 0.35, 0.85),
          "yellow": (0.95, 0.85, 0.2), "orange": (0.95, 0.55, 0.15), "purple": (0.55, 0.25, 0.7),
          "white": (0.92, 0.92, 0.92), "black": (0.12, 0.12, 0.12)}
# graspable primitives: (shape, half-sizes [3] for box or (r, r, h) for cylinder, mass kg)
OBJECTS = {"block": ("box", (0.025, 0.025, 0.025), 0.1), "mug": ("cylinder", (0.04, 0.04, 0.05), 0.25),
           "can": ("cylinder", (0.033, 0.033, 0.06), 0.3), "box": ("box", (0.04, 0.06, 0.04), 0.2),
           "peg": ("box", (0.012, 0.012, 0.04), 0.05), "bowl": ("cylinder", (0.07, 0.07, 0.03), 0.3),
           "sponge": ("box", (0.05, 0.03, 0.015), 0.05)}
IDENT = (1.0, 0.0, 0.0, 0.0)
DRAWER_TRAVEL_M = 0.2

# per-family layout defaults (overridden by robot.params of the same name when Task 3 provides them)
LAYOUT = {
    "arm": {"reach_m": 0.8, "table_x_m": 0.55, "table_z_m": 0.2, "half_len_m": 0.1, "base_height_m": 0.0},
    "mobile_manipulator": {"reach_m": 0.7, "table_x_m": 0.85, "table_z_m": 0.7, "arena_m": 8.0,
                           "corridor_w_m": 1.6, "half_len_m": 0.35, "base_height_m": 0.1},
    "mobile_base": {"arena_m": 8.0, "corridor_w_m": 1.5, "track_len_m": 30.0, "track_w_m": 1.5, "lane_w_m": 1.0,
                    "half_len_m": 0.3, "base_height_m": 0.1},
    "rover2d": {"arena_m": 10.0, "corridor_w_m": 1.4, "track_len_m": 25.0, "track_w_m": 1.4, "lane_w_m": 1.0,
                "half_len_m": 0.25, "base_height_m": 0.1},
    "kart": {"arena_m": 40.0, "corridor_w_m": 5.0, "track_len_m": 120.0, "track_w_m": 4.0, "lane_w_m": 3.5,
             "half_len_m": 0.9, "base_height_m": 0.16},
    "quadruped": {"arena_m": 8.0, "corridor_w_m": 1.5, "step_h_m": 0.08, "half_len_m": 0.35, "base_height_m": 0.3},
    "humanoid": {"arena_m": 6.0, "corridor_w_m": 1.5, "step_h_m": 0.08, "half_len_m": 0.2, "base_height_m": 0.75},
    "quadrotor": {"gate_half_m": 0.5, "half_len_m": 0.2, "base_height_m": 0.1},
}


def layout(robot: RobotSpec, key: str) -> float:
    return float(robot.params.get(key, LAYOUT[robot.family][key]))


def quat_yaw(th: float) -> tuple[float, float, float, float]:
    return (float(np.cos(th / 2)), 0.0, 0.0, float(np.sin(th / 2)))


def _f(v) -> str:
    return " ".join(f"{float(x):.5g}" for x in np.atleast_1d(v))


@dataclass
class _B:
    """Accumulates MJCF bodies and the matching entities."""
    rng: np.random.Generator
    xml: list[str] = field(default_factory=list)
    ents: list[Entity] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)

    def ent(self, id: str, label: str, kind: str, pos, quat=IDENT, extent=None, points=None,
            parent: str | None = None) -> None:
        self.ents.append(Entity(id, label, kind, np.r_[np.asarray(pos, float), np.asarray(quat, float)],
                                None if extent is None else np.asarray(extent, float),
                                None if points is None else np.asarray(points, float), parent))

    def color(self) -> str:
        # a color not used yet in this scene (objects stay distinguishable by label)
        free = [c for c in COLORS if c not in self.colors] or list(COLORS)
        c = free[int(self.rng.integers(len(free)))]
        self.colors.append(c)
        return c

    @staticmethod
    def geom(name: str, type: str, size, pos=(0, 0, 0), rgba=(0.6, 0.6, 0.6, 1.0), mass: float | None = None,
             quat=None, contact: bool = True) -> str:
        s = f'<geom name="{name}" type="{type}" size="{_f(size)}" pos="{_f(pos)}" rgba="{_f(rgba)}"'
        if mass is not None:
            s += f' mass="{mass:.4g}"'
        if quat is not None:
            s += f' quat="{_f(quat)}"'
        if not contact:
            s += ' contype="0" conaffinity="0"'
        return s + "/>"

    @staticmethod
    def body(name: str, pos, inner: list[str], quat=IDENT, free: bool = False) -> str:
        head = f'<body name="{name}" pos="{_f(pos)}" quat="{_f(quat)}">'
        return "\n".join([head] + ([f'<freejoint name="{name}_free"/>'] if free else []) + inner + ["</body>"])

    def obj(self, type: str, k: int, pos_xy, z_support: float, yaw: float = 0.0, color: str | None = None) -> str:
        # free graspable object resting on a surface at height z_support; returns its id
        shape, half, mass = OBJECTS[type]
        c = color or self.color()
        id = f"{type}_{k}"
        size = half if shape == "box" else (half[0], half[2])
        pos = (pos_xy[0], pos_xy[1], z_support + half[2])
        q = quat_yaw(yaw)
        self.xml.append(self.body(id, pos, [self.geom(f"{id}_g0", shape, size, rgba=(*COLORS[c], 1), mass=mass)],
                                  q, free=True))
        self.ent(id, f"{c} {type}", "object", pos, q, half)
        return id

    def fixture(self, id: str, label: str, pos, half, quat=IDENT, rgba=(0.55, 0.45, 0.35, 1.0),
                extra: list[str] | None = None) -> None:
        # static box fixture (extra: more geoms in the same body)
        g = [self.geom(f"{id}_g0", "box", half, rgba=rgba)] + (extra or [])
        self.xml.append(self.body(id, pos, g, quat))
        self.ent(id, label, "fixture", pos, quat, half)

    def region(self, id: str, label: str, center, half, yaw: float = 0.0, ground: bool = True,
               rgba=(0.2, 0.8, 0.3, 0.25)) -> None:
        q = quat_yaw(yaw)
        site = f'<site name="{id}_site" type="box" size="{_f(half)}" rgba="{_f(rgba)}"/>'
        self.xml.append(self.body(id, center, [site], q))
        pts = None
        if ground:
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.asarray(half[:2])
            xy = corners @ R.T + np.asarray(center[:2])
            pts = np.c_[xy, np.full(4, center[2])]
        self.ent(id, label, "region", center, q, half, pts)

    def walls(self, id: str, label: str, polylines: list[np.ndarray], height: float, thick: float,
              rgba=(0.5, 0.5, 0.55, 1.0)) -> None:
        # static wall body at the origin with one box per polyline segment; points = all vertices
        g = []
        for line in polylines:
            for a, b in zip(line[:-1], line[1:]):
                d = b[:2] - a[:2]
                L = float(np.linalg.norm(d))
                if L < 1e-6:
                    continue
                mid = 0.5 * (a[:2] + b[:2])
                g.append(self.geom(f"{id}_g{len(g)}", "box", (L / 2 + thick / 2, thick / 2, height / 2),
                                   (mid[0], mid[1], height / 2), rgba, quat=quat_yaw(float(np.arctan2(d[1], d[0])))))
        self.xml.append(self.body(id, (0, 0, 0), g))
        pts = np.concatenate([line[:, :2] for line in polylines])
        self.ent(id, label, "fixture", (0, 0, height / 2), IDENT, None, np.c_[pts, np.zeros(len(pts))])

    def path(self, id: str, label: str, pts: np.ndarray) -> None:
        p = np.c_[pts[:, :2], np.zeros(len(pts))] if pts.shape[1] == 2 else pts
        self.ent(id, label, "path", p[0], IDENT, None, p)

    def done(self, kind: str, seed: int) -> tuple[str, Scene]:
        r = self.rng
        c1, c2 = r.uniform(0.25, 0.55, 3), r.uniform(0.15, 0.4, 3)
        light_dir = (r.uniform(-0.4, 0.4), r.uniform(-0.4, 0.4), -1.0)
        head = [f'<mujoco model="{kind}_{seed}">', '<compiler angle="radian"/>', "<asset>",
                f'<texture name="floor_tex" type="2d" builtin="checker" rgb1="{_f(c1)}" rgb2="{_f(c2)}" '
                'width="256" height="256"/>',
                '<material name="floor_mat" texture="floor_tex" texrepeat="20 20"/>', "</asset>", "<worldbody>",
                f'<light name="sun" pos="0 0 8" dir="{_f(light_dir)}" diffuse="{_f(r.uniform(0.6, 1.0, 3))}" '
                'directional="true"/>',
                '<body name="floor"><geom name="floor_g0" type="plane" size="100 100 0.1" '
                'material="floor_mat"/></body>']
        self.ent("floor", "floor", "fixture", (0, 0, 0), IDENT, (100.0, 100.0, 0.0))
        return "\n".join(head + self.xml + ["</worldbody>", "</mujoco>"]), Scene(list(self.ents), None, 0.0)


# ---- placement helpers ----

def _scatter(rng, n: int, lo, hi, min_sep: float, avoid: list[tuple[np.ndarray, float]] = (),
             ok=lambda p: True, tries: int = 2000) -> list[np.ndarray]:
    # n xy points uniform in the box [lo, hi], pairwise >= min_sep apart and clear of (center, radius) discs
    pts: list[np.ndarray] = []
    for _ in range(tries):
        if len(pts) == n:
            break
        p = rng.uniform(lo, hi)
        if not ok(p) or any(np.linalg.norm(p - q) < min_sep for q in pts) or \
                any(np.linalg.norm(p - c) < r for c, r in avoid):
            continue
        pts.append(p)
    if len(pts) < n:
        raise RuntimeError(f"could not place {n} items in {lo}..{hi}")
    return pts


def _offset(poly: np.ndarray, d: float, closed: bool) -> np.ndarray:
    # polyline [N, 2] offset by d to the left (miter at vertices)
    nxt = np.roll(poly, -1, 0) if closed else np.r_[poly[1:], poly[-1:] + (poly[-1] - poly[-2])]
    prv = np.roll(poly, 1, 0) if closed else np.r_[poly[:1] - (poly[1] - poly[0]), poly[:-1]]
    t1 = (poly - prv) / np.linalg.norm(poly - prv, axis=1, keepdims=True)
    t2 = (nxt - poly) / np.linalg.norm(nxt - poly, axis=1, keepdims=True)
    t = t1 + t2
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    cos_half = np.clip(np.sum(t * t2, axis=1, keepdims=True), 0.5, 1.0)
    return poly + nrm * d / cos_half


def _resample(poly: np.ndarray, ds: float, closed: bool) -> np.ndarray:
    p = np.r_[poly, poly[:1]] if closed else poly
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(int(s[-1] / ds), 4)
    q = np.linspace(0, s[-1], n, endpoint=not closed)
    return np.c_[np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])]


def _arclen(poly: np.ndarray, closed: bool) -> float:
    p = np.r_[poly, poly[:1]] if closed else poly
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# ---- manipulation scenes ----

def _table(b: _B, robot: RobotSpec, id: str = "table_0", y: float = 0.0) -> tuple[float, float, np.ndarray]:
    tx, tz = layout(robot, "table_x_m"), layout(robot, "table_z_m")
    half = np.array([0.3, 0.45, 0.025])
    legs = [b.geom(f"{id}_g{k + 1}", "box", (0.02, 0.02, max(tz - 0.05, 0.005) / 2),
                   (sx * 0.27, sy * 0.42, -(tz - 0.05) / 2 - 0.025), (0.4, 0.3, 0.2, 1))
            for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])] if tz > 0.06 else []
    b.fixture(id, "table", (tx, y, tz - 0.025), half, extra=legs)
    return tx, tz, half


def _on_table(b: _B, robot: RobotSpec, n: int, tx: float, ty: float, half, avoid=(),
              min_sep: float = 0.11) -> list[np.ndarray]:
    # fixed-base arms: keep objects inside the reach annulus; mobile manipulators drive up to the table
    reach = layout(robot, "reach_m") if robot.family == "arm" else np.inf
    margin = 0.06
    lo, hi = np.array([tx - half[0] + margin, ty - half[1] + margin]), np.array([tx + half[0] - margin,
                                                                                  ty + half[1] - margin])
    return _scatter(b.rng, n, lo, hi, min_sep, list(avoid), ok=lambda p: robot.family != "arm" or 0.3 <= np.linalg.norm(p) <= 0.85 * reach)


def _tabletop(b: _B, robot: RobotSpec) -> None:
    tx, tz, half = _table(b, robot)
    top = tz
    r = b.rng
    # fixtures on the table: plate (place target), hole block (insertion), two zones (push, slide, rearrange)
    spots = _on_table(b, robot, 4, tx, 0.0, half, min_sep=0.17)
    b.fixture("plate_0", "plate", (*spots[0], top + 0.005), (0.06, 0.06, 0.005), rgba=(0.95, 0.95, 0.9, 1))
    hole = 0.018  # half-width of the hole, peg half-width 0.012
    wall, depth = 0.02, 0.04
    hx, hy = spots[1]
    hz = top + depth
    hg = []
    for k, (dx, dy, sx, sy) in enumerate([(hole + wall / 2, 0, wall / 2, hole + wall), (-hole - wall / 2, 0, wall / 2,
                                          hole + wall), (0, hole + wall / 2, hole, wall / 2),
                                         (0, -hole - wall / 2, hole, wall / 2)]):
        hg.append(b.geom(f"hole_0_g{k}", "box", (sx, sy, depth), (dx, dy, 0), (0.3, 0.3, 0.35, 1)))
    b.xml.append(b.body("hole_0", (hx, hy, hz), hg))
    b.ent("hole_0", "hole", "fixture", (hx, hy, hz), IDENT, (hole, hole, depth))
    zc = [b.color() for _ in range(2)]
    for k in range(2):
        b.region(f"zone_{k}", f"{zc[k]} area", (*spots[2 + k], top + 0.05), (0.07, 0.07, 0.05),
                 rgba=(*COLORS[zc[k]], 0.3))
    # objects: two blocks, mug, peg, box, plus up to two distractors
    types = ["block", "block", "mug", "peg", "box"] + list(r.choice(["can", "sponge", "bowl"],
                                                                      int(r.integers(0, 3)), replace=False))
    pts = _on_table(b, robot, len(types), tx, 0.0, half, avoid=[(s, 0.12) for s in spots])
    count: dict[str, int] = {}
    for t, p in zip(types, pts):
        k = count[t] = count.get(t, -1) + 1
        b.obj(t, k, p, top, float(r.uniform(-np.pi, np.pi)))
    # person for handover, standing beside the table, and the hand-off region in front of them
    py = -(half[1] + 0.35)
    b.fixture("person_0", "person", (tx, py, 0.85), (0.15, 0.15, 0.85), rgba=(0.8, 0.65, 0.5, 1))
    b.region("handoff_0", "person's hand", (tx - 0.05, py + 0.25, top + 0.3), (0.08, 0.08, 0.08), ground=False)


def _shelf(b: _B, robot: RobotSpec) -> None:
    tx, tz, half = _table(b, robot, y=0.25)
    r = b.rng
    pts = _on_table(b, robot, 4, tx, 0.25, half)
    for t, p in zip(["block", "can", "mug", "box"], pts):
        b.obj(t, 0, p, tz, float(r.uniform(-np.pi, np.pi)))
    # shelf unit to the right: three boards, the lowest at table height
    sx, sy = tx + 0.05, -0.45
    gap = float(r.uniform(0.22, 0.3))
    bh = np.array([0.18, 0.2, 0.01])
    sides = [b.geom(f"shelf_0_g{k}", "box", (0.18, 0.01, (tz + 2 * gap) / 2 + 0.05),
                    (0, s * 0.21, -(tz) / 2 + gap), (0.45, 0.35, 0.25, 1)) for k, s in enumerate((1, -1))]
    b.xml.append(b.body("shelf_0", (sx, sy, tz), sides))
    b.ent("shelf_0", "shelf", "fixture", (sx, sy, tz), IDENT, (0.18, 0.22, (tz + 2 * gap) / 2 + 0.05))
    for k, name in enumerate(["bottom shelf", "middle shelf", "top shelf"]):
        b.fixture(f"shelf_0_l{k}", name, (sx, sy, tz + k * gap - 0.01), bh, rgba=(0.6, 0.5, 0.35, 1))


def _kitchen(b: _B, robot: RobotSpec) -> None:
    tx, tz = layout(robot, "table_x_m"), layout(robot, "table_z_m")
    half = np.array([0.3, 0.6, tz / 2])
    b.fixture("counter_0", "counter", (tx, 0.0, tz / 2), half, rgba=(0.85, 0.85, 0.8, 1))
    r = b.rng
    # wall cabinet with a hinged door on the far-right end of the counter, door facing the robot (-x)
    cy, ch = -0.4, np.array([0.15, 0.15, 0.15])
    cpos = np.array([tx + 0.1, cy, tz + ch[2]])
    b.fixture("cabinet_0", "cabinet", cpos, ch, rgba=(0.5, 0.35, 0.25, 1))
    hinge = cpos + [-ch[0] - 0.01, ch[1], 0]
    door_half = (0.01, ch[1], ch[2])
    handle = b.body("door_0_handle", (-0.03, -2 * ch[1] + 0.03, 0),
                    [b.geom("door_0_handle_g0", "box", (0.01, 0.01, 0.04), rgba=(0.8, 0.8, 0.8, 1))])
    door = b.body("door_0", hinge, ['<joint name="door_0_joint" type="hinge" axis="0 0 -1" range="0 1.6" '
                                    'damping="0.5"/>',
                                    b.geom("door_0_g0", "box", door_half, (0, -ch[1], 0), (0.6, 0.4, 0.3, 1),
                                           mass=0.5), handle])
    b.xml.append(door)
    b.ent("door_0", "cabinet door", "fixture", hinge + [0, -ch[1], 0], IDENT, door_half)
    b.ent("door_0_handle", "cabinet door handle", "fixture", hinge + [-0.03, -2 * ch[1] + 0.03, 0], IDENT,
          (0.01, 0.01, 0.04), parent="door_0")
    reach_half = np.array([half[0], 0.45, 0.0])
    pts = _on_table(b, robot, 4, tx, 0.15, reach_half)
    for t, p in zip(["mug", "bowl", "sponge", "can"], pts):
        b.obj(t, 0, p, tz, float(r.uniform(-np.pi, np.pi)))
    sp = _on_table(b, robot, 1, tx, 0.15, reach_half, avoid=[(p, 0.12) for p in pts])[0]
    b.region("spill_0", "spill", (*sp, tz + 0.01), (0.06, 0.06, 0.01), rgba=(0.5, 0.35, 0.1, 0.4))


def _drawer_cabinet(b: _B, robot: RobotSpec) -> None:
    tx, tz = layout(robot, "table_x_m"), layout(robot, "table_z_m")
    r = b.rng
    # cabinet whose front face is at x = front; two drawers stacked under its top at table height
    top = max(tz, 0.3)
    front = tx + 0.1
    ch = np.array([0.2, 0.3, top / 2])
    cpos = np.array([front + ch[0], 0.0, top / 2])
    b.fixture("cabinet_0", "cabinet", cpos, ch, rgba=(0.5, 0.4, 0.3, 1))
    L = DRAWER_TRAVEL_M
    dh = min(0.06, top / 6)
    inner = np.array([0.12, 0.2, dh])
    for k, (zc, open_) in enumerate([(top - 2.5 * dh, False), (top - 6 * dh, True)]):
        id = f"drawer_{k}"
        x0 = front - inner[0] - 0.02 - (L if open_ else 0.0)  # drawer interior centre x
        rng_ = "-%g 0" % L if open_ else "0 %g" % L
        t = 0.008
        g = [b.geom(f"{id}_g0", "box", (inner[0], inner[1], t), (0, 0, -inner[2]), mass=0.3),
             b.geom(f"{id}_g1", "box", (t, inner[1], inner[2]), (inner[0], 0, 0)),
             b.geom(f"{id}_g2", "box", (t, inner[1] + 0.02, inner[2] + 0.01), (-inner[0], 0, 0),
                    (0.6, 0.45, 0.3, 1)),
             b.geom(f"{id}_g3", "box", (inner[0], t, inner[2]), (0, inner[1], 0)),
             b.geom(f"{id}_g4", "box", (inner[0], t, inner[2]), (0, -inner[1], 0))]
        hpos = (-inner[0] - 0.03, 0, 0)
        handle = b.body(f"{id}_handle", hpos, [b.geom(f"{id}_handle_g0", "box", (0.01, 0.05, 0.01),
                                                      rgba=(0.8, 0.8, 0.8, 1))])
        b.xml.append(b.body(id, (x0, 0, zc), [f'<joint name="{id}_joint" type="slide" axis="-1 0 0" range="{rng_}" '
                                               'damping="2"/>'] + g + [handle]))
        b.ent(id, f"{'bottom' if k else 'top'} drawer", "fixture", (x0, 0, zc), IDENT, inner)
        b.ent(f"{id}_handle", f"{'bottom' if k else 'top'} drawer handle", "fixture", (x0 + hpos[0], 0, zc),
              IDENT, (0.01, 0.05, 0.01), parent=id)
    # objects on the cabinet top; the robot side half only
    pts = _scatter(r, 2, np.array([front + 0.04, -0.22]), np.array([front + 0.1, 0.22]), 0.11)
    for t, p in zip(["block", "mug"], pts):
        b.obj(t, 0, p, top, float(r.uniform(-np.pi, np.pi)))


# ---- ground scenes ----

def _floor_obstacles(b: _B, robot: RobotSpec) -> None:
    A = layout(robot, "arena_m")
    r = b.rng
    goal = _scatter(r, 1, np.array([0.35 * A, -0.4 * A]), np.array([0.8 * A, 0.4 * A]), 0)[0]
    gh = max(0.15 * A / 8, 0.3)
    b.region("goal_0", "goal area", (*goal, 1.0), (gh, gh, 1.0))
    # dock against the far wall
    dy = float(r.uniform(-0.3 * A, 0.3 * A))
    dpos = (0.95 * A, dy, 0.25)
    b.fixture("dock_0", "charging dock", dpos, (0.1, 0.3, 0.25), rgba=(0.2, 0.2, 0.25, 1))
    sx = 0.95 * A - 0.1 - layout(robot, "half_len_m") - 0.05
    b.region("dock_0_slot", "dock slot", (sx, dy, 1.0), (0.1, 0.1, 1.0), rgba=(0.9, 0.9, 0.2, 0.3))
    n = int(r.integers(3, 9))
    avoid = [(np.zeros(2), 1.0), (goal, gh + 0.6), (np.array([sx, dy]), 0.9)]
    pts = _scatter(r, n, np.array([0.1 * A, -0.5 * A]), np.array([0.85 * A, 0.5 * A]), 0.8, avoid)
    for k, p in enumerate(pts):
        h = float(r.uniform(0.15, 0.4)) * max(A / 8, 1.0)
        half = r.uniform(0.1, 0.35, 2) * max(A / 8, 1.0)
        b.fixture(f"obstacle_{k}", "obstacle", (*p, h), (*half, h), quat_yaw(float(r.uniform(0, np.pi))),
                  rgba=(0.6, 0.3, 0.2, 1))
    if robot.family == "quadruped":
        # payload resting on the robot's back
        z = layout(robot, "base_height_m") + 0.12
        b.xml.append(b.body("payload_0", (0, 0, z), [b.geom("payload_0_g0", "box", (0.08, 0.06, 0.04),
                                                             rgba=(0.9, 0.6, 0.1, 1), mass=0.5)], free=True))
        b.ent("payload_0", "payload box", "object", (0, 0, z), IDENT, (0.08, 0.06, 0.04))


def _terrain(b: _B, robot: RobotSpec) -> None:
    r = b.rng
    sh = layout(robot, "step_h_m")
    # rough band of low rocks
    pts = _scatter(r, int(r.integers(6, 14)), np.array([1.0, -1.5]), np.array([3.0, 1.5]), 0.35)
    for k, p in enumerate(pts):
        h = float(r.uniform(0.01, 0.6 * sh)) / 2
        b.fixture(f"rock_{k}", "rock", (*p, h), (*r.uniform(0.08, 0.2, 2), h), quat_yaw(float(r.uniform(0, np.pi))),
                  rgba=(0.45, 0.42, 0.4, 1))
    # three steps up to a platform
    x0, depth = 4.0, float(r.uniform(0.35, 0.5))
    for k in range(3):
        h = (k + 1) * sh / 2
        b.fixture(f"step_{k}", "step", (x0 + (k + 0.5) * depth, 0, h), (depth / 2, 1.0, h), rgba=(0.55, 0.55, 0.6, 1))
    px = x0 + 3 * depth + 1.0
    b.fixture("platform_0", "platform", (px, 0, 1.5 * sh), (1.0, 1.0, 1.5 * sh), rgba=(0.5, 0.5, 0.55, 1))
    b.region("top_0", "platform top", (px, 0, 3 * sh + 1.0), (0.6, 0.6, 1.0))
    gy = float(r.choice([-1, 1])) * float(r.uniform(2.0, 3.0))
    b.region("goal_0", "goal area", (float(r.uniform(3.0, 4.0)), gy, 1.0), (0.4, 0.4, 1.0))


def _course(b: _B, robot: RobotSpec) -> None:
    r = b.rng
    w = layout(robot, "corridor_w_m")
    seg = 3.0 * w
    pts, th = [np.array([-1.0 * w, 0.0]), np.zeros(2)], 0.0
    for _ in range(int(r.integers(4, 8))):
        pts.append(pts[-1] + seg * float(r.uniform(0.8, 1.5)) * np.array([np.cos(th), np.sin(th)]))
        th = float(np.clip(th + r.uniform(-np.pi / 4, np.pi / 4), -np.pi * 0.6, np.pi * 0.6))
    pts.append(pts[-1] + seg * np.array([np.cos(th), np.sin(th)]))
    line = _resample(np.array(pts), w / 2, closed=False)
    left, right = _offset(line, w / 2, False), _offset(line, -w / 2, False)
    # one wall body: left side, end cap behind the start, right side (finish end stays open)
    wall = np.r_[left[::-1], right]
    b.walls("course_0_wall", "course walls", [wall], 0.4 * max(w / 1.5, 1.0) ** 0.5, 0.1)
    b.path("course_0", "course", line[1:] if np.linalg.norm(line[0]) > 0 else line)
    end, prev = line[-1], line[-3]
    yaw = float(np.arctan2(*(end - prev)[::-1]))
    b.region("finish_0", "finish area", (*(end - 0.6 * w * (end - prev) / np.linalg.norm(end - prev)), 1.0),
             (0.4 * w, 0.45 * w, 1.0), yaw)


def _track(b: _B, robot: RobotSpec) -> None:
    r = b.rng
    L, w, hl = layout(robot, "track_len_m"), layout(robot, "track_w_m"), layout(robot, "half_len_m")
    u = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ab = float(r.uniform(0.45, 0.8))
    rad = 1 + sum(float(r.uniform(0, 0.12)) * np.cos(k * u + float(r.uniform(0, 2 * np.pi))) for k in (2, 3))
    loop = np.c_[np.cos(u) * rad, ab * np.sin(u) * rad]
    loop *= L / _arclen(loop, True)
    loop = _resample(loop, w / 4, closed=True)
    # start at index 0, heading +x, counter-clockwise around the loop
    t0 = loop[1] - loop[0]
    a = -float(np.arctan2(t0[1], t0[0]))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    loop = (loop - loop[0]) @ R.T
    b.path("track_0", "track", loop)
    left, right = _offset(loop, w / 2 + 0.05, True), _offset(loop, -w / 2 - 0.05, True)
    b.walls("track_0_wall", "track walls", [np.r_[left, left[:1]], np.r_[right, right[:1]]], 0.3, 0.1)
    # barrier just behind the start blocks reversing; the finish box sits behind the barrier, reached by a lap
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[loop, loop[:1]], axis=0), axis=1))]
    total = s[-1]

    def at(dist: float) -> tuple[np.ndarray, float]:
        q = total + dist
        i = int(np.searchsorted(s, q) - 1) % len(loop)
        d = loop[(i + 1) % len(loop)] - loop[i]
        return loop[i], float(np.arctan2(d[1], d[0]))

    bp, byaw = at(-(hl + 0.4))
    b.fixture("barrier_0", "start barrier", (*bp, 0.3), (0.05, w / 2 + 0.05, 0.3), quat_yaw(byaw),
              rgba=(0.9, 0.2, 0.2, 1))
    flen = max(2.0 * hl, 1.0)
    fp, fyaw = at(-(hl + 0.4) - hl - 0.3 - flen / 2)
    b.region("finish_0", "finish box", (*fp, 1.0), (flen / 2, w / 2, 1.0), fyaw)


def _road(b: _B, robot: RobotSpec) -> None:
    r = b.rng
    lw, hl = layout(robot, "lane_w_m"), layout(robot, "half_len_m")
    Lr = 40 * hl + 10 * lw
    bend = float(r.uniform(-0.4, 0.4))
    # road centreline between two lanes: straight for 60% then a gentle bend; robot starts in the right lane at
    # the origin; a parking strip for parked cars runs along the right curb
    ctr = [np.array([-2 * hl, lw / 2])]
    th = 0.0
    for k in range(40):
        if k > 24:
            th += bend / 15
        ctr.append(ctr[-1] + (Lr / 40) * np.array([np.cos(th), np.sin(th)]))
    ctr = np.array(ctr)
    car = np.array([max(2.2 * hl, 0.4), 0.4 * lw, 0.4 * max(hl, 0.5)])
    strip = 2 * car[1] + 0.2
    b.walls("road_0_curb", "curb", [_offset(ctr, lw + 0.05, False), _offset(ctr, -lw - strip - 0.05, False)],
            0.15, 0.1)
    lane = _offset(ctr, -lw / 2, False)
    b.path("lane_0", "right lane", lane)
    # parked cars along the right curb with one free slot
    slot_i = int(r.integers(1, 4))
    x = 6 * hl + lw
    for k in range(5):
        cx = x + k * (2 * car[0] + 2.4 * hl + 0.4)
        p = np.array([cx, -lw / 2 - 0.1 - car[1]])
        if k == slot_i:
            b.region("slot_0", "parking spot", (*p, 1.0), (car[0], car[1], 1.0), 0.0, rgba=(0.9, 0.9, 0.2, 0.3))
        else:
            b.fixture(f"car_{k}", "parked car", (*p, car[2]), car, rgba=(*COLORS[b.color()], 1))
    # a cone in the lane past the parked cars, finish in the right lane at the end
    ci = int(0.75 * len(lane))
    b.fixture("cone_0", "traffic cone", (*lane[ci], 0.2), (0.12, 0.12, 0.2), rgba=(1.0, 0.45, 0.0, 1))
    end, prev = lane[-2], lane[-4]
    b.region("finish_0", "end of the road", (*end, 1.0), (max(2 * hl, 0.6), 0.45 * lw, 1.0),
             float(np.arctan2(*(end - prev)[::-1])))


# ---- air scene ----

def _airspace(b: _B, robot: RobotSpec) -> None:
    r = b.rng
    g = layout(robot, "gate_half_m")
    W, H = 2.5, 4.0
    n_g = int(r.integers(2, 5))
    x0, dx = 3.0, 4.0
    xe = x0 + dx * n_g + 2.0
    rail = np.array([[x0, W], [xe, W], [xe, -W], [x0, -W]])
    cg = [b.geom(f"corridor_0_g{k}", "box", (*hs, H / 2), (*c, H / 2), (0.5, 0.5, 0.55, 0.4))
          for k, (c, hs) in enumerate([((0.5 * (x0 + xe), W), (0.5 * (xe - x0), 0.05)),
                                       ((0.5 * (x0 + xe), -W), (0.5 * (xe - x0), 0.05)),
                                       ((xe, 0.0), (0.05, W))])]
    cg.append(b.geom(f"corridor_0_g{len(cg)}", "box", (0.5 * (xe - x0), W, 0.05), (0.5 * (x0 + xe), 0, H + 0.05),
                     (0.5, 0.5, 0.55, 0.2)))
    b.xml.append(b.body("corridor_0", (0, 0, 0), cg))
    b.ent("corridor_0", "corridor", "fixture", (0.5 * (x0 + xe), 0, H / 2), IDENT, (0.5 * (xe - x0), W, H / 2),
          np.c_[rail, np.zeros(4)])
    for k in range(n_g):
        gx = x0 + dx * (k + 0.5)
        yo, zo = float(r.uniform(-W + g + 0.4, W - g - 0.4)), float(r.uniform(g + 0.5, H - g - 0.5))
        # wall filling the corridor cross-section with a square opening (half-size g) at (yo, zo)
        geoms = []
        lo_w, hi_w = yo - g, yo + g  # opening y range
        boxes = [(-W, lo_w, 0, H), (hi_w, W, 0, H), (lo_w, hi_w, 0, zo - g), (lo_w, hi_w, zo + g, H)]
        for j, (y0, y1, z0, z1) in enumerate(boxes):
            geoms.append(b.geom(f"gate_{k}_g{j}", "box", (0.05, (y1 - y0) / 2, (z1 - z0) / 2),
                                (0, (y0 + y1) / 2 - yo, (z0 + z1) / 2 - zo), (*COLORS["orange"], 1)))
        b.xml.append(b.body(f"gate_{k}", (gx, yo, zo), geoms))
        b.ent(f"gate_{k}", f"gate {k + 1}", "fixture", (gx, yo, zo), IDENT, (0.05, g, g))
    b.region("finish_0", "finish area", (xe - 1.0, 0, H / 2), (0.6, W - 0.3, H / 2 - 0.2), ground=False)
    # inspection tower and its viewpoint, hover point, free-flight goal
    tpos = np.array([float(r.uniform(-2, 2)), -6.0])
    th = float(r.uniform(4.0, 7.0))
    b.fixture("tower_0", "tower", (*tpos, th / 2), (0.5, 0.5, th / 2), rgba=(0.6, 0.6, 0.65, 1))
    ang = float(r.uniform(0.25 * np.pi, 0.75 * np.pi))  # viewpoint on the corridor-facing side
    vp = tpos + 2.0 * np.array([np.cos(ang), np.sin(ang)])
    vz = float(r.uniform(1.5, th - 0.5))
    b.region("view_0", "tower viewpoint", (*vp, vz), (0.3, 0.3, 0.3), yaw=ang + np.pi, ground=False)
    b.region("hover_0", "hover point", (float(r.uniform(-4, -2)), float(r.uniform(2, 4)), float(r.uniform(1.0, 2.5))),
             (0.15, 0.15, 0.15), ground=False)
    b.region("goal_0", "landing zone", (float(r.uniform(-6, -3)), float(r.uniform(5, 8)), float(r.uniform(1, 3))),
             (0.5, 0.5, 0.5), ground=False)


_KINDS = {"tabletop": _tabletop, "shelf": _shelf, "kitchen": _kitchen, "drawer_cabinet": _drawer_cabinet,
          "floor_obstacles": _floor_obstacles, "terrain": _terrain, "course": _course, "track": _track,
          "road": _road, "airspace": _airspace}
assert set(_KINDS) == set(SCENE_KINDS)


def make_scene(kind: str, robot: RobotSpec, seed: int) -> tuple[str, Scene]:
    """(MJCF xml of the world without the robot, ground-truth Scene at t = 0)."""
    if kind not in _KINDS:
        raise ValueError(f"unknown scene kind {kind!r}")
    b = _B(np.random.default_rng([seed, SCENE_KINDS.index(kind)]))
    _KINDS[kind](b, robot)
    return b.done(kind, seed)
