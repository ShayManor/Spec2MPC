"""Real race-track geometry for karts (TUM racetrack database, F1TENTH racetracks) and gate courses for quadrotors.

Assets: <ASSET_ROOT>/tracks/{racetrack-database, f1tenth_racetracks, rpg_time_optimal}. Both track sources give a
closed centerline with right and left widths (`x_m, y_m, w_tr_right_m, w_tr_left_m`). F1TENTH tracks are sized for
1/10-scale cars, so they are scaled by kart width / F1TENTH car width. A full lap has no predicate translation
(end-state predicates cannot tell a lap from staying put), so each card is a forward track segment: start on the
centerline, reach a finish line a fraction of a lap ahead, never touch the walls built on the track boundaries.

Quadrotors: the UZH time-optimal race track (rpg_time_optimal/tracks/track.yaml, 7 gates) plus generated loops of
gates. Gate order has no predicate (only the finish region and gate and floor contacts are tested), so it is a
dropped conjunct; the route through the gate centers carries it to the executor.
"""
from __future__ import annotations

import functools

import numpy as np
import yaml

from spec2mpc.core import Entity, Scene, TaskCard, TaskPublic
from spec2mpc.tasks.sources import Translation, pred, write_coverage
from spec2mpc.tasks.sources import _nav as N

SOURCE = "tracks"
TUM = N.ASSET_ROOT / "tracks" / "racetrack-database" / "tracks"
F1T = N.ASSET_ROOT / "tracks" / "f1tenth_racetracks"
UZH = N.ASSET_ROOT / "tracks" / "rpg_time_optimal" / "tracks" / "track.yaml"
F1T_SCALE = 1.1 / 0.31  # nominal kart width (v1 EVC kart) / F1TENTH car width
KART_VMAX_MPS = 17.5  # v1 EVC kart nominal top speed (configs/robots/kart.yaml)
SEG_FRAC = (0.05, 0.2)  # segment length as a lap fraction (< 0.5 keeps forward the short way)
SEG_MAX_M = 1500.0
WALL_PAD_M = 30.0  # walls extend this far beyond both segment ends
WALL_STEP_M = {"tum": 5.0, "f1tenth": 2.0}
FINISH_HALF_LEN_M = 2.0

QUAD_SCALE = {"skydio_x2": 1.0, "crazyflie2": 0.25}  # course scale per member (crazyflie flies indoors)
QUAD_VMAX_MPS = {"skydio_x2": 6.0, "crazyflie2": 1.5}
GATE_HALF_M = 0.5  # inner half-size of a square gate at scale 1
GATE_BAR_M = 0.05
GEN_GATES = (4, 8)
GEN_RADIUS_M = (4.0, 10.0)
GEN_HEIGHT_M = (1.0, 3.0)

KART_STYLES = {
    "default": ["Drive this stretch of the track to the finish line without touching the walls.",
                "Take the kart along the track to the finish marker, stay off the walls."],
    "fast": ["Drive this section of the track as fast as you can, no wall contact.",
             "Push hard and get to the finish line quickly without hitting the barriers."],
    "gentle": ["Drive smoothly along the track to the finish line, no harsh braking or jerky steering.",
               "Take this part of the circuit gently to the finish marker, keep the ride comfortable."],
}
QUAD_STYLES = {
    "default": ["Fly the course through the gates to the finish without touching any gate.",
                "Go through the gates in order and reach the end of the course without a collision."],
    "fast": ["Race through the gate course to the finish as quickly as you can, no gate contact.",
             "Fly the gates in order as fast as possible and do not crash."],
    "gentle": ["Fly the gate course smoothly to the finish, no aggressive moves.",
               "Take the drone through the gates calmly and steadily to the end of the course."],
}


def available() -> bool:
    return TUM.exists() and F1T.exists()


def _name(s: str) -> str:
    return s.replace(" ", "_")


@functools.lru_cache(maxsize=None)
def track_names() -> dict[str, str]:
    # task name -> csv path
    out = {f"tum_{_name(p.stem)}": str(p) for p in sorted(TUM.glob("*.csv"))}
    for d in sorted(p for p in F1T.iterdir() if p.is_dir()):
        c = d / f"{d.name}_centerline.csv"
        if c.exists():
            out[f"f1tenth_{_name(d.name)}"] = str(c)
    return out


@functools.lru_cache(maxsize=None)
def centerline(task: str) -> np.ndarray:
    # [N, 4] x, y, w_right, w_left in m (F1TENTH scaled to kart size)
    a = np.loadtxt(track_names()[task], delimiter=",", comments="#")
    return a * F1T_SCALE if task.startswith("f1tenth_") else a


def _tangent(p: np.ndarray) -> np.ndarray:
    # unit tangents of a closed polyline [N, 2]
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _kart(pub: TaskPublic) -> tuple[str, Scene, dict]:
    _, task, sseed = N.parse_task_id(pub.task_id)
    rng = np.random.default_rng(sseed)
    src = task.split("_")[0]
    cl = N.resample(centerline(task), WALL_STEP_M[src], closed=True)  # [M, 4]
    lap = N.arclen(np.vstack([cl, cl[:1]]))
    seg = min(rng.uniform(*SEG_FRAC) * lap, SEG_MAX_M)
    step = WALL_STEP_M[src]
    i0 = int(rng.integers(len(cl)))
    n_seg = int(round(seg / step))
    n_pad = int(round(WALL_PAD_M / step))
    idx = np.arange(i0 - n_pad, i0 + n_seg + n_pad + 1) % len(cl)
    xy, wr, wl = cl[:, :2], cl[:, 2], cl[:, 3]
    tg = _tangent(xy)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)  # left normal
    left, right = xy + nrm * wl[:, None], xy - nrm * wr[:, None]
    org = xy[i0].copy()  # recentre on the start
    w = N.World(f"tracks_{task}", floor_half_m=float(np.abs(xy - org).max() + 50))
    w.wall("wall_left", left[idx] - org)
    w.wall("wall_right", right[idx] - org)
    i1 = (i0 + n_seg) % len(cl)
    yaw0, yaw1 = (float(np.arctan2(tg[i][1], tg[i][0])) for i in (i0, i1))
    fin = N.rect(xy[i1] - org, FINISH_HALF_LEN_M, 0.5 * (wl[i1] + wr[i1]), yaw1)
    fin[:, :2] += nrm[i1] * 0.5 * (wl[i1] - wr[i1])
    rt = np.c_[xy[idx[n_pad:n_pad + n_seg + 1]] - org, np.zeros(n_seg + 1)]

    def wall(id: str, pts: np.ndarray) -> Entity:
        p3 = np.c_[pts - org, np.full(len(pts), 0.5)]
        return Entity(id, id.replace("_", " "), "fixture", N.pose7(p3.mean(0)), np.array([0.1, 0.1, 0.5]), p3, None)

    ents = [N.start([0, 0, 0], yaw0), N.region("finish", fin, yaw=yaw1), N.path("route", rt),
            wall("wall_left", left[idx]), wall("wall_right", right[idx])]
    return w.xml(), N.scene_of(ents), {"seg_m": n_seg * step}


def _gates_uzh() -> np.ndarray:
    g = np.asarray(yaml.safe_load(UZH.read_text())["gates"], np.float64)
    _, first = np.unique(g, axis=0, return_index=True)
    return g[np.sort(first)]  # one lap: the yaml repeats the 7 gates for 3 laps


def _gates_gen(rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(GEN_GATES[0], GEN_GATES[1] + 1))
    a = np.sort(rng.uniform(0, 2 * np.pi, k)) if rng.random() < 0.3 else np.linspace(0, 2 * np.pi, k, endpoint=False)
    r = rng.uniform(*GEN_RADIUS_M, k)
    return np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(*GEN_HEIGHT_M, k)], 1)


def _quad(pub: TaskPublic) -> tuple[str, Scene, dict]:
    _, task, sseed = N.parse_task_id(pub.task_id)
    rng = np.random.default_rng(sseed)
    s = QUAD_SCALE[pub.robot.name]
    gates = (_gates_uzh() if task == "gates_uzh" else _gates_gen(rng)) * s
    h, b = GATE_HALF_M * s, GATE_BAR_M * s
    gates[:, 2] = np.maximum(gates[:, 2], h + 3 * b)  # keep the gate frame above the floor
    # start one gate spacing before gate 0, finish one spacing past the last gate
    d0 = gates[1] - gates[0]
    st = gates[0] - d0 / np.linalg.norm(d0) * 2.0 * s
    st[2] = max(st[2], 1.0 * s)
    dl = gates[-1] - gates[-2]
    fin = gates[-1] + dl / np.linalg.norm(dl) * 2.0 * s
    rt = np.vstack([st, gates, fin])
    w = N.World(f"tracks_{task}", floor_half_m=float(np.abs(rt[:, :2]).max() + 10))
    ents = [N.start(st, float(np.arctan2(*(gates[0] - st)[1::-1]))),
            N.box_region("finish", fin, np.array([1.0, 1.0, 1.0]) * s), N.path("route", rt),
            Entity("floor", "floor", "fixture", N.pose7([0, 0, 0]), np.array([w.floor_half_m, w.floor_half_m, 0.0]),
                   None, None)]
    for k, g in enumerate(gates):
        nxt, prv = rt[k + 2], rt[k]
        d = nxt - prv
        yaw = float(np.arctan2(d[1], d[0]))  # gate normal along the direction of travel
        c, si = np.cos(yaw), np.sin(yaw)
        lat = np.array([-si, c, 0.0])
        gid = f"gate_{k}"
        for j, (off, size) in enumerate([(np.array([0, 0, h + b]), [b, h + 2 * b, b]),
                                         (np.array([0, 0, -h - b]), [b, h + 2 * b, b]),
                                         (lat * (h + b), [b, b, h]), (-lat * (h + b), [b, b, h])]):
            w.geom(f"{gid}:{j}", "box", size, list(g + off), yaw, "0.9 0.5 0.1 1")
        ents.append(Entity(gid, f"gate {k + 1}", "fixture", N.pose7(g, yaw), np.array([b, h + 2 * b, h + 2 * b]),
                           None, None))
    return w.xml(), N.scene_of(ents), {"len_m": N.arclen(rt) + float(np.sum(np.abs(np.diff(rt[:, 2])))),
                                       "n_gates": len(gates)}


def _build(pub: TaskPublic) -> tuple[str, Scene, dict]:
    return _quad(pub) if N.parse_task_id(pub.task_id)[1].startswith("gates_") else _kart(pub)


def scene(card: TaskCard | TaskPublic) -> tuple[str, Scene]:
    """MJCF world xml (no robot) and the Scene for a tracks card."""
    xml, sc, _ = _build(card.public if isinstance(card, TaskCard) else card)
    return xml, sc


def _kart_test(info: dict, style: str) -> tuple[list[dict], list[dict], float]:
    t_ref = info["seg_m"] / KART_VMAX_MPS
    preds = [pred("in_region", "base", "finish"), pred("no_contact", "robot", "wall_left"),
             pred("no_contact", "robot", "wall_right")]
    graded = [{"term": "time", "args": [], "good": 1.5 * t_ref, "bad": 6.0 * t_ref}]
    if style == "fast":
        preds.append(pred("time_below", t_s=round(3.0 * t_ref, 2)))
    if style == "gentle":
        graded = [{"term": "accel", "args": ["base"], "good": 2.0, "bad": 8.0},
                  {"term": "jerk", "args": ["base"], "good": 5.0, "bad": 40.0}]
    return preds, graded, min(10.0 * t_ref + 10.0, 600.0)


def _quad_test(info: dict, style: str, member: str) -> tuple[list[dict], list[dict], float]:
    t_ref = info["len_m"] / QUAD_VMAX_MPS[member]
    preds = [pred("in_region", "base", "finish"), pred("no_contact", "robot", "floor")]
    preds += [pred("no_contact", "robot", f"gate_{k}") for k in range(info["n_gates"])]
    graded = [{"term": "time", "args": [], "good": 1.5 * t_ref, "bad": 6.0 * t_ref}]
    if style == "fast":
        preds.append(pred("time_below", t_s=round(3.0 * t_ref, 2)))
    if style == "gentle":
        graded = [{"term": "tilt", "args": ["base"], "good": 0.2, "bad": 0.8},
                  {"term": "jerk", "args": ["base"], "good": 5.0, "bad": 50.0}]
    return preds, graded, min(10.0 * t_ref + 10.0, 600.0)


def translations() -> list[Translation]:
    kart = [pred("in_region", "base", "finish"), pred("no_contact", "robot", "wall_left"),
            pred("no_contact", "robot", "wall_right")]
    out = [Translation(t, kart, ["base", "finish", "wall_left", "wall_right"], KART_STYLES["default"][0],
                       "drive_course", 600.0, ["full_lap: segment of a lap instead"], "kart") for t in track_names()]
    quad = [pred("in_region", "base", "finish"), pred("no_contact", "robot", "floor")]
    out += [Translation(t, quad, ["base", "finish", "floor"], QUAD_STYLES["default"][0], "fly_course", 600.0,
                        ["gate_order: route only"], "quadrotor") for t in ("gates_uzh", "gates_gen")]
    return out


def coverage() -> dict:
    tr = translations()
    sub = {k: sum(t.task.startswith(k) for t in tr) for k in ("tum_", "f1tenth_", "gates_")}
    return write_coverage(SOURCE, [t.task for t in tr], tr, {}, {"per_subsource": sub})


def cards(n: int, seed: int) -> list[TaskCard]:
    """n cards: kart segments over the TUM and F1TENTH tracks and quadrotor gate courses (seeded, cycling)."""
    if not available():
        raise RuntimeError(f"track assets missing under {TUM.parent}")
    rng = np.random.default_rng(seed)
    names = sorted(track_names()) + ["gates_uzh", "gates_gen"]
    order = rng.permutation(len(names))
    coverage()
    out = []
    for j in range(n):
        task = names[int(order[j % len(names)])]
        sseed = int(rng.integers(0, 2**31 - 1))
        quad = task.startswith("gates_")
        rb = N.robot("quadrotor", N.pick(rng, N.members("quadrotor")), sseed) if quad else N.robot("kart", "kart",
                                                                                                    sseed)
        styles = QUAD_STYLES if quad else KART_STYLES
        style = N.pick(rng, sorted(styles))
        pub = TaskPublic(N.task_id(SOURCE, task, sseed), rb.family, rb, sseed, "", None)
        _, _, info = _build(pub)
        preds, graded, horizon = _quad_test(info, style, rb.name) if quad else _kart_test(info, style)
        out.append(N.card(SOURCE, task, sseed, rb, N.pick(rng, styles[style]), "fly_course" if quad else
                          "drive_course", preds, graded, horizon))
    return out
