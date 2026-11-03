from __future__ import annotations

import re

import numpy as np

from spec2mpc.core.types import Entity, Scene
from spec2mpc.perception.v2.grasps import add_grasps

# Image perception for benchmarks: OWLv2 open-vocabulary boxes, depth back-projection, grasp proposals.

MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEFAULT_LABELS = ["mug", "cup", "bowl", "plate", "bottle", "can", "box", "block", "ball", "fruit",
                  "drawer", "door", "handle", "basket", "tray", "pan", "pot", "book", "tool"]
MAX_PTS = 1024
_CACHE: dict[str, tuple] = {}


def _detector(model_id: str) -> tuple:
    if model_id not in _CACHE:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = Owlv2ForObjectDetection.from_pretrained(model_id).to(dev).eval()
        _CACHE[model_id] = (Owlv2Processor.from_pretrained(model_id), model, dev)
    return _CACHE[model_id]


def detect(rgb: np.ndarray, labels: list[str], threshold: float = 0.15,
           model_id: str = MODEL_ID) -> list[tuple[str, float, np.ndarray]]:
    """(label, score, box [4] x0 y0 x1 y1 px) per detection, class-agnostic NMS at IoU 0.5."""
    import torch
    from PIL import Image

    proc, model, dev = _detector(model_id)
    H, W = rgb.shape[:2]
    inputs = proc(text=[labels], images=Image.fromarray(rgb), return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    S = max(H, W)  # OWLv2 pads to a square, so boxes are in padded-image coordinates
    r = proc.post_process_grounded_object_detection(out, threshold=threshold, target_sizes=[(S, S)],
                                                    text_labels=[labels])[0]
    dets = sorted(zip(r["text_labels"], r["scores"].tolist(), r["boxes"].cpu().numpy()), key=lambda x: -x[1])
    keep: list[tuple[str, float, np.ndarray]] = []
    for lab, sc, b in dets:
        b = np.clip(b, 0, [W, H, W, H])
        if all(_iou(b, k[2]) < 0.5 for k in keep):
            keep.append((lab, sc, b))
    return keep


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def _cam_to_world(cam_pose: np.ndarray) -> np.ndarray:
    # [4, 4] camera to world (OpenCV axes: x right, y down, z forward); accepts [4, 4] or [7] xyz + wxyz
    if cam_pose.shape == (4, 4):
        return cam_pose
    w, x, y, z = cam_pose[3:] / np.linalg.norm(cam_pose[3:])
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = cam_pose[:3]
    return T


def _backproject(depth: np.ndarray, K: np.ndarray, T: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # world points [N, 3] and their depths [N] for valid pixels inside box
    x0, y0, x1, y1 = np.round(box).astype(int)
    v, u = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1]
    ok = np.isfinite(z) & (z > 0)
    u, v, z = u[ok] + 0.5, v[ok] + 0.5, z[ok]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return pc @ T[:3, :3].T + T[:3, 3], z


def _segment(depth: np.ndarray, K: np.ndarray, T: np.ndarray, box: np.ndarray) -> np.ndarray:
    # [N, 3] world points of the detected object: depth-consistent with the box center, above the support plane
    bw, bh = box[2] - box[0], box[3] - box[1]
    inner = box + np.array([0.25 * bw, 0.25 * bh, -0.25 * bw, -0.25 * bh])
    _, z_in = _backproject(depth, K, T, inner)
    if len(z_in) == 0:
        return np.zeros((0, 3))
    d0 = float(np.median(z_in))
    size_m = max(bw, bh) * d0 / K[0, 0]
    pts, z = _backproject(depth, K, T, box)
    pts = pts[np.abs(z - d0) < max(0.03, size_m)]
    # support plane (z up): dominant height in a ring around the box, if the ring is flat
    H, W = depth.shape
    big = np.clip(box + np.array([-0.3 * bw, -0.3 * bh, 0.3 * bw, 0.3 * bh]), 0, [W, H, W, H])
    ring, _ = _backproject(depth, K, T, big)
    if len(ring) and len(pts):
        hist, edges = np.histogram(ring[:, 2], bins=np.arange(ring[:, 2].min(), ring[:, 2].max() + 0.01, 0.005))
        z_sup = edges[np.argmax(hist)] + 0.0025
        if hist.max() > 0.2 * len(ring) and z_sup < np.percentile(pts[:, 2], 90):
            pts = pts[pts[:, 2] > z_sup + 0.008]
    return pts


def _complete(pts: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # center [3] (box midpoint) and points mirrored across the vertical plane through it, facing the camera
    c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    v = c[:2] - cam[:2]
    v = np.array([*(v / (np.linalg.norm(v) + 1e-9)), 0.0])
    s = (pts - c) @ v
    return c, np.concatenate([pts, pts - 2 * s[:, None] * v])


def perceive_images(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_pose: np.ndarray,
                    labels_hint: list[str] | None, threshold: float = 0.15, n_grasps: int = 8,
                    model_id: str = MODEL_ID, t_s: float = 0.0) -> Scene:
    """Scene from one RGB-D view: rgb [H, W, 3] uint8, depth [H, W] m along the optical axis,
    K [3, 3] intrinsics, cam_pose camera to world ([4, 4] or [7], OpenCV axes).

    Each detection becomes an "object" entity "<label>_<k>" with world pose (identity rotation),
    extent, and a cloud completed by mirror symmetry; grasps come from add_grasps.
    """
    labels = labels_hint or DEFAULT_LABELS
    T = _cam_to_world(np.asarray(cam_pose, dtype=float))
    rng = np.random.default_rng(0)
    ents: list[Entity] = []
    counts: dict[str, int] = {}
    for lab, _, box in detect(rgb, labels, threshold, model_id):
        pts = _segment(depth, K, T, box)
        if len(pts) < 24:
            continue
        c, full = _complete(pts, T[:3, 3])
        if len(full) > MAX_PTS:
            full = full[rng.choice(len(full), MAX_PTS, replace=False)]
        key = re.sub(r"\W+", "_", lab.strip().lower())
        k = counts.get(key, 0)
        counts[key] = k + 1
        ents.append(Entity(f"{key}_{k}", lab, "object", np.array([*c, 1.0, 0, 0, 0]),
                           np.abs(full - c).max(axis=0), full, None))
    return add_grasps(Scene(ents, rgb, t_s), n=n_grasps)
