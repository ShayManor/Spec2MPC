from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from spec2mpc.core import types as T

ROBOT_CFG_DIR = Path(__file__).parent.parent.parent / "configs" / "v2" / "robots"
_TYPES = {c.__name__: c for c in (T.Entity, T.Scene, T.RobotSpec, T.Term, T.Stage, T.Program,
                                  T.TaskPublic, T.TaskTest, T.TaskCard, T.StepResult, T.Episode)}


def to_jsonable(obj: Any) -> Any:
    # dataclasses -> {"__type__": name, fields...}; arrays -> {"__ndarray__": list, dtype, shape}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {"__type__": type(obj).__name__, **d}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def from_jsonable(d: Any) -> Any:
    if isinstance(d, list):
        return [from_jsonable(v) for v in d]
    if not isinstance(d, dict):
        return d
    if "__ndarray__" in d:
        return np.asarray(d["__ndarray__"], dtype=d["dtype"]).reshape(d["shape"])
    if "__type__" in d:
        cls = _TYPES[d["__type__"]]
        return cls(**{k: from_jsonable(v) for k, v in d.items() if k != "__type__"})
    return {k: from_jsonable(v) for k, v in d.items()}


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(obj)))


def load_json(path: str | Path) -> Any:
    return from_jsonable(json.loads(Path(path).read_text()))


def load_family_config(family: str) -> dict:
    # configs/v2/robots/<family>.yaml: family, solver, floors, ranges, members
    return yaml.safe_load((ROBOT_CFG_DIR / f"{family}.yaml").read_text())
