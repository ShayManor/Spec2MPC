import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch
from omegaconf import DictConfig
from torch import Tensor

SLOT_NAMES = ("t_max_s", "v_max_mps", "a_lat_max_g", "energy_max_Wh", "generic")
D_SLOT = 2 * len(SLOT_NAMES)

WORDS = {w: float(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDS["half"] = 0.5

# unit text (lowercase) -> (slot name, factor to SI / slot unit)
UNITS = {
    **{u: ("t_max_s", 1.0) for u in ("s", "sec", "secs", "second", "seconds")},
    **{u: ("t_max_s", 60.0) for u in ("min", "mins", "minute", "minutes")},
    **{u: ("t_max_s", 3600.0) for u in ("h", "hr", "hrs", "hour", "hours")},
    "m/s": ("v_max_mps", 1.0),
    "km/h": ("v_max_mps", 1 / 3.6),
    "kph": ("v_max_mps", 1 / 3.6),
    "mph": ("v_max_mps", 0.44704),
    "g": ("a_lat_max_g", 1.0),
    "wh": ("energy_max_Wh", 1.0),
    "kwh": ("energy_max_Wh", 1000.0),
}

NUMBER = re.compile(
    r"(?<![\w.])(?:(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|(?P<word>" + "|".join(WORDS) + r"))(?![\w.]*\d)"
    r"(?:(?<=half)\s+an?(?=\s))?"
    r"[\s-]*(?P<unit>[^\W\d_]+(?:/[^\W\d_]+)?)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Slot:
    name: str
    value: float
    unit: str


def parse_slots(text: str) -> list[Slot]:
    slots = []
    for m in NUMBER.finditer(text):
        try:
            value = float(m["digits"].replace(",", "")) if m["digits"] else WORDS[m["word"].lower()]
        except (ValueError, OverflowError):
            continue
        unit = m["unit"] or ""
        name, factor = UNITS.get(unit.lower(), ("generic", 1.0))
        if math.isfinite(value * factor):
            slots.append(Slot(name, value * factor, unit))
    return slots


def slots_dict(slots: list[Slot]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in slots:
        if s.name != "generic":
            out.setdefault(s.name, s.value)
    return out


def slot_vector(slots: list[Slot], slot_cfg: DictConfig) -> Tensor:
    # returns [D_SLOT]: per name [present, log1p(value / scale)]
    first: dict[str, float] = {}
    for s in slots:
        first.setdefault(s.name, s.value)
    v = torch.zeros(D_SLOT, dtype=torch.float32)
    for i, name in enumerate(SLOT_NAMES):
        if name in first:
            v[2 * i] = 1.0
            v[2 * i + 1] = math.log1p(first[name] / float(slot_cfg.scales[name]))
    return v


def slot_matrix(texts: Sequence[str], slot_cfg: DictConfig) -> Tensor:
    # returns [N, D_SLOT]
    if not len(texts):
        return torch.zeros(0, D_SLOT, dtype=torch.float32)
    return torch.stack([slot_vector(parse_slots(t), slot_cfg) for t in texts])
