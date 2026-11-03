"""Commands: template commands for generated cards, LLM paraphrases (gpt-6.1-sol), and the threshold-leak check."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np

from spec2mpc.core import Scene, TaskCard, TaskTest

MODEL = "gpt-6.1-sol"
# llm(instructions, user_text, json_schema) -> JSON text; injectable so tests never call the API
LLM = Callable[[str, str, dict], str]

# predicate and graded params that are thresholds (pose coordinates and penalty weights are not)
THRESHOLD_KEYS = ("tol_m", "tol_rad", "min", "t_s", "max_tilt_rad", "min_height_m", "good", "bad")

# unit -> factor to SI (m, rad, s, fraction, N, J)
_UNITS = {"mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3, "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2,
          "m": 1.0, "meter": 1.0, "metre": 1.0, "km": 1e3, "deg": np.pi / 180, "degree": np.pi / 180,
          "°": np.pi / 180, "rad": 1.0, "radian": 1.0, "s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0,
          "minute": 60.0, "%": 0.01, "percent": 0.01, "n": 1.0, "newton": 1.0, "j": 1.0, "joule": 1.0,
          "kj": 1e3, "g": 9.81, "mps": 1.0}
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "hundred": 100, "half": 0.5, "quarter": 0.25}
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*([a-z°%]+)?", re.I)
# no-unit numbers are compared in every unit a threshold could be phrased in
_BARE = (1.0, 1e2, 1e3, 180 / np.pi, 1 / 60)


def thresholds(test: TaskTest) -> list[float]:
    out = []
    for d in list(test.predicates) + list(test.graded):
        params = d.get("params", d)
        for k in THRESHOLD_KEYS:
            v = params.get(k)
            if isinstance(v, (int, float)) and float(v) != 0.0:
                out.append(abs(float(v)))
    return out


def _numbers(text: str) -> list[tuple[float, float | None]]:
    # (value, SI factor or None when no unit)
    out: list[tuple[float, float | None]] = []
    for m in _NUM.finditer(text):
        u = (m.group(2) or "").lower()
        u = u[:-1] if u.endswith("s") and u[:-1] in _UNITS else u
        out.append((float(m.group(1)), _UNITS.get(u)))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _WORDS:
            out.append((float(_WORDS[w]), None))
    return out


def has_threshold(command: str, test: TaskTest, rel: float = 0.02) -> bool:
    """True if the command states any of the test's threshold values (in any common unit)."""
    ts = thresholds(test)
    for v, f in _numbers(command):
        cands = [v * f] if f is not None else [v / s for s in _BARE]
        if any(abs(c - t) <= rel * t for c in cands for t in ts):
            return True
    return False


def template_command(templates: tuple[str, ...], slots: dict[str, str], style_phrase: str,
                     rng: np.random.Generator) -> str:
    cmd = templates[int(rng.integers(len(templates)))].format(**slots)
    return f"{cmd} {style_phrase}".strip() if style_phrase else cmd


# ---- LLM paraphrases ----

INSTRUCTIONS = """You write short commands that a robot orchestrator issues to a skill executor \
(for example "grab the red mug", "drive to the dock gently"). Rewrite the given command in several different ways. \
Keep exactly the same task, the same objects and places (describe them by their given colour and name), and the same \
manner (gentle, fast, careful, energy saving, or plain). Never state any number, distance, tolerance, angle, time, \
speed, force, or limit, and never invent constraints that the command does not state. One imperative sentence each, \
as a person would say it."""


def _schema() -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["commands"],
            "properties": {"commands": {"type": "array", "items": {"type": "string"}}}}


class OpenAIJSON:
    """gpt-6.1-sol through the Responses API with a strict JSON schema (key from OPENAI_API_KEY)."""

    def __init__(self, client: Any = None, model: str = MODEL, effort: str = "low", max_output_tokens: int = 8000):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model, self.effort, self.max_output_tokens = client, model, effort, max_output_tokens

    def __call__(self, instructions: str, user: str, schema: dict) -> str:
        for _ in range(3):
            r = self.client.responses.create(
                model=self.model, max_output_tokens=self.max_output_tokens, instructions=instructions,
                input=[{"role": "user", "content": [{"type": "input_text", "text": user}]}],
                reasoning={"effort": self.effort},
                text={"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}})
            if r.status == "completed":
                return r.output_text
        raise RuntimeError(f"{self.model}: no answer (status={r.status})")


def _user_text(card: TaskCard, scene: Scene | None, n: int) -> str:
    pub = card.public
    lines = [f"robot: {pub.robot.name} ({pub.family})", f"command: {pub.command}"]
    if scene is not None:
        labels = sorted({e.label for e in scene.entities if e.kind in ("object", "fixture", "region")})
        lines.append("things in the scene: " + ", ".join(labels))
    lines.append(f"write {n} different rewrites")
    return "\n".join(lines)


def draft_commands(card: TaskCard, n: int, llm: LLM | None = None, scene: Scene | None = None,
                   rounds: int = 3) -> list[str]:
    """Up to n paraphrases of the card's command; any that states a test threshold is rejected."""
    llm = llm or OpenAIJSON()
    if scene is None:
        from spec2mpc.tasks.cards import build_scene

        scene = build_scene(card.public)[1]
    seen = {card.public.command.strip().lower()}
    out: list[str] = []
    for _ in range(rounds):
        if len(out) >= n:
            break
        raw = json.loads(llm(INSTRUCTIONS, _user_text(card, scene, n - len(out)), _schema()))
        for c in raw.get("commands", []):
            c = " ".join(str(c).split())
            if c and c.lower() not in seen and not has_threshold(c, card.test):
                seen.add(c.lower())
                out.append(c)
    return out[:n]
