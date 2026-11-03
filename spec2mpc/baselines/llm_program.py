"""Baseline B: LLM-composed program. gpt-6.1-sol zero-shot writes the MPC program from the Task 9 executor prompt
plus the vocabulary documentation, under the same json_schema (OpenAI strict structured output) and validate.

Responses are cached by (command, robot.name, scene hash, feedback) in data/v2/llm_program_cache.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from spec2mpc.core import ExecutorError, Program, RobotSpec, Scene, to_jsonable
from spec2mpc.executor.decode import parse_program
from spec2mpc.executor.prompt import build_prompt
from spec2mpc.orchestrator.agent import image_url
from spec2mpc.program import json_schema, validate

MODEL = "gpt-6.1-sol"
EFFORT = "high"
CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "v2" / "llm_program_cache.json"


def openai_schema(schema: dict) -> dict:
    # json_schema(robot, scene) in the subset OpenAI strict mode accepts: every property required (optional ones
    # become nullable), and `items: false` replaced by the union of the positional enums (lengths stay fixed by
    # minItems/maxItems and positions are checked again by validate)
    if isinstance(schema, list):
        return [openai_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    n = {k: openai_schema(v) for k, v in schema.items()}
    if n.get("items") is False:
        vals = sorted({v for s in n.get("prefixItems", []) for v in s["enum"]}) or [""]
        n["items"] = {"type": "string", "enum": vals}
    if n.get("type") == "object" and "properties" in n:
        req = n.get("required", [])
        n["properties"] = {k: v if k in req else {"anyOf": [v, {"type": "null"}]} for k, v in n["properties"].items()}
        n["required"] = list(n["properties"])
    return n


def from_openai(d: Any) -> Any:
    # drop the nulls openai_schema allowed for omitted optional params
    if isinstance(d, list):
        return [from_openai(v) for v in d]
    if isinstance(d, dict):
        return {k: from_openai(v) for k, v in d.items() if v is not None}
    return d


def scene_hash(scene: Scene) -> str:
    h = hashlib.sha256(json.dumps(to_jsonable(Scene(scene.entities, None, scene.t_s)), sort_keys=True).encode())
    if scene.image is not None:
        img = np.ascontiguousarray(scene.image)
        h.update(str(img.shape).encode() + img.tobytes())
    return h.hexdigest()


def to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    # build_prompt chat messages -> (instructions, Responses API input); text and image parts kept in order
    instructions = "\n\n".join(p["text"] for m in messages if m["role"] == "system" for p in m["content"])
    inp = [{"role": m["role"], "content": [
        {"type": "input_image", "image_url": image_url(np.asarray(p["image"]))} if p["type"] == "image"
        else {"type": "input_text", "text": p["text"]} for p in m["content"]]}
        for m in messages if m["role"] != "system"]
    return instructions, inp


class LLMProgramExecutor:
    """Executor: one strict structured-output call, validate, retry once, else ExecutorError("invalid_program")."""

    def __init__(self, client: Any = None, model: str = MODEL, effort: str = EFFORT,
                 cache_path: str | Path = CACHE_PATH, retries: int = 1, max_output_tokens: int = 16000,
                 name: str = "llm_program"):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model, self.effort, self.retries = client, model, effort, retries
        self.max_output_tokens, self.name = max_output_tokens, name
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._lock = threading.Lock()  # guards the cache dict and file

    def key(self, command: str, robot: RobotSpec, scene: Scene, feedback: str | None) -> str:
        return hashlib.sha256(json.dumps([command, robot.name, scene_hash(scene), feedback]).encode()).hexdigest()

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self.cache, indent=1))
        tmp.replace(self.cache_path)

    def act(self, command: str, robot: RobotSpec, scene: Scene, feedback: str | None) -> Program:
        k = self.key(command, robot, scene, feedback)
        with self._lock:
            if k in self.cache:
                return Program.from_json(self.cache[k]["program"])
        instructions, inp = to_responses_input(build_prompt(command, robot, scene, feedback, docs=True))
        fmt = {"type": "json_schema", "name": "program", "schema": openai_schema(json_schema(robot, scene)),
               "strict": True}
        errors: list[str] = []
        for _ in range(1 + self.retries):  # Review Focus 3: retry once, then raise
            r = self.client.responses.create(model=self.model, max_output_tokens=self.max_output_tokens,
                                             instructions=instructions, input=inp,
                                             reasoning={"effort": self.effort}, text={"format": fmt})
            p = None
            if r.status == "completed":
                try:
                    p = parse_program(json.dumps(from_openai(json.loads(r.output_text))))
                except ValueError:
                    p = None
            errs = [f"unparseable({r.status})"] if p is None else validate(p, robot, scene)
            if not errs:
                with self._lock:
                    self.cache[k] = {"command": command, "robot": robot.name, "program": p.to_json()}
                    self._save()
                return p
            errors += errs
        raise ExecutorError("invalid_program", ",".join(sorted(set(errors))))
