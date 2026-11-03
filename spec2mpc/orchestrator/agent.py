"""Orchestrator agent: picks the next skill command for a goal from the camera image and the command history."""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Protocol

import numpy as np

MODEL = "gpt-6.1-sol"
EFFORT = "high"
DONE = "DONE"

SYSTEM = """You are the orchestrator of a robot. You hold an overall goal and issue one short skill command at a \
time to an executor that controls the robot (for example "grab the red mug", "drive to the dock gently", \
"open the top drawer"). After each command you see its execution status and a fresh camera image.
Issue the next single command that makes progress toward the goal, or set done=true when the goal is achieved \
or no further command can help. One action per command; describe objects as they appear in the image."""


def output_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["done", "command"],
            "properties": {"done": {"type": "boolean"}, "command": {"type": "string"}}}


def user_text(goal: str, history: list[tuple[str, str]]) -> str:
    lines = [f"goal: {goal}", "commands so far:"]
    lines += [f"{i + 1}. {c!r} -> {s}" for i, (c, s) in enumerate(history)] or ["none"]
    return "\n".join(lines)


def image_url(image: np.ndarray) -> str:
    # [H, W, 3] uint8 -> PNG data URL
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Agent(Protocol):
    def next(self, goal: str, image: np.ndarray | None, history: list[tuple[str, str]]) -> str: ...


class OpenAIAgent:
    """gpt-6.1-sol via the Responses API; returns the next command or DONE."""

    def __init__(self, client: Any = None, model: str = MODEL, effort: str = EFFORT,
                 max_output_tokens: int = 16000):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model, self.effort, self.max_output_tokens = client, model, effort, max_output_tokens

    def next(self, goal: str, image: np.ndarray | None, history: list[tuple[str, str]]) -> str:
        content: list[dict] = [{"type": "input_text", "text": user_text(goal, history)}]
        if image is not None:
            content.append({"type": "input_image", "image_url": image_url(image)})
        for _ in range(3):
            r = self.client.responses.create(
                model=self.model, max_output_tokens=self.max_output_tokens, instructions=SYSTEM,
                input=[{"role": "user", "content": content}], reasoning={"effort": self.effort},
                text={"format": {"type": "json_schema", "name": "next_command", "schema": output_schema(),
                                 "strict": True}})
            if r.status == "completed":
                break
        else:
            raise RuntimeError(f"orchestrator agent: no answer (status={r.status})")
        out = json.loads(r.output_text)
        return DONE if out["done"] or not out["command"].strip() else out["command"].strip()


class RecordedAgent:
    """Replays a fixed command transcript (tests and executor comparisons); DONE when exhausted."""

    def __init__(self, transcript: list[str]):
        self.transcript = list(transcript)
        self.histories: list[list[tuple[str, str]]] = []

    def next(self, goal: str, image: np.ndarray | None, history: list[tuple[str, str]]) -> str:
        self.histories.append(list(history))
        i = len(self.histories) - 1
        return self.transcript[i] if i < len(self.transcript) else DONE
