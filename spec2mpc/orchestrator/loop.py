"""Orchestrator loop: goal -> commands -> executor -> env, identical commands and feedback for every executor.

The env is any object with run_program(program, max_s) and run_actions(policy, max_s) (SimEnv, or a benchmark
client's wrapper). If it also has observe() -> Scene, the scene is refreshed after each command; if it has
qpos() -> np.ndarray, action-chunk executors receive it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from spec2mpc.core import ActionChunkSource, Episode, Executor, ExecutorError, Program, RobotSpec, Scene, to_jsonable
from spec2mpc.orchestrator.agent import DONE, Agent, OpenAIAgent

Post = Callable[[str, dict], tuple[int, dict]]


def feedback_text(command: str, status: str) -> str:
    # the one feedback string every executor receives
    return f"previous command {command!r} ended with status {status}"


def _urllib_post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class HTTPExecutor:
    """Executor client for the executor server's POST /act; non-200 replies raise ExecutorError(error)."""

    def __init__(self, url: str, name: str, episode_id: str, post: Post | None = None):
        self.url, self.name, self.episode_id = url.rstrip("/"), name, episode_id
        self.post = post or _urllib_post

    def act(self, command: str, robot: RobotSpec, scene: Scene, feedback: str | None) -> Program:
        code, body = self.post(self.url + "/act", {
            "executor": self.name, "episode_id": self.episode_id, "command": command,
            "robot": to_jsonable(robot), "scene": to_jsonable(scene), "feedback": feedback})
        if code != 200 or "program" not in body:
            raise ExecutorError(body.get("error", f"http_{code}"), body.get("message", ""))
        return Program.from_json(body["program"])


def _chunk_policy(src: ActionChunkSource, qpos: Callable[[], np.ndarray]) -> Callable[[Scene], np.ndarray]:
    buf: list[np.ndarray] = []

    def policy(scene: Scene) -> np.ndarray:
        if not buf:
            buf.extend(src.next_chunk(scene, qpos()))  # [T, act_dim]
        return buf.pop(0)  # [act_dim]

    return policy


def _execute(command: str, env: Any, executor: Executor, robot: RobotSpec, scene: Scene, feedback: str | None,
             max_s: float, task_id: str) -> Episode:
    t0 = time.time()
    for attempt in range(2):  # Review Focus 3: retry once, then report failure
        try:
            out = executor.act(command, robot, scene, feedback)
            break
        except ExecutorError as e:
            status = f"executor_error:{e.code}"
            feedback = feedback_text(command, status)
    else:
        return Episode(task_id, executor.name, [], {}, status, time.time() - t0)
    if isinstance(out, Program):
        ep = env.run_program(out, max_s)
    else:
        ep = env.run_actions(_chunk_policy(out, getattr(env, "qpos", lambda: np.zeros(0))), max_s)
    return Episode(task_id or ep.task_id, executor.name, ep.programs, ep.trace, ep.status, ep.wall_s)


def _executor(executor: Executor | None, url: str, name: str, post: Post | None) -> Executor:
    return executor if executor is not None else HTTPExecutor(url, name, uuid.uuid4().hex, post)


def run_command(command: str, env: Any, executor_url: str, executor_name: str, *, robot: RobotSpec,
                scene: Scene, feedback: str | None = None, max_s: float = 60.0, task_id: str = "",
                executor: Executor | None = None, post: Post | None = None) -> Episode:
    """One atomic command (generated-suite tasks), no decomposition by the agent."""
    ex = _executor(executor, executor_url, executor_name, post)
    return _execute(command, env, ex, robot, scene, feedback, max_s, task_id)


def run_goal(goal: str, env: Any, executor_url: str, executor_name: str, max_commands: int = 12, *,
             robot: RobotSpec, scene: Scene, agent: Agent | None = None, max_s: float = 60.0, task_id: str = "",
             executor: Executor | None = None, post: Post | None = None) -> list[Episode]:
    """Agent issues commands until DONE or max_commands; one executor episode id for the whole goal."""
    agent = agent or OpenAIAgent()
    ex = _executor(executor, executor_url, executor_name, post)
    history: list[tuple[str, str]] = []
    episodes: list[Episode] = []
    feedback: str | None = None
    for _ in range(max_commands):
        command = agent.next(goal, scene.image, history)
        if command == DONE:
            break
        ep = _execute(command, env, ex, robot, scene, feedback, max_s, task_id)
        episodes.append(ep)
        history.append((command, ep.status))
        feedback = feedback_text(command, ep.status)
        if hasattr(env, "observe"):
            scene = env.observe()
    return episodes
