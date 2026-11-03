"""Executor server. Wire format matches spec2mpc/orchestrator/loop.py HTTPExecutor.

POST /act   {executor, episode_id, command, robot, scene, feedback} -> 200 {"program": Program.to_json()}
            (an action-chunk executor instead stores its ActionChunkSource for the episode -> 200 {"chunks": true})
POST /chunk {executor, episode_id, scene, qpos} -> 200 {"actions": [[...], ...]}
POST /end   {episode_id} drops the episode state.
Failures are non-200 with {"error": <ExecutorError code>, "message": str}. State is keyed by episode_id, so
concurrent clients never share feedback or chunk sources (Review Focus 5).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

from spec2mpc.core import ActionChunkSource, Executor, ExecutorError, Program, from_jsonable


@dataclass
class EpisodeState:
    feedback: list[str] = field(default_factory=list)  # every feedback string this episode received, in order
    programs: list[Program] = field(default_factory=list)
    chunks: ActionChunkSource | None = None


def _err(code: str, status: int, msg: str = "") -> Any:
    return jsonify({"error": code, "message": msg}), status


def create_app(executors: dict[str, Executor]) -> Flask:
    app = Flask(__name__)
    episodes: dict[str, EpisodeState] = {}
    lock = threading.Lock()  # guards the episodes dict only; executors handle their own concurrency

    def state(eid: str) -> EpisodeState:
        with lock:
            return episodes.setdefault(eid, EpisodeState())

    def body(*keys: str) -> dict:
        d = request.get_json(silent=True)
        if not isinstance(d, dict) or any(k not in d for k in keys):
            raise ExecutorError("bad_request", f"need {keys}")
        if d["executor"] not in executors:
            raise ExecutorError("unknown_executor", str(d["executor"]))
        return d

    @app.errorhandler(ExecutorError)
    def executor_error(e: ExecutorError) -> Any:
        return _err(e.code, {"bad_request": 400, "unknown_executor": 404, "no_chunk_source": 409}.get(e.code, 422),
                    str(e))

    @app.post("/act")
    def act() -> Any:
        d = body("executor", "episode_id", "command", "robot", "scene")
        st = state(str(d["episode_id"]))
        if d.get("feedback"):
            st.feedback.append(str(d["feedback"]))
        robot, scene = from_jsonable(d["robot"]), from_jsonable(d["scene"])
        fb = "\n".join(st.feedback) or None
        out = executors[d["executor"]].act(d["command"], robot, scene, fb)
        if isinstance(out, Program):
            st.programs.append(out)
            return jsonify({"program": out.to_json()})
        st.chunks = out
        return jsonify({"chunks": True})

    @app.post("/chunk")
    def chunk() -> Any:
        d = body("executor", "episode_id", "scene", "qpos")
        src = state(str(d["episode_id"])).chunks
        ex = executors[d["executor"]]
        if src is None and isinstance(ex, ActionChunkSource):
            src = ex
        if src is None:
            raise ExecutorError("no_chunk_source", "call /act first")
        a = src.next_chunk(from_jsonable(d["scene"]), np.asarray(d["qpos"], dtype=np.float64))  # [T, act_dim]
        return jsonify({"actions": np.asarray(a, dtype=np.float64).tolist()})

    @app.post("/end")
    def end() -> Any:
        d = request.get_json(silent=True) or {}
        with lock:
            episodes.pop(str(d.get("episode_id")), None)
        return jsonify({"ok": True})

    app.config["episodes"] = episodes  # exposed for tests and logging
    return app


def main(cfg: dict, lora_path: str | None = None, names: tuple[str, ...] | None = None) -> None:
    # serve the VLM executor (vLLM backend) and/or the "llm_program" baseline on cfg["server"]["host"]:port
    executors: dict[str, Executor] = {}
    for n in names or (cfg.get("name", "spec2mpc"),):
        if n == "llm_program":
            from spec2mpc.baselines import llm_program

            executors[n] = llm_program.LLMProgramExecutor()
            continue
        from spec2mpc.executor.decode import make_executor

        ex = make_executor(cfg, lora_path)
        executors[ex.name] = ex
    create_app(executors).run(cfg["server"]["host"], int(cfg["server"]["port"]), threaded=True)
