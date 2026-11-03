"""Stub executor server for benchmark smoke runs: POST /act returns one fixed hold program.

usage: python stub_executor.py [port]   (same request/response format as the Task 9 executor server)
"""
from __future__ import annotations

import sys
import threading

from flask import Flask, jsonify, request

from spec2mpc.core import Program, Stage, Term

HOLD = Program([Stage([Term("smooth", [], {"w": 1.0})], Term("timeout", [], {"t_s": 3.0}))])


def create_app() -> Flask:
    app = Flask("stub_executor")
    calls: dict[str, int] = {}
    lock = threading.Lock()

    @app.post("/act")
    def act():
        body = request.get_json()
        if not body or "episode_id" not in body or "command" not in body:
            return jsonify({"error": "bad_request"}), 400
        with lock:
            calls[body["episode_id"]] = calls.get(body["episode_id"], 0) + 1
        return jsonify({"program": HOLD.to_json(), "calls": calls[body["episode_id"]]})

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765, threaded=True)
