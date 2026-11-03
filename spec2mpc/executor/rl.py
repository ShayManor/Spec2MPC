"""Executor GRPO (TRL GRPOTrainer + LoRA, vLLM generation) with the MuJoCo test score as reward.

Rewards come from remote env workers over HTTP, so this module never reads task tests:
POST {url}/score {task_id, seed, program: Program.to_json()} -> {"success": bool, "score": float, "status": str}.
Prompt records (JSONL, public fields only): {"task_id", "seed", "command", "robot", "scene"} as in sft.py.
trl, peft, datasets and vllm are imported lazily: they exist only in the Modal image.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from spec2mpc.core import Program, from_jsonable
from spec2mpc.executor.decode import parse_program, split_output
from spec2mpc.executor.prompt import build_prompt, prompt_text
from spec2mpc.executor.sft import (filter_kwargs, last_checkpoint, load_backbone, lora_config, read_jsonl,
                                   time_checkpoint_callback)

Post = Callable[[str, dict], dict]


def _post(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


class RewardClient:
    """Round-robin client for env workers; only success, score, and status are read from replies."""

    def __init__(self, urls: list[str], post: Post | None = None):
        if not urls:
            raise ValueError("no env worker urls")
        self._urls = itertools.cycle([u.rstrip("/") for u in urls])
        self._lock = threading.Lock()
        self.post = post or _post

    def score(self, task_id: str, seed: int, program: Program) -> dict:
        with self._lock:
            url = next(self._urls)
        r = self.post(url + "/score", {"task_id": task_id, "seed": int(seed), "program": program.to_json()})
        return {"success": bool(r["success"]), "score": float(r["score"]), "status": str(r.get("status", ""))}


def _completion_text(c: Any) -> str:
    # TRL passes str completions, or [{"role": "assistant", "content": ...}] for conversational prompts
    if isinstance(c, str):
        return c
    content = c[-1]["content"]
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def make_reward_fn(client: RewardClient, invalid_reward: float = -1.0, workers: int = 16) -> Callable[..., list[float]]:
    """GRPO reward: invalid or unparseable programs get invalid_reward; valid ones the env worker's score."""
    from spec2mpc.program import validate

    def one(text: str, task_id: str, seed: int, robot: str, scene: str) -> float:
        p = parse_program(split_output(text).text)
        if p is None or validate(p, from_jsonable(json.loads(robot)), from_jsonable(json.loads(scene))):
            return invalid_reward
        return client.score(task_id, seed, p)["score"]

    def reward(prompts: list, completions: list, task_id: list[str], seed: list[int], robot: list[str],
               scene: list[str], **_: Any) -> list[float]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(one, [_completion_text(c) for c in completions], task_id, seed, robot, scene))

    reward.__name__ = "mujoco_test_score"
    return reward


def to_grpo_row(rec: dict, prompt_kwargs: dict | None = None) -> dict:
    """Prompt record -> {"prompt": chat, "image"?, task_id, seed, robot, scene (JSON strings for the reward)}."""
    robot, scene = from_jsonable(rec["robot"]), from_jsonable(rec["scene"])
    msgs = build_prompt(rec["command"], robot, scene, rec.get("feedback"), **(prompt_kwargs or {}))
    images = [p["image"] for m in msgs for p in m["content"] if p["type"] == "image"]
    row: dict[str, Any] = {"task_id": rec["task_id"], "seed": int(rec.get("seed", 0)),
                           "robot": json.dumps(rec["robot"]), "scene": json.dumps(rec["scene"])}
    if images:
        row["prompt"] = [{"role": m["role"], "content": [{"type": "image"} if p["type"] == "image" else p
                                                         for p in m["content"]]} for m in msgs]
        row["image"] = images[0]
    else:
        row["prompt"] = [{"role": m["role"], "content": prompt_text([m])} for m in msgs]
    return row


def _start_vllm_server(cfg: dict) -> subprocess.Popen:
    # TRL's vLLM server on the generation GPU; the trainer syncs LoRA-merged weights into it every step
    rc, b = cfg["rl"], cfg["backbone"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(rc["vllm_gpu"])}
    proc = subprocess.Popen(["trl", "vllm-serve", "--model", b["model_id"], "--revision", b["revision"],
                             "--port", str(rc["vllm_port"]), "--gpu_memory_utilization", "0.9"], env=env)
    deadline = time.time() + rc.get("vllm_start_timeout_s", 1800)
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{rc['vllm_port']}/health/", timeout=5)
            return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("trl vllm-serve exited")
            time.sleep(10)
    proc.kill()
    raise RuntimeError("trl vllm-serve did not come up")


def main(cfg: dict, on_save: Callable[[], None] | None = None) -> str:
    """GRPO from cfg rl.init_adapter (an SFT adapter) or the base model; returns the final adapter dir."""
    rc = cfg["rl"]
    server = None
    if rc["vllm_mode"] == "server":
        server = _start_vllm_server(cfg)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rc["train_gpu"])  # before torch initializes CUDA
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    try:
        rows = [to_grpo_row(r, cfg.get("prompt")) for r in read_jsonl(rc["prompts"])]
        model, proc = load_backbone(cfg)
        peft_cfg = lora_config(cfg)
        if rc.get("init_adapter"):
            from peft import PeftModel

            model, peft_cfg = PeftModel.from_pretrained(model, rc["init_adapter"], is_trainable=True), None
        args = GRPOConfig(**filter_kwargs(GRPOConfig, dict(
            output_dir=rc["output_dir"], learning_rate=rc["lr"], num_generations=rc["num_generations"],
            per_device_train_batch_size=rc["batch_size"], gradient_accumulation_steps=rc["grad_accum"],
            max_completion_length=rc["max_completion_length"], beta=rc["beta"], temperature=rc["temperature"],
            num_train_epochs=rc["epochs"], bf16=True, gradient_checkpointing=True, logging_steps=1,
            use_vllm=True, vllm_mode=rc["vllm_mode"], vllm_server_host="127.0.0.1",
            vllm_server_port=rc["vllm_port"], save_strategy="steps", save_steps=10**9, save_total_limit=2,
            report_to="none", seed=rc.get("seed", 0))))
        reward = make_reward_fn(RewardClient(rc["env_urls"]), rc["invalid_reward"], rc.get("reward_workers", 16))
        trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=args, train_dataset=Dataset.from_list(rows),
                              processing_class=proc, peft_config=peft_cfg,
                              callbacks=[time_checkpoint_callback(rc["checkpoint_minutes"], on_save)])
        trainer.train(resume_from_checkpoint=last_checkpoint(rc["output_dir"]))
        final = str(Path(rc["output_dir"]) / "final")
        trainer.save_model(final)
        if on_save is not None:
            on_save()
        return final
    finally:
        if server is not None:
            server.terminate()
