"""Executor SFT (TRL SFTTrainer + LoRA) and the zero-shot SFT gate.

SFT records (JSONL, public fields only): {"command", "robot": to_jsonable(RobotSpec), "scene": to_jsonable(Scene),
"program": Program.to_json(), "rationale": str | None, "feedback": str | None}. oracle_to_sft builds them from
Task 8's oracle files, reading only the best program and its success flag.
Checkpoints are saved every cfg checkpoint_minutes (30) to output_dir (a Modal volume) and resumed from there.
trl, peft and datasets are imported lazily: they exist only in the Modal image.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spec2mpc.core import ExecutorError, Program, from_jsonable
from spec2mpc.executor.decode import THINK_CLOSE, THINK_OPEN
from spec2mpc.executor.prompt import build_prompt, prompt_text


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def oracle_to_sft(oracle_dir: str | Path, publics: list[dict], require_success: bool = True) -> list[dict]:
    """publics: train-split prompt records {task_id, command, robot, scene}; oracle_dir: <task_id>.json files."""
    out = []
    for rec in publics:
        f = Path(oracle_dir) / f"{rec['task_id']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if require_success and not bool((d.get("best_score") or {}).get("success", False)):
            continue
        best = d["best"]
        prog = from_jsonable(best) if "__type__" in best else Program.from_json(best)
        out.append({"command": rec["command"], "robot": rec["robot"], "scene": rec["scene"],
                    "program": prog.to_json(), "rationale": d.get("rationale"), "feedback": None})
    return out


def target_text(program: Program, rationale: str | None) -> str:
    # assistant turn: reasoning trace, then the program JSON
    return f"{THINK_OPEN}\n{(rationale or '').strip()}\n{THINK_CLOSE}\n\n{json.dumps(program.to_json())}"


def to_sft_row(rec: dict, prompt_kwargs: dict | None = None) -> dict:
    """One record -> {"messages": chat with the assistant target, "images": list of PIL images}."""
    msgs = build_prompt(rec["command"], from_jsonable(rec["robot"]), from_jsonable(rec["scene"]),
                        rec.get("feedback"), **(prompt_kwargs or {}))
    images = [p["image"] for m in msgs for p in m["content"] if p["type"] == "image"]
    if images:
        msgs = [{"role": m["role"], "content": [{"type": "image"} if p["type"] == "image" else p
                                                for p in m["content"]]} for m in msgs]
        tgt: Any = [{"type": "text", "text": target_text(Program.from_json(rec["program"]), rec.get("rationale"))}]
    else:
        msgs = [{"role": m["role"], "content": prompt_text([m])} for m in msgs]
        tgt = target_text(Program.from_json(rec["program"]), rec.get("rationale"))
    return {"messages": msgs + [{"role": "assistant", "content": tgt}], "images": images}


def filter_kwargs(cls: type, kw: dict) -> dict:
    # keep only the fields this TRL version's config dataclass accepts
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(kw) - names)
    if dropped:
        print(f"[executor] {cls.__name__} ignores {dropped}")
    return {k: v for k, v in kw.items() if k in names}


def time_checkpoint_callback(minutes: float, on_save: Callable[[], None] | None = None) -> Any:
    """TrainerCallback that saves every `minutes` of wall-clock and calls on_save (e.g. a Modal volume commit)."""
    from transformers import TrainerCallback

    class TimeCheckpoint(TrainerCallback):
        def __init__(self) -> None:
            self.last = time.time()

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if time.time() - self.last >= minutes * 60:
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            self.last = time.time()
            if on_save is not None:
                on_save()
            return control

    return TimeCheckpoint()


def last_checkpoint(output_dir: str | Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir)) if Path(output_dir).is_dir() else None


def load_backbone(cfg: dict) -> tuple[Any, Any]:
    # bf16 VLM backbone and its processor at the pinned revision
    import torch
    import transformers

    b = cfg["backbone"]
    cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(b["model_id"], revision=b["revision"], dtype=torch.bfloat16,
                                attn_implementation=cfg.get("attn_implementation", "sdpa"))
    proc = transformers.AutoProcessor.from_pretrained(b["model_id"], revision=b["revision"])
    return model, proc


def lora_config(cfg: dict) -> Any:
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM")


def main(cfg: dict, on_save: Callable[[], None] | None = None) -> str:
    """SFT on oracle programs; returns the output dir holding the final adapter."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    sc = cfg["sft"]
    rows = [to_sft_row(r, cfg.get("prompt")) for r in read_jsonl(sc["data"])]
    if not any(r["images"] for r in rows):
        rows = [{"messages": r["messages"]} for r in rows]
    model, proc = load_backbone(cfg)
    args = SFTConfig(**filter_kwargs(SFTConfig, dict(
        output_dir=sc["output_dir"], num_train_epochs=sc["epochs"], learning_rate=sc["lr"],
        per_device_train_batch_size=sc["batch_size"], gradient_accumulation_steps=sc["grad_accum"],
        max_length=sc["max_length"], bf16=True, gradient_checkpointing=True, logging_steps=10,
        save_strategy="steps", save_steps=10**9, save_total_limit=2, report_to="none",
        assistant_only_loss=False, completion_only_loss=None, seed=sc.get("seed", 0))))
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=proc,
                         peft_config=lora_config(cfg),
                         callbacks=[time_checkpoint_callback(sc["checkpoint_minutes"], on_save)])
    trainer.train(resume_from_checkpoint=last_checkpoint(sc["output_dir"]))
    final = str(Path(sc["output_dir"]) / "final")
    trainer.save_model(final)
    if on_save is not None:
        on_save()
    return final


def zero_shot_eval(cfg: dict, executor: Any = None, client: Any = None, records: list[dict] | None = None) -> dict:
    """SFT gate: zero-shot success on R0 train prompts; skip SFT when success >= cfg zero_shot.gate_success.

    executor defaults to the vLLM executor with vocabulary docs and three examples; client defaults to
    rl.RewardClient on the env worker urls. Writes the summary to zero_shot.out and per-task rows beside it.
    """
    from spec2mpc.executor.rl import RewardClient

    zc = cfg["zero_shot"]
    if executor is None:
        from spec2mpc.executor.decode import make_executor

        executor = make_executor({**cfg, "prompt": {**cfg.get("prompt", {}), **zc["prompt"]}})
    client = client or RewardClient(zc["env_urls"])
    records = records if records is not None else read_jsonl(zc["prompts"])[: zc.get("max_tasks")]
    rows = []
    for rec in records:
        t0 = time.time()
        try:
            prog = executor.act(rec["command"], from_jsonable(rec["robot"]), from_jsonable(rec["scene"]), None)
            r = client.score(rec["task_id"], rec.get("seed", 0), prog)
            row = {"valid": True, "success": bool(r["success"]), "score": float(r["score"]),
                   "status": r.get("status", "")}
        except ExecutorError as e:
            row = {"valid": False, "success": False, "score": 0.0, "status": f"executor_error:{e.code}"}
        rows.append({"task_id": rec["task_id"], "wall_s": time.time() - t0, **row})
    n = max(len(rows), 1)
    rate = sum(r["success"] for r in rows) / n
    summary = {"n": len(rows), "success_rate": rate, "validity_rate": sum(r["valid"] for r in rows) / n,
               "mean_score": sum(r["score"] for r in rows) / n, "gate": zc["gate_success"],
               "decision": "skip_sft" if rate >= zc["gate_success"] else "run_sft"}
    if zc.get("out"):
        Path(zc["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(zc["out"]).write_text(json.dumps(summary, indent=1))
        write_jsonl(rows, Path(zc["out"]).with_suffix(".rows.jsonl"))
    return summary
