"""Modal app for the executor: smoke, zero-shot SFT gate, SFT, GRPO. Every run passes the $100 budget guard.

Run from the repo root, e.g.
  ~/.local/bin/modal run spec2mpc/executor/modal_app.py::smoke
  ~/.local/bin/uv run --with modal modal run spec2mpc/executor/modal_app.py::sft --data data/v2/sft.jsonl
(the config-driven entrypoints need pyyaml locally, hence uv run --with modal). Top-level imports are stdlib
plus an optional modal, so the budget guard imports and tests without modal installed.

Budget ledger data/v2/budget.jsonl, one JSON object per line {ts, run, kind, usd, note}; a run's cost is the usd
of its last line (kind "reserve" at start with the estimate, "final" at the end with the measured cost). A run
refuses to start when the logged total plus its estimate exceeds the cap. Each function's timeout equals the
hours its estimate assumed, so measured cost never exceeds the estimate.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # local tests: guard only
    modal = None

ROOT = Path(__file__).resolve().parent.parent.parent
BUDGET_PATH = ROOT / "data" / "v2" / "budget.jsonl"
CAP_USD = 100.0
APP_NAME, VOLUME, VOL_DIR = "languagespec-executor", "languagespec-executor", "/vol"
SMOKE_EST_USD = 0.25  # CPU container for a few minutes plus the image build

# Modal image only; never installed locally. trl 1.17.1 supports vllm < 0.33; vllm 0.32.0 pins torch 2.13
PIP = ["vllm==0.32.0", "trl==1.17.1", "peft==0.21.2", "accelerate==1.15.0", "datasets==5.1.0",
       "jsonschema", "flask", "pyyaml", "pillow", "numpy"]


class BudgetExceeded(RuntimeError):
    pass


def ledger(path: str | Path = BUDGET_PATH) -> dict[str, float]:
    # run id -> cost (usd of its last line)
    p = Path(path)
    runs: dict[str, float] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                runs[d["run"]] = float(d["usd"])
    return runs


def budget_total(path: str | Path = BUDGET_PATH) -> float:
    return sum(ledger(path).values())


def log_cost(run: str, usd: float, kind: str, note: str = "", path: str | Path = BUDGET_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run, "kind": kind,
           "usd": round(float(usd), 4), "note": note}
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def reserve(name: str, estimate_usd: float, cap_usd: float = CAP_USD, path: str | Path = BUDGET_PATH,
            note: str = "") -> str:
    """Refuse (BudgetExceeded) if logged total + estimate > cap; else log the reservation and return a run id."""
    total = budget_total(path)
    if total + estimate_usd > cap_usd:
        raise BudgetExceeded(f"{name}: logged ${total:.2f} + estimate ${estimate_usd:.2f} > cap ${cap_usd:.2f}")
    run = f"{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    log_cost(run, estimate_usd, "reserve", note, path)
    return run


def estimate_usd(mc: dict, job: str) -> tuple[float, float]:
    # (usd, hours) for a job in cfg["modal"]: n_gpu x timeout x rate, plus CPU, times the margin
    j = mc["jobs"][job]
    h = float(j["timeout_h"])
    usd = (j["n_gpu"] * mc["usd_per_gpu_hour"][j["gpu"]] + mc["usd_per_cpu_hour"]) * h * mc["margin"]
    return usd, h


def measured_usd(mc: dict, job: str, elapsed_s: float) -> float:
    j = mc["jobs"][job]
    return (j["n_gpu"] * mc["usd_per_gpu_hour"][j["gpu"]] + mc["usd_per_cpu_hour"]) * elapsed_s / 3600 * mc["margin"]


def _load_cfg(path: str) -> dict:
    import yaml

    return yaml.safe_load((ROOT / path).read_text())


if modal is not None:
    app = modal.App(APP_NAME)
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install(*PIP)
             .env({"HF_HOME": f"{VOL_DIR}/hf", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_python_source("spec2mpc"))

    @app.function(image=image, timeout=900, volumes={VOL_DIR: vol})
    def smoke_remote(model_id: str, revision: str) -> dict:
        # imports the image and returns versions; downloads only the backbone config.json
        import importlib.metadata as md

        out: dict[str, Any] = {k: md.version(k) for k in ("torch", "transformers", "trl", "peft", "vllm",
                                                          "accelerate", "datasets")}
        import transformers

        c = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
        out["backbone"] = {"model_type": c.model_type, "architectures": c.architectures}
        import spec2mpc.executor.rl  # noqa: F401  (package imports in the image)

        out["spec2mpc_executor_import"] = "ok"
        return out

    def _commit() -> None:
        vol.commit()

    @app.function(image=image, gpu="H100", timeout=6 * 3600, volumes={VOL_DIR: vol})
    def sft_remote(cfg: dict) -> str:
        from spec2mpc.executor import sft

        return sft.main(cfg, on_save=_commit)

    @app.function(image=image, gpu="H100:2", timeout=6 * 3600, volumes={VOL_DIR: vol})
    def rl_remote(cfg: dict) -> str:
        from spec2mpc.executor import rl

        return rl.main(cfg, on_save=_commit)

    @app.function(image=image, gpu="H100", timeout=2 * 3600, volumes={VOL_DIR: vol})
    def zero_shot_remote(cfg: dict) -> dict:
        from spec2mpc.executor.sft import zero_shot_eval

        out = zero_shot_eval(cfg)
        vol.commit()
        return out

    def _guarded(job: str, fn: Any, cfg: dict) -> Any:
        mc = cfg["modal"]
        est, hours = estimate_usd(mc, job)
        j = mc["jobs"][job]
        run = reserve(job, est, mc["cap_usd"], ROOT / mc["budget_path"], f"{j['n_gpu']}x{j['gpu']} <= {hours} h")
        t0 = time.time()
        try:
            gpu = j["gpu"] if j["n_gpu"] == 1 else f"{j['gpu']}:{j['n_gpu']}"
            return fn.with_options(gpu=gpu, timeout=int(hours * 3600)).remote(cfg)
        finally:
            usd = measured_usd(mc, job, time.time() - t0)
            log_cost(run, usd, "final", f"measured {time.time() - t0:.0f} s", ROOT / mc["budget_path"])

    def _upload(local: str | None, remote: str) -> None:
        if local:
            with vol.batch_upload(force=True) as b:
                b.put_file(str(ROOT / local), remote.removeprefix(VOL_DIR))

    @app.local_entrypoint()
    def smoke(model_id: str = "Qwen/Qwen3.8-27B", revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0") -> None:
        run = reserve("smoke", SMOKE_EST_USD, note="cpu image import and versions")
        t0 = time.time()
        try:
            print(json.dumps(smoke_remote.remote(model_id, revision), indent=1))
        finally:
            # CPU container: about $0.20 per hour including memory
            log_cost(run, max(0.01, 0.20 * (time.time() - t0) / 3600 * 1.15), "final",
                     f"measured {time.time() - t0:.0f} s wall incl. image build")

    @app.local_entrypoint()
    def zero_shot(config: str = "configs/v2/executor.yaml", prompts: str = "") -> None:
        cfg = _load_cfg(config)
        _upload(prompts or None, cfg["zero_shot"]["prompts"])
        print(json.dumps(_guarded("zero_shot", zero_shot_remote, cfg), indent=1))

    @app.local_entrypoint()
    def sft(config: str = "configs/v2/executor.yaml", data: str = "") -> None:
        cfg = _load_cfg(config)
        _upload(data or None, cfg["sft"]["data"])
        print(_guarded("sft", sft_remote, cfg))

    @app.local_entrypoint()
    def rl(config: str = "configs/v2/executor.yaml", prompts: str = "") -> None:
        cfg = _load_cfg(config)
        _upload(prompts or None, cfg["rl"]["prompts"])
        print(_guarded("rl", rl_remote, cfg))
