"""Executor decoding: a reasoning trace, then a program constrained by json_schema(robot, scene), then validate.

Backends: VLLMGenerator (production, vLLM guided JSON on Modal) and HFGenerator (transformers; a schema-walking
sampler that scores only the choices the schema allows, used locally and in tests). vllm is imported lazily.
"""
from __future__ import annotations

import base64
import io
import json
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from spec2mpc.core import ExecutorError, Program, RobotSpec, Scene
from spec2mpc.executor.prompt import build_prompt, prompt_text

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@dataclass(frozen=True)
class Generation:
    reasoning: str
    text: str  # program JSON text, possibly truncated at the token limit


class Generator(Protocol):
    def generate(self, messages: list[dict], schema: dict, n: int) -> list[Generation]: ...


def split_output(raw: str) -> Generation:
    # "<think>r</think> {json}" -> (r, json); no think block means no reasoning
    head, sep, tail = raw.partition(THINK_CLOSE)
    if not sep:
        return Generation("", raw.strip())
    return Generation(head.replace(THINK_OPEN, "").strip(), tail.strip())


def parse_program(text: str) -> Program | None:
    # None for truncated or malformed JSON or a wrong shape
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return Program.from_json(json.loads(t))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _program_api() -> tuple[Callable[[RobotSpec, Scene], dict], Callable[[Program, RobotSpec, Scene], list[str]]]:
    from spec2mpc.program import json_schema, validate

    return json_schema, validate


class VLMExecutor:
    """Executor: generate (reasoning, program) under the schema, validate, retry once, else ExecutorError."""

    def __init__(self, generator: Generator, name: str = "spec2mpc", n_samples: int = 1, retries: int = 1,
                 prompt_kwargs: dict | None = None):
        self.generator, self.name, self.n_samples, self.retries = generator, name, n_samples, retries
        self.prompt_kwargs = prompt_kwargs or {}
        self.last_reasoning: str = ""
        self._lock = threading.Lock()  # one generation at a time on the shared model

    def act(self, command: str, robot: RobotSpec, scene: Scene, feedback: str | None) -> Program:
        json_schema, validate = _program_api()
        messages = build_prompt(command, robot, scene, feedback, **self.prompt_kwargs)
        schema = json_schema(robot, scene)
        errors: list[str] = []
        for _ in range(1 + self.retries):  # Review Focus 3: retry once, then raise
            with self._lock:
                gens = self.generator.generate(messages, schema, self.n_samples)
            valid: list[tuple[Program, Generation]] = []
            for g in gens:
                p = parse_program(g.text)
                errs = ["unparseable"] if p is None else validate(p, robot, scene)
                if errs:
                    errors += errs
                else:
                    valid.append((p, g))
            if valid:
                # self-consistency: the most frequent program text among valid samples
                best, _ = Counter(p.to_text() for p, _ in valid).most_common(1)[0]
                p, g = next((p, g) for p, g in valid if p.to_text() == best)
                self.last_reasoning = g.reasoning
                return p
        raise ExecutorError("invalid_program", ",".join(sorted(set(errors))))


# ---- schema-walking constrained sampler (backend-agnostic) ----

Scorer = Callable[[str, list[str]], np.ndarray]  # (context, candidates) -> [n_cand] scores (mean logprob)


class SchemaSampler:
    """Emits JSON valid under a schema subset by choosing among allowed continuations at each choice point.

    Supported: type (object, array, string, number, integer, boolean, null), enum, const, properties,
    required, prefixItems, items (schema or false), minItems/maxItems, minimum/maximum (and exclusive),
    anyOf/oneOf, $ref to #/$defs or #/definitions. Every declared property is emitted.
    """

    def __init__(self, score: Scorer, rng: np.random.Generator, temperature: float = 1.0, n_numbers: int = 11,
                 max_extra_items: int = 4):
        self.score, self.rng, self.temperature = score, rng, temperature
        self.n_numbers, self.max_extra_items = n_numbers, max_extra_items

    def sample(self, schema: dict, prefix: str) -> str:
        self.root, self.prefix, self.out = schema, prefix, []
        self._value(schema)
        return "".join(self.out)

    # helpers
    def _w(self, s: str) -> None:
        self.out.append(s)

    def _pick(self, cands: list[str]) -> int:
        if len(cands) == 1:
            return 0
        z = np.asarray(self.score(self.prefix + "".join(self.out), cands), dtype=np.float64)
        z = z / max(self.temperature, 1e-6)
        p = np.exp(z - z.max())
        return int(self.rng.choice(len(cands), p=p / p.sum()))

    def _resolve(self, s: Any) -> dict:
        if s is True or s == {}:
            return {}
        while "$ref" in s:
            node: Any = self.root
            for k in s["$ref"].removeprefix("#/").split("/"):
                node = node[k]
            s = node
        return s

    def _lead(self, s: Any) -> str:
        # the fixed text a branch starts with (scores anyOf branches)
        s = self._resolve(s)
        if "const" in s:
            return json.dumps(s["const"])
        if "enum" in s:
            return json.dumps(s["enum"][0]) if len(s["enum"]) == 1 else ""
        if s.get("type") == "object" or "properties" in s:
            props = s.get("properties", {})
            if not props:
                return "{"
            k = next(iter(props))
            return "{" + json.dumps(k) + ": " + self._lead(props[k])
        if s.get("type") == "array":
            return "["
        return ""

    def _value(self, s: Any) -> None:
        s = self._resolve(s)
        if "const" in s:
            return self._w(json.dumps(s["const"]))
        if "enum" in s:
            c = [json.dumps(v) for v in s["enum"]]
            return self._w(c[self._pick(c)])
        branches = s.get("anyOf") or s.get("oneOf")
        if branches:
            leads = [self._lead(b) for b in branches]
            i = self._pick(leads) if len(set(leads)) == len(leads) else int(self.rng.integers(len(branches)))
            return self._value(branches[i])
        t = s.get("type", "object" if "properties" in s else None)
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), "null")
        if t == "object":
            return self._object(s)
        if t == "array":
            return self._array(s)
        if t in ("number", "integer"):
            return self._number(s, t == "integer")
        if t == "boolean":
            c = ["true", "false"]
            return self._w(c[self._pick(c)])
        if t == "string":
            return self._w('"' + "x" * max(s.get("minLength", 0), 1) + '"')  # free strings: none in json_schema
        return self._w("null")

    def _object(self, s: dict) -> None:
        self._w("{")
        for i, (k, sub) in enumerate(s.get("properties", {}).items()):
            self._w((", " if i else "") + json.dumps(k) + ": ")
            self._value(sub)
        self._w("}")

    def _array(self, s: dict) -> None:
        pre = s.get("prefixItems", [])
        items = s.get("items", {})
        lo = s.get("minItems", 0)
        hi = s.get("maxItems", len(pre) if items is False else max(lo, len(pre)) + self.max_extra_items)
        if items is False:
            hi = min(hi, len(pre))
        self._w("[")
        i = 0
        while i < hi:
            if i >= lo:
                sub = pre[i] if i < len(pre) else items
                more = ", " if i else self._lead(sub) or " "
                if self._pick(["]", more]) == 0:
                    break
            self._w(", " if i else "")
            self._value(pre[i] if i < len(pre) else items)
            i += 1
        self._w("]")

    def _number(self, s: dict, integer: bool) -> None:
        lo, hi = float(s.get("minimum", -1.0)), float(s.get("maximum", 1.0))
        if "exclusiveMinimum" in s:
            lo = float(s["exclusiveMinimum"])
        if "exclusiveMaximum" in s:
            hi = float(s["exclusiveMaximum"])
        lo_ok = (lambda v: v > lo) if "exclusiveMinimum" in s else (lambda v: v >= lo)
        hi_ok = (lambda v: v < hi) if "exclusiveMaximum" in s else (lambda v: v <= hi)
        if integer:
            vals = np.unique(np.round(np.linspace(np.ceil(lo), np.floor(hi), self.n_numbers))).astype(int)
            c = [str(int(v)) for v in vals if lo_ok(v) and hi_ok(v)]
        else:
            c = [repr(round(float(v), 6)) for v in np.linspace(lo, hi, self.n_numbers)]
            c = [x for x in dict.fromkeys(c) if lo_ok(float(x)) and hi_ok(float(x))]
        if not c:
            raise ValueError(f"empty number range {s}")
        self._w(c[self._pick(c)])


def _crop_to(cache: Any, n: int) -> None:
    # keep the first n cached positions (transformers 5 crop takes a negative count to remove)
    extra = cache.get_seq_length() - n
    if extra > 0:
        cache.crop(-extra)


class HFScorer:
    """Mean log-probability of each candidate continuation under a transformers causal LM.

    The context only grows while one JSON is emitted, so its KV cache is kept and extended; checkpoints let a new
    sample that shares the prompt crop back to it instead of re-encoding.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model, self.tok = model, tokenizer
        self.cache: Any = None
        self.marks: list[tuple[str, int, Any]] = []  # (context so far, cache length, last-position logprobs [V])

    def _advance(self, context: str) -> Any:
        import torch
        from transformers import DynamicCache

        while self.marks and not context.startswith(self.marks[-1][0]):
            self.marks.pop()
        if not self.marks:
            self.cache, done = DynamicCache(), ""
        else:
            done, n, _ = self.marks[-1]
            _crop_to(self.cache, n)
        new = context[len(done):]
        ids = self.tok(new, add_special_tokens=False)["input_ids"] if new else []
        if ids:
            t0 = self.cache.get_seq_length()
            x = torch.tensor([ids], device=self.model.device)  # [1, n]
            pos = torch.arange(t0, t0 + len(ids), device=self.model.device)[None]  # [1, n]
            out = self.model(input_ids=x, past_key_values=self.cache, position_ids=pos, use_cache=True,
                             logits_to_keep=1)
            self.marks.append((context, self.cache.get_seq_length(), torch.log_softmax(out.logits[0, -1].float(), -1)))
        return self.marks[-1][2]

    def __call__(self, context: str, cands: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            last = self._advance(context)  # [V]
            cs = [self.tok(c, add_special_tokens=False)["input_ids"] or [self.tok.eos_token_id] for c in cands]
            B, L, T = len(cs), max(len(c) for c in cs), self.cache.get_seq_length()
            dev = self.model.device
            ids = torch.tensor([c + [c[-1]] * (L - len(c)) for c in cs], device=dev)  # [B, L] right-padded
            am = torch.cat([torch.ones(B, T, dtype=torch.long, device=dev),
                            torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in cs], device=dev)], 1)  # [B, T+L]
            pos = (T + torch.arange(L, device=dev))[None].expand(B, L)  # [B, L]
            self.cache.batch_repeat_interleave(B)
            try:
                logits = self.model(input_ids=ids, attention_mask=am, past_key_values=self.cache, position_ids=pos,
                                    use_cache=True).logits  # [B, L, V]
            finally:
                _crop_to(self.cache, T)
                self.cache.batch_select_indices(torch.tensor([0], device=dev))
            logp = torch.log_softmax(logits.float(), -1)  # [B, L, V], position j predicts token j+1
            out = np.zeros(B)
            for b, c in enumerate(cs):
                lp = [float(last[c[0]])] + [float(logp[b, j - 1, c[j]]) for j in range(1, len(c))]
                out[b] = sum(lp) / len(lp)
            return out


def _flatten(messages: list[dict]) -> list[dict]:
    # text-only chat (images dropped) for tokenizers without a vision processor
    return [{"role": m["role"], "content": prompt_text([m])} for m in messages]


def _render(tokenizer: Any, messages: list[dict]) -> str:
    # chat prompt ending inside an open think block
    s = tokenizer.apply_chat_template(_flatten(messages), tokenize=False, add_generation_prompt=True)
    return s if s.rstrip().endswith(THINK_OPEN) else s + THINK_OPEN + "\n"


class HFGenerator:
    """transformers backend: free reasoning up to max_reasoning_tokens, then SchemaSampler for the program."""

    def __init__(self, model: Any, tokenizer: Any, max_reasoning_tokens: int = 256, temperature: float = 0.7,
                 seed: int = 0, n_numbers: int = 11, max_extra_items: int = 4):
        self.model, self.tok = model, tokenizer
        self.max_reasoning_tokens, self.temperature = max_reasoning_tokens, temperature
        self.rng = np.random.default_rng(seed)
        self.sampler = SchemaSampler(HFScorer(model, tokenizer), self.rng, temperature, n_numbers, max_extra_items)

    def _reason(self, prompt: str) -> str:
        if self.max_reasoning_tokens <= 0:
            return ""
        import torch

        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_reasoning_tokens, do_sample=True,
                                      temperature=self.temperature, stop_strings=[THINK_CLOSE], tokenizer=self.tok,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split(THINK_CLOSE)[0]

    def generate(self, messages: list[dict], schema: dict, n: int) -> list[Generation]:
        prompt = _render(self.tok, messages)
        out = []
        for _ in range(n):
            r = self._reason(prompt)
            out.append(Generation(r.strip(), self.sampler.sample(schema, f"{prompt}{r}\n{THINK_CLOSE}\n\n")))
        return out


def _png_data_uri(img: Any) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VLLMGenerator:
    """vLLM backend: phase 1 reasoning stopped at </think>, phase 2 the program under guided JSON (xgrammar)."""

    def __init__(self, model_id: str, revision: str, max_reasoning_tokens: int = 1024, max_program_tokens: int = 1024,
                 temperature: float = 0.7, lora_path: str | None = None, seed: int = 0, **llm_kwargs: Any):
        from vllm import LLM

        self.llm = LLM(model=model_id, revision=revision, seed=seed, enable_lora=lora_path is not None,
                       **llm_kwargs)
        self.tok = self.llm.get_tokenizer()
        self.max_reasoning_tokens, self.max_program_tokens = max_reasoning_tokens, max_program_tokens
        self.temperature = temperature
        self.lora = None
        if lora_path is not None:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("executor", 1, lora_path)

    def _json_params(self, schema: dict, n: int) -> Any:
        from vllm import SamplingParams

        kw: dict[str, Any] = {"n": n, "temperature": self.temperature, "max_tokens": self.max_program_tokens}
        try:  # vLLM >= 0.11
            from vllm.sampling_params import StructuredOutputsParams

            kw["structured_outputs"] = StructuredOutputsParams(json=schema)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            kw["guided_decoding"] = GuidedDecodingParams(json=schema)
        return SamplingParams(**kw)

    def _prompt(self, messages: list[dict]) -> dict:
        images = [p["image"] for m in messages for p in m["content"] if p["type"] == "image"]
        chat = [{"role": m["role"], "content": [
            {"type": "image_url", "image_url": {"url": _png_data_uri(p["image"])}} if p["type"] == "image"
            else {"type": "text", "text": p["text"]} for p in m["content"]]} for m in messages]
        text = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        text = text if text.rstrip().endswith(THINK_OPEN) else text + THINK_OPEN + "\n"
        out: dict[str, Any] = {"prompt": text}
        if images:
            out["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
        return out

    def generate(self, messages: list[dict], schema: dict, n: int) -> list[Generation]:
        from vllm import SamplingParams

        p = self._prompt(messages)
        think = self.llm.generate([p], SamplingParams(n=n, temperature=self.temperature, stop=[THINK_CLOSE],
                                                      max_tokens=self.max_reasoning_tokens),
                                  lora_request=self.lora, use_tqdm=False)[0].outputs
        reasons = [o.text.split(THINK_CLOSE)[0] for o in think]
        cont = [dict(p, prompt=f"{p['prompt']}{r}\n{THINK_CLOSE}\n\n") for r in reasons]
        progs = self.llm.generate(cont, self._json_params(schema, 1), lora_request=self.lora, use_tqdm=False)
        return [Generation(r.strip(), o.outputs[0].text) for r, o in zip(reasons, progs)]


def make_executor(cfg: dict, lora_path: str | None = None) -> VLMExecutor:
    # cfg: configs/v2/executor.yaml as a dict
    b, d = cfg["backbone"], cfg["decode"]
    if d.get("backend", "vllm") == "vllm":
        gen: Generator = VLLMGenerator(b["model_id"], b["revision"], d["max_reasoning_tokens"],
                                       d["max_program_tokens"], d["temperature"], lora_path, **cfg.get("vllm", {}))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(b["model_id"], revision=b["revision"])
        model = AutoModelForCausalLM.from_pretrained(b["model_id"], revision=b["revision"])
        gen = HFGenerator(model, tok, d["max_reasoning_tokens"], d["temperature"])
    return VLMExecutor(gen, cfg.get("name", "spec2mpc"), d.get("n_samples", 1), d.get("retries", 1),
                       cfg.get("prompt", {}))
