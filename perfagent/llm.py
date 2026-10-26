"""LLM backends behind one interface.

Gemini, Groq and Ollama all speak the OpenAI chat-completions protocol with tool calling, so
one client covers them. ``ScriptedPolicy`` implements the same interface without a model: a
hand-written expert strategy used as a baseline and to test the agent loop offline.

API keys come from the environment (``GEMINI_API_KEY``, ``GROQ_API_KEY``), never from code.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
PREFERRED = {
    "gemini": (r"gemini-[\d.]+-flash$", r"gemini-[\d.]+-flash", r"flash"),
    "groq": (r"llama-3\.3-70b", r"70b", r"llama"),
    "ollama": (r"qwen2\.5", r"llama", r""),
}


@dataclass
class Reply:
    content: str | None
    tool_calls: list[dict]  # {"id", "name", "args"}
    usage: dict = field(default_factory=dict)
    raw_message: dict = field(default_factory=dict)


def _pick_model(ids: list[str], provider: str) -> str:
    ids = [i.split("/")[-1] for i in ids]
    bad = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview", "vision", "guard")
    clean = sorted((i for i in ids if not any(b in i for b in bad)), reverse=True)
    for pat in PREFERRED[provider]:
        for i in clean:
            if re.search(pat, i):
                return i
    raise RuntimeError(f"no suitable {provider} model in {ids[:20]}")


class OpenAICompatLLM:
    def __init__(self, provider: str, model: str | None = None, temperature: float = 0.0, max_retries: int = 8):
        from openai import OpenAI

        base, env = PROVIDERS[provider]
        key = os.environ.get(env, "") if env else "ollama"
        if env and not key:
            raise RuntimeError(f"set {env} in the environment to use {provider}")
        self.client = OpenAI(api_key=key, base_url=base, max_retries=0, timeout=120)
        self.provider = provider
        self.model = model or _pick_model([m.id for m in self.client.models.list().data], provider)
        self.temperature = temperature
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                        temperature=self.temperature)
                break
            except (RateLimitError, APIConnectionError) as e:
                err = e
            except APIStatusError as e:
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise
                err = e
            m = re.search(r"retry in ([\d.]+)s", str(err), re.IGNORECASE)
            time.sleep(float(m.group(1)) + 1 if m else delay)
            delay = min(delay * 2, 60)
        else:
            raise RuntimeError(f"{self.name} kept failing: {err}")
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            calls.append({"id": tc.id or f"call_{uuid.uuid4().hex[:8]}", "name": tc.function.name, "args": args})
        usage = {"in": getattr(r.usage, "prompt_tokens", 0) or 0, "out": getattr(r.usage, "completion_tokens", 0) or 0}
        raw = {"role": "assistant", "content": msg.content or ""}
        if calls:
            raw["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in calls]
        return Reply(msg.content, calls, usage, raw)


class ScriptedPolicy:
    """A hand-written expert that drives the same tools through the same message protocol.

    It reads the task's parameters directly (an LLM must parse the prompt), so it is an upper
    reference for strategy, not a fair LLM competitor. Used as a baseline and for offline tests.
    """

    name = "scripted-expert"

    def __init__(self, task):
        self.task = task
        self.plan = self._plan()
        self.pending: list[dict] = []

    # The policy is a generator of tool calls; each chat() advances it with the last tool results.
    def _plan(self):
        t, p = self.task, self.task.params
        if t.kind == "bottleneck":
            r = yield ("simulate_step", {k: p[k] for k in ("model", "hw", "tp", "precision", "phase", "batch", "tokens")})
            yield ("submit_answer", {"answer": {"op": r["top_ops"][0]["name"], "bound": r["dominant_bound"]},
                                     "rationale": "slowest op and dominant bound from one simulation"})
        elif t.kind == "sizing":
            need = p["users"] * p["ctx"]
            for tp, pp in sorted(((tp, pp) for tp in (1, 2, 4, 8) for pp in (1, 2, 4)), key=lambda x: (x[0] * x[1], x[1])):
                r = yield ("check_fit", {"model": p["model"], "hw": p["hw"], "tp": tp, "pp": pp, "precision": p["precision"]})
                if r.get("fits") and r["kv_capacity_tokens"] >= need:
                    yield ("submit_answer", {"answer": {"devices": tp * pp, "tp": tp, "pp": pp},
                                             "rationale": "smallest layout whose KV capacity covers the demand"})
                    return
        elif t.kind == "crossover":
            lo, hi = 1, 4096
            while lo < hi:
                mid = (lo + hi) // 2
                r = yield ("roofline_ops", {"model": p["model"], "hw": p["hw"], "precision": p["precision"],
                                            "phase": "decode", "batch": mid, "tokens": 1024})
                bound = next(o["bound"] for o in r["ops"] if o["name"] == "gate_up")
                lo, hi = (lo, mid) if bound == "compute" else (mid + 1, hi)
            yield ("submit_answer", {"answer": {"batch": lo}, "rationale": "binary search on gate_up's bound"})
        elif t.kind == "compare":
            res = {}
            for k in ("a", "b"):
                lay = p[k]
                r = yield ("simulate_step", {"model": p["model"], "hw": lay["hw"], "tp": lay["tp"],
                                             "precision": lay["precision"], "phase": "decode",
                                             "batch": p["batch"], "tokens": p["ctx"]})
                res[k] = r["tok_s_per_user"]
            f = "a" if res["a"] >= res["b"] else "b"
            yield ("submit_answer", {"answer": {"faster": f, "ratio": max(res.values()) / min(res.values())},
                                     "rationale": "two simulations"})
        elif t.kind == "dse":
            yield from self._dse()

    def _dse(self):
        p = self.task.params
        moe = p["model"] in ("mixtral-8x7b", "deepseek-v3")
        best, best_cfg, budget = 0.0, None, 30
        for hw in p["hws"]:
            for prec in p["precs"]:
                for tp in (1, 2, 4, 8):
                    r = yield ("check_fit", {"model": p["model"], "hw": hw, "tp": tp, "precision": prec})
                    if not r.get("fits"):
                        continue
                    # largest batch first; stop at the first one that meets the SLA
                    for batch in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
                        if budget <= 1:
                            break
                        cfg = {"model": p["model"], "hw": hw, "tp": tp, "pp": 1, "precision": prec, "batch": batch,
                               "prompt": 2048, "output": 256, "expert_parallel": moe and tp > 1}
                        r = yield ("evaluate_layout", cfg)
                        budget -= 1
                        if r.get("fits") and r["tok_s_per_user"] >= p["sla"]:
                            if r["tok_s_per_device"] > best:
                                best, best_cfg = r["tok_s_per_device"], cfg
                            break
                    break  # smallest fitting TP per hw/precision only
        ans = ({k: best_cfg[k] for k in ("hw", "precision", "tp", "pp", "expert_parallel", "batch")}
               if best_cfg else {"hw": p["hws"][0], "precision": p["precs"][0], "tp": 8, "pp": 1,
                                 "expert_parallel": False, "batch": 1})
        yield ("submit_answer", {"answer": ans, "rationale": "smallest fitting TP, largest batch meeting the SLA"})

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        last = messages[-1]
        result = None
        if last.get("role") == "tool":
            try:
                result = json.loads(last["content"])
            except (json.JSONDecodeError, TypeError):
                result = {}
        try:
            name, args = self.plan.send(result) if self.pending or last.get("role") == "tool" else next(self.plan)
        except StopIteration:
            return Reply("done", [], {"in": 0, "out": 0}, {"role": "assistant", "content": "done"})
        self.pending = [1]
        cid = f"call_{uuid.uuid4().hex[:8]}"
        raw = {"role": "assistant", "content": "",
               "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return Reply(None, [{"id": cid, "name": name, "args": args}], {"in": 0, "out": 0}, raw)
