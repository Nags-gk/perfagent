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


class QuotaExhausted(RuntimeError):
    """The provider's daily (not per-minute) quota is spent: retrying today is pointless."""


# A daily quota, or a model the key gets no free quota for at all ("limit: 0"): retrying is pointless.
_DAILY = re.compile(r"per ?day|PerDay|daily|limit: ?0\b", re.IGNORECASE)


@dataclass
class Reply:
    content: str | None
    tool_calls: list[dict]  # {"id", "name", "args"}
    usage: dict = field(default_factory=dict)
    raw_message: dict = field(default_factory=dict)


_BAD = ("image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview", "vision", "guard", "whisper")


def _rank_models(ids: list[str], provider: str) -> list[str]:
    """Candidate chat models, best first: preferred patterns (newest first), then '-lite' variants."""
    ids = [i.split("/")[-1] for i in ids]
    clean = sorted((i for i in ids if not any(b in i for b in _BAD)), reverse=True)
    out: list[str] = []
    for lite in (False, True):
        for pat in PREFERRED[provider]:
            for i in clean:
                if ("lite" in i) == lite and re.search(pat, i) and i not in out:
                    out.append(i)
    if not out:
        raise RuntimeError(f"no suitable {provider} model in {ids[:20]}")
    return out


def _pick_model(ids: list[str], provider: str) -> str:
    return _rank_models(ids, provider)[0]


class OpenAICompatLLM:
    def __init__(self, provider: str, model: str | None = None, temperature: float = 0.0, max_retries: int = 8):
        from openai import OpenAI

        base, env = PROVIDERS[provider]
        key = os.environ.get(env, "") if env else "ollama"
        if env and not key:
            raise RuntimeError(f"set {env} in the environment to use {provider}")
        self.client = OpenAI(api_key=key, base_url=base, max_retries=0, timeout=120)
        self.provider = provider
        self.skipped: dict[str, str] = {}
        self.model = model or self._first_working(_rank_models([m.id for m in self.client.models.list().data], provider))
        self.temperature = temperature
        self.max_retries = max_retries
        self.requests = 0
        self.retries = 0

    def _first_working(self, candidates: list[str], limit: int = 6) -> str:
        """Send each candidate a tiny request; use the first one this key can actually call.

        The newest model is often overloaded (503) or has no free-tier quota (429 "limit: 0"),
        and the model list alone does not say so.
        """
        from openai import APIConnectionError, APIStatusError

        for m in candidates[:limit]:
            for attempt in range(2):
                try:
                    self.client.chat.completions.create(model=m, max_tokens=5, temperature=0,
                                                        messages=[{"role": "user", "content": "Reply with OK."}])
                    if self.skipped:
                        print(f"  model probe: skipped {self.skipped}", flush=True)
                    print(f"  model probe: using {m}", flush=True)
                    return m
                except (APIStatusError, APIConnectionError) as e:
                    msg = str(e)
                    code = getattr(e, "status_code", 0)
                    if code == 429 and not _DAILY.search(msg) and attempt == 0:
                        time.sleep(15)  # per-minute limit: try this model once more
                        continue
                    self.skipped[m] = f"{code or 'conn'}: {msg[:160]}"
                    break
        raise RuntimeError(f"no {self.provider} model is callable with this key; tried {self.skipped}")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                self.requests += 1
                r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                        temperature=self.temperature)
                break
            except (RateLimitError, APIConnectionError) as e:
                err = e
            except APIStatusError as e:
                if e.status_code not in (429, 500, 502, 503, 504):
                    body = e.body if e.body is not None else getattr(e.response, "text", "")
                    raise RuntimeError(f"{self.name} HTTP {e.status_code}: {str(body)[:600]}") from e
                err = e
            if _DAILY.search(str(err)):
                raise QuotaExhausted(f"{self.name}: daily quota exhausted ({str(err)[:300]})") from err
            self.retries += 1
            m = re.search(r"retry in ([\d.]+)s", str(err), re.IGNORECASE)
            time.sleep(float(m.group(1)) + 1 if m else delay)
            delay = min(delay * 2, 60)
        else:
            raise RuntimeError(f"{self.name} kept failing: {str(err)[:400]}")
        msg = r.choices[0].message
        # Echo the assistant message back exactly as received: Gemini 3 attaches a thought signature
        # to each tool call (extra_content.google.thought_signature) and rejects a follow-up request
        # with HTTP 400 if it is missing.
        raw = msg.model_dump(exclude_none=True)
        raw["role"] = "assistant"
        raw.setdefault("content", "")
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            cid = tc.id or f"call_{uuid.uuid4().hex[:8]}"
            raw["tool_calls"][i]["id"] = cid
            calls.append({"id": cid, "name": tc.function.name, "args": args})
        usage = {"in": getattr(r.usage, "prompt_tokens", 0) or 0, "out": getattr(r.usage, "completion_tokens", 0) or 0}
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
