"""OpenTelemetry spans plus a readable event log for each agent run.

Spans go to an in-memory exporter and are saved with the trace (name, parent, start/end,
attributes), so a run can be inspected offline or forwarded to any OTLP backend. The event
log is what the dashboard's trace viewer replays.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _attr(v):
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
        return list(v)
    return json.dumps(v, default=str)[:2000]


class _Span:
    def __init__(self, span):
        self._s = span

    def set(self, **kw):
        for k, v in kw.items():
            self._s.set_attribute(k, _attr(v))


class Tracer:
    def __init__(self, task_id: str, agent: str):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer("perfagent")
        self.task_id, self.agent = task_id, agent
        self.events: list[dict] = []
        self.t0 = time.time()

    @contextmanager
    def span(self, name: str, **attrs):
        with self.tracer.start_as_current_span(name) as s:
            w = _Span(s)
            w.set(**attrs)
            yield w

    def event(self, kind: str, **data):
        self.events.append({"t": round(time.time() - self.t0, 3), "type": kind, **data})

    def export(self) -> dict:
        spans = []
        for s in self.exporter.get_finished_spans():
            spans.append({"name": s.name, "span_id": format(s.context.span_id, "016x"),
                          "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                          "start_ns": s.start_time, "end_ns": s.end_time, "attributes": dict(s.attributes or {})})
        spans.sort(key=lambda x: x["start_ns"])
        return {"task": self.task_id, "agent": self.agent, "events": self.events, "otel_spans": spans}
