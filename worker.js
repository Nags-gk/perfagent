// Runs InferSim and PerfAgent's tools in Pyodide, off the main thread.
let py = null;
let api = null;

const HELPERS = `
import json
from perfagent.tools import ToolBox, ToolError
from perfagent.tasks import build_suite

TASKS = {t.id: t for t in build_suite()}
STATE = {"box": ToolBox(budget=30)}

def reset(budget):
    STATE["box"] = ToolBox(budget=int(budget))
    return json.dumps({"ok": True})

def call(name, args_json):
    box = STATE["box"]
    try:
        out = box.call(name, json.loads(args_json))
    except ToolError as e:
        out = {"error": str(e)}
    except TypeError as e:
        out = {"error": f"bad arguments: {e}"}
    return json.dumps(dict(out, _used=box.used, _budget=box.budget))

def grade(task_id, answer_json):
    return json.dumps(TASKS[task_id].grade(json.loads(answer_json)))

def verify(task_id, answer_json):
    t, a = TASKS[task_id], json.loads(answer_json)
    if t.kind != "dse":
        return json.dumps({"problem": None})
    p = t.params
    if a.get("hw") not in p["hws"] or a.get("precision") not in p["precs"]:
        return json.dumps({"problem": "the submitted hardware or precision is not allowed"})
    try:
        r = STATE["box"].call("evaluate_layout", {"model": p["model"], "hw": a["hw"], "tp": a.get("tp", 1),
            "pp": a.get("pp", 1), "precision": a["precision"], "batch": a["batch"],
            "expert_parallel": a.get("expert_parallel", False), "prompt": 2048, "output": 256})
    except (ToolError, KeyError, TypeError) as e:
        return json.dumps({"problem": None if "budget" in str(e) else str(e)})
    if not r.get("fits"):
        return json.dumps({"problem": "the submitted layout does not fit"})
    if r["tok_s_per_user"] < p["sla"]:
        return json.dumps({"problem": f"the submitted layout gives {r['tok_s_per_user']} tok/s per user, below the SLA"})
    return json.dumps({"problem": None})
`;

async function boot(indexURL) {
  const { loadPyodide } = await import(indexURL + "pyodide.mjs");
  py = await loadPyodide({ indexURL });
  const files = await (await fetch("py/manifest.json")).json();
  for (const f of files) {
    const text = await (await fetch("py/" + f)).text();
    const path = "/home/pyodide/" + f;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, text);
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  py.runPython(HELPERS);
  api = { reset: py.globals.get("reset"), call: py.globals.get("call"), grade: py.globals.get("grade"), verify: py.globals.get("verify") };
  return { python: py.runPython("import sys; sys.version.split()[0]") };
}

self.onmessage = async (e) => {
  const { id, type, fn, args, indexURL } = e.data;
  try {
    if (type === "boot") {
      self.postMessage({ id, ok: true, result: await boot(indexURL) });
      return;
    }
    const out = api[fn](...args);
    self.postMessage({ id, ok: true, result: JSON.parse(out) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
