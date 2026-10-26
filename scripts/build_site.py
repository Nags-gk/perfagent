"""Copy the pure-Python packages the browser needs (InferSim and PerfAgent's tools/tasks)
into site/py and write a manifest the Pyodide worker loads."""

import json
import shutil
from pathlib import Path

import infersim

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "site" / "py"


def main() -> None:
    shutil.rmtree(PY, ignore_errors=True)
    shutil.copytree(Path(infersim.__file__).parent, PY / "infersim", ignore=shutil.ignore_patterns("__pycache__"))
    (PY / "perfagent").mkdir(parents=True)
    for f in ("__init__.py", "tools.py", "tasks.py"):
        shutil.copy(ROOT / "perfagent" / f, PY / "perfagent" / f)
    from perfagent.agent import SYSTEM, submit_spec
    from perfagent.tasks import build_suite
    from perfagent.tools import TOOL_SPECS

    spec = {"system": SYSTEM, "tools": TOOL_SPECS, "submit": {t.id: submit_spec(t) for t in build_suite()}}
    (ROOT / "site" / "data" / "agent_spec.json").write_text(json.dumps(spec))
    files = sorted(str(p.relative_to(PY)) for p in PY.rglob("*") if p.is_file())
    (PY / "manifest.json").write_text(json.dumps(files))
    print(f"site/py: {len(files)} files")


if __name__ == "__main__":
    main()
