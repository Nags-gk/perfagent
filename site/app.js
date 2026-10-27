// PerfAgent dashboard: results, traces, and a live in-browser agent (Gemini + Pyodide tools).

const PYODIDE = new URLSearchParams(location.search).get("pyodide") || "https://cdn.jsdelivr.net/npm/pyodide@314.0.7/";
const GEMINI = new URLSearchParams(location.search).get("gemini") || "https://generativelanguage.googleapis.com/v1beta";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (x) => (100 * x).toFixed(0) + "%";
const kpi = (v, k) => `<div class="kpi"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`;

let R = null, SPEC = null, worker = null, ready = false, seq = 0;
const pending = new Map();
const AGENT_KIND = (a) => (a.startsWith("random") || a.startsWith("optuna") ? "search" : a.startsWith("scripted") ? "rules" : "llm");
const LABEL = { "random-search": "Random search", "optuna-tpe": "Optuna TPE", "scripted-expert": "Rule-based strategy" };
const label = (a) => LABEL[a] || a.replace(/^gemini:/, "Gemini agent: ").replace(/^groq:/, "Groq agent: ");
const COLORS = () => ({ llm: css("--attention"), search: css("--comm"), rules: css("--other") });
const agentColor = (a, i) => (AGENT_KIND(a) === "llm" ? [css("--attention"), css("--afd"), css("--disaggregated")][i % 3] : COLORS()[AGENT_KIND(a)]);

// ---------- worker (Pyodide) ----------
function rpc(fn, ...args) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, fn, args });
  });
}
function boot() {
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    e.data.ok ? p.resolve(e.data.result) : p.reject(new Error(e.data.error));
  };
  const id = ++seq;
  pending.set(id, {
    resolve: (r) => { ready = true; $("#engine").textContent = `Simulator live: Python ${r.python} in your browser`; $("#engine").className = "pill pill-ok"; $("#live-run").disabled = false; },
    reject: (e) => { $("#engine").textContent = "Simulator failed to load"; $("#engine").className = "pill pill-err"; $("#live-status").textContent = e.message; },
  });
  worker.postMessage({ id, type: "boot", indexURL: PYODIDE });
}

// ---------- results ----------
function agents() {
  const order = { llm: 0, rules: 1, search: 2 };
  return Object.keys(R.summary).sort((a, b) => order[AGENT_KIND(a)] - order[AGENT_KIND(b)] || a.localeCompare(b));
}
function renderBoard() {
  const dse = R.tasks.filter((t) => t.kind === "dse").length, exact = R.tasks.length - dse;
  $("#results-hint").textContent = `${R.tasks.length} tasks (${dse} design-space, ${exact} exact-answer). Budget: ${R.budget} simulations per task. Updated ${R.generated}.`;
  const rows = agents().map((a) => {
    const s = R.summary[a], d = s.by_kind.dse;
    const ex = Object.entries(s.by_kind).filter(([k]) => k !== "dse");
    const exN = ex.reduce((n, [, v]) => n + v.n, 0), exOk = ex.reduce((n, [, v]) => n + v.success_rate * v.n, 0);
    const seeds = AGENT_KIND(a) === "search" ? " (5 seeds)" : "";
    return `<tr><td><strong>${esc(label(a))}</strong><br><span class="dim">${AGENT_KIND(a) === "llm" ? "LLM agent" : AGENT_KIND(a) === "search" ? "search baseline" + seeds : "hand-written, reads task parameters"}</span></td>
      <td class="num">${d ? pct(d.success_rate) : "-"}</td><td class="num">${d ? pct(d.mean_score) : "-"}</td><td class="num">${d ? d.mean_simulations.toFixed(1) : "-"}</td>
      <td class="num">${exN ? `${Math.round(exOk)}/${exN}` : "-"}</td><td class="num">${s.llm_calls || "-"}</td><td class="num">${s.tokens_in ? ((s.tokens_in + s.tokens_out) / 1e3).toFixed(0) + "k" : "-"}</td></tr>`;
  }).join("");
  $("#board").innerHTML = `<tr><th>agent</th><th class="num">design space: within 95% of optimum</th><th class="num">mean fraction of optimum</th><th class="num">simulations used</th><th class="num">exact-answer tasks solved</th><th class="num">LLM calls</th><th class="num">tokens</th></tr>${rows}`;
}
function renderDSEChart() {
  const tasks = R.tasks.filter((t) => t.kind === "dse"), as = agents();
  const mean = (a, t) => { const rs = R.rows.filter((r) => r.agent === a && r.task === t.id); return rs.length ? rs.reduce((x, r) => x + r.score, 0) / rs.length : null; };
  $("#dse-legend").innerHTML = as.map((a, i) => `<span><i style="background:${agentColor(a, i)}"></i>${esc(label(a))}</span>`).join("");
  $("#dse-chart").innerHTML = tasks.map((t) => `<div style="margin-bottom:10px"><div class="dim mono" style="font-size:11px">${esc(t.id.replace(/^dse-/, ""))}</div>` +
    as.map((a, i) => { const v = mean(a, t); return v === null ? "" : `<div class="kbar"><span class="dim">${esc(label(a)).slice(0, 16)}</span><span class="track"><span class="fill" style="display:block;width:${(100 * v).toFixed(1)}%;background:${agentColor(a, i)}"></span></span><span class="mono">${pct(v)}</span></div>`; }).join("") + "</div>").join("");
  $("#dse-chart").style.height = "auto";
}
function renderKindChart() {
  const kinds = ["dse", "bottleneck", "sizing", "crossover", "compare"], as = agents();
  $("#kind-chart").innerHTML = kinds.map((k) => `<div style="margin-bottom:10px"><div class="dim mono" style="font-size:11px">${k}</div>` +
    as.map((a, i) => { const v = R.summary[a].by_kind[k]; return v ? `<div class="kbar"><span class="dim">${esc(label(a)).slice(0, 16)}</span><span class="track"><span class="fill" style="display:block;width:${(100 * v.success_rate).toFixed(1)}%;background:${agentColor(a, i)}"></span></span><span class="mono">${pct(v.success_rate)}</span></div>` : ""; }).join("") + "</div>").join("");
}
function renderTaskTable() {
  const as = agents();
  const cell = (a, t) => {
    const rs = R.rows.filter((r) => r.agent === a && r.task === t.id);
    if (!rs.length) return `<td class="cell dim">-</td>`;
    const sc = rs.reduce((x, r) => x + r.score, 0) / rs.length, ok = rs.filter((r) => r.success).length;
    const txt = rs.length > 1 ? `${ok}/${rs.length}` : rs[0].success ? "pass" : "fail";
    const tr = rs[0].trace_file ? `data-trace="${esc(a)}|${esc(t.id)}"` : "";
    return `<td class="cell ${ok === rs.length ? "ok" : ok ? "" : "bad"}" ${tr} title="${esc(rs[0].detail)}">${txt}${t.kind === "dse" ? `<br><span class="dim">${pct(sc)}</span>` : ""}</td>`;
  };
  $("#task-table").innerHTML = `<tr><th>task</th><th>type</th>${as.map((a) => `<th>${esc(label(a))}</th>`).join("")}</tr>` +
    R.tasks.map((t) => `<tr><td title="${esc(t.prompt)}" class="mono" style="font-size:12px">${esc(t.id)}</td><td>${t.kind}</td>${as.map((a) => cell(a, t)).join("")}</tr>`).join("");
  $("#task-table").addEventListener("click", (e) => {
    const c = e.target.closest("[data-trace]");
    if (!c) return;
    const [a, t] = c.dataset.trace.split("|");
    $("#tr-agent").value = a; fillTraceTasks(); $("#tr-task").value = t; showTrace(); location.hash = "#trace";
  });
}

// ---------- traces ----------
function eventHTML(ev) {
  if (ev.type === "llm") {
    const calls = (ev.tool_calls || []).map((c) => `${c.name}(${JSON.stringify(c.args)})`).join("\n");
    return `<div class="ev llm"><div class="h">${ev.t}s  model${ev.tokens && ev.tokens.in ? `  ${ev.tokens.in} in / ${ev.tokens.out} out tokens` : ""}</div>${ev.text ? `<pre>${esc(ev.text)}</pre>` : ""}${calls ? `<pre>${esc(calls)}</pre>` : ""}</div>`;
  }
  if (ev.type === "tool") {
    let res = ev.result;
    try { res = JSON.stringify(JSON.parse(ev.result), null, 1); } catch (e) {}
    return `<div class="ev tool"><div class="h">${ev.t}s  tool ${esc(ev.name)}</div><details><summary>${esc(JSON.stringify(ev.args)).slice(0, 160)}</summary><pre>${esc(res)}</pre></details></div>`;
  }
  return `<div class="ev verify"><div class="h">${ev.t}s  verifier</div><pre>${esc(ev.problem ? "rejected: " + ev.problem : "accepted")}</pre></div>`;
}
function fillTraceTasks() {
  const a = $("#tr-agent").value;
  const ts = R.rows.filter((r) => r.agent === a && r.trace_file).map((r) => r.task);
  $("#tr-task").innerHTML = ts.map((t) => `<option>${esc(t)}</option>`).join("");
}
async function showTrace() {
  const a = $("#tr-agent").value, t = $("#tr-task").value;
  const row = R.rows.find((r) => r.agent === a && r.task === t);
  if (!row) return;
  const task = R.tasks.find((x) => x.id === t);
  $("#tr-kpis").innerHTML = kpi(row.success ? "pass" : "fail", row.detail) + kpi(row.simulations, "simulations") + kpi(row.llm_calls || "-", "model calls") + kpi(row.seconds + " s", "wall time");
  const tr = await (await fetch("data/" + row.trace_file)).json();
  $("#tr-events").innerHTML = `<div class="ev"><div class="h">task</div><pre>${esc(task.prompt)}</pre></div>` + tr.events.map(eventHTML).join("") +
    `<div class="ev"><div class="h">answer</div><pre>${esc(JSON.stringify(row.answer))}${row.rationale ? "\n" + esc(row.rationale) : ""}</pre></div><p class="note">${tr.otel_spans.length} OpenTelemetry spans recorded for this run.</p>`;
}
function initTraces() {
  const as = agents().filter((a) => R.rows.some((r) => r.agent === a && r.trace_file));
  $("#tr-agent").innerHTML = as.map((a) => `<option value="${esc(a)}">${esc(label(a))}</option>`).join("");
  $("#tr-agent").addEventListener("change", () => { fillTraceTasks(); showTrace(); });
  $("#tr-task").addEventListener("change", showTrace);
  fillTraceTasks();
  showTrace();
}

// ---------- live agent (Gemini REST, tools in Pyodide) ----------
async function gemini(path, key, body) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const r = await fetch(GEMINI + path, { method: body ? "POST" : "GET", headers: { "x-goog-api-key": key, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    if (r.ok) return r.json();
    const text = await r.text();
    if (r.status === 429 || r.status >= 500) {
      const m = text.match(/"retryDelay":\s*"(\d+)s"/);
      const wait = m ? +m[1] + 1 : 8 * (attempt + 1);
      $("#live-status").textContent = `Rate limited by the API; retrying in ${wait}s...`;
      await new Promise((ok) => setTimeout(ok, wait * 1000));
      continue;
    }
    throw new Error(`Gemini API ${r.status}: ${text.slice(0, 300)}`);
  }
  throw new Error("Gemini API kept failing");
}
async function pickModel(key) {
  // The newest model is often overloaded or has no free quota for this key, so probe in order.
  const d = await gemini("/models?pageSize=200", key);
  const bad = ["image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview", "vision"];
  const ids = (d.models || []).filter((m) => (m.supportedGenerationMethods || []).includes("generateContent")).map((m) => m.name.replace("models/", ""))
    .filter((n) => n.includes("flash") && !bad.some((b) => n.includes(b))).sort().reverse();
  const ranked = ids.filter((n) => !n.includes("lite")).concat(ids.filter((n) => n.includes("lite")));
  const skipped = [];
  for (const m of ranked.slice(0, 6)) {
    $("#live-status").textContent = `Checking ${m}...`;
    const r = await fetch(`${GEMINI}/models/${m}:generateContent`, { method: "POST", headers: { "x-goog-api-key": key, "content-type": "application/json" },
      body: JSON.stringify({ contents: [{ role: "user", parts: [{ text: "Reply with OK." }] }], generationConfig: { maxOutputTokens: 5 } }) });
    if (r.ok) return m;
    skipped.push(`${m} (${r.status})`);
  }
  throw new Error("No Gemini model is callable with this key. Tried: " + skipped.join(", "));
}
async function runLive(e) {
  e.preventDefault();
  const f = new FormData($("#live-form"));
  const key = f.get("key"), taskId = f.get("task"), budget = +f.get("budget");
  const task = R.tasks.find((t) => t.id === taskId);
  const log = $("#live-events");
  const add = (ev) => { log.insertAdjacentHTML("beforeend", eventHTML(ev)); };
  $("#live-panel").hidden = false; log.innerHTML = `<div class="ev"><div class="h">task</div><pre>${esc(task.prompt)}</pre></div>`;
  $("#live-run").disabled = true;
  const t0 = performance.now(), t = () => ((performance.now() - t0) / 1000).toFixed(1);
  let calls = 0, tokIn = 0, tokOut = 0, used = 0, verified = 0, nudges = 0, answer = null;
  const status = (s) => { $("#live-status").textContent = s; $("#live-kpis").innerHTML = kpi(used + "/" + budget, "simulations") + kpi(calls, "model calls") + kpi(((tokIn + tokOut) / 1e3).toFixed(1) + "k", "tokens"); };
  try {
    await rpc("reset", budget);
    status("Choosing a model...");
    const model = await pickModel(key);
    const fdecls = SPEC.tools.concat([SPEC.submit[taskId]]);
    const fmt = JSON.stringify(Object.fromEntries(Object.entries(SPEC.submit[taskId].parameters.properties.answer.properties).map(([k]) => [k, "..."])));
    const system = SPEC.system.replace("{answer_format}", fmt);
    const contents = [{ role: "user", parts: [{ text: task.prompt }] }];
    for (let turn = 0; turn < 40 && !answer; turn++) {
      status(`${model}: thinking (turn ${turn + 1})...`);
      const d = await gemini(`/models/${model}:generateContent`, key, { systemInstruction: { parts: [{ text: system }] }, contents, tools: [{ functionDeclarations: fdecls }], toolConfig: { functionCallingConfig: { mode: "AUTO" } }, generationConfig: { temperature: 0 } });
      calls++; tokIn += d.usageMetadata?.promptTokenCount || 0; tokOut += d.usageMetadata?.candidatesTokenCount || 0;
      const content = d.candidates?.[0]?.content || { role: "model", parts: [] };
      contents.push(content);
      const fcs = (content.parts || []).filter((p) => p.functionCall).map((p) => p.functionCall);
      const text = (content.parts || []).filter((p) => p.text).map((p) => p.text).join("\n");
      add({ type: "llm", t: t(), text, tool_calls: fcs.map((c) => ({ name: c.name, args: c.args || {} })), tokens: { in: d.usageMetadata?.promptTokenCount, out: d.usageMetadata?.candidatesTokenCount } });
      if (!fcs.length) {
        if (nudges++ >= 2) break;
        contents.push({ role: "user", parts: [{ text: "Use the tools to get numbers, then call submit_answer." }] });
        continue;
      }
      const responses = [];
      for (const c of fcs) {
        let result;
        if (c.name === "submit_answer") {
          const a = c.args?.answer || {};
          const need = Object.keys(SPEC.submit[taskId].parameters.properties.answer.properties);
          const missing = need.filter((k) => !(k in a) && !["pp", "expert_parallel"].includes(k));
          if (missing.length) result = { error: `answer is missing ${missing.join(", ")}` };
          else {
            const v = await rpc("verify", taskId, JSON.stringify(a));
            add({ type: "verify", t: t(), problem: v.problem });
            if (v.problem && verified++ < 1) result = { error: `Verification failed: ${v.problem}. Fix it and submit again.` };
            else { answer = a; result = { status: "received" }; }
          }
        } else {
          result = await rpc("call", c.name, JSON.stringify(c.args || {}));
          if (result._used !== undefined) used = result._used;
          delete result._used; delete result._budget;
          add({ type: "tool", t: t(), name: c.name, args: c.args || {}, result: JSON.stringify(result) });
        }
        const fr = { name: c.name, response: { result } };
        if (c.id) fr.id = c.id;
        responses.push({ functionResponse: fr });
      }
      contents.push({ role: "user", parts: responses });
      status(`${model}: turn ${turn + 1} done`);
    }
    const g = answer ? await rpc("grade", taskId, JSON.stringify(answer)) : { success: false, detail: "no answer submitted" };
    log.insertAdjacentHTML("beforeend", `<div class="ev ${g.success ? "" : "verify"}"><div class="h">result</div><pre>${esc(JSON.stringify(answer))}\n${g.success ? "PASS" : "FAIL"}: ${esc(g.detail)}</pre></div>`);
    status(`Done with ${model} in ${t()} s. ${g.success ? "Correct." : "Not correct."}`);
  } catch (err) {
    $("#live-status").textContent = "Error: " + err.message;
  } finally {
    $("#live-run").disabled = !ready;
  }
}

// ---------- architecture ----------
function renderArch() {
  const n = (s) => `<span class="node">${s}</span>`, a = `<span class="arrow">&rarr;</span>`;
  $("#arch").innerHTML = [n("task prompt"), a, n("LLM (Gemini / Groq / Ollama)"), a, n("tools: check_fit, simulate_step,<br>roofline_ops, evaluate_layout"), a, n("InferSim"), a, n("submit_answer<br>(typed schema)"), a, n("verifier<br>(re-simulates)"), a, n("grader<br>(exhaustive truth)")].join("");
}

async function main() {
  [R, SPEC] = await Promise.all([fetch("data/results.json").then((r) => r.json()), fetch("data/agent_spec.json").then((r) => r.json())]);
  renderBoard(); renderDSEChart(); renderKindChart(); renderTaskTable(); initTraces(); renderArch();
  $("#live-form select[name=task]").innerHTML = R.tasks.map((t) => `<option value="${esc(t.id)}">${esc(t.id)}</option>`).join("");
  $("#live-form select[name=task]").value = R.tasks.find((t) => t.kind === "bottleneck").id;
  $("#live-form").addEventListener("submit", runLive);
  const links = [...document.querySelectorAll("#nav a")];
  const obs = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) links.forEach((l) => l.classList.toggle("on", l.getAttribute("href") === "#" + e.target.id)); }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((s) => obs.observe(s));
  boot();
}
main();
