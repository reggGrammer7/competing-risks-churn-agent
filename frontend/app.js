// API_BASE now comes from config.js (loaded before this file) -- edit that
// one file when deploying, not this one.
const CAUSES = ["Price sensitivity", "Dissatisfaction", "Competitive loss", "Non-behavioral"];
const CAUSE_COLORS = {
  "Price sensitivity": "#6ea8fe",
  "Dissatisfaction": "#ff8a80",
  "Competitive loss": "#f0c98a",
  "Non-behavioral": "#85e0a3",
};

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------
document.querySelectorAll(".tabBtn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabBtn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tabPanel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// ---------------------------------------------------------------------
// Generic "run model" buttons -> fetch -> dispatch to a render function
// ---------------------------------------------------------------------
document.querySelectorAll(".loadBtn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const out = btn.nextElementSibling;
    out.innerHTML = `<p class="hint">Running (fits a real model server-side, may take a few seconds)...</p>`;
    try {
      const res = await fetch(`${API_BASE}${btn.dataset.endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      window[btn.dataset.render](data, out);
    } catch (err) {
      out.innerHTML = errorHtml(err);
    }
  });
});

function errorHtml(err) {
  return `<p class="hint">Error contacting backend at ${API_BASE}: ${err.message}<br/>Is uvicorn running? (uvicorn backend.main:app --reload)</p>`;
}

// Converts the agent's markdown-style **bold** into real <strong> HTML,
// instead of showing literal asterisks -- both the template answers (the
// ranking feature deliberately marks its winner with **bold**) and an
// LLM's own writing style tend to use this. Escapes other HTML first so
// answer text can never inject markup of its own.
function renderAnswerText(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function statCard(value, label) {
  return `<div class="statCard"><div class="statValue">${value}</div><div class="statLabel">${label}</div></div>`;
}

// ---------------------------------------------------------------------
// 1. Strawman
// ---------------------------------------------------------------------
function renderStrawman(data, out) {
  const bars = data.top_shap_features
    .map((f) => {
      const maxAbs = data.top_shap_features[0].mean_abs_shap || 1;
      const pct = Math.max(4, (f.mean_abs_shap / maxAbs) * 100);
      return `<div class="barRow">
        <div class="barLabel" title="${f.feature}">${f.feature}</div>
        <div class="barTrack"><div class="barFill" style="width:${pct}%"></div></div>
        <div class="barValue">${f.mean_abs_shap.toFixed(3)}</div>
      </div>`;
    })
    .join("");
  out.innerHTML = `
    <div class="statRow">${statCard((data.test_accuracy * 100).toFixed(1) + "%", "Test accuracy (not the full picture -- see caveat)")}</div>
    <p class="hint" style="margin-top:0.5rem;">Average feature impact on the classifier's output, across ~200 test customers (SHAP mean |value|):</p>
    ${bars}
    <div class="caveatBox"><b>Caveat:</b> ${data.caveat}</div>
  `;
}

// ---------------------------------------------------------------------
// 2. Kaplan-Meier
// ---------------------------------------------------------------------
function renderKM(data, out) {
  out.innerHTML = `<canvas id="kmChart" width="720" height="320"></canvas><p class="hint" style="margin-top:0.5rem;">${data.note}</p>`;
  const canvas = document.getElementById("kmChart");
  drawLineChart(canvas, [
    { label: "All-cause survival probability", color: "#6ea8fe", points: data.months.map((m, i) => ({ x: m, y: data.survival_prob[i] })) },
  ], { xLabel: "Month", yLabel: "Survival probability", yAsPercent: true, yMax: 1 });
}

// ---------------------------------------------------------------------
// 3. CIF (single cause + compare-all)
// ---------------------------------------------------------------------
const cifButtonsWrap = document.getElementById("cifCauseButtons");
CAUSES.forEach((cause, i) => {
  const b = document.createElement("button");
  b.textContent = cause;
  b.dataset.cause = cause;
  if (i === 0) b.classList.add("activeCause");
  b.addEventListener("click", () => {
    document.querySelectorAll("#cifCauseButtons button").forEach((x) => x.classList.remove("activeCause"));
    b.classList.add("activeCause");
    loadSingleCif(cause);
  });
  cifButtonsWrap.appendChild(b);
});

async function loadSingleCif(cause) {
  const canvas = document.getElementById("cifChart");
  const withCi = document.getElementById("ciToggleBox").checked;
  try {
    if (withCi) {
      const res = await fetch(`${API_BASE}/tools/cif-ci/${encodeURIComponent(cause)}?n_boot=100`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      drawLineChart(canvas, [
        {
          label: cause, color: CAUSE_COLORS[cause],
          points: d.months.map((m, i) => ({ x: m, y: d.cif[i] })),
          band: { lower: d.ci_lower, upper: d.ci_upper },
        },
      ], { xLabel: "Month", yLabel: "Cumulative incidence", yAsPercent: true, title: `${cause} -- CIF with 95% bootstrap CI (n=${d.n_bootstrap})` });
    } else {
      const res = await fetch(`${API_BASE}/tools/cif/${encodeURIComponent(cause)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      drawLineChart(canvas, [
        { label: cause, color: CAUSE_COLORS[cause], points: d.months.map((m, i) => ({ x: m, y: d.cif[i] })) },
      ], { xLabel: "Month", yLabel: "Cumulative incidence", yAsPercent: true, title: `${cause} -- cumulative incidence over time` });
    }
  } catch (err) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
}
document.getElementById("ciToggleBox").addEventListener("change", () => {
  const active = document.querySelector("#cifCauseButtons .activeCause");
  loadSingleCif(active ? active.dataset.cause : CAUSES[0]);
});
loadSingleCif(CAUSES[0]);

document.getElementById("compareBtn").addEventListener("click", async () => {
  const canvas = document.getElementById("compareChart");
  const note = document.getElementById("crossoverNote");
  canvas.classList.remove("hidden");
  try {
    const res = await fetch(`${API_BASE}/tools/compare-causes`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    const series = CAUSES.map((c) => ({
      label: c, color: CAUSE_COLORS[c], points: d[c].months.map((m, i) => ({ x: m, y: d[c].cif[i] })),
    }));
    drawLineChart(canvas, series, { xLabel: "Month", yLabel: "Cumulative incidence", yAsPercent: true, title: "All causes -- who's the leading cause, and when", legend: true });
    note.innerHTML = crossoverSummary(d);
    note.classList.remove("hidden");
  } catch (err) {
    note.innerHTML = errorHtml(err);
    note.classList.remove("hidden");
  }
});

// Figures out, at each shared timepoint, which cause has the highest CIF,
// then collapses consecutive timepoints with the same leader into ranges --
// this is the actual answer to "is there a plot supporting which cause
// dominates when," computed from the real numbers instead of eyeballed.
function crossoverSummary(byCause) {
  const months = byCause[CAUSES[0]].months;
  const leaders = months.map((_, i) => {
    let best = CAUSES[0], bestVal = -1;
    CAUSES.forEach((c) => {
      if (byCause[c].cif[i] > bestVal) { bestVal = byCause[c].cif[i]; best = c; }
    });
    return best;
  });
  const ranges = [];
  let start = months[0], current = leaders[0];
  for (let i = 1; i <= leaders.length; i++) {
    if (i === leaders.length || leaders[i] !== current) {
      const end = i === leaders.length ? months[months.length - 1] : months[i - 1];
      ranges.push({ start, end, cause: current });
      if (i < leaders.length) { start = months[i]; current = leaders[i]; }
    }
  }
  const items = ranges.map((r) => `<li><b>Month ${r.start}\u2013${r.end}:</b> ${r.cause} has the highest cumulative incidence</li>`).join("");
  return `<b>Leading cause by time range</b> (computed from the actual CIF curves above):<ul>${items}</ul>`;
}

// ---------------------------------------------------------------------
// 4. Cox hazard ratios
// ---------------------------------------------------------------------
function renderCox(data, out) {
  const sections = CAUSES.map((cause) => {
    const rows = (data.top_covariates_by_cause[cause] || [])
      .map((r) => {
        const hrClass = r.hazard_ratio > 1 ? "hrUp" : "hrDown";
        const arrow = r.hazard_ratio > 1 ? "\u2191 raises risk" : "\u2193 lowers risk";
        return `<tr><td>${r.covariate}</td><td class="${hrClass}">${r.hazard_ratio} (${arrow})</td><td>${r.p_value}</td></tr>`;
      })
      .join("");
    return `<h3 style="margin-bottom:0.2rem;">${cause}</h3>
      <table class="dataTable">
        <thead><tr><th>Covariate</th><th>Hazard ratio</th><th>p-value</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }).join("<div class=\"divider\"></div>");
  out.innerHTML = sections;
}

// ---------------------------------------------------------------------
// 5. RSF
// ---------------------------------------------------------------------
function renderRSF(data, out) {
  out.innerHTML = `
    <div class="statRow">
      ${statCard(data.concordance_index.toFixed(3), "Concordance index (0.5 = random, 1.0 = perfect ranking)")}
      ${statCard(data.n_trees, "Trees in the forest")}
    </div>
    <p class="hint">${data.model}</p>
  `;
}

// ---------------------------------------------------------------------
// 7. Evaluation
// ---------------------------------------------------------------------
function renderEvaluation(data, out) {
  const rows = Object.values(data.by_cause).map((r) => `
    <h3 style="margin-bottom:0.2rem;">${r.cause}</h3>
    <p class="hint" style="margin-top:0;">${r.n_events_in_test} events in a test set of ${r.n_test} (${data.test_size * 100}% held out)</p>
    <table class="dataTable">
      <thead><tr><th>Model</th><th>C-index</th><th>95% CI</th><th>Integrated Brier score</th></tr></thead>
      <tbody>
        <tr><td>Baseline (constant risk)</td><td>${r.baseline_c_index}</td><td>&mdash;</td><td>&mdash;</td></tr>
        <tr><td>Cox (cause-specific)</td><td>${r.cox_c_index}</td><td>[${r.cox_c_index_ci[0]}, ${r.cox_c_index_ci[1]}]</td><td>${r.cox_integrated_brier_score}</td></tr>
        <tr><td>Random Survival Forest</td><td>${r.rsf_c_index}</td><td>[${r.rsf_c_index_ci[0]}, ${r.rsf_c_index_ci[1]}]</td><td>${r.rsf_integrated_brier_score}</td></tr>
      </tbody>
    </table>`).join("<div class=\"divider\"></div>");
  out.innerHTML = `
    <p class="hint">${data.n_train} train / ${data.n_test} test, ${data.n_bootstrap} bootstrap resamples per confidence interval.</p>
    ${rows}
    <div class="caveatBox">C-index measures ranking quality; Integrated Brier score also penalizes poor calibration. A cause scoring at or below baseline means no real signal was found for it in this data -- for a cause that's genuinely close to random with respect to the available covariates, that's the framework working correctly, not a bug.</div>
  `;
}

// ---------------------------------------------------------------------
// 8. Data validation
// ---------------------------------------------------------------------
function renderValidation(data, out) {
  const statusColor = data.passed ? "#85e0a3" : "#ff8a80";
  const list = (items, label) => items.length
    ? `<p class="hint" style="margin-bottom:0.2rem;"><b>${label}:</b></p><ul class="traceList">${items.map((c) => `<li><b>${c.name}:</b> ${c.message}</li>`).join("")}</ul>`
    : "";
  out.innerHTML = `
    <div class="statRow">${statCard(`<span style="color:${statusColor}">${data.passed ? "PASSED" : "FAILED"}</span>`, "Overall validation status")}</div>
    ${list(data.errors, "Errors (block the pipeline)")}
    ${list(data.warnings, "Warnings (flagged, don't block)")}
    ${!data.errors.length && !data.warnings.length ? '<p class="hint">No issues found.</p>' : ""}
  `;
}

// ---------------------------------------------------------------------
// 6. Agent
// ---------------------------------------------------------------------
const questionBox = document.getElementById("question");

// Auto-grow: the box gets TALLER as the question gets longer, instead of
// scrolling horizontally with earlier text hidden off-screen. Reset height
// to auto first so shrinking (deleting text) also works, not just growing.
function autoGrowQuestionBox() {
  questionBox.style.height = "auto";
  questionBox.style.height = `${questionBox.scrollHeight}px`;
}
questionBox.addEventListener("input", autoGrowQuestionBox);

// Enter submits (matching the old <input>'s behavior); Shift+Enter inserts
// a real newline instead, since a plain textarea would otherwise treat
// every Enter as a newline and never submit.
questionBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("askBtn").click();
  }
});

document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  if (!question) return;
  const out = document.getElementById("agentOutput");
  out.innerHTML = `<p class="hint">Thinking...</p>`;
  try {
    const res = await fetch(`${API_BASE}/agent/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const t = data.trace;

    if (t.task_type === "clarification_needed") {
      out.innerHTML = `
        <p style="white-space:pre-wrap;">${renderAnswerText(data.answer)}</p>
        <p class="hint">The question was too vague to extract a cause, time horizon, or segment -- ask again with more detail (this endpoint has no memory of this exchange).</p>
      `;
      return;
    }

    if (t.task_type === "invalid_horizon") {
      out.innerHTML = `<p style="white-space:pre-wrap;">${renderAnswerText(data.answer)}</p>`;
      return;
    }

    if (t.task_type === "unsupported_segment") {
      out.innerHTML = `
        <p style="white-space:pre-wrap;">${renderAnswerText(data.answer)}</p>
        <p class="hint">Requested attribute not in this dataset's schema: "${t.requested_segment}".</p>
      `;
      return;
    }

    const conf = t.detected_cause_confidence;
    const confClass = conf >= 0.08 ? "confHigh" : "confLow";
    const causeDisplay = Array.isArray(t.detected_cause) ? t.detected_cause.join(" vs ") : (t.detected_cause || "all causes");
    const confLabel = t.detected_cause ? (conf >= 0.08 ? "confident match" : "low-confidence match") : "no cause specified";
    out.innerHTML = `
      <p style="white-space:pre-wrap;">${renderAnswerText(data.answer)}</p>
      <ul class="traceList">
        <li><b>Task type:</b> ${t.task_type === "instance_level" ? "instance-level prediction" : "population analysis"}</li>
        <li><b>Detected cause:</b> ${causeDisplay} <span class="confBadge ${confClass}">${confLabel}${t.detected_cause ? ` (${(conf * 100).toFixed(1)}%)` : ""}</span></li>
        <li><b>Time horizon:</b> ${t.wants_full_curve ? "full curve" : `${t.time_horizon_months} months`}</li>
        <li><b>Segment:</b> ${t.segment} (n=${t.n_customers_in_segment})</li>
        <li><b>Tool calls:</b> ${t.tool_calls.join(", ")}</li>
        <li><b>Methodology doc used:</b> ${t.methodology_doc_used}</li>
        ${t.policy_doc_used ? `<li><b>Policy doc used:</b> ${t.policy_doc_used}</li>` : ""}
      </ul>
    `;
  } catch (err) {
    document.getElementById("agentOutput").innerHTML = errorHtml(err);
  }
});

// ---------------------------------------------------------------------
// 9. Predict a customer (instance-level)
// ---------------------------------------------------------------------
document.getElementById("predictBtn").addEventListener("click", async () => {
  const out = document.getElementById("predictOut");
  out.innerHTML = `<p class="hint">Predicting...</p>`;
  const senior = document.getElementById("pf-senior").value;
  const body = {
    contract: document.getElementById("pf-contract").value || null,
    internet_service: document.getElementById("pf-internet").value || null,
    tech_support: document.getElementById("pf-techsupport").value || null,
    payment_method: document.getElementById("pf-payment").value || null,
    senior_citizen: senior === "" ? null : senior === "true",
    dependents: document.getElementById("pf-dependents").value || null,
    paperless_billing: document.getElementById("pf-paperless").value || null,
    phone_service: document.getElementById("pf-phoneservice").value || null,
    multiple_lines: document.getElementById("pf-multiplelines").value || null,
    cause: document.getElementById("pf-cause").value || null,
    time_horizon_months: parseInt(document.getElementById("pf-horizon").value, 10) || 12,
  };
  try {
    const res = await fetch(`${API_BASE}/predict-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const r = await res.json();
    const specified = Object.entries(r.profile_specified);
    const filled = Object.entries(r.profile_filled_with_population_defaults);
    out.innerHTML = `
      <div class="statRow">
        ${statCard((r.churn_probability_at_horizon * 100).toFixed(1) + "%", `Predicted churn probability by month ${r.horizon_months_used}`)}
        ${statCard((r.survival_probability_at_horizon * 100).toFixed(1) + "%", "Predicted survival probability")}
        ${r.median_expected_tenure_months !== null ? statCard(r.median_expected_tenure_months.toFixed(0), "Median expected tenure (months)") : ""}
        ${r.cause ? statCard(r.cause_specific_relative_risk + "x", `Relative risk: ${r.cause}`) : ""}
      </div>
      ${specified.length ? `<p class="hint"><b>You specified:</b> ${specified.map(([k, v]) => `${k}=${v}`).join(", ")}</p>` : `<p class="hint">Nothing specified -- this is the fully population-typical profile.</p>`}
      ${filled.length ? `<p class="hint"><b>Filled with population defaults:</b> ${filled.map(([k, v]) => `${k}=${v}`).join(", ")}</p>` : ""}
      ${r.median_expected_tenure_note ? `<p class="hint">${r.median_expected_tenure_note}</p>` : ""}
      <p class="caveatBox">This is one constructed representative profile built from the fields above (with population defaults for anything left unset), not a prediction for a real individual customer.</p>
    `;
  } catch (err) {
    out.innerHTML = errorHtml(err);
  }
});

// ---------------------------------------------------------------------
// 10. Descriptive analytics
// ---------------------------------------------------------------------
function renderDescriptive(data, out) {
  const cards = Object.entries(data.columns).map(([colName, info]) => {
    if (info.type === "numeric") {
      const maxCount = Math.max(...info.histogram.counts, 1);
      const bars = info.histogram.labels.map((label, i) => {
        const pct = Math.max(3, (info.histogram.counts[i] / maxCount) * 100);
        return `<div class="barRow">
          <div class="barLabel" style="width:110px;font-size:0.72rem;" title="${label}">${label}</div>
          <div class="barTrack"><div class="barFill" style="width:${pct}%"></div></div>
          <div class="barValue">${info.histogram.counts[i]}</div>
        </div>`;
      }).join("");
      return `<div class="descCard">
        <h3>${colName} <span class="typeBadge">numeric</span></h3>
        <div class="statRow" style="margin:0.5rem 0;">
          ${statCard(info.mean, "Mean")}
          ${statCard(info.median, "Median")}
          ${statCard(info.std, "Std dev")}
          ${statCard(`${info.min}\u2013${info.max}`, "Range")}
        </div>
        ${bars}
      </div>`;
    }
    const maxCount = Math.max(...info.bar_chart.counts, 1);
    const bars = info.bar_chart.labels.map((label, i) => {
      const pct = Math.max(3, (info.bar_chart.counts[i] / maxCount) * 100);
      return `<div class="barRow">
        <div class="barLabel" title="${label}">${label}</div>
        <div class="barTrack"><div class="barFill" style="width:${pct}%"></div></div>
        <div class="barValue">${info.bar_chart.counts[i]}</div>
      </div>`;
    }).join("");
    return `<div class="descCard">
      <h3>${colName} <span class="typeBadge">categorical</span></h3>
      <p class="hint" style="margin:0.2rem 0;">${info.n_unique} unique value(s)</p>
      ${bars}
    </div>`;
  }).join("<div class=\"divider\"></div>");

  const excludedNote = data.id_like_columns_excluded.length
    ? ` (excluded as ID-like: ${data.id_like_columns_excluded.join(", ")})`
    : "";
  out.innerHTML = `
    <p class="hint">${data.n_rows} rows, ${data.n_columns_described} columns described${excludedNote}.</p>
    ${cards}
  `;
}

// ---------------------------------------------------------------------
// Reusable chart: gridlines, real tick labels, optional CI band, legend.
// ---------------------------------------------------------------------
function niceTicks(min, max, count) {
  if (max <= min) max = min + 1;
  const range = max - min;
  const rawStep = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const ticks = [];
  let t = Math.ceil(min / step) * step;
  while (t <= max + 1e-9) {
    ticks.push(Math.round(t * 1000) / 1000);
    t += step;
  }
  return ticks;
}

function drawLineChart(canvas, series, opts) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const padding = { left: 60, right: 20, top: opts.title ? 34 : 16, bottom: 46 };
  const w = canvas.width - padding.left - padding.right;
  const h = canvas.height - padding.top - padding.bottom;

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  let allY = series.flatMap((s) => s.points.map((p) => p.y));
  series.forEach((s) => { if (s.band) allY = allY.concat(s.band.upper); });
  const xMin = 0, xMax = Math.max(...allX);
  const yMin = 0, yMax = opts.yMax !== undefined ? opts.yMax : Math.max(0.02, ...allY) * 1.15;

  const xTicks = niceTicks(xMin, xMax, 6);
  const yTicks = niceTicks(yMin, yMax, 5);

  const xToPx = (x) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * w;
  const yToPx = (y) => padding.top + h - ((y - yMin) / (yMax - yMin || 1)) * h;

  // gridlines + tick labels
  ctx.strokeStyle = "#1d212a";
  ctx.fillStyle = "#9aa1ac";
  ctx.font = "11px sans-serif";
  yTicks.forEach((ty) => {
    const py = yToPx(ty);
    ctx.beginPath();
    ctx.moveTo(padding.left, py);
    ctx.lineTo(padding.left + w, py);
    ctx.stroke();
    const label = opts.yAsPercent ? `${(ty * 100).toFixed(0)}%` : ty.toString();
    ctx.textAlign = "right";
    ctx.fillText(label, padding.left - 8, py + 3);
  });
  xTicks.forEach((tx) => {
    const px = xToPx(tx);
    ctx.textAlign = "center";
    ctx.fillText(tx.toString(), px, padding.top + h + 16);
  });

  // axes
  ctx.strokeStyle = "#3a4050";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + h);
  ctx.lineTo(padding.left + w, padding.top + h);
  ctx.stroke();

  // axis titles
  ctx.fillStyle = "#c7cbd4";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(opts.xLabel || "", padding.left + w / 2, padding.top + h + 36);
  ctx.save();
  ctx.translate(16, padding.top + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(opts.yLabel || "", 0, 0);
  ctx.restore();

  if (opts.title) {
    ctx.fillStyle = "#e6e8ec";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(opts.title, padding.left, 18);
  }

  // CI band (drawn under the line)
  series.forEach((s) => {
    if (!s.band) return;
    ctx.fillStyle = hexToRgba(s.color, 0.18);
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = xToPx(p.x), y = yToPx(s.band.upper[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    for (let i = s.points.length - 1; i >= 0; i--) {
      const x = xToPx(s.points[i].x), y = yToPx(s.band.lower[i]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  });

  // lines
  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = xToPx(p.x), y = yToPx(p.y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // legend
  if (opts.legend && series.length > 1) {
    let lx = padding.left + w - 160, ly = padding.top + 8;
    series.forEach((s) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = "#e6e8ec";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(s.label, lx + 16, ly + 1);
      ly += 16;
    });
  }
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
