"""Offline per-allocation regret versus cohort-level extraction scatter plots."""

import html
import math
from pathlib import Path

import pandas as pd

from patient_assignment.proxy_plots import LABELS, SYMBOLS, color

METRICS = {
    "qwk": ("regret_vs_qwk", "Extraction QWK against proxy"),
    "exact_accuracy_pct": (
        "regret_vs_exact_extraction",
        "Exact extraction agreement with proxy (%)",
    ),
}


def build_scatter(frame, accuracy, resolved, metric):
    import plotly.graph_objects as go

    data = frame.copy()
    if "policy" not in data:
        data["policy"] = "deterministic"
    if "policy_configuration_id" not in data:
        data["policy_configuration_id"] = "deterministic"
    scores = accuracy[["model_id", metric]].rename(columns={metric: "extraction_score"})
    data = data.merge(scores, on="model_id", how="left", validate="many_to_one")
    data["extraction_score"] = pd.to_numeric(data.extraction_score, errors="coerce")
    data["proxy_regret"] = pd.to_numeric(data.proxy_regret, errors="coerce")
    data["valid_score"] = data.extraction_score.map(math.isfinite)
    data["valid_regret"] = data.proxy_regret.map(math.isfinite)
    data["success"] = data.solver_status.eq("OPTIMAL")
    fig = go.Figure()
    seen = set()
    configs = data.groupby(["model_id", "policy"]).policy_configuration_id.nunique()
    for (model, policy, config, profile), subset in data.groupby(
        ["model_id", "policy", "policy_configuration_id", "profile_id"], sort=True
    ):
        good = subset[subset.success & subset.valid_score & subset.valid_regret]
        pair = f"{model} / {policy}"
        label = pair if configs[(model, policy)] == 1 else f"{pair} / {config}"
        caseloads = ", ".join(
            f"{p['id']}: {p['initial_count']}" for p in resolved["profiles"].get(profile, [])
        )
        fields = [
            "model_id",
            "policy",
            "policy_configuration_id",
            "chunk_id",
            "profile_id",
            "selection_seed",
            "solution_hash",
            "source_run",
        ]
        custom = [
            [html.escape(str(r.get(k, ""))) for k in fields] + [html.escape(caseloads)]
            for r in good.to_dict("records")
        ]
        hover = "<br>".join(k + ": %{customdata[" + str(i) + "]}" for i, k in enumerate(fields))
        hover += "<br>Initial caseloads: %{customdata[8]}<br>Extraction score: %{x}<br>Proxy regret: %{y}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=good.extraction_score.tolist(),
                y=good.proxy_regret.tolist(),
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=label not in seen and bool(len(good)),
                marker={
                    "color": color(label),
                    "symbol": SYMBOLS.get(profile, "circle"),
                    "size": 7,
                    "opacity": 0.55,
                },
                customdata=custom,
                hovertemplate=hover,
                meta={
                    "profile": profile,
                    "model_policy": pair,
                    "total": len(subset),
                    "shown": len(good),
                    "failed": int((~subset.success).sum()),
                    "missing": int(
                        (subset.success & ~(subset.valid_score & subset.valid_regret)).sum()
                    ),
                },
            )
        )
        if len(good):
            seen.add(label)
    fig.update_layout(
        template="plotly_white",
        height=650,
        title="Proxy regret vs " + METRICS[metric][1],
        xaxis_title=METRICS[metric][1],
        yaxis_title="Proxy-benchmark regret",
        legend={"title": "Model / policy", "groupclick": "togglegroup"},
        margin={"t": 90, "b": 80},
        hovermode="closest",
    )
    return fig


def scatter_page(fig, metric):
    controls = []
    for field, label in [
        ("profile", "Initial caseload / starting profile"),
        ("model_policy", "Model / policy"),
    ]:
        values = sorted({t.meta[field] for t in fig.data})
        options = "".join(
            '<option value="'
            + html.escape(v, quote=True)
            + '">'
            + html.escape(LABELS.get(v, v) if field == "profile" else v)
            + "</option>"
            for v in values
        )
        controls.append(
            f'<label>{label} <select data-filter="{field}"><option value="">All</option>'
            + options
            + "</select></label>"
        )
    script = """<script>
(() => {
 const gd = document.querySelector('.plotly-graph-div');
 const controls = [...document.querySelectorAll('[data-filter]')];
 const hidden = new Set();
 function apply() {
   const eligible = gd.data.map(t => controls.every(c => !c.value || t.meta[c.dataset.filter] === c.value));
   const seen = new Set();
   const showlegend = gd.data.map((t,i) => {
     if (!eligible[i] || !t.meta.shown || seen.has(t.legendgroup)) return false;
     seen.add(t.legendgroup); return true;
   });
   const visible = gd.data.map((t,i) => !eligible[i] ? false : hidden.has(t.legendgroup) ? 'legendonly' : true);
   const counts = {total:0, shown:0, failed:0, missing:0};
   gd.data.forEach((t,i) => {if (visible[i] === true) Object.keys(counts).forEach(k => counts[k] += t.meta[k]);});
   document.getElementById('coverage').textContent = `${counts.shown}/${counts.total} saved allocation rows plotted; ${counts.failed} failed; ${counts.missing} certified rows lack a finite extraction score or regret.`;
   Plotly.restyle(gd, {visible, showlegend});
 }
 controls.forEach(c => c.addEventListener('change', apply));
 gd.on('plotly_legendclick', ev => {
   const key = gd.data[ev.curveNumber].legendgroup;
   if (hidden.has(key)) hidden.delete(key); else hidden.add(key);
   apply(); return false;
 });
 gd.on('plotly_legenddoubleclick', () => false);
 apply();
})();
</script>"""
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>'
        + html.escape(METRICS[metric][1])
        + "</title><style>body{font:15px system-ui;margin:24px}label{display:inline-block;margin:8px}select{padding:6px}</style></head><body>"
        + "".join(controls)
        + '<p id="coverage"></p>'
        + fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            config={"responsive": True, "displaylogo": False},
        )
        + "<p>Each point is one saved certified allocation result, including all selection seeds "
        "and repeated assignments. Extraction scores are fixed per model over the report cohort; "
        "filters do not recalculate them. Combined reports use the union of unique clients. "
        "Vertical stacks and overlapping points are expected; no jitter or averaging is applied. "
        "These repeated solver results are not independent extraction observations. Undefined "
        "QWK is omitted from the QWK plot and counted above.</p>" + script + "</body></html>"
    )


def write_scatters(frame, accuracy, resolved, output):
    output = Path(output)
    (output / "figures").mkdir(exist_ok=True)
    links = []
    for metric, (name, _) in METRICS.items():
        fig = build_scatter(frame, accuracy, resolved, metric)
        target = f"figures/{name}.html"
        (output / target).write_text(scatter_page(fig, metric), encoding="utf-8")
        links.append(f'<li><a href="{target}">{name.replace("_", " ")}</a></li>')
    return links
