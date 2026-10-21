"""Offline policy comparisons regenerated solely from checksummed saved artifacts."""

import html
import json
from pathlib import Path

import pandas as pd

from .inputs import file_hash
from .proxy_inputs import select
from .proxy_plots import CAPTION, color
from .reporting import fresh_directory

FILTERS = [
    "model_id",
    "policy",
    "policy_configuration_id",
    "profile_id",
    "chunk_id",
    "scenario_count",
    "scenario_seed",
    "gamma_fraction",
    "probability_threshold",
    "temperature",
]
HOVERS = [
    "model_id",
    "policy",
    "policy_configuration_id",
    "chunk_id",
    "profile_id",
    "selection_seed",
    "scenario_seed",
    "scenario_count",
    "gamma_fraction",
    "gamma_effective",
    "probability_threshold",
    "temperature",
    "benchmark_id",
    "reference_seed",
    "solver_seed",
    "proxy_regret",
    "exact_assignment_pct",
    "selected_policy_objective",
    "solution_hash",
    "certified",
    "holdout_seed",
    "holdout_fingerprint",
]


def load_results(source):
    source = Path(source)
    manifest = json.loads((source / "artifacts.json").read_text(encoding="utf-8"))
    required = {
        "results.csv",
        "allocations.csv",
        "reference_allocations.csv",
        "accuracy_vs_proxy.csv",
        "resolved_config.json",
        "validation_report.json",
        "benchmark_optima.json",
        "policy_grid.csv",
    }
    if not required <= set(manifest):
        raise ValueError("Saved policy manifest lacks required artifacts")
    for name, expected in manifest.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Missing or corrupt saved policy artifact: {name}")
    resolved = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(source / "results.csv", dtype=str, keep_default_na=False)
    columns = ["model_id", "policy_configuration_id", "chunk_id", "profile_id", "selection_seed"]
    expected = {
        (m, c["policy_configuration_id"], chunk, p, str(seed))
        for m in resolved["models"]
        for c in resolved["policy_grid"]
        for chunk in resolved["chunks"]
        for p in resolved["profiles"]
        for seed in {*resolved["selection_seeds"], resolved["chunk_seed"]}
    }
    if frame.duplicated(columns).any() or set(map(tuple, frame[columns].values)) != expected:
        raise ValueError("Missing, duplicate or unexpected policy selection rows")
    for field in ("certified", "is_seed_sample", "is_chunk_evaluation"):
        if not set(frame[field]) <= {"true", "false"}:
            raise ValueError("Invalid saved certification or role flags")
        frame[field] = frame[field].eq("true")
    if set(frame.run_fingerprint) != {resolved["run_fingerprint"]}:
        raise ValueError("Saved policy rows belong to a different run fingerprint")
    for row in frame.to_dict("records"):
        seed = int(row["selection_seed"])
        if row["is_seed_sample"] != (seed in resolved["selection_seeds"]) or row[
            "is_chunk_evaluation"
        ] != (seed == resolved["chunk_seed"]):
            raise ValueError("Saved roles disagree with selection seed configuration")
        if row["certified"] != (row["solver_status"] == "OPTIMAL"):
            raise ValueError("Inconsistent saved certification")
    for field in (
        "proxy_regret",
        "exact_assignment_pct",
        "selected_policy_objective",
        "holdout_mean",
        "holdout_cvar",
    ):
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame.loc[frame.certified, ["proxy_regret", "exact_assignment_pct"]].isna().any().any():
        raise ValueError("Certified rows have missing metrics")
    return resolved, frame


def chart(frame, metric, xfield, mode, facet_cols, faceted=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = list(dict.fromkeys(zip(frame.chunk_id, frame.profile_id))) if faceted else [None]
    if not panels:
        panels = [None]
    cols = min(facet_cols, len(panels))
    fig = make_subplots(
        rows=(len(panels) + cols - 1) // cols,
        cols=cols,
        subplot_titles=[" / ".join(p) if p else "All selected chunks" for p in panels],
        shared_yaxes=True,
    )
    # progress-only: begin
    seen_legend = set()
    configuration_counts = frame.groupby("policy").policy_configuration_id.nunique()
    # progress-only: end
    groups = ["model_id", "policy_configuration_id", "chunk_id", "profile_id"]
    for identity, subset in frame.groupby(groups, sort=True):
        row = subset.iloc[0]
        panel = panels.index((row.chunk_id, row.profile_id)) if faceted else 0
        metadata = {k: str(row.get(k, "")) for k in FILTERS}
        metadata.update(
            n_total=len(subset),
            n_success=int(subset.certified.sum()),
            n_unique=int(subset.loc[subset.certified, "solution_hash"].nunique()),
        )
        good = subset[subset.certified]
        if good.empty or metric not in good:
            continue
        label = f"{row.model_id} / {row.policy_configuration_id}"
        # progress-only: begin
        label = f"{row.model_id} / {row.policy}"
        if configuration_counts[row.policy] > 1:
            if row.policy == "stochastic":
                label += f" (W={row.get('scenario_count', '')}, seed={row.get('scenario_seed', '')}, T={row.get('temperature', '')})"
            elif row.policy == "robust":
                budget = row.get("gamma_fraction", "") or row.get("gamma_effective", "")
                label += f" (Gamma={budget}, threshold={row.get('probability_threshold', '')})"
        legend_key = f"{row.model_id}/{row.policy_configuration_id}"
        # progress-only: end
        custom = [[str(r.get(k, "")) for k in HOVERS] for r in good.to_dict("records")]
        hover = "<br>".join(k + ": %{customdata[" + str(i) + "]}" for i, k in enumerate(HOVERS))
        hover += f"<br>Successful/attempted: {len(good)}/{len(subset)}<br>Distinct assignments: {metadata['n_unique']}<extra></extra>"
        xs = [label] * len(good) if xfield == "method" else good[xfield].tolist()
        kwargs = {
            "x": xs,
            "y": good[metric].tolist(),
            "name": label,
            "legendgroup": label,
            "meta": metadata,
            "customdata": custom,
            "hovertemplate": hover,
            "marker": {"color": color(label)},
        }
        if mode == "Box":
            trace = go.Box(**kwargs, boxpoints="all", jitter=0.2, pointpos=0)
        else:
            trace = go.Scatter(**kwargs, mode="markers", marker_symbol="circle", opacity=0.75)
        # progress-only: begin
        trace.legendgroup = legend_key
        trace.showlegend = legend_key not in seen_legend
        seen_legend.add(legend_key)
        # progress-only: end
        fig.add_trace(trace, row=panel // cols + 1, col=panel % cols + 1)
    fig.update_layout(
        template="plotly_white",
        height=max(500, 350 * ((len(panels) + cols - 1) // cols)),
        boxmode="group",
        title=metric.replace("_", " "),
        legend={"orientation": "h"},
        margin={"b": 160},
    )
    fig.update_yaxes(title_text=metric.replace("_", " "), matches="y")
    fig.update_xaxes(title_text=xfield.replace("_", " "))
    # progress-only: begin
    fig.update_layout(
        height=max(650, 450 * ((len(panels) + cols - 1) // cols)),
        legend={"orientation": "v", "x": 1.02, "y": 1, "xanchor": "left",
                "yanchor": "top", "groupclick": "togglegroup", "font": {"size": 11}},
        margin={"b": 150, "r": 260, "t": 90},
    )
    fig.update_xaxes(automargin=True, tickangle=-25 if xfield == "method" else 0)
    if metric == "exact_assignment_pct":
        fig.update_yaxes(title_text="Assignment agreement with proxy (%)", range=[0, 100])
    # progress-only: end
    return fig


def standalone(fig, frame, title):
    controls = []
    for key in FILTERS:
        values = sorted({str(v) for v in frame.get(key, []) if str(v)})
        controls.append(
            "<label>"
            + html.escape(key)
            + ' <select data-filter="'
            + key
            + '">'
            + '<option value="">All</option>'
            + "".join(
                '<option value="' + html.escape(v, quote=True) + '">' + html.escape(v) + "</option>"
                for v in values
            )
            + "</select></label> "
        )
    # Independent configuration selectors allow an actual M0/M2/M3 matched comparison.
    for policy in ("stochastic", "robust"):
        ids = sorted(set(frame.loc[frame.policy == policy, "policy_configuration_id"]))
        if ids:
            controls.append(
                "<label>"
                + policy
                + ' configuration <select data-filter="policy_configuration_id" data-policy="'
                + policy
                + '">'
                + '<option value="">All</option>'
                + "".join(
                    '<option value="'
                    + html.escape(v, quote=True)
                    + '">'
                    + html.escape(v)
                    + "</option>"
                    for v in ids
                )
                + "</select></label> "
            )
    records = [
        {k: str(r.get(k, "")) for k in FILTERS} | {"certified": bool(r["certified"])}
        for r in frame.to_dict("records")
    ]
    records_json = json.dumps(records).replace("<", "\\u003c")
    script = """<script>
const records = RECORDS;
const controls = [...document.querySelectorAll('[data-filter]')];
const gd = document.querySelector('.plotly-graph-div');
function matches(row) { return controls.every(c =>
 !c.value || (c.dataset.policy && row.policy !== c.dataset.policy) || row[c.dataset.filter] === c.value); }
function apply() {
 const selected = records.filter(matches);
 document.getElementById('coverage').textContent = selected.filter(r=>r.certified).length + '/' + selected.length + ' certified observations; failures retained in denominator.';
 Plotly.restyle(gd, {visible: gd.data.map(t => matches(t.meta))});
}
if (MATCHED) {
 controls.forEach(c => {
  if ((c.dataset.policy || ['model_id','chunk_id','profile_id'].includes(c.dataset.filter)) && c.options.length > 1) c.selectedIndex = 1;
 });
}
controls.forEach(c => c.addEventListener('change', apply));
apply();
</script>""".replace("RECORDS", records_json).replace(
        "MATCHED", "true" if title.startswith("matched_") else "false"
    )
    # progress-only: begin
    # Recompute legend representatives after filtering, so hiding the first chunk
    # does not remove a model/policy from the legend.
    script = script.replace(
        "Plotly.restyle(gd, {visible: gd.data.map(t => matches(t.meta))});",
        """const visible = gd.data.map(t => matches(t.meta));
 const seen = new Set();
 const showlegend = gd.data.map((t, i) => {
   if (!visible[i] || seen.has(t.legendgroup)) return false;
   seen.add(t.legendgroup); return true;
 });
 Plotly.restyle(gd, {visible, showlegend});""",
    )
    # progress-only: end
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{font:14px system-ui;margin:24px}label{display:inline-block;margin:6px}select{max-width:230px}</style></head><body>"
        + "".join(controls)
        + '<p id="coverage"></p>'
        + fig.to_html(full_html=False, include_plotlyjs=True)
        + "<p>"
        + html.escape(CAPTION)
        + "</p><p>Holdout distributions are model-specific simulations. Training replicates and optimal-selection seeds are separate sources of variation. Panels and paired differences are descriptive; no clinical independence or confidence intervals are assumed.</p>"
        + script
        + "</body></html>"
    )


# progress-only: begin
def ordinary_figures(frame, resolved, mode, facet_cols):
    """Reuse the deterministic report's profile grouping and sampling roles."""
    from .proxy_plots import build_figure

    display = frame.copy()
    labels = {}
    configurations = list(dict.fromkeys(frame.policy_configuration_id))
    for (model, configuration), data in frame.groupby(
        ["model_id", "policy_configuration_id"], sort=False
    ):
        policy = data.iloc[0].policy
        label = f"{model} / {policy}"
        if frame.loc[frame.policy == policy, "policy_configuration_id"].nunique() > 1:
            label += f" (config {configurations.index(configuration) + 1})"
        labels[(model, configuration)] = label
    display["model_id"] = [
        labels[(r.model_id, r.policy_configuration_id)] for r in frame.itertuples()
    ]
    models = list(dict.fromkeys(display.model_id))
    chunks = list(dict.fromkeys(display.chunk_id))
    profiles = [p for p in resolved["profiles"] if p in set(display.profile_id)]
    result = {}
    for name, metric, faceted in (
        ("regret_by_chunk", "proxy_regret", True),
        ("assignment_by_chunk", "exact_assignment_pct", True),
        ("regret_across_chunks", "proxy_regret", False),
        ("assignment_across_chunks", "exact_assignment_pct", False),
    ):
        fig = build_figure(
            display, resolved, models, chunks, profiles, mode, facet_cols, metric, faceted
        )
        for trace in fig.data:
            configuration = next(c for (m, c), label in labels.items() if label == trace.name)
            trace.hovertemplate = trace.hovertemplate.replace(
                "<extra>", "<br>Policy configuration: " + html.escape(configuration) + "<extra>"
            )
        fig.update_xaxes(title_text="Model / policy", tickangle=-30, automargin=True)
        fig.update_layout(legend_title_text="Model / policy", margin={"b": 160})
        result[name] = fig
    return result


def render_ordinary(source, output, frame, resolved, mode, facet_cols):
    from .proxy_plots import standalone as ordinary_page

    coverage = (
        f"{int(frame.certified.sum())}/{len(frame)} certified allocations; "
        f"common proxy benchmark: {resolved['benchmark_id']}. "
        "Failed allocations remain in the coverage denominator."
    )
    links = []
    for name, fig in ordinary_figures(frame, resolved, mode, facet_cols).items():
        target = f"figures/{name}_{mode}.html"
        (output / target).write_text(ordinary_page(fig, name, coverage), encoding="utf-8")
        links.append(f'<li><a href="{target}">{name.replace("_", " ")} ({mode})</a></li>')
    accuracy = pd.read_csv(Path(source) / "accuracy_vs_proxy.csv", keep_default_na=False)
    accuracy = accuracy[accuracy.model_id.isin(frame.model_id.unique())]
    (output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Proxy analysis</title>'
        '<style>body{font:15px system-ui;margin:32px}li{margin:12px}td,th{padding:8px}</style>'
        '</head><body><h1>Regret and assignment plots</h1><p>'
        + html.escape(coverage)
        + '</p><p>Four figures: open each link below. Each includes all selected models and '
        'policies, with the original Starting Parameter dropdown and profile symbols.</p><ul>'
        + ''.join(links)
        + '</ul><h2>Accuracy against the proxy</h2><p>Saved unique-cohort accuracy; '
        'optimization policies do not change input classification accuracy. Plot filters '
        'do not redefine the original cohort.</p>'
        + accuracy.to_html(index=False, escape=True)
        + '<p>' + html.escape(CAPTION) + '</p></body></html>', encoding="utf-8"
    )

# progress-only: end
def render(
    source,
    output,
    mode,
    facet_cols,
    models=None,
    chunk_ids=None,
    profile_ids=None,
    policies=None,
    configurations=None,
    allow_existing=False,
    # progress-only: begin
    extended_plots=False,
    # progress-only: end
):
    resolved, frame = load_results(source)
    for field, allowed, requested in (
        ("model_id", resolved["models"], models),
        ("chunk_id", resolved["chunks"], chunk_ids),
        ("profile_id", resolved["profiles"], profile_ids),
        ("policy", list(dict.fromkeys(c["policy"] for c in resolved["policy_grid"])), policies),
        (
            "policy_configuration_id",
            [c["policy_configuration_id"] for c in resolved["policy_grid"]],
            configurations,
        ),
    ):
        values = select(dict.fromkeys(allowed), requested, field)
        frame = frame[frame[field].isin(values)]
    output = Path(output) if allow_existing else fresh_directory(output)
    (output / "figures").mkdir(exist_ok=True)
    # progress-only: begin
    if not extended_plots:
        render_ordinary(source, output, frame, resolved, mode, facet_cols)
        return
    # progress-only: end
    links = []

    def save(name, data, metric, x="method", faceted=True, style=mode):
        fig = chart(data, metric, x, style, facet_cols, faceted)
        target = f"figures/{name}_{mode}.html"
        (output / target).write_text(standalone(fig, data, name), encoding="utf-8")
        links.append(f'<li><a href="{target}">{html.escape(name)}</a></li>')

    samples = frame[frame.is_seed_sample]
    across = frame[frame.is_chunk_evaluation]
    for prefix, metric in (("regret", "proxy_regret"), ("assignment", "exact_assignment_pct")):
        save(prefix + "_by_chunk", samples, metric)
        save(prefix + "_across_chunks", across, metric, faceted=False)
        if resolved["uncertainty"]["plots"]["include_parameter_sweeps"]:
            for policy, x in (("stochastic", "scenario_count"), ("robust", "gamma_fraction")):
                data = samples[samples.policy == policy].copy()
                if policy == "robust" and len(data) and not data[x].astype(bool).any():
                    x = "gamma_effective"
                if len(data):
                    data[x] = pd.to_numeric(data[x])
                    save(prefix + "_" + policy + "_sensitivity", data, metric, x)
    if resolved["uncertainty"]["plots"]["include_policy_comparisons"]:
        save("matched_policy_comparison", samples, "proxy_regret")
    if resolved["uncertainty"]["plots"]["include_assignment_comparisons"]:
        save("matched_assignment_comparison", samples, "exact_assignment_pct")
    diversity = []
    for _, data in samples.groupby(
        ["model_id", "policy_configuration_id", "chunk_id", "profile_id"]
    ):
        seen = set()
        for index, row in enumerate(
            data.sort_values("sample_index", key=lambda x: pd.to_numeric(x)).to_dict("records"), 1
        ):
            if row["certified"]:
                seen.add(row["solution_hash"])
            diversity.append(row | {"samples_attempted": index, "unique_solutions": len(seen)})
    if diversity:
        save(
            "solution_diversity",
            pd.DataFrame(diversity),
            "unique_solutions",
            "samples_attempted",
            style="Point",
        )
    stochastic = samples[samples.policy == "stochastic"]
    replicate_rows = []
    for _, data in stochastic.groupby(
        ["model_id", "policy_configuration_id", "chunk_id", "profile_id"]
    ):
        good = data[data.certified]
        if len(good):
            replicate_rows.append(
                good.iloc[0].to_dict() | {"proxy_regret": good.proxy_regret.mean()}
            )
    if replicate_rows:
        save(
            "between_training_replicates_mean_regret",
            pd.DataFrame(replicate_rows),
            "proxy_regret",
            "scenario_seed",
            style="Point",
        )
    for metric in ("holdout_mean", "holdout_cvar"):
        if metric in samples and samples[metric].notna().any():
            save(metric, samples, metric)
    coverage = f"{int(frame.certified.sum())}/{len(frame)} certified allocations; common point proxy: {resolved['benchmark_id']}."
    (output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"></head><body><h1>Uncertainty-aware proxy analysis</h1><p>'
        + html.escape(coverage)
        + "</p><ul>"
        + "".join(links)
        + "</ul><p>"
        + html.escape(CAPTION)
        + "</p></body></html>",
        encoding="utf-8",
    )
