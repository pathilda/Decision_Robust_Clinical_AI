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
