"""Self-contained Plotly reports from saved results only (no Gurobi imports)."""

import html
import json
import math
from pathlib import Path

import pandas as pd

from .experiment_inputs import digest
from .inputs import file_hash
from .proxy_inputs import PROFILES, select
from .reporting import fresh_directory

LABELS = {
    "balanced": "Balanced",
    "unbalanced": "Unbalanced",
    "highly_unbalanced": "Highly unbalanced",
}
SYMBOLS = {"balanced": "circle", "unbalanced": "diamond", "highly_unbalanced": "square"}
CAPTION = (
    "The benchmark is a proxy, not human ground truth. Assignment agreement depends on the chosen "
    "fixed benchmark-optimal allocation; zero regret does not imply 100% assignment agreement. "
    "Random linear tie-breaks are not uniform samples from the optimal set; repeated allocations are retained. "
    "Cross-chunk performance describes the observed patient groups. "
    "Boxes use linear quartiles and standard 1.5 IQR whiskers, which are not confidence intervals "
    "or exact min/max bounds. All observations, including outliers and singletons, are shown."
)
HOVER = (
    "Model: %{customdata[0]}<br>Proxy benchmark: %{customdata[1]}<br>Chunk: %{customdata[2]}"
    "<br>Starting profile: %{customdata[3]}<br>Selection seed: %{customdata[4]}"
    "<br>Proxy regret: %{customdata[5]}<br>Assignment agreement: %{customdata[6]}%"
    "<br>Objective: %{customdata[7]}<br>Group observations: %{customdata[8]}<extra></extra>"
)


def color(mid):
    # Configuration identity, independent of candidate order, plot filters and benchmark.
    return f"hsl({int(digest(mid)[:8], 16) % 360},62%,42%)"


def load_results(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
    required = {
        "results.csv",
        "allocations.csv",
        "reference_allocations.csv",
        "accuracy_vs_proxy.csv",
        "resolved_config.json",
        "validation_report.json",
        "benchmark_optima.json",
    }
    if not required <= set(manifest):
        raise ValueError("Saved results manifest lacks required selection data")
    for name in required:
        if not (directory / name).is_file() or file_hash(directory / name) != manifest[name]:
            raise ValueError(
                f"Missing or corrupt saved results: {name}; plotting never reruns optimization"
            )
    resolved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "validation_report.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "results.csv", dtype=str, keep_default_na=False)
    expected = {
        (m, c, p, str(s))
        for m in resolved["models"]
        for c in resolved["chunks"]
        for p in resolved["profiles"]
        for s in {*resolved["selection_seeds"], resolved["chunk_seed"]}
    }
    columns = ["model_id", "chunk_id", "profile_id", "selection_seed"]
    if frame.duplicated(columns).any() or set(map(tuple, frame[columns].values)) != expected:
        raise ValueError("Missing, duplicate or unexpected selection rows in saved results")
    for field in ("proxy_regret", "exact_assignment_pct", "nominal_optimum", "proxy_optimum"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    for field in ("is_seed_sample", "is_chunk_evaluation"):
        if not set(frame[field]) <= {"true", "false"}:
            raise ValueError("Invalid saved selection role flags")
        frame[field] = frame[field].eq("true")
    for row in frame.to_dict("records"):
        seed = int(row["selection_seed"])
        if row["is_seed_sample"] != (seed in resolved["selection_seeds"]) or row[
            "is_chunk_evaluation"
        ] != (seed == resolved["chunk_seed"]):
            raise ValueError("Saved selection roles disagree with configured seeds")
    good = frame.solver_status.eq("OPTIMAL")
    if frame.loc[good, ["proxy_regret", "exact_assignment_pct"]].isna().any().any():
        raise ValueError("Certified selection data contains missing metrics")
    if not set(frame.solver_status) <= {"OPTIMAL", "FAILED"}:
        raise ValueError("Unknown certification status")
    return resolved, report, frame


def build_figure(frame, resolved, models, chunks, profiles, mode, facet_cols, metric, faceted):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = chunks if faceted else [None]
    cols = min(facet_cols, len(groups)) if faceted else 1
    rows = math.ceil(len(groups) / cols)
    titles = [f"Chunk {html.escape(c)}" for c in groups] if faceted else [""]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_yaxes="all",
        subplot_titles=titles,
        horizontal_spacing=min(0.08, 0.4 / cols),
    )
    role = "is_seed_sample" if faceted else "is_chunk_evaluation"
    selected = frame.loc[frame[role]]
    successful = selected.loc[selected.solver_status.eq("OPTIMAL")]
    all_values = successful[metric].dropna()
    if metric == "exact_assignment_pct":
        yrange = [0, 100]
        ylabel = "Exact Assignment Agreement (%)"
    else:
        lo = min(0, float(all_values.min())) if len(all_values) else 0
        hi = max(1, float(all_values.max())) if len(all_values) else 1
        yrange = [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.1]
        ylabel = "Proxy-benchmark regret"
    for k, chunk in enumerate(groups):
        row, col = k // cols + 1, k % cols + 1
        missing = []
        for mi, mid in enumerate(models):
            for profile in profiles:
                data = successful.loc[
                    successful.model_id.eq(mid) & successful.profile_id.eq(profile)
                ]
                if chunk is not None:
                    data = data.loc[data.chunk_id.eq(chunk)]
                offset = (PROFILES.index(profile) - 1) * 0.23
                x = [mi + offset] * len(data)
                if mode == "Point":
                    x = [
                        value
                        + (int(digest([str(v) for v in record])[:8], 16) / (2**32 - 1) - 0.5) * 0.08
                        for value, record in zip(
                            x, data[["chunk_id", "selection_seed", "profile_id", "model_id"]].values
                        )
                    ]
                custom = data[
                    [
                        "model_id",
                        "benchmark_id",
                        "chunk_id",
                        "profile_id",
                        "selection_seed",
                        "proxy_regret",
                        "exact_assignment_pct",
                        "objective",
                    ]
                ].values.tolist()
                custom = [record + [len(data)] for record in custom]
                shared = {
                    "x": x,
                    "y": data[metric].tolist(),
                    "name": mid,
                    "legendgroup": mid,
                    "showlegend": k == 0 and profile == profiles[0],
                    "meta": {"model": mid, "profile": profile, "chunk": chunk, "n": len(data)},
                    "customdata": custom,
                    "hovertemplate": HOVER,
                    "marker": {"color": color(mid), "symbol": SYMBOLS[profile], "size": 7},
                }
                if mode == "Point":
                    trace = go.Scatter(**shared, mode="markers")
                else:
                    # All points provide patient-group/seed hover even for singleton boxes.
                    trace = go.Box(
                        **shared,
                        width=0.18,
                        quartilemethod="linear",
                        boxpoints="all",
                        jitter=0,
                        pointpos=0,
                        line={"color": color(mid)},
                        fillcolor=color(mid),
                        opacity=0.7,
                        offsetgroup=profile,
                        alignmentgroup=mid,
                    )
                fig.add_trace(trace, row=row, col=col)
                if data.empty:
                    missing.append(f"{mid}/{LABELS[profile]}")
        if missing:
            axis = "" if k == 0 else str(k + 1)
            fig.add_annotation(
                xref=f"x{axis} domain",
                yref=f"y{axis} domain",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                text="No certified observations: " + html.escape(", ".join(missing)),
                font={"size": 10, "color": "#9b2226"},
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(models))),
            ticktext=[html.escape(m) for m in models],
            range=[-0.6, len(models) - 0.4],
            title_text="Candidate configuration",
            row=row,
            col=col,
        )
        fig.update_yaxes(range=yrange, title_text=ylabel if col == 1 else None, row=row, col=col)
    subtitle = (
        "selection draws within each chunk"
        if faceted
        else f"cross-chunk performance at selection seed {resolved['chunk_seed']}"
    )
    fig.update_layout(
        title={
            "text": f"{ylabel} — proxy benchmark {html.escape(resolved['benchmark_id'])}<br>"
            f"<sup>{subtitle}; objective: {resolved['objective']}</sup>"
        },
        template="plotly_white",
        height=max(520, 370 * rows + 150),
        margin={"t": 120, "b": 80},
        boxmode="overlay",
        hovermode="closest",
        legend={"title": "Configuration (click to toggle)", "groupclick": "togglegroup"},
    )
    return fig


def standalone(fig, title, coverage):
    """One visibility state combines profile dropdown and model legend toggles."""
    import plotly.io as pio

    body = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    script = """
<script>
(() => {
 const gd = document.querySelector('.plotly-graph-div');
 const menu = document.getElementById('starting-parameter');
 const hidden = new Set();
 const sampleCount = document.getElementById('sample-count');
 function apply() {
   const profile = menu.value;
   const visible = gd.data.map(t => hidden.has(t.meta.model) ? 'legendonly' :
       (profile === 'All' || t.meta.profile === profile));
   const legends = new Set();
   const show = gd.data.map(t => {
     const eligible = profile === 'All' || t.meta.profile === profile;
     if (!eligible || t.meta.n === 0 || legends.has(t.meta.model)) return false;
     legends.add(t.meta.model); return true;
   });
   sampleCount.textContent = 'Visible observations: ' + gd.data.reduce((n,t,i) =>
       n + (visible[i] === true ? t.meta.n : 0), 0);
   return Plotly.restyle(gd, {visible: visible, showlegend: show});
 }
 menu.addEventListener('change', apply);
 gd.on('plotly_legendclick', ev => {
   const model = gd.data[ev.curveNumber].meta.model;
   if (hidden.has(model)) hidden.delete(model); else hidden.add(model);
   apply(); return false;
 });
 gd.on('plotly_legenddoubleclick', () => false);
 apply();
})();
</script>"""
    options = '<option value="All">All</option>' + "".join(
        f'<option value="{p}">{LABELS[p]}</option>' for p in PROFILES
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;margin:24px;color:#17212b} "
        "select{padding:6px} .note{max-width:1200px;line-height:1.5}</style></head><body>"
        + "<label for='starting-parameter'>Starting Parameter: </label>"
        + f"<select id='starting-parameter'>{options}</select> "
        + "<span>Profile key: ● Balanced · ◆ Unbalanced · ■ Highly unbalanced; left/center/right offsets.</span>"
        + "<p id='sample-count'></p><p>"
        + html.escape(coverage)
        + "</p>"
        + body
        + "<p class='note'>"
        + html.escape(CAPTION)
        + "</p>"
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
    allow_existing=False,
):
    resolved, report, frame = load_results(source)
    models = list(select(dict.fromkeys(resolved["models"]), models, "models"))
    chunks = list(select(resolved["chunks"], chunk_ids, "chunks"))
    profiles = list(select(resolved["profiles"], profile_ids, "starting parameters"))
    frame = frame.loc[
        frame.model_id.isin(models) & frame.chunk_id.isin(chunks) & frame.profile_id.isin(profiles)
    ]
    output = Path(output) if allow_existing else fresh_directory(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    successful = int(frame.solver_status.eq("OPTIMAL").sum())
    coverage = (
        f"Proxy benchmark: {resolved['benchmark_id']}. Selected coverage: {successful}/{len(frame)} "
        f"certified allocations; {len(frame) - successful} failed. Run: {report['feasible_instances']}/"
        f"{report['total_instances']} capacity-feasible instances. "
        "Profiles excluded by a plot filter have no observations in this report."
    )
    links = []
    for name, metric, faceted in (
        ("regret_by_chunk", "proxy_regret", True),
        ("assignment_by_chunk", "exact_assignment_pct", True),
        ("regret_across_chunks", "proxy_regret", False),
        ("assignment_across_chunks", "exact_assignment_pct", False),
    ):
        fig = build_figure(
            frame, resolved, models, chunks, profiles, mode, facet_cols, metric, faceted
        )
        target = f"figures/{name}_{mode}.html"
        (output / target).write_text(standalone(fig, name, coverage), encoding="utf-8")
        links.append(f'<li><a href="{target}">{name.replace("_", " ")} ({mode})</a></li>')
    accuracy = (
        pd.read_csv(
            Path(source) / "accuracy_vs_proxy.csv",
            dtype={"model_id": str, "benchmark_id": str},
            keep_default_na=False,
        )
        .set_index("model_id")
        .loc[models]
        .reset_index()
    )
    # Plot filters never recalculate the original unique-cohort accuracy table.
    accuracy = accuracy.rename(
        columns={
            "exact_accuracy_pct": "Exact Accuracy (proxy agreement)",
            "qwk": "QWK (proxy agreement)",
        }
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Proxy benchmark analysis</title>"
        "<style>body{font:15px system-ui;margin:32px}td,th{padding:8px}table{border-collapse:collapse}</style>"
        "</head><body><h1>Proxy benchmark: "
        + html.escape(resolved["benchmark_id"])
        + "</h1><p>"
        + html.escape(coverage)
        + "</p><h2>Exact Accuracy (proxy agreement) and QWK (proxy agreement)</h2>"
        + "<p>Accuracy uses the unique cohort of the saved run, once per patient; plot filters do not "
        "redefine that cohort. Empty QWK values are NA; qwk_status explains why.</p>"
        + accuracy.to_html(index=False, escape=True)
        + "<ul>"
        + "".join(links)
        + "</ul><p>"
        + html.escape(CAPTION)
        + "</p></body></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def regenerate(args):
    resolved = json.loads((args.results / "resolved_config.json").read_text(encoding="utf-8"))
    if "policy_grid" in resolved:
        from .proxy_policy_plots import render as render_policies

        render_policies(
            args.results,
            args.output,
            args.show_type,
            args.facet_cols,
            args.models,
            args.chunk_ids,
            args.starting_parameters,
            args.policies,
            args.policy_configuration_ids,
            # progress-only: begin
            # Forward this display-only option after the positional filters.
            extended_plots=getattr(args, "extended_plots", False),
            # progress-only: end
        )
        return
    if args.policies and args.policies != ["deterministic"] or args.policy_configuration_ids:
        raise ValueError("Saved legacy results contain only deterministic policy")
    render(
        args.results,
        args.output,
        args.show_type,
        args.facet_cols,
        args.models,
        args.chunk_ids,
        args.starting_parameters,
    )
    print(f"Saved-data-only report: {args.output / 'index.html'}")
