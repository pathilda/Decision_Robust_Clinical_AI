"""Combine completed policy reports offline; never import or invoke optimization."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from patient_assignment.proxy_plots import standalone
from patient_assignment.proxy_policy_plots import load_results, ordinary_figures


def combine(sources, output, mode="Box", facet_cols=3):
    if len(sources) < 2:
        raise ValueError("Provide at least two original completed result folders")
    if facet_cols < 1:
        raise ValueError("facet-cols must be positive")
    loaded = [load_results(Path(source)) for source in sources]
    reference = loaded[0][0]
    # Conservative compatibility: require the same inputs and experimental settings.
    ignored = {"chunks", "run_fingerprint"}
    chunks = {}
    frames = []
    for source, (resolved, frame) in zip(sources, loaded):
        differences = [
            k
            for k in reference.keys() | resolved.keys()
            if k not in ignored and reference.get(k) != resolved.get(k)
        ]
        if differences:
            raise ValueError(
                f"Incompatible saved settings in {source}: {', '.join(sorted(differences))}"
            )
        overlap = chunks.keys() & resolved["chunks"].keys()
        if overlap:
            raise ValueError(
                f"Overlapping chunk IDs would double-count observations: {sorted(overlap)}"
            )
        chunks.update(resolved["chunks"])
        frame = frame.copy()
        frame["source_run"] = str(source)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    resolved = dict(reference, chunks=chunks)
    output = Path(output)
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output} already exists")
    figures = ordinary_figures(frame, resolved, mode, facet_cols)
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    coverage = (
        f"{int(frame.certified.sum())}/{len(frame)} certified allocations; "
        f"{len(chunks)} chunks; proxy benchmark: {reference['benchmark_id']}. "
        "Combined from completed reports without optimization. Failures remain in the denominator."
    )
    links = []
    for name, fig in figures.items():
        target = f"figures/{name}_{mode}.html"
        (output / target).write_text(standalone(fig, name, coverage), encoding="utf-8")
        links.append(f'<li><a href="{target}">{name.replace("_", " ")}</a></li>')
    frame.to_csv(output / "combined_observations.csv", index=False)
    metrics = [
        m
        for m in ("proxy_regret", "exact_assignment_pct", "holdout_mean", "holdout_cvar")
        if m in frame
    ]
    group = ["model_id", "policy", "policy_configuration_id", "profile_id"]
    for name, flag in (
        ("selection_summary", "is_seed_sample"),
        ("across_chunk_summary", "is_chunk_evaluation"),
    ):
        selected = frame[frame[flag]]
        counts = selected.groupby(group).agg(
            n_attempted=("certified", "size"), n_certified=("certified", "sum")
        )
        stats = (
            selected[selected.certified]
            .groupby(group)[metrics]
            .agg(["mean", "median", "std", "min", "max"])
        )
        stats.columns = ["_".join(c) for c in stats.columns]
        counts.join(stats).reset_index().to_csv(output / f"{name}.csv", index=False)
        links.append(f'<li><a href="{name}.csv">{name.replace("_", " ")}</a></li>')
    provenance = {
        "source_runs": [str(p) for p in sources],
        "source_fingerprints": [r["run_fingerprint"] for r, _ in loaded],
        "chunk_ids": list(chunks),
        "optimization_executed": False,
    }
    (output / "combined_sources.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Combined regret and assignment report</h1><p>'
        + html.escape(coverage)
        + "</p><ul>"
        + "".join(links)
        + "</ul><p>Selection summaries describe optimal-selection draws; across-chunk summaries "
        "use the designated chunk-evaluation seed. These are descriptive, not independent clinical "
        "observations. Classification accuracy is not pooled from per-run percentages.</p>",
        encoding="utf-8",
    )
    print(f"No optimization executed. Four plots and two summary tables: {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show_type", choices=["Point", "Box"], default="Box")
    parser.add_argument("--facet-cols", type=int, default=3)
    args = parser.parse_args()
    try:
        combine(args.results, args.output, args.show_type, args.facet_cols)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
