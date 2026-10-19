"""Plots from saved tables only. No solver imports or initialization."""

import json
import math

import pandas as pd

from .inputs import file_hash
from .reporting import fresh_directory, write_json

OVERVIEW_METRICS = ["regret", "disagreement", "benchmark_range", "count_range"]


def aggregate_overview(data):
    """Average draws within chunks first, then weight all chunks equally per cell."""
    keys = ["model_id", "objective", "profile_id", "strategy", "oracle", "nominal_slack"]
    valid = data.loc[data.certified.eq(True)].copy()
    grouped = valid.groupby([*keys, "chunk_id"], dropna=False, sort=True)
    chunks = grouped[OVERVIEW_METRICS].mean()
    chunks["draws"] = grouped.size()
    chunks["patient_count"] = grouped.patient_count.first()
    chunks = chunks.reset_index()
    grouped = chunks.groupby(keys, dropna=False, sort=True)
    means = grouped[OVERVIEW_METRICS].mean()
    means["chunks"] = grouped.size()
    means["draws"] = grouped.draws.sum()
    means["min_chunk_size"] = grouped.patient_count.min()
    means["max_chunk_size"] = grouped.patient_count.max()
    means["weighting"] = "equal chunk weight after averaging draws within each chunk"
    return chunks, means.reset_index()


def plot_overview(data, config, args, output, save):
    """Two compact figures per strategy; objectives stay separate and all models share axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    chunks, means = aggregate_overview(data)
    chunks.to_csv(output / "overview_chunk_means.csv", index=False)
    means.to_csv(output / "overview_means.csv", index=False)
    profiles = [
        p for p in ("balanced", "unbalanced", "highly_unbalanced") if p in set(data.profile_id)
    ]
    profiles += sorted(set(data.profile_id) - set(profiles))
    labels = []
    for profile in profiles:
        clinicians = config["profiles"][profile]["clinicians"]
        loads = [c["initial_complexity"] for c in clinicians]
        labels.append(profile.replace("_", " ") + f"\ninitial range {max(loads) - min(loads):g}")
    models = sorted(data.model_id.unique())
    palette = plt.get_cmap("tab10")
    model_colors = {mid: palette(i % 10) for i, mid in enumerate(models)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    benchmark = config["benchmark"]
    regret_label = (
        "Mean proxy-benchmark regret"
        if benchmark["kind"] == "proxy"
        else "Mean clinical-reference regret"
    )
    agreement_label = (
        "Assignment disagreement with proxy benchmark"
        if benchmark["kind"] == "proxy"
        else "Assignment disagreement with clinical reference"
    )
    for strategy, selected in means.groupby("strategy", sort=True):
        objectives = [
            mode
            for mode in ("complexity-range", "lexicographic")
            if mode in set(selected.objective)
        ]
        available_chunks = data.loc[data.strategy == strategy, "chunk_id"].nunique()
        minimum_chunks = int(selected.chunks.min())
        coverage = f"All {available_chunks} saved chunks; equal weight per chunk"
        if minimum_chunks != available_chunks:
            coverage += (
                f"; incomplete cells use {minimum_chunks}–{available_chunks} chunks (see CSV)"
            )
        sizes = sorted(data.loc[data.strategy == strategy, "patient_count"].unique())
        coverage += "; chunk sizes: " + ", ".join(str(int(n)) for n in sizes)
        strategy_label = (
            "matched — benchmark-informed diagnostic" if strategy == "matched" else strategy
        )
        if strategy == "robust" and (selected.nominal_slack > 0).any():
            strategy_label = "near-optimal robust (positive nominal slack)"
        for family, metrics, ylabels in (
            (
                "regret_assignments",
                ["regret", "disagreement"],
                [regret_label + " [load units]", agreement_label],
            ),
            (
                "actual_loads",
                ["benchmark_range", "count_range"],
                [
                    "Mean benchmark-evaluated load range [load units]",
                    "Mean final count range [patients]",
                ],
            ),
        ):
            fig, axes = plt.subplots(
                len(objectives),
                2,
                figsize=(11.5, 3.4 * len(objectives) + 1.2),
                squeeze=False,
                sharex=True,
                sharey="col",
            )
            handles, legends = [], []
            for row, objective in enumerate(objectives):
                for col, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
                    ax = axes[row, col]
                    for index, mid in enumerate(models):
                        part = selected.loc[
                            (selected.objective == objective) & (selected.model_id == mid)
                        ]
                        if part.empty:
                            continue
                        # Saved experiments have one oracle/slack setting per model/strategy.
                        if part.profile_id.duplicated().any():
                            raise ValueError(
                                "Overview cannot mix different oracle/slack settings in one cell"
                            )
                        values = part.set_index("profile_id").reindex(profiles)[metric]
                        label = mid + (" (benchmark)" if mid == benchmark["id"] else "")
                        if part.oracle.any():
                            label += " [ORACLE]"
                        (line,) = ax.plot(
                            range(len(profiles)),
                            values,
                            marker=markers[index % len(markers)],
                            color=model_colors[mid],
                            linewidth=1.8,
                            markersize=6,
                            label=label,
                        )
                        if row == 0 and col == 0:
                            handles.append(line)
                            legends.append(label)
                    ax.set_title(objective)
                    ax.set_ylabel(ylabel)
                    ax.set_xticks(range(len(profiles)), labels)
                    ax.tick_params(axis="x", labelbottom=True)
                    ax.grid(axis="y", alpha=0.2)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.set_ylim(bottom=0)
                    if metric == "disagreement":
                        ax.set_ylim(0, 1)
                        ax.yaxis.set_major_formatter(PercentFormatter(1))
                    threshold = (
                        args.regret_threshold
                        if metric == "regret"
                        else args.disagreement_threshold
                        if metric == "disagreement"
                        else None
                    )
                    if threshold is not None:
                        ax.axhline(threshold, color="gray", linestyle=":")
            fig.suptitle(
                f"{benchmark['kind'].upper()} benchmark: {benchmark['id']} | {strategy_label}\n"
                f"{coverage}\nLoad units: {config['scoring']['scale']}",
                fontsize=11,
            )
            fig.legend(
                handles,
                legends,
                loc="lower center",
                ncol=min(3, len(legends)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.015),
            )
            if family == "regret_assignments":
                fig.text(
                    0.5,
                    0.002,
                    "Assignment disagreement depends on the chosen optimal assignments; lines connect profile means.",
                    ha="center",
                    fontsize=8,
                )
            save(fig, f"overview_{family}_{strategy}", rect=(0, 0.12, 1, 0.98))


def plot_results(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.max_instances_per_figure < 1:
        raise ValueError("max-instances-per-figure must be positive")
    for value in (args.regret_threshold, args.disagreement_threshold):
        if value is not None and not math.isfinite(value):
            raise ValueError("Plot thresholds must be finite")
    config = json.loads((args.results / "resolved_config.json").read_text(encoding="utf-8"))
    data = pd.read_csv(
        args.results / "solutions.csv", dtype={"model_id": str, "chunk_id": str, "profile_id": str}
    )
    bounds = pd.read_csv(
        args.results / "regret_bounds.csv",
        dtype={"model_id": str, "chunk_id": str, "profile_id": str},
    )
    data = data.loc[data.certified.eq(True)].copy()
    for arg, column in (
        ("models", "model_id"),
        ("profile_ids", "profile_id"),
        ("objectives", "objective"),
        ("selection", "strategy"),
    ):
        selected = getattr(args, arg)
        if selected:
            data = data.loc[data[column].isin(selected)]
    if data.empty:
        raise ValueError("No certified selections match the plot filters")
    output = fresh_directory(args.output)
    benchmark = config["benchmark"]
    regret_label = (
        "Proxy-benchmark regret" if benchmark["kind"] == "proxy" else "Clinical-reference regret"
    )
    units = config["scoring"]["scale"]
    title_base = f"{benchmark['kind'].upper()} benchmark: {benchmark['id']} | units: {units}"
    if data.oracle.any():
        title_base += "\nORACLE diagnostics present (see candidate flags in saved tables)"
    if (data.nominal_slack > 0).any():
        title_base += "\nRobust selections use positive nominal slack: near-optimal robust"
    files, omitted = [], []

    def save(fig, name, rect=None):
        fig.tight_layout(rect=rect)
        for fmt in args.format:
            path = output / f"{name}.{fmt}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)

    def thresholds(ax, disagreement=False):
        if args.regret_threshold is not None:
            ax.axhline(
                args.regret_threshold,
                color="gray",
                linestyle=":",
                label="explicit regret threshold",
            )
        if disagreement and args.disagreement_threshold is not None:
            ax.axvline(args.disagreement_threshold, color="gray", linestyle=":")

    marker = {"random": "o", "robust": "*", "matched": "s"}
    color = {"random": "#3679ad", "robust": "#c64737", "matched": "#7d639f"}
    data["instance"] = data.chunk_id + " / " + data.profile_id
    data["diagnostic"] = data.strategy.map(
        lambda s: "matched diagnostic" if s == "matched" else "random and robust"
    )
    for family in args.plots:
        if family == "overview":
            plot_overview(data, config, args, output, save)
        elif family in {"regret-envelope", "assignment-regret"}:
            for group_index, ((mid, mode, diagnostic), part) in enumerate(
                data.groupby(["model_id", "objective", "diagnostic"], sort=True)
            ):
                instances = sorted(part.instance.unique())
                for page, start in enumerate(
                    range(0, len(instances), args.max_instances_per_figure)
                ):
                    ids = instances[start : start + args.max_instances_per_figure]
                    selected = part.loc[part.instance.isin(ids)]
                    fig, ax = plt.subplots(figsize=(max(7, len(ids) * 0.75), 5))
                    if family == "regret-envelope":
                        for index, iid in enumerate(ids):
                            subset = selected.loc[selected.instance == iid]
                            first = subset.iloc[0]
                            eb = bounds.loc[
                                (bounds.model_id == mid)
                                & (bounds.objective == mode)
                                & (bounds.chunk_id == first.chunk_id)
                                & (bounds.profile_id == first.profile_id)
                                & bounds.complete.eq(True)
                            ]
                            # An original-optimal envelope must not visually bound near-optimal robust selections.
                            if not eb.empty and not (subset.nominal_slack > 0).any():
                                ax.vlines(
                                    index,
                                    eb.minimum_regret.iloc[0],
                                    eb.maximum_regret.iloc[0],
                                    color="black",
                                    linewidth=3,
                                )
                            for strategy, dots in subset.groupby("strategy"):
                                ax.scatter(
                                    [index] * len(dots),
                                    dots.regret,
                                    marker=marker[strategy],
                                    color=color[strategy],
                                    alpha=0.6,
                                    s=65 if strategy == "robust" else 22,
                                    label=strategy if index == 0 else None,
                                )
                        ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
                        ax.set_xlabel(
                            "Chunk / profile; lines = certified exact envelope (not a confidence interval)"
                        )
                    else:
                        for strategy, dots in selected.groupby("strategy"):
                            ax.scatter(
                                dots.disagreement,
                                dots.regret,
                                marker=marker[strategy],
                                color=color[strategy],
                                alpha=0.65,
                                label=strategy,
                            )
                        ax.set_xlabel(
                            "Patient assignment disagreement fraction (tie-break dependent)"
                        )
                    thresholds(ax, family == "assignment-regret")
                    ax.set_ylabel(regret_label + " [load units]")
                    ax.set_title(f"{mid} | {mode} | {diagnostic}\n{title_base}")
                    ax.legend()
                    save(fig, f"{family}_{group_index}_{page}")
        elif family == "profile-effect":
            # First reduce draws within each instance, then connect the same chunk across profiles.
            means = data.groupby(
                ["model_id", "objective", "strategy", "chunk_id", "profile_id", "patient_count"],
                as_index=False,
            )[["regret", "benchmark_range", "count_range"]].mean()
            order = list(config["profiles"])
            preferred = [p for p in ("balanced", "unbalanced", "highly_unbalanced") if p in order]
            order = preferred + [p for p in order if p not in preferred]
            for group_index, ((mid, mode, strategy, size), part) in enumerate(
                means.groupby(["model_id", "objective", "strategy", "patient_count"])
            ):
                chunks = sorted(part.chunk_id.unique())
                for page, start in enumerate(range(0, len(chunks), args.max_instances_per_figure)):
                    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
                    for chunk in chunks[start : start + args.max_instances_per_figure]:
                        subset = (
                            part.loc[part.chunk_id == chunk].set_index("profile_id").reindex(order)
                        )
                        for ax, metric, label in zip(
                            axes,
                            ("regret", "benchmark_range", "count_range"),
                            (
                                regret_label + " [load units]",
                                "Benchmark-evaluated range [load units]",
                                "Final count range [patients]",
                            ),
                        ):
                            ax.plot(order, subset[metric], marker="o", alpha=0.7, label=chunk)
                            ax.set_ylabel(label)
                            ax.tick_params(axis="x", rotation=35)
                    thresholds(axes[0])
                    axes[-1].legend(title="chunk")
                    fig.suptitle(
                        f"{mid} | {mode} | {strategy} | chunk size {size}; per-instance means\n{title_base}"
                    )
                    save(fig, f"{family}_{group_index}_{page}")
        else:
            eligible = data.loc[data.strategy.isin(["random", "robust"])]
            paired = []
            for key, part in eligible.groupby(
                ["model_id", "objective", "instance", "stress_set_id", "heldout_set_id"], sort=True
            ):
                if {"random", "robust"} <= set(part.strategy):
                    paired.append((key, part))
            if not paired:
                omitted.append("robustness: no paired random/robust selections")
            for page, start in enumerate(range(0, len(paired), args.max_instances_per_figure)):
                entries = paired[start : start + args.max_instances_per_figure]
                # Each row is a distinct instance/stress set. No pooling across different scenarios.
                has_heldout = any(p.heldout_w.notna().any() for _, p in entries)
                metrics = ["stress_w", "regret"] + (["heldout_w"] if has_heldout else [])
                fig, axes = plt.subplots(
                    len(entries),
                    len(metrics),
                    figsize=(5 * len(metrics), 3 * len(entries)),
                    squeeze=False,
                )
                for row_index, (key, part) in enumerate(entries):
                    for column, metric in enumerate(metrics):
                        ax = axes[row_index, column]
                        random_mean = part.loc[part.strategy == "random", metric].mean()
                        robust_values = part.loc[part.strategy == "robust", metric].dropna()
                        for value in robust_values:
                            ax.plot(
                                [0, 1],
                                [random_mean, value],
                                marker="o",
                                color=color["robust"],
                                alpha=0.65,
                            )
                        ax.set_xticks([0, 1], ["Random mean", "Robust selection"])
                        ax.set_ylabel(
                            {
                                "stress_w": "In-set worst range W [load units]",
                                "regret": regret_label + " [load units]",
                                "heldout_w": "Held-out W [load units]",
                            }[metric]
                        )
                        ax.set_title(
                            f"{key[0]} | {key[1]} | {key[2]}\n"
                            f"{'held-out' if metric == 'heldout_w' else 'construction'} stress set "
                            f"{key[4][:8] if metric == 'heldout_w' else key[3][:8]}"
                            + (" | ORACLE" if part.oracle.any() else "")
                        )
                        if metric == "regret":
                            thresholds(ax)
                fig.suptitle(title_base, y=1.01)
                save(fig, f"{family}_{page}")
    write_json(
        output / "plot_manifest.json",
        {
            "source_run": config["run_fingerprint"],
            "source_hashes": {
                name: file_hash(args.results / name)
                for name in ("solutions.csv", "regret_bounds.csv")
            },
            "options": vars(args),
            "files": files,
            "omitted": omitted,
            "dot_subsampling": False,
            "plot_seed": args.plot_seed,
            "statistics": "Draws aggregated within instances first; no confidence intervals; unequal chunk sizes stratified in profile plots; overview combines all chunks with explicitly equal chunk weights",
        },
    )
    print(f"Saved {len(files)} plots to {output}")
    return 0
