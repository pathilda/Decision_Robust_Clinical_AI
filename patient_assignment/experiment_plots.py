"""Plots from saved tables only. No solver imports or initialization."""

import json
import math

import pandas as pd

from .inputs import file_hash
from .reporting import fresh_directory, write_json


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

    def save(fig, name):
        fig.tight_layout()
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
        if family in {"regret-envelope", "assignment-regret"}:
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
            "statistics": "Draws aggregated within instances first; no confidence intervals; unequal chunk sizes stratified in profile plots",
        },
    )
    print(f"Saved {len(files)} plots to {output}")
    return 0
