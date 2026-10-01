"""Write publication-ready tables and figures for an agreement comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .analysis import ComparisonResults


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


CATEGORY_COLORS = ("#4C78A8", "#F2CF5B", "#E45756")


def write_report(
    results: ComparisonResults,
    *,
    output_dir: Path,
    categories: tuple[str, ...],
    cohort: str,
    id_column: str,
    category_column: str,
    title: str,
) -> list[Path]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [
        _write_csv(destination / "category_distributions.csv", results.distributions),
        _write_matrix_csv(
            destination / "exact_agreement_matrix.csv",
            results.exact_agreement,
        ),
        _write_matrix_csv(
            destination / "quadratic_weighted_kappa_matrix.csv",
            results.quadratic_weighted_kappa,
        ),
        _write_matrix_csv(destination / "pairwise_sample_size_matrix.csv", results.pairwise_n),
        _write_csv(destination / "input_summary.csv", results.input_summary),
    ]
    config_path = destination / "comparison_config.json"
    config: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "cohort": cohort,
        "analysis_client_count": results.analysis_client_count,
        "id_column": id_column,
        "category_column": category_column,
        "ordered_categories": list(categories),
        "exact_agreement_unit": "percent",
        "kappa_weighting": "quadratic",
        "inputs": results.input_summary.to_dict(orient="records"),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(config_path)

    summary_path = destination / "README.md"
    summary_path.write_text(
        _summary_markdown(results, title=title, cohort=cohort),
        encoding="utf-8",
    )
    outputs.append(summary_path)

    distribution_plot = destination / "category_distributions.png"
    _plot_distributions(
        results.distributions,
        categories=categories,
        title=f"{title}: category distributions",
        destination=distribution_plot,
    )
    outputs.append(distribution_plot)

    agreement_plot = destination / "exact_agreement_heatmap.png"
    _plot_heatmap(
        results.exact_agreement,
        title=f"{title}: exact agreement (%)",
        destination=agreement_plot,
        value_format=".1f",
        value_suffix="%",
        vmin=0,
        vmax=100,
        cmap="Blues",
    )
    outputs.append(agreement_plot)

    kappa_plot = destination / "quadratic_weighted_kappa_heatmap.png"
    _plot_heatmap(
        results.quadratic_weighted_kappa,
        title=f"{title}: quadratic-weighted Cohen's kappa",
        destination=kappa_plot,
        value_format=".3f",
        value_suffix="",
        vmin=-1,
        vmax=1,
        cmap="RdYlBu",
    )
    outputs.append(kappa_plot)
    return outputs


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def _write_matrix_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=True, float_format="%.6f")
    return path


def _plot_distributions(
    frame: pd.DataFrame,
    *,
    categories: tuple[str, ...],
    title: str,
    destination: Path,
) -> None:
    configurations = frame["Configuration"].astype(str).tolist()
    figure_height = max(3.6, 0.55 * len(configurations) + 1.8)
    figure, axis = plt.subplots(figsize=(9.5, figure_height), constrained_layout=True)
    left = np.zeros(len(frame))
    colors = [CATEGORY_COLORS[index % len(CATEGORY_COLORS)] for index in range(len(categories))]
    for category, color in zip(categories, colors, strict=True):
        values = frame[f"{category} %"].astype(float).to_numpy()
        axis.barh(configurations, values, left=left, label=category, color=color)
        for row, value in enumerate(values):
            if value >= 6:
                text_color = "#1F2937" if color == "#F2CF5B" else "white"
                axis.text(
                    left[row] + value / 2,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )
        left += values
    axis.set_xlim(0, 100)
    axis.set_xlabel("Clients (%)")
    axis.set_ylabel("")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(3, len(categories)))
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(destination, dpi=300, facecolor="white")
    plt.close(figure)


def _plot_heatmap(
    frame: pd.DataFrame,
    *,
    title: str,
    destination: Path,
    value_format: str,
    value_suffix: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    values = frame.to_numpy(dtype=float)
    size = max(6.2, 0.9 * len(frame) + 2.4)
    figure, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    color_map = matplotlib.colormaps[cmap].copy()
    color_map.set_bad("#E6E6E6")
    image = axis.imshow(np.ma.masked_invalid(values), cmap=color_map, vmin=vmin, vmax=vmax)
    labels = frame.index.astype(str).tolist()
    axis.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = values[row, column]
            text = "—" if np.isnan(value) else f"{value:{value_format}}{value_suffix}"
            if np.isnan(value):
                text_color = "#1F2937"
            else:
                scaled = np.clip((value - vmin) / (vmax - vmin), 0, 1)
                red, green, blue, _ = color_map(scaled)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.5 else "#1F2937"
            axis.text(column, row, text, ha="center", va="center", color=text_color)
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    color_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    color_bar.outline.set_visible(False)
    figure.savefig(destination, dpi=300, facecolor="white")
    plt.close(figure)


def _summary_markdown(
    results: ComparisonResults,
    *,
    title: str,
    cohort: str,
) -> str:
    if cohort == "intersection":
        cohort_note = (
            f"All tables use the {results.analysis_client_count} clients present in every input."
        )
    else:
        cohort_note = "Each comparison uses the clients available in that pair."
    return (
        f"# {title}\n\n"
        f"{cohort_note}\n\n"
        "This report describes agreement among model configurations; without reference labels, "
        "it does not measure accuracy or identify a best model.\n\n"
        "- `category_distributions.csv`: category counts and percentages by configuration.\n"
        "- `exact_agreement_matrix.csv`: pairwise exact agreement in percent.\n"
        "- `quadratic_weighted_kappa_matrix.csv`: pairwise quadratic-weighted Cohen's kappa.\n"
        "- `pairwise_sample_size_matrix.csv`: number of shared clients used in each cell.\n"
        "- PNG files: publication-ready versions of the distribution and matrix figures.\n"
    )
