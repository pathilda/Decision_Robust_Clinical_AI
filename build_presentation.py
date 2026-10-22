"""Build the supplied discussion slides as a Quarto presentation using saved HTML plots.

No inference or optimization is performed; missing assignment scatter uses saved data only.
"""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "presentations" / "decision_robust"
PLOTS = {
    "regret_qwk": ("regret_vs_qwk", "Regret / QWK"),
    "regret_across": ("regret_across_chunks", "Regret / model"),
    "assignment_across": ("assignment_across_chunks", "Assignment / model"),
}
OPTIONAL = {"assignment_qwk": ("assignment_vs_qwk", "Assignment / QWK")}
SECTIONS = {
    "ACCURACY_PLOTS": ["regret_qwk", "assignment_qwk"],
    "DISTRIBUTION_PLOTS": ["regret_across", "assignment_across"],
}


def assignment_scatter(directory):
    """Fill the missing plot strictly from already saved allocation rows and QWK."""
    import pandas as pd

    from proxy_extraction_scatter import build_scatter, scatter_page

    root = Path(directory).resolve()
    if root.name == "figures":
        root = root.parent
    results = next(
        (root / n for n in ["combined_observations.csv", "results.csv"] if (root / n).is_file()),
        None,
    )
    scores = next(
        (root / n for n in ["extraction_qwk.csv", "accuracy_vs_proxy.csv"] if (root / n).is_file()),
        None,
    )
    if not results or not scores:
        raise ValueError(
            "assignment_vs_qwk.html is missing. Supply a full saved report directory "
            "containing results.csv or combined_observations.csv and extraction_qwk.csv "
            "or accuracy_vs_proxy.csv, or supply the prebuilt plot with --plot-map."
        )
    frame = pd.read_csv(results, dtype=str, keep_default_na=False)
    accuracy = pd.read_csv(scores, dtype={"model_id": str}, keep_default_na=False)
    config = root / "resolved_config.json"
    resolved = (
        json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {"profiles": {}}
    )
    fig = build_scatter(frame, accuracy, resolved, "qwk", outcome="exact_assignment_pct")
    return scatter_page(fig, "qwk"), {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [results, scores]
    }


def resolve_plots(directory, style="Box", overrides=None):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Plot directory does not exist: {directory}")
    overrides = overrides or {}
    unknown = overrides.keys() - (PLOTS.keys() | OPTIONAL.keys())
    if unknown:
        raise ValueError(f"Unknown plot-map keys: {sorted(unknown)}")
    found, missing = {}, []
    for key, (stem, _) in (PLOTS | OPTIONAL).items():
        if key in overrides:
            path = (directory / overrides[key]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError(f"Plot-map path must stay within --plots-dir: {key}")
            candidates = [path] if path.is_file() else []
        else:
            name = (
                f"{stem}_{style}.html" if key.endswith(("across", "by_chunk")) else f"{stem}.html"
            )
            candidates = [
                p for p in (directory / name, directory / "figures" / name) if p.is_file()
            ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous plot {key}; specify its relative path in --plot-map")
        if not candidates:
            if key in PLOTS or key in overrides:
                raise ValueError(
                    f"Missing plot {key} in {directory}; use --plot-style or --plot-map"
                )
            missing.append(key)
            continue
        path = candidates[0].resolve()
        if not path.is_relative_to(directory) or path.suffix.lower() != ".html":
            raise ValueError(f"Expected an HTML plot inside --plots-dir: {path}")
        found[key] = path
    return found, missing


def tabs(section, keys, found):
    chosen = [key for key in keys if key in found]
    controls, panels = [], []
    for index, key in enumerate(chosen):
        title = html.escape((PLOTS | OPTIONAL)[key][1])
        selected = index == 0
        panel_id = section.lower() + "-" + key
        controls.append(
            f'<button type="button" role="tab" id="tab-{panel_id}" '
            f'aria-controls="{panel_id}" aria-selected="{str(selected).lower()}" '
            f'tabindex="{0 if selected else -1}">{title}</button>'
        )
        panels.append(
            f'<div class="plot-panel" role="tabpanel" id="{panel_id}" '
            f'aria-labelledby="tab-{panel_id}"' + ("" if selected else " hidden") + ">"
            f'<iframe data-plot-url="plots/{key}.html" title="{title}" loading="lazy" '
            "allowfullscreen></iframe></div>"
        )
    return (
        '```{=html}\n<div class="plot-tabs">\n'
        '<div class="plot-tab-controls" role="tablist" aria-label="Result figures">'
        + "".join(controls)
        + f'<a class="plot-open" href="plots/{chosen[0]}.html" target="_blank" rel="noopener">'
        "Open full-size</a></div>" + "".join(panels) + "\n</div>\n```"
    )


def build(plots_dir, output, style="Box", plot_map=None, quarto=None, prepare_only=False):
    found, missing = resolve_plots(plots_dir, style, plot_map)
    generated = None
    if "assignment_qwk" not in found:
        generated = assignment_scatter(plots_dir)
        found["assignment_qwk"] = None
        missing = []
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Choose a new output folder; refusing to overwrite {output}")
    executable = quarto or shutil.which("quarto")
    if not prepare_only and not executable:
        raise ValueError(
            "Quarto CLI is required. Install from https://quarto.org/docs/get-started/ "
            "or pass --quarto PATH. Use --prepare-only to create editable sources only."
        )
    if executable and not Path(executable).is_file():
        executable = shutil.which(executable)
        if not executable:
            raise ValueError("Cannot find the specified Quarto executable")
    if executable:
        executable = str(Path(executable).resolve())
    source = (TEMPLATE / "slides.qmd").read_text(encoding="utf-8-sig")
    for section, keys in SECTIONS.items():
        source = source.replace("@@" + section + "@@", tabs(section, keys, found))
    if "@@" in source:
        raise ValueError("Unresolved presentation placeholder")
    output.mkdir(parents=True)
    (output / "plots").mkdir()
    (output / "source").mkdir()
    (output / "slides.qmd").write_text(source, encoding="utf-8")
    for name in ["theme.scss", "interactions.html"]:
        shutil.copy2(TEMPLATE / name, output / name)
    shutil.copy2(TEMPLATE / "source" / "Slides.tex", output / "source" / "Slides.tex")
    manifest = {
        "plot_style": style,
        "rendered": False,
        "plots": {},
        "missing_optional_plots": missing,
        "source_tex_sha256": hashlib.sha256(
            (TEMPLATE / "source" / "Slides.tex").read_bytes()
        ).hexdigest(),
        "notes": [
            "Original text and equations preserved except user-requested removal of Analysis III and narrowed plot captions.",
            "The original iff?alse material is excluded and retained in source/Slides.tex.",
            "Plot captions are preserved; tabs occupy the original figure placeholders.",
            "Missing assignment-vs-QWK HTML is generated only from saved rows and QWK, never invented.",
            "Existing plot files are copied unchanged; this build never runs inference or optimization.",
        ],
    }
    for key, path in found.items():
        if path is None:
            target = output / "plots" / f"{key}.html"
            target.write_text(generated[0], encoding="utf-8")
            manifest["plots"][key] = {
                "generated_from_saved_data": generated[1],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            continue
        shutil.copy2(path, output / "plots" / f"{key}.html")
        manifest["plots"][key] = {
            "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "README.md").write_text(
        "# Presentation\n\nOpen `slides.html` in a browser. Share this entire folder: "
        "the `plots/` files must remain alongside it.\n\n"
        "Use arrow keys outside embedded plots to navigate, Escape for overview, "
        "and the plot tabs or Open full-size links for results. "
        "Scroll view is available by adding `?view=scroll` before the URL hash.\n\n"
        "Edit `slides.qmd` for content and `theme.scss` for style; rerender with "
        "`quarto render slides.qmd`. These text sources can be collaboratively edited "
        "through version control; the HTML is a presentation, not a live coediting service.\n\n"
        "For local serving: `python -m http.server 8000 --directory .` and open "
        "`http://localhost:8000/slides.html`. No inference or solver is used.\n",
        encoding="utf-8",
    )

    def save_manifest():
        (output / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    save_manifest()
    if not prepare_only:
        completed = subprocess.run(
            [str(executable), "render", "slides.qmd"], cwd=output, check=False
        )
        if completed.returncode:
            raise RuntimeError(f"Quarto rendering failed; editable sources remain at {output}")
        if not (output / "slides.html").is_file():
            raise RuntimeError("Quarto did not produce slides.html")
        manifest["rendered"] = True
        save_manifest()
    if missing:
        print("Not available in this plot directory (not fabricated): " + ", ".join(missing))
    print(
        ("Prepared editable sources: " if prepare_only else "Presentation: ")
        + str(output / ("slides.qmd" if prepare_only else "slides.html"))
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir", type=Path, required=True, help="Report directory or its figures/ directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-style", choices=["Box", "Point"], default="Box")
    parser.add_argument(
        "--plot-map",
        type=Path,
        help="Optional JSON object mapping plot keys to relative HTML paths",
    )
    parser.add_argument("--quarto", help="Path to Quarto executable when not on PATH")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write QMD, assets and plot copies without rendering",
    )
    args = parser.parse_args()
    try:
        overrides = json.loads(args.plot_map.read_text(encoding="utf-8")) if args.plot_map else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("--plot-map must contain a JSON object")
        build(
            args.plots_dir, args.output, args.plot_style, overrides, args.quarto, args.prepare_only
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
