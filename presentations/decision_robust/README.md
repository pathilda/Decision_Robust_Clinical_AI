# Discussion presentation

The editable Quarto conversion of the supplied `Slides.tex` uses its wording,
equations, colors and active slide order. As requested, Analysis III is removed.
The remaining deck has 16 slides, including the unchanged blank Next Possible
Steps slide. The original source, including excluded `iffalse` material, is
retained in `source/Slides.tex`. No new empirical claims are introduced.

## Build with your results

Install the Quarto CLI: https://quarto.org/docs/get-started/ . Run from the repository:

```bash
python build_presentation.py \
  --plots-dir Inital_Results/Qwen_CoT_Run7_Combined_WithScatters \
  --output presentations/generated/run7
```

The default expects Box plots. `--plots-dir` accepts a complete report directory
or its `figures/` subdirectory. Open `presentations/generated/run7/slides.html`.
Each build needs a new output directory; existing reports are never overwritten.
No optimization or inference is run. Quarto is not a pip package.

The deck uses exactly four figures:

| Slide | Figures |
|---|---|
| Analysis I | Regret vs QWK; assignment agreement vs QWK |
| Analysis II | Regret vs model/policy; assignment agreement vs model/policy (Box) |

Analysis II uses the existing `*_across_chunks_Box.html` views and their original
sampling convention: the designated chunk-evaluation seed. Analysis I retains
all saved certified allocation rows from the scatter report. Filters and hover
remain interactive; QWK is the model's cohort-level score, not a separate value
for each allocation.

Required saved plot filenames: `regret_vs_qwk.html`,
`regret_across_chunks_Box.html`, `assignment_across_chunks_Box.html`.
If `assignment_vs_qwk.html` is missing, it is generated offline from
`combined_observations.csv` or `results.csv` and `extraction_qwk.csv` or
`accuracy_vs_proxy.csv` in the report directory. This calculation does not read
original inference files or run the solver. If a `resolved_config.json` is
present, initial clinician caseloads are also included in hover. The profile
filter is available either way. If only HTML plots were copied, include the
assignment scatter HTML too, or supply the full report directory.

A JSON `--plot-map` can explicitly map `regret_qwk`, `assignment_qwk`,
`regret_across`, `assignment_across` to relative HTML paths inside `--plots-dir`.
Ambiguous or missing plots fail before rendering. Do not map Point plots to Box
slots: the builder preserves existing plot content without changing its meaning.

## Present and collaborate

- Arrow keys outside the embedded plots navigate slides; Escape opens overview.
- Each results slide has two tabs, plus an Open full-size link.
- Use `slides.html?view=scroll` for scrolling mode.
- Share the entire generated directory, including `plots/`, not only slides.html.
- Edit `slides.qmd`, `theme.scss` and `interactions.html`; rerender with
  `quarto render slides.qmd`. These sources work with Git/shared editing; no live
  multiuser coediting service is installed.
- If local-file embedding is blocked, use `python -m http.server 8000 --directory
  presentations/generated/run7` and open http://localhost:8000/slides.html .
- `--quarto PATH` accepts a portable CLI (on Windows prefer `quarto.exe`).
- `--prepare-only` copies the plots and produces editable QMD without rendering.

`build_manifest.json` records plot source paths, hashes, and any saved CSVs used
to generate the assignment scatter. Existing plot HTML is copied byte-for-byte.
Generated presentation bundles can be large because each original Plotly HTML
contains its own embedded plotting library.

The title date and blinded-author line are omitted. An interactive analysis roadmap
follows the regret definition. Hover, keyboard focus and click reveal precise stage
details in a fixed panel below the diagram; no external libraries are required.
