# Reusable model agreement comparison

This command compares any two or more ordinal classification outputs without treating any model
as ground truth. Every input must contain one unique row per client and, by default, these columns:

```text
Client AlayaCare Client ID
treatment_category
```

## Five-configuration comparison

From the repository root:

```bash
python -m model_comparison \
  --input "QS=/path/to/results/qwen/client_classifications.xlsx" \
  --input "QR=/path/to/results/qwen_brief_reasoning/client_classifications.xlsx" \
  --input "MS=/path/to/results/medgemma/client_classifications.xlsx" \
  --input "MR=/path/to/results/medgemma_brief_reasoning/client_classifications.xlsx" \
  --input "OD=/path/to/results/opendecider/client_classifications.xlsx" \
  --output /path/to/results/model_comparison \
  --title "Treatment-difficulty model agreement"
```

The default `--cohort intersection` uses only client IDs present in all inputs. This gives every
distribution and matrix cell the same client cohort. Use `--cohort pairwise` only when retaining
the maximum available sample for each pair is more important than a common analysis cohort.

## Outputs

```text
category_distributions.csv
category_distributions.png
exact_agreement_matrix.csv
exact_agreement_heatmap.png
quadratic_weighted_kappa_matrix.csv
quadratic_weighted_kappa_heatmap.png
pairwise_sample_size_matrix.csv
input_summary.csv
comparison_config.json
README.md
```

Exact agreement is stored as a percentage. Cohen's kappa uses quadratic weights and the category
order supplied to the command. The default order is Easy, Moderate, Severe.

## Future comparisons

Repeat `--input "LABEL=PATH"` for any number of configurations. CSV, Excel, and Parquet inputs are
accepted. For a different ordinal task, repeat `--category` from lowest to highest level:

```bash
python -m model_comparison \
  --input "Model-A=/path/a.csv" \
  --input "Model-B=/path/b.csv" \
  --category low \
  --category medium \
  --category high \
  --output /path/to/comparison
```

This is an agreement analysis. Without external reference labels, it cannot establish accuracy or
identify a best model.
