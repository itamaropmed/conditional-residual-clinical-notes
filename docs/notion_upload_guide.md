# Notion Upload Guide

Use this package as a compact project record for the clean 31.7703 model.

## Suggested Notion Page Structure

Create one top-level page:

```text
Strict 31.7703 No-LLM Duration Model
```

Recommended subpages:

| Subpage | Upload or paste |
| --- | --- |
| Overview | `README.md` |
| Full Model Card | `docs/strict_31_7703_clean_method.md` |
| Post-Hoc Interpretability | `docs/posthoc_interpretability_31_7703_report.md` |
| Source Manifest | `MANIFEST.md` |
| Final Result | `results/final_result.json` |
| Leakage Audit | `results/strict_validation.json` |
| Raw Note Audit | `results/raw_strict_high_signal_audit.json` |
| Interpretability Tables | `results/posthoc_interpretability/` |
| Code | Attach the zip file or upload the `src/` folder files |

## What To Emphasize

The short version for collaborators:

- final MAE is `31.7702636563`
- base models are tabular-only
- final model adds a strict pre-op raw-note residual correction
- old 11 LLM features are not used anywhere
- notes obey `note_date < scheduled_in_room - 2 days`
- train/test split has zero patient overlap
- residual training uses patient-grouped OOF predictions

## Best Upload Option

Upload the zip:

```text
strict_31_7703_project.zip
```

Then paste `README.md` into the Notion page body and attach the zip for reproducibility.
