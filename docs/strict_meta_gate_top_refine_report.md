# Strict Meta-Gate Top Refinement Report

## Result

This pass found a model that improves both train OOF and held-out relative to the previous safe procedure-family gate.

```text
strict PLS baseline:
train OOF MAE: 32.7964177885
held-out MAE: 31.7702636563

previous safe family gate:
train OOF MAE: 32.7692796199
held-out MAE: 31.6075223696

new OOF-selected top-refine gate:
train OOF MAE: 32.7156839203
held-out MAE: 31.5564180731
```

The new model improves:

```text
vs strict PLS held-out:       +0.2138455833 MAE
95% CI vs strict PLS:         [+0.1006, +0.3262]

vs procedure-family held-out: +0.0511042965 MAE
95% CI vs procedure-family:   [-0.0108, +0.1123]
```

The improvement over the strict PLS model is clearly positive on this held-out set. The improvement over the 31.6075 family gate is directionally positive but still modest.

## Selected Model

Selected by train OOF:

```text
topref_pair__pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00__signed_resid_hgb8_x_family_cap_gamma__delta_hgb4_proc6_none_n80_m0.02
```

Readable version:

1. Start with candidate A:

```text
pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00
```

This is the prior strict meta-gate OOF winner. It blends the procedure-family model with a risk-family model using patient-crossfit OOF logic.

2. Blend candidate A with candidate B:

```text
signed_resid_hgb8_x_family_cap_gamma
```

3. The blend gate is based on:

```text
delta_hgb4_proc = risk_abs_hgb4_x_family_cap_gamma - procedure_family_gamma_positive
```

4. `delta_hgb4_proc` is split into 6 quantile bins. The bin edges learned from training were:

```text
-16.7446
-1.2492
-0.3816
-0.0305
 0.0751
 0.6576
16.2156
```

5. The global blend uses 37% of the signed-residual candidate:

```text
prediction = 0.63 * candidate_A + 0.37 * signed_resid_hgb8_x_family
```

6. Three bins override the global weight:

```text
delta_hgb4_proc6_0: signed-residual weight = 1.00
delta_hgb4_proc6_1: signed-residual weight = 0.00
delta_hgb4_proc6_5: signed-residual weight = 0.00
other bins:          signed-residual weight = 0.37
```

Interpretation: when the risk-HGB family model is much lower than the procedure-family model, fully trust the signed-residual candidate; in two other disagreement bins, stay with candidate A; otherwise use a moderate blend.

## Leakage Validation

This selected model is leakage-controlled:

- It does not read the old 11 LLM feature file.
- The final top-refine script reads only saved OOF/test prediction CSVs from the strict meta-gate stage.
- Upstream candidates were produced from the strict tabular + pre-cutoff-note PLS manifold pipeline.
- Train/held-out patient overlap is 0.
- The top-refine blend weights are selected inside patient GroupKFold splits.
- For each OOF validation fold, score-bin edges and local blend weights are learned only from the fold's training side.
- Held-out outcomes are used only for final scoring.

This is feasible to reproduce: run the strict PLS manifold pipeline, run the strict risk/family meta-gate stage, then run this top-refine pair-blend stage.

## What Improved

Held-out gains by procedure family for the OOF-selected model:

```text
AF/PVI ablation:                    +0.7341 vs strict, +0.1522 vs family gate
Other ablation:                     +0.1480 vs strict, +0.1484 vs family gate
Diagnostic/testing:                 +0.2783 vs strict, +0.0487 vs family gate
VT/PVC ablation:                    +0.0437 vs strict, +0.0175 vs family gate
Device implant:                     -0.0424 vs strict, -0.0022 vs family gate
Device revision-generator-upgrade:  +0.0734 vs strict, -0.0082 vs family gate
Lead/device extraction-removal:     -0.0959 vs strict, -0.3367 vs family gate
```

The strongest reliable gains are still in ablation families, especially AF/PVI.

## Exploratory Held-Out Best

The best held-out row from the fine-grid search was:

```text
topref_pair__procedure_family_gamma_positive__signed_resid_hgb8_x_family_cap_gamma__candidate_spread7_family_n80_m0.02
train OOF MAE: 32.7806355030
held-out MAE: 31.5234029457
```

This is not the selected final candidate because its train OOF is worse than the procedure-family gate. It is a promising lockbox candidate only.

## Recommendation

Use the OOF-selected top-refine model as the current best leakage-free candidate:

```text
train OOF MAE: 32.7156839203
held-out MAE: 31.5564180731
```

Report the 31.5234 model only as exploratory. It is useful evidence that more signal exists, but selecting it from held-out would not be clean.

## Output Files

```text
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine.py
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine_results.csv
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine_summary.json
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine_selected_predictions.csv
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine_candidate_predictions_slim.csv
outputs/strict_meta_gate_top_refine/strict_meta_gate_top_refine_report.md
```

