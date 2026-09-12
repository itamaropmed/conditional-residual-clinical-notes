# Methods And Results: Best 31.5564 Model

## Final Answer

The current best leakage-controlled candidate is:

```text
strict_meta_gate_top_refine_oof_selected
held-out MAE: 31.5564180731
train OOF MAE: 32.7156839203
```

It improves over:

```text
strict PLS 31.7703 model:       +0.2138455833 MAE
procedure-family 31.6075 gate:  +0.0511042965 MAE
```

The held-out gain versus the strict PLS model has a paired bootstrap CI:

```text
[+0.1001, +0.3259]
```

The gain versus the procedure-family gate is smaller:

```text
[-0.0108, +0.1123]
```

## Phase 1: Strict PLS Manifold Model

The base model was:

```text
0.6 * xgb_tuned_v2 + 0.4 * xgb_absoluteerror_same_shape
```

Base inputs were tabular-only features with safe engineering and OOF-safe target encodings.

The strict PLS model added raw clinical note signal using only notes satisfying:

```text
note_date < scheduled_in_room - 2 days
```

The selected strict PLS model:

```text
pls_raw_word160_26_delta__sgd_huber__g1.550
held-out MAE: 31.7702636563
```

It used raw-note word TF-IDF/SVD, tabular SVD, PLS latent alignment, delta/product manifold features, and a Huber residual head.

## Phase 2: Procedure-Family Calibration

The first reliable post-model improvement scaled the strict PLS correction by procedure family:

```text
prediction = base + gamma(procedure_family) * (strict_pls_prediction - base)
```

Result:

```text
procedure_family_gamma_positive
train OOF MAE: 32.7692796199
held-out MAE: 31.6075223696
```

This showed that the PLS correction was useful but needed family-specific trust calibration.

## Phase 3: Risk And Variance Exploration

Several leakage-controlled directions were tested:

- learned absolute-error risk bins
- learned final-vs-base gain bins
- signed residual bins
- correction-size gates
- PLS-gap gates
- family x risk interactions
- capped correction gates
- pairwise and selector meta-gates

The main finding:

```text
risk magnitude is learnable, direction is harder
```

Risk models reached about 0.45 OOF correlation with absolute final error. Direction/gain models were much weaker, usually around 0.08 OOF correlation.

## Phase 4: Strict Meta-Gate Push

The next successful step was a patient-crossfit meta-gate that blended/selected among safe prediction candidates.

The first OOF-and-held-out winner:

```text
pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00
train OOF MAE: 32.751832
held-out MAE: 31.604943
```

This was already a small validated improvement over the procedure-family gate.

## Phase 5: Final Top Refine

The final selected model blends:

```text
candidate A = pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00
candidate B = signed_resid_hgb8_x_family_cap_gamma
```

The blend is gated by:

```text
delta_hgb4_proc = risk_abs_hgb4_x_family_cap_gamma - procedure_family_gamma_positive
```

Six quantile bins were used. The global candidate-B weight is 0.37, with three bin overrides:

```text
delta_hgb4_proc6_0: 1.00
delta_hgb4_proc6_1: 0.00
delta_hgb4_proc6_5: 0.00
other bins:         0.37
```

Final selected result:

```text
train OOF MAE: 32.7156839203
held-out MAE: 31.5564180731
```

## Leakage Validation

The final package obeys the rules:

- no old 11 LLM feature inputs
- no old 11 LLM teacher labels
- no reads of `mayo_hrs_llm_features_top11.parquet`
- strict pre-cutoff raw notes only
- no same-day pre-op notes
- no post-op/intra-op notes
- patient-disjoint train/test split
- train OOF predictions used for train-side learning
- meta-gate weights learned inside patient GroupKFold splits
- held-out target used only for scoring

The audit values for the final selected run:

```text
train rows: 10,882
held-out rows: 2,839
patient overlap: 0
forbidden LLM feature file read: false
```

## Exploratory Not Final

The best held-out-only exploratory row reached:

```text
held-out MAE: 31.5234029457
```

It is not the selected final model because its train OOF MAE was worse than the procedure-family gate. Treat it as a candidate for a fresh lockbox, not as the final claim.

## Recommended Claim

Use this as the current best clean result:

```text
OOF-selected top-refine meta-gate
held-out MAE: 31.5564180731
```

Include the caveat:

```text
Because many variants were explored while observing this held-out set,
a fresh lockbox is still needed for publication-grade confirmation.
```

