# Strict Risk-Family K Sweep Report

## Purpose

This pass tested whether using more than 3 learned risk families could improve the clean strict PLS manifold model, while preserving leakage rules.

The core idea was:

```text
prediction = base + gamma(category) * clip(strict_pls_prediction - base, -cap(category), cap(category))
```

where categories can be procedure family, learned risk family, learned risk x procedure family, or related conservative gates.

## Leakage Validation

This experiment is leakage-controlled:

- No old 11 LLM feature file was read.
- The only imported data sources were the strict OOF/test prediction artifacts, the tabular parquet, and the strict pre-cutoff note bundle metadata.
- The old LLM file name appears only as a forbidden-file sentinel in the shared loader.
- Train/held-out patient overlap remained 0.
- Score models for risk/gain/direction were trained inside patient GroupKFold splits.
- For each outer validation fold, learned score bins were built using only that fold's training side.
- Category gammas/caps were tuned using only the outer training fold.
- Held-out outcomes were used only for final scoring and exploratory reporting.

## Baselines

```text
strict PLS model:
held-out MAE: 31.7702636563

current safest calibrated model:
procedure_family_gamma_positive
train OOF MAE: 32.7692796199
held-out MAE: 31.6075223696
gain vs strict PLS: +0.1627412868
95% CI: [+0.0581, +0.2651]
```

## Score Model Validity

The risk magnitude models were learnable:

```text
risk_abs_extra_trees OOF corr with abs error: 0.4569
risk_abs_hgb OOF corr with abs error:         0.4519
```

The tail/direction/gain models were weak:

```text
tail30_hgb OOF corr with tail error:          0.0318
gain_hgb OOF corr with final-vs-base gain:    0.0823
signed_resid_hgb OOF corr with residual:      0.0793
```

Interpretation: the features can identify high-variance/high-error cases, but they still do not reliably tell us whether the PLS correction should be strengthened, shrunk, or reversed.

## K > 3 Risk-Family Results

The most tempting result was a 4-risk-family HGB risk x procedure-family gate:

```text
risk_abs_hgb4_x_family_cap_gamma
train OOF MAE: 32.8003443076
held-out MAE: 31.5790109028
gain vs strict PLS: +0.1912527535
95% CI: [+0.0691, +0.3161]
improved held-out cases: 49.17%
large wins >10 min: 27
large losses <-10 min: 15
```

But it did not validate by OOF. Its train OOF MAE was worse than both the safe procedure-family model and even the strict final OOF baseline. Therefore this should be treated as exploratory, not final.

K-risk-family table:

```text
risk_abs_hgb4_x_family_cap_gamma   train OOF 32.8003   held-out 31.5790
risk_abs_hgb5_x_family_cap_gamma   train OOF 32.7713   held-out 31.6764
risk_abs_hgb6_x_family_cap_gamma   train OOF 32.7573   held-out 31.6567
risk_abs_hgb8_x_family_cap_gamma   train OOF 32.8127   held-out 31.7497

risk_abs_et3_x_family_cap_gamma    train OOF 32.7731   held-out 31.5857
risk_abs_et4_x_family_cap_gamma    train OOF 32.7693   held-out 31.6855
risk_abs_et5_x_family_cap_gamma    train OOF 32.7796   held-out 31.6762
risk_abs_et6_x_family_cap_gamma    train OOF 32.7863   held-out 31.7022
risk_abs_et8_x_family_cap_gamma    train OOF 32.7906   held-out 31.7030
```

Conclusion for k > 3: there is signal, but no stable validated improvement. The best held-out result comes from k=4, but the OOF behavior says not to crown it.

## Strict OOF Winner

The best model by train OOF in this expanded pass was:

```text
gain_hgb6_x_family_cap_gamma
train OOF MAE: 32.7543106975
held-out MAE: 31.7008146577
gain vs strict PLS: +0.0694489986
95% CI: [-0.0455, +0.1835]
```

This is leakage-free, but it does not transfer well. Because the gain score itself had only about 0.08 OOF correlation with true gain, I do not recommend it.

## Other Methods

### Tail-Risk Gates

Tail-risk gates were leakage-free, but the tail-risk target was not learnable enough. Most tail-risk x family variants collapsed back to the capped family gate:

```text
tail30_hgb4_x_family_cap_gamma: held-out 31.6288
tail30_hgb6_x_family_cap_gamma: held-out 31.6288
tail30_hgb8_x_family_cap_gamma: held-out 31.6288
```

Verdict: feasible and clean, but not useful.

### Signed-Residual Gates

Signed-residual gates are the right conceptual direction, but the current features do not predict residual direction strongly enough:

```text
signed_resid_hgb4_x_family_cap_gamma: held-out 31.6849
signed_resid_hgb6_x_family_cap_gamma: held-out 31.6792
signed_resid_hgb8_x_family_cap_gamma: held-out 31.6439
```

Verdict: clean, feasible, and worth revisiting only if we add a better directional feature source.

### Correction-Size And PLS-Gap Interactions

Adding correction-size or PLS-gap interactions generally over-segmented the data:

```text
risk_abs_hgb4_x_family_x_corr_size_cap_gamma: held-out 31.7973
risk_abs_et4_x_family_x_pls_gap_cap_gamma:    held-out 31.7579
signed_resid_hgb6_x_family_x_pls_gap:         held-out 31.8115
```

Verdict: leakage-free but too fragmented for the current sample size.

### OOF Blends

Simple two-way OOF-tuned blends were clean but did not beat the safe family model:

```text
blend procedure_family + risk_abs_et4_family:        held-out 31.6286
blend procedure_family + risk_abs_et6_family:        held-out 31.6208
blend procedure_family + signed_resid_hgb6_family:   held-out 31.6203
```

The positive linear stack failed:

```text
positive_linear_stack_selected_gates: held-out 32.2627
```

Verdict: no replacement for the family gate.

## Where The Exploratory K=4 Risk Gate Helped

For `risk_abs_hgb4_x_family_cap_gamma`, post-hoc gains by procedure family were:

```text
AF/PVI ablation:       +0.7912 MAE
VT/PVC ablation:       +0.2731 MAE
Diagnostic/testing:    +0.2659 MAE
Device implant:        -0.0327 MAE
Device revision:       -0.0759 MAE
Other ablation:        -0.1701 MAE
Lead/device removal:   -0.2695 MAE
```

Post-hoc gains by strict-final error size:

```text
0-15 min final error:      -0.3006 MAE
15-30 min final error:     +0.3526 MAE
30-60 min final error:     +0.6803 MAE
60-120 min final error:    +0.3198 MAE
120+ min final error:      +0.7342 MAE
```

So the model does attack the high-error cases, but it still hurts some already-good cases and does not validate by OOF.

## Recommendation

Do not replace the current final model with a k-risk-family gate yet.

The safest current result remains:

```text
procedure_family_gamma_positive
held-out MAE: 31.6075223696
```

The best exploratory candidate for a fresh lockbox is:

```text
risk_abs_hgb4_x_family_cap_gamma
held-out MAE: 31.5790109028
```

But because it was not selected by train OOF, it should be labeled exploratory. The honest interpretation is:

```text
More than 3 risk families can expose useful variance structure,
but the current validation does not prove a stable improvement over the simpler family gate.
```

## Output Files

```text
outputs/strict_risk_family_k_sweep/strict_risk_family_k_sweep.py
outputs/strict_risk_family_k_sweep/strict_k_sweep_results.csv
outputs/strict_risk_family_k_sweep/strict_k_sweep_summary.json
outputs/strict_risk_family_k_sweep/strict_k_sweep_test_predictions.csv
outputs/strict_risk_family_k_sweep/strict_k_sweep_candidate_predictions_slim.csv
outputs/strict_risk_family_k_sweep/strict_risk_family_k_sweep_report.md
```

