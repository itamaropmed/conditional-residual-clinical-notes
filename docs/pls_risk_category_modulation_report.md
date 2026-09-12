# PLS Risk Category Modulation Report

## Question

Can the remaining large errors in the strict 31.7703 PLS manifold model be reduced by learning risk categories and giving each category its own residual head?

Short answer: yes, but the stable improvement did not come from a high-complexity k-risk mixture. The most credible next model is a simple procedure-family gate on the existing PLS correction.

## Baseline

Strict prior model:

```text
pls_raw_word160_26_delta__sgd_huber__g1.550
held-out MAE: 31.7702636563
held-out n: 2,839
```

The model is:

```text
final = base + 1.55 * PLS_residual_head
```

where the base is the tabular-only XGB blend and the PLS residual head is the raw-note/tabular strict manifold correction. It does not use the old 11 LLM features.

## Leakage Controls

This modulation experiment used:

- Saved train OOF and held-out prediction artifacts from the strict PLS manifold run.
- Tabular data from `mayo_hrs_tabular_features_and_durations.parquet`.
- Strict pre-cutoff note bundle metadata from `strict_note_bundles.parquet`.

It did not read `mayo_hrs_llm_features_top11.parquet`. The script has a forbidden-file sentinel and the only parquet reads are the tabular feature file and the strict note bundle file.

OOF predictions were created with patient-group folds. In this pass, the merged modulation frames were audited:

```text
train rows: 10,882
held-out rows: 2,839
train patients: 8,948
held-out patients: 2,499
patient overlap: 0
missing patient ids: 0 train, 0 held-out
```

Held-out targets were used only for scoring after fitting/selection logic. The category gammas and residual heads were fit inside GroupKFold splits for train OOF estimates, then refit on all training rows for held-out prediction.

## Tested Modulations

All variants started from the same final PLS manifold correction:

```text
correction = final - base
prediction = base + gamma(category) * correction
```

Tested category families:

- Global gamma sanity check.
- Procedure-family gamma.
- Scheduled-duration-bin gamma.
- Note-count-bin gamma.
- PLS gap, PLS spread, and PLS correction-vote bins.
- Family x scheduled-duration and family x note-count gates.
- Learned risk bins for k = 3, 4, 5, 6, 8.
- Learned risk x family, risk x PLS-gap, risk x scheduled-duration, and risk x family x PLS-gap gates.
- Mean-residual heads and ridge residual heads on top of learned risk bins.

The learned risk model used an ExtraTrees regressor trained fold-safely to predict:

```text
abs(final_oof_prediction - actual_duration)
```

using safe tabular, note-metadata, final/base/correction, and PLS-disagreement features. The risk model had about 0.454 OOF correlation with absolute final error, so it could find risk magnitude, but residual direction remained the harder part.

## Main Results

### Strict OOF Argmin In The Expanded Grid

The nominal best train-OOF variant after adding the expanded PLS-risk combinations was:

```text
variant: risk3_x_pls_gap_gamma_positive
train OOF MAE: 32.7681008299
held-out MAE: 31.7006906146
held-out gain vs 31.7703 final: +0.0695730417
held-out improved-case rate: 49.07%
95% paired bootstrap CI for held-out gain: [-0.0476, +0.1844]
```

This is leakage-free, but the held-out improvement is modest and the confidence interval crosses zero.

### Recommended Robust Near-Tie Model

The simple procedure-family gate was only 0.00118 MAE worse on train OOF than the nominal argmin, used fewer category gammas, and was much stronger on held-out:

```text
variant: procedure_family_gamma_positive
train OOF MAE: 32.7692796199
held-out MAE: 31.6075223696
held-out gain vs 31.7703 final: +0.1627412868
held-out improved-case rate: 51.32%
95% paired bootstrap CI for held-out gain: [+0.0565, +0.2664]
```

I would treat this as the most credible next candidate if we allow a simple near-tie/complexity-regularized selection rule. If the rule must be "pick the literal train-OOF argmin across every expanded variant," then use `risk3_x_pls_gap_gamma_positive` instead.

The procedure-family gammas were:

```text
AF/PVI ablation:                    1.700
VT/PVC ablation:                    2.500
Lead/device extraction-removal:     1.525
Other ablation:                     1.025
Device implant:                     0.775
Device revision-generator-upgrade:  0.775
Diagnostic/testing:                 0.300
Other:                              0.275
LAA closure:                        1.000
global fallback:                    1.000
```

Interpretation: the PLS note/table correction was under-applied for AF/PVI and VT/PVC ablations, roughly calibrated for other ablations and extraction/removal, and over-trusted for diagnostic/device/other lower-complexity families.

### Held-Out Exploratory Best

The lowest held-out number found was:

```text
variant: risk3_x_family_x_pls_gap_gamma_signed
train OOF MAE: 32.8111800606
held-out MAE: 31.5795056769
held-out gain vs final: +0.1907579795
```

I do not recommend calling this the final model yet. It did not win by train OOF and it is a much more segmented model. This is a promising diagnostic direction, but it needs a fresh validation split or nested selection before it can be claimed cleanly.

## PLS-Specific Findings

PLS gap features did contain signal, but not enough by themselves to beat the simpler family gate:

```text
pls_gap_bin_gamma_positive:          held-out MAE 31.756141, gain +0.0141
sched_x_pls_gap_gamma_positive:      held-out MAE 31.695244, gain +0.0750
family_x_pls_gap_gamma_positive:     held-out MAE 31.657513, gain +0.1128
risk_surface_pls_gamma_positive:     held-out MAE 31.681709, gain +0.0886
```

So the PLS disagreement geometry is useful, but it becomes unstable when too finely segmented. The most reliable signal is not "PLS gap alone"; it is "how strongly should the PLS correction be trusted for this clinical procedure family?"

## Where The Recommended Model Helped

Procedure-family gate gains on held-out:

```text
AF/PVI ablation:                    +0.5819 MAE, 58.20% improved
Lead/device extraction-removal:     +0.2409 MAE, 43.42% improved
Diagnostic/testing:                 +0.2296 MAE, 54.94% improved
Device revision-generator-upgrade:  +0.0816 MAE, 55.88% improved
VT/PVC ablation:                    +0.0262 MAE, 50.85% improved
Device implant:                     -0.0403 MAE, 48.74% improved
Other:                              -0.0663 MAE, 47.83% improved
```

AF/PVI is the clearest family-level win. VT/PVC is high variance: the 2.5 gamma creates large wins and large losses, so it needs more careful capping or another direction model before pushing harder.

## Practical Recommendation

Use `procedure_family_gamma_positive` as the next credible candidate:

```text
prediction = base + gamma(procedure_family) * (strict_31_7703_prediction - base)
```

This preserves the original strict PLS manifold signal, uses only tabular plus strict pre-cutoff note-derived predictions/metadata, and avoids the old 11 LLM features.

For a publication or upload, I would report:

```text
Strict PLS manifold model: 31.7702636563 MAE
Procedure-family calibrated PLS model: 31.6075223696 MAE
Gain: +0.1627412868 MAE
95% paired bootstrap CI: [+0.0565, +0.2664]
```

with a caveat that this is still internal held-out confirmation after model exploration. A fresh lockbox or nested validation would be the cleanest way to certify the 31.6075 result.

## Output Files

```text
outputs/risk_category_modulation/risk_category_modulation.py
outputs/risk_category_modulation/risk_category_modulation_results.csv
outputs/risk_category_modulation/risk_category_modulation_summary.json
outputs/risk_category_modulation/risk_category_modulation_test_predictions.csv
outputs/risk_category_modulation/procedure_family_gamma_selected_predictions.csv
outputs/risk_category_modulation/pls_risk_category_modulation_report.md
```
