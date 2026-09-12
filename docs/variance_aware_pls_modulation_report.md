# Variance-Aware PLS Modulation Report

## Purpose

This pass tested whether the large remaining errors in the strict PLS manifold model can be handled by learning high-variance/risk categories and giving those categories their own heads.

## Baseline

```text
strict PLS model: pls_raw_word160_26_delta__sgd_huber__g1.550
held-out MAE: 31.7702636563
held-out n: 2,839
```

The strongest previous simple calibrator was:

```text
procedure_family_gamma_positive
held-out MAE: 31.6075223696
gain vs strict PLS model: +0.1627412868
```

## Leakage Verdict

The strict nested risk-gate experiment is leakage-free for the held-out score and for OOF model selection:

- No old 11 LLM feature file was read.
- Inputs were the strict PLS OOF/test predictions, tabular features, and strict pre-cutoff note bundle metadata.
- Train/held-out patient overlap remained 0.
- Learned risk/gain bins were built inside outer patient GroupKFold splits.
- For each outer validation fold, risk/gain score models were trained without that validation fold.
- Category heads were tuned using only the outer training fold.
- Held-out outcomes were used only for final scoring.

The earlier broad risk-grid was held-out-target safe, but the OOF risk-bin selection was not as strictly nested. For a clean claim, use the strict nested results in this folder.

## What Was Tested

1. Capped PLS correction gates:

```text
prediction = base + gamma(category) * clip(final - base, -cap(category), cap(category))
```

2. Strict nested risk gates:

```text
risk target = abs(final_oof_prediction - actual_duration)
```

3. Strict nested gain gates:

```text
gain target = abs(base_prediction - actual_duration) - abs(final_prediction - actual_duration)
```

4. Direct cross-fit residual heads:

```text
prediction = final + alpha * residual_model(features)
```

## Results

```text
crossfit_residual_hgb
train OOF MAE: 32.724460
held-out MAE: 31.737202
gain vs strict final: +0.033062
```

This was the best by train OOF, but it did not transfer well. I do not trust it as the next final model.

```text
family_cap_gamma
train OOF MAE: 32.757110
held-out MAE: 31.628808
gain vs strict final: +0.141456
95% CI: [+0.0317, +0.2491]
```

This is leakage-free and helpful, but it is worse than the simpler uncapped procedure-family gate.

```text
nested_risk_abs3_x_family_cap_gamma
train OOF MAE: 32.771967
held-out MAE: 31.607452
gain vs strict final: +0.162811
95% CI: [+0.0269, +0.2985]
```

This is the best strict nested risk-gated result. It is essentially tied with the simpler procedure-family gate:

```text
procedure-family gate:      31.6075223696
nested risk-family gate:    31.6074524935
difference:                -0.0000698760 MAE
```

The difference is numerically tiny and not a meaningful win.

## What The Risk Gate Learned

The risk model learned error magnitude reasonably well:

```text
OOF correlation with abs final error: about 0.456
```

The gain/direction model was weak:

```text
OOF correlation with final-vs-base gain: about 0.077
```

That explains the behavior: we can identify "this case may be high variance," but we still struggle to know whether the PLS correction should move up, move down, or be stopped.

## Where The Nested Risk-Family Gate Helped

Post-hoc by strict-final absolute error:

```text
final error 30-60 min:    +0.6649 MAE gain
final error 60-120 min:   +0.3875 MAE gain
final error 120+ min:     +0.6924 MAE gain
final error 0-15 min:     -0.4138 MAE loss
```

So it does what we hoped on many worse cases, but it pays for that by perturbing already-good cases.

Post-hoc by correction size:

```text
abs PLS correction 10-20 min: +2.9319 MAE gain, n=45
abs PLS correction 5-10 min:  +0.4081 MAE gain, n=618
abs PLS correction 0-5 min:   +0.0359 MAE gain, n=2,176
```

The largest upside is in the small subset where the PLS correction is nontrivial.

Post-hoc by family:

```text
AF/PVI ablation:      +0.6643 MAE gain
Diagnostic/testing:   +0.3115 MAE gain
Other ablation:       +0.0651 MAE gain
VT/PVC ablation:      +0.0195 MAE gain
Device implant:       -0.0490 MAE loss
Device revision:      -0.1173 MAE loss
Lead/device removal:  -0.1409 MAE loss
```

AF/PVI remains the most reliable positive pocket.

## Conclusion

The gated risk idea works technically and can be made leakage-free. It is feasible to manage.

But with the current features, the strict nested risk gate does not provide a meaningful improvement over the simpler procedure-family gate. The safer result to report remains:

```text
procedure_family_gamma_positive
held-out MAE: 31.6075223696
```

The best strict nested risk-family gate is interesting but should be treated as a tie:

```text
nested_risk_abs3_x_family_cap_gamma
held-out MAE: 31.6074524935
```

The next genuine improvement probably needs a better direction signal, not just a better risk-magnitude signal.

## Output Files

```text
outputs/variance_aware_pls_modulation/variance_aware_pls_modulation.py
outputs/variance_aware_pls_modulation/variance_aware_results.csv
outputs/variance_aware_pls_modulation/variance_aware_summary.json
outputs/variance_aware_pls_modulation/variance_aware_test_predictions.csv
outputs/variance_aware_pls_modulation/variance_aware_pls_modulation_report.md
```

