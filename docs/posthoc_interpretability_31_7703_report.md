# Post-Hoc Interpretability Report: Strict 31.7703 Model

This report explains where the fixed clean `pls_raw_word160_26_delta` model improves over the tabular base, and what its note/tabular manifold correction appears to be doing.

The analysis uses the already-generated held-out predictions only. It does not retrain, retune, or select a new model. It does not read or use the old 11 LLM feature file.

## Model Analyzed

Final prediction:

```text
prediction = 0.6 * xgb_tuned_v2
           + 0.4 * xgb_absoluteerror_same_shape
           + 1.550 * sgd_huber_residual(pls_raw_word160_26_delta_features)
```

Held-out split:

| Item | Value |
| --- | ---: |
| Held-out cases | 2,839 |
| Split | temporal, patient-disjoint |
| Patient overlap | 0 |
| Old 11 LLM features used | false |
| Note rule | `note_date < scheduled_in_room - 2 days` |

Final result:

| Metric | Tabular base | Final model | Change |
| --- | ---: | ---: | ---: |
| MAE | 32.239636 | 31.770264 | +0.469372 gain |
| P90 absolute error | 69.605281 | 69.826732 | -0.221451 |
| P95 absolute error | 94.371794 | 93.229405 | +1.142389 gain |

The final model improves 53.47% of held-out cases and worsens 46.53%. Median case-level gain is 0.354 minutes. The net gain is not because every case gets better; it comes from more large wins than large losses:

| Case-level event | Count |
| --- | ---: |
| Gain > 10 minutes | 28 |
| Loss < -10 minutes | 6 |

## Main Behavioral Finding

The residual correction is small but directional:

| Quantity | Value |
| --- | ---: |
| Mean correction, final minus base | -2.168475 min |
| Median correction, final minus base | -2.235328 min |
| Mean absolute correction | 3.358979 min |
| Correlation with true residual | 0.158227 |
| Direction aligned with true residual | 56.29% |

The model mostly learns to pull predictions downward. That is good when the tabular base overestimates, and harmful when the tabular base underestimates.

| Base error direction | N | Base MAE | Final MAE | Gain | Improved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base overestimated | 1,558 | 28.748879 | 26.525336 | +2.223543 | 74.26% |
| Base underestimated | 1,281 | 36.485225 | 38.149341 | -1.664115 | 28.18% |

Interpretation: the note manifold is finding a real "shorter-than-tabular-expectation" signal. It is weaker at detecting hidden complexity that should push long cases upward.

## Where The Model Helps Most

By broad procedure family:

| Procedure family | N | Base MAE | Final MAE | Gain | Improved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF/PVI ablation | 677 | 34.408913 | 33.101606 | +1.307307 | 61.30% |
| Lead/device extraction-removal | 76 | 49.141493 | 48.414816 | +0.726677 | 46.05% |
| VT/PVC ablation | 295 | 54.195904 | 53.859330 | +0.336574 | 53.22% |
| Device implant | 712 | 25.519770 | 25.187739 | +0.332031 | 52.67% |
| Other ablation | 365 | 36.598240 | 36.294633 | +0.303607 | 52.05% |

The cleanest subgroup win is AF/PVI ablation. It is large enough to matter, has many cases, and improves more than 61% of cases.

By individual procedure names with at least 10 held-out cases:

| Procedure | N | Gain | Improved rate | Mean correction |
| --- | ---: | ---: | ---: | ---: |
| PPM temporary permanent insertion, single | 64 | +1.963007 | 71.88% | -3.230922 |
| ICD lead revision, RV | 13 | +1.941241 | 61.54% | +2.382448 |
| Extraction, intermediate risk | 13 | +1.840645 | 46.15% | -1.541665 |
| PPM implant, leadless | 48 | +1.784111 | 72.92% | -2.282729 |
| Ablation, atrial flutter, right | 39 | +1.767313 | 66.67% | -3.787292 |
| Cardioneural ablation | 40 | +1.552711 | 60.00% | -5.933032 |
| Ablation, PVI | 672 | +1.258921 | 61.01% | -3.766615 |

Reflection: the model is strongest in procedures where notes may clarify plan details, patient pathway, rhythm context, or whether the case is likely to be simpler than the tabular schedule suggests.

## Where The Model Hurts

Broad families with negative gain:

| Procedure family | N | Base MAE | Final MAE | Gain | Improved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Device revision-generator-upgrade | 306 | 23.197446 | 23.296549 | -0.099103 | 47.06% |
| Diagnostic/testing | 233 | 23.667749 | 23.906559 | -0.238810 | 45.92% |

Individual procedure names with the clearest negative gain:

| Procedure | N | Gain | Improved rate | Mean correction |
| --- | ---: | ---: | ---: | ---: |
| NIPS | 18 | -2.217755 | 33.33% | +3.518807 |
| PPM generator change, BIV | 11 | -1.670248 | 27.27% | -1.360109 |
| Ablation, atrial tachycardia, right | 16 | -1.239956 | 43.75% | -2.213164 |
| Loop recorder explant | 14 | -1.070143 | 28.57% | -2.891434 |
| Ablation, AVNRT | 17 | -1.064700 | 41.18% | -3.723075 |
| ICD implant, single chamber | 65 | -0.832784 | 32.31% | -0.646028 |

Reflection: these are groups where the correction often moves in the wrong direction or adds noise. Many are shorter or more protocolized workflows, where the tabular base already knows much of the duration and the notes may add less stable signal.

## Duration Bands

The correction is best for shorter and mid-duration cases, and weak for long cases.

| Actual duration band | N | Gain | Improved rate | Mean base residual, actual minus base | Mean correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 150-209 | 694 | +1.317720 | 60.37% | -11.438791 | -2.742666 |
| <90 | 451 | +0.957888 | 61.86% | -14.337984 | -1.876409 |
| 90-149 | 795 | +0.576157 | 55.47% | -10.079503 | -1.764352 |
| 420+ | 63 | +0.327135 | 49.21% | +120.743299 | +0.497135 |
| 210-299 | 605 | -0.244477 | 45.45% | +8.166959 | -2.702643 |
| 300-419 | 231 | -1.492222 | 31.60% | +51.156443 | -1.732424 |

This is the clearest weakness: long cases often need an upward correction, but the final residual model tends to pull down.

## Note Availability And Note Volume

The model does not simply improve because there are more notes. It improves most when there is enough note signal but not only maximum-volume note text.

| Note count bin | N | Gain | Improved rate | Mean correction |
| --- | ---: | ---: | ---: | ---: |
| 13-23 notes | 260 | +1.447760 | 57.31% | -3.891986 |
| 24-note cap | 2,482 | +0.380047 | 53.10% | -1.949034 |
| 6-12 notes | 65 | +0.337923 | 55.38% | -3.191692 |
| 1-5 notes | 26 | -0.381581 | 46.15% | -3.854985 |

By note character volume:

| Note chars bin | N | Gain | Improved rate |
| --- | ---: | ---: | ---: |
| 8k-18k | 359 | +0.589984 | 54.32% |
| 18k-32k | 2,061 | +0.485193 | 53.76% |
| 32k+ | 367 | +0.330521 | 51.50% |
| <8k | 46 | -0.029357 | 50.00% |

Reflection: middle-rich note histories seem best. Very sparse notes do not help. Very large note bundles still help, but probably contain more template/noise.

## Note Types Associated With Higher Gains

Presence of these note types is associated with higher gains:

| Note type present | N present | Gain when present | Gain lift vs absent |
| --- | ---: | ---: | ---: |
| Result Encounter Note | 123 | +1.002289 | +0.557051 |
| Outpatient Summary | 248 | +0.989685 | +0.570116 |
| Nursing Note | 315 | +0.952389 | +0.543298 |
| Referral Triage Note | 604 | +0.886432 | +0.529769 |
| Documentation Clarification | 61 | +0.772030 | +0.309303 |
| Consults - Outpatient | 1,815 | +0.671208 | +0.559582 |

Note types associated with weaker or negative gains:

| Note type present | N present | Gain when present | Gain lift vs absent |
| --- | ---: | ---: | ---: |
| Transfer Note | 63 | -0.204806 | -0.689478 |
| Anesthesia Procedure Notes | 122 | -0.152159 | -0.649439 |
| ED Notes | 91 | -0.077631 | -0.565117 |
| ED Procedure Note | 77 | +0.036761 | -0.444672 |
| Anesthesia Preprocedure Evaluation | 280 | +0.110499 | -0.398140 |

These are associations, not causal claims. Note type presence is confounded with patient pathway and procedure mix. Still, it suggests the useful signal is often in outpatient planning, referral/triage, result summaries, and consult context rather than procedure/anesthesia notes.

## Clinical Regex Concept Flags

Simple concept flags did not explain the full model gain by themselves.

| Concept flag present | N present | Gain when present | Gain lift vs absent |
| --- | ---: | ---: | ---: |
| Congenital/complex anatomy | 241 | +0.439941 | -0.032161 |
| Access support | 377 | +0.267049 | -0.233305 |
| VT/PVC instability | 1,157 | +0.221037 | -0.419157 |
| Redo/prior ablation | 442 | +0.217993 | -0.297733 |
| Device/lead complexity | 345 | +0.143508 | -0.370941 |

Reflection: the model is not just rediscovering a few obvious binary flags. The gain appears to come from broader, distributed note/tabular geometry: procedure context, note source, rhythm/device language, and whether the note view agrees or disagrees with the tabular expectation.

## Integrated-Gradient-Equivalent Attribution

Classical Integrated Gradients is designed for differentiable neural networks. The final 31.7703 model is not a neural net; it is:

```text
TF-IDF -> SVD -> PLS manifold -> delta features -> standardized linear SGD Huber residual head
```

For a linear residual head, Integrated Gradients from a zero baseline in the final standardized latent feature space is exactly:

```text
contribution_j = standardized_feature_j * coefficient_j * gamma
```

I refit the exact final selected model path and verified it reproduces the saved final prediction column exactly:

| Check | Value |
| --- | ---: |
| Refit final MAE | 31.7702636563 |
| Saved final MAE | 31.7702636563 |
| Max abs prediction difference | 5.68e-14 |
| Mean abs prediction difference | 2.82e-15 |

Attribution by final geometry block:

| Geometry block | Mean absolute contribution | Mean signed contribution |
| --- | ---: | ---: |
| `diff = note_score - tab_score` | 8.302475 | -0.526788 |
| `absdiff = abs(note_score - tab_score)` | 5.046046 | -0.021501 |
| `product = note_score * tab_score` | 2.768311 | -0.017423 |

Interpretation: the dominant signal is directional disagreement between the note manifold and the tabular manifold. Absolute mismatch also matters. The product/alignment term matters less.

Top latent PLS components by total absolute contribution:

| PLS component | Mean absolute contribution | Mean signed contribution |
| --- | ---: | ---: |
| 7 | 1.207244 | +0.015236 |
| 12 | 1.097794 | -0.015332 |
| 9 | 1.014642 | -0.049590 |
| 13 | 0.929243 | -0.009664 |
| 26 | 0.842400 | -0.011313 |
| 16 | 0.839913 | -0.451751 |
| 17 | 0.838225 | -0.172638 |
| 24 | 0.771576 | +0.040384 |

Approximate token anchors for these components include device/loop-recorder/syncope language, VT/PVC/amiodarone/pacing language, SVT/AV-node language, AF/PVI/TEE/anticoagulation instruction language, and bundle-branch/conduction language. These are approximate anchors from SVD/PLS back-projection, not direct clinical causal statements.

## Practical Interpretation

The final model is best understood as a calibrated note/tabular disagreement detector.

It works when the notes indicate that the tabular/schedule model is too high, especially in AF/PVI and certain implant/leadless/temporary pacing workflows. It struggles when the base is already underestimating long or complex cases, because the learned correction still often moves downward.

The strongest next interpretability follow-ups would be:

1. Counterfactual note-source ablation: remove each note type from the fitted pipeline and measure prediction movement.
2. Sentence-level occlusion: remove top-matching sentences within strict pre-op notes and measure residual-correction change.
3. Family-calibrated correction gates: learn when to allow upward corrections for long/underestimated cases, using only train OOF.
4. Latent SHAP or Linear SHAP on the 78 final features: this should agree closely with the IG-equivalent attribution above.
5. If a future neural note encoder is trained, then true token-level Integrated Gradients, attention rollout, and TCAV-style concept vectors become appropriate.

## Files Produced

Main report:

```text
outputs/posthoc_interpretability_31_7703_report.md
```

Supporting tables:

```text
outputs/posthoc_interpretability/family_summary.csv
outputs/posthoc_interpretability/procedure_summary.csv
outputs/posthoc_interpretability/actual_duration_bin_summary.csv
outputs/posthoc_interpretability/base_direction_summary.csv
outputs/posthoc_interpretability/note_count_bin_summary.csv
outputs/posthoc_interpretability/note_chars_bin_summary.csv
outputs/posthoc_interpretability/doc_type_summary.csv
outputs/posthoc_interpretability/concept_flags_summary.csv
outputs/posthoc_interpretability/latent_block_attribution_summary.csv
outputs/posthoc_interpretability/latent_component_attribution_summary.csv
outputs/posthoc_interpretability/latent_integrated_gradient_equivalent_features.csv
outputs/posthoc_interpretability/latent_component_approximate_terms.csv
```

Scripts:

```text
work/no_llm_notes/posthoc_interpretability_analysis.py
work/no_llm_notes/latent_attribution_31_7703.py
```

## Caution

This is post-hoc held-out interpretation. It is useful for understanding the already selected model, but these subgroup findings should not be used to tune a new final model unless the next evaluation is nested or performed on a fresh temporal holdout.
