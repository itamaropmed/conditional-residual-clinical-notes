# CRCNL v2 — 30.6866 min holdout MAE

Rebuild of the Conditional Residual Clinical Notes Learner duration model.

| | published CRCNL | **v2** |
|---|---|---|
| Held-out MAE | 31.5564 | **30.6866** |
| Improvement | — | **+0.8698 min (2.76%)** |
| Held-out cases | 2,839 | 2,839 (identical split) |
| Training cases | 10,882 | 10,882 |
| Patient overlap | 0 | 0 |

For scale: the published four-stage ladder gained **+0.683 min** in total over
its tabular base. This single change is larger than all of it.

Related: [DSC-127](https://opmed-ai.atlassian.net/browse/DSC-127)

---

## What actually moved the number

**The `procedures` list.** Every case carries a list like

```
['ABLATION - PVI', 'ABLATION - SVC', 'ABLATION - WACA', '3D MAPPING - CARTO']
```

The shipped pipeline used only `primary_procedure_name` and discarded the
rest. But `CRYO`, `3D MAPPING - NAVX`, `CARTOSOUND` and the other entries are
technique and equipment markers, and equipment is time. Multi-hot encoding
every procedure code appearing ≥30 times adds 90 binary features.

**Measured gain: +0.5070 min, 95% CI [+0.2993, +0.7045]** — a single
pre-registered comparison, same model with and without the block.

**Two MAE specs, five seeds, median-aggregated.** Duration is right-skewed, so
squared and pseudo-Huber losses chase the conditional mean while MAE wants the
median. Using only absolute-error objectives and taking the median across fits
is worth a further **+0.0960 [+0.0189, +0.1748]**.

## The ladder

| step | holdout MAE |
|---|---|
| tabular, no lag block | 33.1322 |
| + shipped lag block | 31.4387 |
| + procedure multi-hot | 30.8965 |
| MAE-only objectives | 30.7816 |
| **+ second spec, median-aggregated** | **30.6866** |

## What did not work

Recency weighting, the overrun target parameterisation, quantile regression,
a full causal expanding-history engine, booking-bias features, and **both
clinical-note channels** — the incumbent regex flags and a rebuilt graded
extractor. All measured, all rejected, all written up in
[`EXPERIMENTS_THAT_FAILED.md`](EXPERIMENTS_THAT_FAILED.md).

The note finding matters most: the rebuilt extractor is 2.2× better than regex
at predicting residual *magnitude* (corr 0.090 vs 0.041), but neither channel
improves MAE against a base containing the lag block and the procedure
multi-hot. Those two features encode procedural complexity directly — which is
exactly what the notes were being asked to supply.

## Files

| file | what it does |
|---|---|
| `final_model.py` | the 30.6866 model |
| `causal_features.py` | procedure multi-hot + a fully causal history engine |
| `crcnl_data.py` | data layer, temporal patient-disjoint split |
| `run_ensemble.py` | feature-family ablation with paired bootstrap CIs |
| `run_round2.py` | objective and target-parameterisation sweep |
| `run_round3.py` | recency-weighting sweep |
| `stage_a_retrieval.py` | note passage retrieval and reranking |
| `stage_b_extract.py` | graded, negation-aware note extraction |
| `gold_labels.py` | 44 hand-read cases with graded labels |
| `eval_extractor.py` | extractor accuracy against those labels |
| `compare_channels.py` | regex vs graded extraction on the residual |

## Run

```bash
pip install -r requirements.txt
python final_model.py        # reproduces 30.6866
python run_ensemble.py       # the feature ablation
```

Expects `mayo_hrs_tabular_features_with_lags_and_durations.parquet` under the
path set in `crcnl_data.DATA`. No clinical note text is required by the final
model.

## Two things to fix before any external claim

1. **A fresh lockbox.** This holdout has been scored across many
   configurations here, on top of the iterations DSC-127 already records. The
   30.6866 is optimistic by an unknown amount. The +0.5070 multi-hot CI is the
   trustworthy number.
2. **The lag recipe.** DSC-127 Rule 1 still stands: `lag_sp` cannot be
   recomputed for a new case, so this model is not production-scorable as-is.
   `causal_features.py` contains a documented, strictly-past replacement —
   swap it in and re-fit before deployment.
