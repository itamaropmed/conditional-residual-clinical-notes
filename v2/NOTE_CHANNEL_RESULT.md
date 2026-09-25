# The clinical-note channel: a decisive negative result

Run with a real clinical encoder, on the full cohort, on the full holdout,
with paired bootstrap confidence intervals. **The notes do not help.**

## What was run

| | |
|---|---|
| Encoder | Bio_ClinicalBERT (108M params, 768-dim), weights local, no API |
| Cases encoded | **13,186** of 14,066 eligible (93.7% have pre-op notes) |
| Passages encoded | 26,348, mean-pooled and L2-normalised |
| Note cutoff | `scheduled_in_room − 2 days` — known at prediction time |
| Holdout | **2,813 cases**, scored once per configuration |
| Train | 9,743 |

Two note representations, neither of them pattern matching:

1. **Concept-anchor similarity.** 33 canonical clinical sentences across 12
   duration-relevant concepts, written in the register these notes actually
   use, encoded once. Each case gets max and mean cosine similarity to each
   concept's anchors. A note reading *"frequent ventricular ectopy in
   bigeminy"* scores on the PVC anchor without sharing a token with it.
2. **Pooled document vectors**, SVD-reduced to 48 dims each for mean- and
   max-pooling, **fitted on training rows only** (explains 0.899 / 0.821).

## Result

| configuration | features | holdout MAE | gain vs v4b | 95% CI |
|---|---:|---:|---:|---|
| **A — v4b, no notes** | 133 | **30.8770** | — | — |
| B — + anchor similarity | 161 | 31.1235 | **−0.2465** | [−0.4846, −0.0230] |
| C — + document SVD | 230 | 31.4297 | **−0.5527** | [−0.8579, −0.2607] |
| D — + both | 257 | 31.5037 | **−0.6267** | [−0.9382, −0.3342] |

Every note configuration is **significantly worse**, and every confidence
interval excludes zero. Restricting to the 2,808 holdout cases that actually
have notes changes nothing (30.8670 / 31.1129 / 31.4151 / 31.4885).

## Reading the pattern

The damage scales almost exactly with how many features were added — 28
features cost 0.25 min, 97 cost 0.55, 124 cost 0.63. That is the signature of
**variance from added dimensionality**, not of the notes carrying
anti-information. With 9,743 training rows, a gradient-boosted model cannot
absorb a hundred weakly-informative columns for free.

The conclusion is therefore narrower and more useful than "notes are useless":

> At this sample size, against a base that already contains surgeon × procedure
> history and the procedure multi-hot, the note channel carries no signal that
> survives the variance cost of admitting it.

The base already encodes procedural complexity directly. `lag_proc_mean` — a
surgeon's own past durations on the same procedure — is the strongest feature
in the model, and the procedure multi-hot supplies technique and equipment.
That is precisely what one hoped to read out of the narrative.

## What this supersedes

The earlier attempt was on 2,352 sampled training cases with a regex-derived
channel, and was correctly labelled anecdotal. This one is not: full cohort,
full holdout, real clinical encoder, paired CIs. It reaches the same verdict
with much stronger evidence.

The published CRCNL gain of +0.469 min from notes was measured against a base
containing neither the lag block nor the procedure multi-hot. It does not
survive against a base with both.

## What would still be worth trying

Not more note features for pooled MAE. Two other uses:

1. **Tail triage.** Middle-98% MAE is 29.79; the 1%/99% tails are 85.40. If the
   notes can flag *which* cases will overrun badly, that is operationally
   valuable even with pooled MAE unmoved. Different target, different metric.
2. **More training data.** The failure is a variance failure. The same channel
   on 50,000 cases might clear the bar. On 9,743 it does not.

An LLM extractor would produce a comparable handful of structured columns and
faces the same variance arithmetic. `claude_code_extract.py` is provided for
running that test locally, but the prior should be set by this result.
