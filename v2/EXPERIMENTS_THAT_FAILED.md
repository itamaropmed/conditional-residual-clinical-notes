# What did not work

Every lever below was measured on the same temporal patient-disjoint split and
rejected on evidence. They are recorded because a rejected lever is worth as
much to the next person as an accepted one.

## Recency weighting — monotonically harmful

The data rules note ~+21% duration drift across the window, and the holdout is
the most recent slice, so downweighting old cases looked obvious. It is wrong.
Exponential half-life sweep, selected on a validation tail inside training:

| half-life (days) | validation MAE |
|---|---|
| **none (flat)** | **31.5146** |
| 1095 | 31.6528 |
| 730 | 31.8762 |
| 547 | 32.0646 |
| 365 | 32.3936 |
| 240 | 32.3324 |

Monotone in the wrong direction. With 10,882 training cases spread over 39
surgeons and 162 procedures, the variance cost of discarding history exceeds
the bias cost of the drift. More data wins.

## Overrun target parameterisation — clearly worse

Predicting `actual − booked slot` and adding the slot back, instead of
predicting duration directly:

| spec | direct | overrun target |
|---|---|---|
| mae_d7 | 31.4883 | 32.9840 |
| mae_d6 | 31.4879 | 33.0002 |
| q50 | 31.5353 | 33.0859 |

About 1.5 min worse everywhere. The booked slot is already a feature; forcing
the model to use it as an offset removes its freedom to distrust it.

## Quantile regression at τ=0.5 — slightly worse

31.5353 against 31.4879 for the same architecture under a plain absolute-error
objective. No reason to prefer it.

## Pseudo-Huber in the ensemble — costs 0.115 min

Duration is right-skewed. A pseudo-Huber or squared loss targets the
conditional mean; MAE wants the conditional median. Including a pseudo-Huber
spec in a median-of-specs ensemble dragged predictions upward:
30.8965 with it, 30.7816 without.

## Expanding-history and booking-bias features — redundant or harmful

A full causal feature engine was built (`causal_features.py`): strictly-past
expanding means, medians and counts over surgeon, procedure and
surgeon×procedure, at 10 / 30 / all-time windows, plus booking-bias features
(historic overrun and actual/booked ratio per surgeon and per pair).

| configuration | holdout MAE | gain vs lag block |
|---|---|---|
| shipped lag block | 31.4387 | — |
| **+ procedure multi-hot** | **30.8965** | **+0.5422 [+0.3266, +0.7519]** |
| + experience/load | 30.9623 | +0.4764 [+0.1827, +0.7898] |
| + expanding history | 31.2485 | +0.1901 [−0.1399, +0.5228] |
| + everything incl. booking bias | 31.4769 | −0.0382 [−0.3694, +0.3067] |

The expanding-history block is redundant with the shipped `lag_sp` (which is
the same quantity computed at Mayo), and stacking both adds variance without
information. Booking-bias features actively hurt.

The engine is kept in the repository anyway, because it answers DSC-127 Rule 1:
`lag_sp` cannot be recomputed for a new case, so the model as shipped is not
production-scorable. `causal_features.py` reproduces the same signal from
documented inputs and can be scored at prediction time.

## Both clinical-note channels — no MAE gain

| channel | corr with signed residual | corr with \|residual\| | MAE gain |
|---|---|---|---|
| regex binary flags (incumbent) | +0.040 | +0.041 | −0.744 |
| graded extraction (rebuilt) | −0.007 | +0.090 | −0.762 |

The rebuilt extraction is 2.2× better than regex at predicting residual
*magnitude* — the thing notes can actually do, consistent with the documented
finding that magnitude is learnable and direction is not. Neither predicts
direction and neither improves MAE.

The reason is now clear: `lag_sp` is surgeon × primary-procedure expanding
history, and the procedure multi-hot adds the technique and equipment markers.
Between them they encode procedural complexity directly. That is what the notes
were being asked to supply, and they supply it worse.

The published +0.469 min note gain was measured against a base without the lag
block. It does not survive against a base with it.

## Honest caveat on the holdout

This holdout has now been scored across many configurations in this work, on
top of the multiple research iterations DSC-127 already records. The headline
30.6866 is therefore optimistic by an unknown amount. The procedure multi-hot
gain (+0.5070, CI [+0.2993, +0.7045]) is a single clean pre-registered
comparison and is the more trustworthy number. **A fresh lockbox cohort is
required before any external claim.**
