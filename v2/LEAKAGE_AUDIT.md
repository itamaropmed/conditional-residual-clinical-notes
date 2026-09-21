# Leakage audit, and the correction

The v2 result (30.6866) was audited after the fact. **It was not leakage-free.**
Four problems were found; all are fixed in `v3_corrected.py`.

## What was wrong

**1. A post-hoc column used as a feature.** `all_procs_not_performed` was in the
feature set. You only know a procedure was not performed *after* the case.
Removed.

**2. Rule 3 not applied — 105 backfill artefacts, 23 of them in the holdout.**
Rows where the actual duration equals the scheduled duration *to the second*.
These are rows where the actual was never captured and the schedule was written
back into the field. They are free MAE for any model holding the scheduled
duration, and v2 held it. Dropped.

**3. Rules 4/4b not applied — 23 aborted cases (19 in holdout) and 6 rows with a
null primary procedure.** An aborted case measures an abort, not a procedure.
Dropped.

**4. Vocabulary built using holdout rows.** The procedure multi-hot kept codes
appearing >= 30 times *across all rows*, and the categorical rare-level bucketing
counted levels the same way. Ten multi-hot columns existed only because holdout
rows pushed them over the threshold:

```
3D Mapping - EnSite X, ABLATION - ATRIAL FIBRILLATION,
ABLATION - ATRIAL TACHYCARDIA - LEFT, DEBRIDEMENT AND IRRIGATION
PACEMAKER POCKET WOUND, DRUG CHALLENGE - ISOPROTERENOL,
EXTERNAL CARDIOVERSION, EXTRACTION - HIGH RISK, EXTRACTION - LOW RISK,
ICD GENERATOR CHANGE, PPM LEAD REVISION
```

Both vocabularies are now fitted on training rows only: 84 procedure columns
instead of 95.

## What was checked and found clean

**The split is valid.** Forward-only on `scheduled_in_room`:

| | n | range |
|---|---|---|
| train | 10,882 | 2022-09-01 → 2025-08-08 |
| holdout | 2,839 | 2025-08-08 → 2026-03-27 |

Train's last case precedes the holdout's first. Zero shared patients — 474
training cases were dropped because their patient appears in the holdout.

**`procedures` is the BOOKED list, not the performed list** — this was the
critical question, because the whole v2 gain rests on it. Two independent
checks:

- Cases flagged `all_procs_not_performed` still list 1.52 procedures on average
  (vs 2.41 for the rest). A *performed* list would be empty for those cases.
- `case_num_procedures`, an existing tabular column, equals `len(procedures)` in
  **100.0%** of rows — so the list is the same booked count already trusted in
  the feature set.

The multi-hot is legitimate pre-decision information.

## The corrected result

| | v2 (leaky) | **v3 (corrected)** |
|---|---|---|
| holdout MAE | 30.6866 | **30.4811** |
| without procedure multi-hot | 31.1936 | 31.0311 |
| **multi-hot gain** | +0.5070 [+0.2993, +0.7045] | **+0.5500 [+0.3343, +0.7686]** |
| holdout cases | 2,839 | 2,813 |
| features | 152 | 139 |

**The gain survived the correction and got slightly stronger**, which is what you
would expect if it was real signal rather than leakage.

## The comparison you can and cannot make

**Valid**: the +0.5500 min multi-hot gain. Same cohort, same preprocessing, one
feature block toggled, paired bootstrap CI.

**Not valid**: reading 31.5564 − 30.4811 = 1.08 min as the improvement over the
published pipeline. v3 drops 42 rows from the holdout, so the two numbers are on
different cohorts and that difference mixes a real gain with a cohort change.
Re-running the published pipeline on the v3 cohort is required to quote a
like-for-like figure.

## Still outstanding

- **Tails.** Middle-98% MAE is 29.36; the 1%/99% tails are 86.95. Rule 5 asks
  for this stratified split to be reported, not trimmed. Now reported.
- **`bmi_missing` as a time index (Rule 6).** The indicator was added, but the
  null rate is 0.744 in train against 0.566 in holdout, and it has not been
  verified that the model is not leaning on it.
- **The holdout has been scored many times.** A fresh lockbox is still required
  before any external claim.
- **DSC-127 Rule 1.** `lag_sp` cannot be recomputed for a new case, so this is
  not production-scorable until `causal_features.py` replaces it.
