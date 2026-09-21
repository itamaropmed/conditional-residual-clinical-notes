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

---

# v4 — the leak-safe model

The v3 correction left three items open. All are now closed, and each fix is
priced so the cost of safety is explicit.

## Further problems found and fixed

**5. Staffing columns may be post-hoc.** `case_num_providers` correlates +0.604
with the target and +0.358 within procedure, but matches the booked procedure
count in only 88% of rows — so 12% of its value comes from somewhere other than
the booking. It may be the providers who *actually* scrubbed in. The schema does
not say. `case_num_panels` is the same question (35.7% match). **Both dropped by
default.** If Mayo confirms they are booked staffing, they are worth +0.176 min
and can be restored.

**6. Rule 10 not applied.** 2022Q3 has 100% null lag features and 2022Q4 is 38%
null. Training on them teaches the model what to do when lags are absent — a
state that never recurs after 2023. Training now starts 2023-01-01, costing
1,041 rows.

## Verified clean and kept

**`prior_same_proc_duration` is time-honest.** It matches a strictly-past
reconstruction in **100.0%** of rows (corr 1.000, against 0.741 for a
row-inclusive reconstruction). Kept.

**Rule 6 — `bmi_missing` is not acting as a time index.** The concern was real:
the null rate is 0.744 in train against 0.566 in holdout. But in the fitted
model `bmi_missing` ranks **122nd of 133** features with a gain share of 0.0056,
and dropping `bmi` entirely changes nothing measurable (−0.0443, CI
[−0.1567, +0.0677]). The indicator does its job.

Top features, for reference: `lag_proc_mean`, `primary_procedure_name`,
`procedures_dedup`, `sched_room_minutes`, `proc__n_unique`,
`proc__CORONARY ANGIOGRAPHY`, `proc__PULSED FIELD ABLATION`,
`proc__ABLATION - ATRIAL FLUTTER - LEFT`. Three of the top eight are procedure
multi-hot columns.

## The price of leak-safety

| configuration | train n | features | holdout MAE |
|---|---|---|---|
| v3 (2022 onward, staffing kept) | 10,784 | 139 | 30.4811 |
| v4a + Rule 10 burn-in | 9,743 | 135 | 30.7008 |
| **v4b − staffing — LEAK-SAFE** | 9,743 | 133 | **30.8770** |
| v4c − staffing − bmi | 9,743 | 131 | 30.8326 |

Full leak-safety costs **0.396 min** against v3 (CI [−0.596, −0.194]): 0.220 for
the burn-in, 0.176 for the staffing columns. That is the honest price, and it is
worth paying — an inflated backtest that collapses in production costs more.

## The headline claim, measured on the leak-safe config

| | multi-hot gain | 95% CI |
|---|---|---|
| v2 (leaky) | +0.5070 | [+0.2993, +0.7045] |
| v3 (4 fixes) | +0.5500 | [+0.3343, +0.7686] |
| **v4b (fully leak-safe)** | **+0.5179** | **[+0.2949, +0.7389]** |

**The gain is stable across every correction.** That is what a real signal does;
leakage shrinks when you remove it.

Leak-safe final: **30.8770** holdout MAE, 9,743 train / 2,813 holdout.
Middle-98% MAE 29.79, tails 85.40.

## Rule 11 — both numbers

The split purges patients from training, so all 2,813 holdout cases are
new-patient cases and the two Rule 11 numbers coincide at 30.8770. Without the
purge, 352 patients would appear on both sides covering 411 holdout cases; that
plain-forward variant would be the deployment-realistic figure and is expected
to be lower. It has not been run.

## Still outstanding

- **A fresh lockbox.** This holdout has been scored across many configurations.
- **DSC-127 Rule 1.** `lag_sp` cannot be recomputed for a new case —
  `causal_features.py` must replace it before deployment.
- **A schema answer on `case_num_providers` / `case_num_panels`**, worth
  +0.176 min if they turn out to be booked.
- **Comparability.** 30.8770 is on 2,813 cases; the published 31.5564 is on
  2,839. Re-run the published pipeline on this cohort before quoting a
  like-for-like delta.
