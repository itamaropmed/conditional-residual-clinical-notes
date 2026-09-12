# Procedure-Duration Prediction: Complete Data, Architecture, Validation, and Residual-Iteration Guide

## Document purpose

This document describes the complete modeling process that produced the selected held-out mean absolute error of `31.5564180731` minutes. It is written to stand alone in Notion and refers to the companion archive `procedure_duration_31_5564_project.zip`.

The result belongs to the OOF-selected top-refine meta-gate. It is not a single neural network. It is a staged system composed of a tabular foundation, a clinical-note/tabular shared manifold, robust residual learners, procedure/risk calibration, and a final conditional pair blend.

All mathematics in this document uses Notion-compatible inline math delimiters such as `$y_i$`, `$\hat y_i$`, and `$\mathrm{MAE}$`.

## Headline result

| Quantity | Value |
| --- | ---: |
| Selected model | `strict_meta_gate_top_refine_oof_selected` |
| Training OOF cases | `10,882` |
| Held-out cases | `2,839` |
| Train/held-out patient overlap | `0` |
| Training OOF MAE | `32.7156839203` minutes |
| Held-out MAE | `31.5564180731` minutes |
| Gain over the strict PLS manifold | `0.2138455833` minutes |
| Case-level paired-bootstrap interval for that gain | `[0.1001, 0.3259]` minutes |

The selected variant is:

`topref_pair__pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00__signed_resid_hgb8_x_family_cap_gamma__delta_hgb4_proc6_none_n80_m0.02`

The saved held-out prediction file independently recomputes to `31.556418073067334`, matching the recorded result.

## Interpretation of the result

Mean absolute error is measured in minutes. For targets $y_i$ and predictions $\hat y_i$, the metric is $\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|$.

The held-out MAE being lower than the train OOF MAE is not mathematically contradictory. OOF predictions are produced by models trained on only a fraction of the training cohort, whereas the held-out prediction is produced after fitting on the full training cohort. The later cohort can also have a different case mix. This pattern is acceptable, but it does not remove the need for a new lockbox cohort.

## What is and is not included in the archive

The archive contains:

- Source code for every stage in the selected model chain.
- Snapshots of imported helper modules required by the original scripts.
- Run configurations and requirements.
- The final result record.
- Candidate summaries and saved prediction tables.
- Timing and forbidden-input audit artifacts.
- A standalone script that recomputes the `31.5564` result from saved predictions.
- This Notion-ready document.

The archive does not contain raw clinical data, raw note text, patient identifiers outside the existing deidentified result artifacts, fitted model binaries, or production credentials. Raw Mayo parquet files must be supplied separately in an authorized environment.

## Prediction problem

The target is the actual procedure duration in minutes, represented by `case_actual_duration_time`. For case $i$, the target is $y_i=\mathrm{case\_actual\_duration\_time}_i$.

The operational objective is to predict duration before the procedure begins. Inputs therefore need to represent information that would be available at the intended prediction timestamp.

## Raw data sources

### Tabular case data

The tabular source contains `14,195` rows and `55` columns before filtering and splitting.

Identity and split fields:

- `case_durable_id`
- `patient_durable_id`
- `scheduled_in_room`

Case descriptors:

- `case_anesthesia`
- `case_urgency_level`
- `case_surgery_patient_class`
- `case_admission_patient_class`
- `surgeon_durable_id`
- `procedures`
- `case_num_procedures`
- `case_num_panels`
- `case_num_providers`
- `primary_procedure_name`
- `is_redo`
- `prior_same_proc_duration`
- `redo_time_gap_days`

Patient descriptors:

- `patient_sex`
- `patient_smoking_status`
- `patient_age`
- `age_group`
- `bmi`

Comorbidity indicators:

- coronary artery disease, heart failure, arrhythmia, and conduction block
- valve disease, cardiac device, vascular disease, and cardiac inflammation
- ventricular dysfunction and thromboembolism
- pulmonary, renal, metabolic, hypertensive, and hematologic disease
- neurologic, hepatic, oncologic, psychiatric, and musculoskeletal disease
- lifestyle, gastrointestinal, immune, and transplant indicators

Scheduling timestamps:

- `scheduled_setup_start`
- `scheduled_in_room`
- `scheduled_out_of_room`
- `scheduled_cleanup_complete`

Outcome-time fields present in the raw table but excluded from predictors:

- `case_actual_patient_in_room`
- `case_actual_duration_time`
- `case_actual_prep_time`
- `case_actual_procedure_time`
- `case_actual_wrapup_time`

Other excluded identity/outcome-related fields include `patient_birth_date`, `all_procs_not_performed`, case ID, and patient ID.

### Clinical notes

The notes source contains `3,179,288` rows with four raw columns:

- `epic_id`
- `doc_date`
- `doc_type`
- `content_markdown`

Notes are matched to cases by patient and then filtered against a case-specific timestamp. Raw note text is not distributed in the ZIP.

### Explicitly excluded feature source

The former 11-variable LLM feature table is not an input, teacher, label, distillation target, or selection source. Its filename appears only in defensive assertions and audit scans. The selected run records `forbidden_llm_feature_file_read=false`.

## Prospective information boundary

### Case filtering

Rows without `scheduled_in_room`, without the target, or with non-positive target duration are removed. Remaining rows are sorted by scheduled time.

### Temporal patient-disjoint held-out split

Let $t_i$ be the scheduled in-room timestamp and let $q_{0.8}$ be its 80th percentile. Candidate held-out patients are those with cases at or after $q_{0.8}$. The training set contains earlier cases only and excludes every patient assigned to held-out.

The implemented split is therefore:

- train: $t_i<q_{0.8}$ and patient $i$ is not a held-out patient;
- held-out: $t_i\ge q_{0.8}$ and patient $i$ is a held-out patient.

This produces `10,882` training cases, `2,839` held-out cases, and zero patient overlap.

### Internal OOF split

Within the training cohort, most model-selection predictions use five-fold `GroupKFold` with patient ID as the grouping variable. A patient never crosses the fit and validation portions of the same fold.

The internal OOF folds are patient-grouped, not forward-time folds. The top-level held-out set is chronological; the internal OOF estimator is designed primarily to control repeated-patient dependence.

### Note cutoff

For every case, a note is eligible only if $\mathrm{doc\_date}<\mathrm{scheduled\_in\_room}-2\ \mathrm{days}$.

Additional note-cache settings are:

| Setting | Value |
| --- | ---: |
| Timestamp policy | Exact timestamp |
| Same-day preoperative note types | Disabled |
| Postoperative document types | Rejected |
| Maximum lookback | `548` days |
| Maximum notes per case | `24` |
| Maximum characters per note | `2,200` |
| Maximum characters per case | `48,000` |
| Maximum selected sentences | `72` |

Audit results:

| Audit item | Value |
| --- | ---: |
| Raw note rows scanned | `3,179,288` |
| Rows matched by patient | `3,144,540` |
| Postoperative document-type rows rejected | `145,426` |
| Attached eligible note events | `295,463` |
| Cases with a bundle | `13,721` |
| Cases with at least one used note | `12,850` |
| Cutoff violations | `0` |
| Closest note margin before cutoff | `2.5` hours |
| Train note coverage | `92.0511%` |
| Held-out note coverage | `99.7887%` |
| Median selected notes per case | `24` on both splits |
| Median note characters | `24,377` train; `24,702` held-out |

Historical notes can contain words such as """post""" or """discharge""" because they may describe an older event. Timestamp eligibility, rather than a word blacklist alone, is the decisive boundary. The high-signal bundles contained no `actual duration` probe hits in either split, while total-time language remained uncommon. A production audit should retain both timestamp and lexical checks.

## Scheduling-feature caveat

The selected run uses planned scheduling information. The code derives:

- `scheduled_year`, `scheduled_month`, `scheduled_dayofweek`, and `scheduled_hour`;
- `scheduled_is_weekend`;
- `scheduled_room_minutes` from scheduled out-of-room minus scheduled in-room;
- `scheduled_setup_lead_minutes` from scheduled in-room minus scheduled setup start;
- `scheduled_cleanup_minutes` from scheduled cleanup complete minus scheduled out-of-room.

These are acceptable only if the planned timestamps are present and stable at the real prediction moment. They are not actual outcome times. If the deployment question is """predict before a schedule is built,""" these fields must be removed and the entire pipeline must be retrained and reevaluated. The reported `31.5564` score applies to the schedule-informed prediction setting.

## Stage 1: tabular foundation

### Feature preparation

The tabular feature builder removes IDs, birth date, the target, all columns beginning with `case_actual_`, date-suffixed columns, and raw schedule timestamps after deriving the permitted planned-time features.

Boolean variables are converted to integers. Numeric missing values use the training median. Categorical missing values use the most frequent training category. Categories are one-hot encoded with unknown categories ignored and a minimum frequency of `10` where supported.

An explicit `bmi_missing` indicator is added because BMI missingness is informative and common.

### Smoothed target encodings

Procedure and surgeon encodings are computed using only the fitting portion of each OOF fold. For a category $c$ with count $n_c$, category mean $\bar y_c$, global mean $\bar y$, and smoothing constant $k$, the encoding is $\mathrm{TE}(c)=\frac{n_c\bar y_c+k\bar y}{n_c+k}$.

The smoothing constants are `20` for procedure and `15` for surgeon. Unknown validation or held-out categories receive the training global mean.

### Paired XGBoost models

Two tabular XGBoost models share the same tuned tree shape:

- `xgb_tuned_v2` uses `reg:squarederror`;
- `xgb_absoluteerror_same_shape` uses `reg:absoluteerror`.

Core shape parameters are:

| Parameter | Value |
| --- | ---: |
| Trees | `1213` |
| Maximum depth | `8` |
| Learning rate | `0.0137216531` |
| Row subsample | `0.8431891100` |
| Column subsample | `0.5331833452` |
| Minimum child weight | `9.0834809592` |
| L1 regularization | `6.006e-7` |
| L2 regularization | `0.0018093545` |
| Gamma | `1.0186857689` |
| Maximum histogram bins | `212` |

The raw tabular base is $\hat y_i^{base}=0.60\hat y_i^{sq}+0.40\hat y_i^{abs}$. The blend combines conditional-mean sensitivity with a more median-like, outlier-resistant objective.

Raw tabular-base performance is `32.9667124829` train OOF MAE and `32.2396358747` held-out MAE.

## Stage 2: strict clinical-note/tabular manifold

### Why residual learning

The notes should not relearn the complete duration problem. The tabular model already captures procedure identity, schedule, provider, demographics, and comorbidity structure. The note model therefore learns the component the tabular model missed.

For each training case, the residual target is $r_i=y_i-\hat y_i^{base,OOF}$. Using OOF base predictions is essential because an in-sample base residual would be artificially small and structured by overfit.

### Note representation

The selected note view is a word TF-IDF representation with unigrams and bigrams, lowercasing, English stop-word removal, clinical tokenization, `min_df=3`, `max_df=0.97`, sublinear term frequency, and at most `50,000` terms.

TF-IDF for term $j$ in note bundle $i$ is conceptually $\mathrm{tfidf}_{ij}=(1+\log c_{ij})\log\frac{N+1}{df_j+1}$ for nonzero count $c_{ij}$, subject to the library's normalization conventions.

Truncated SVD compresses the sparse note matrix to `160` coordinates. Training statistics are used for scaling. Hand-authored clinical concept indicators extracted from the same eligible text are appended; they are not the old 11 LLM variables.

### Tabular latent view

The preprocessed tabular matrix is reduced with truncated SVD to `96` dimensions, scaled from training statistics, and the first `64` coordinates are supplied to the shared manifold.

### Partial least squares alignment

PLS learns paired directions that maximize covariance between tabular latent coordinates $X$ and note latent coordinates $Z$. For component $h$, it seeks weights $w_h$ and $c_h$ with high $\mathrm{cov}(Xw_h,Zc_h)^2$ under sequential deflation.

The surgical profile fits up to `40` PLS components, and the selected representation uses the first `26` transformed components. Let $t_i$ be the 26-dimensional tabular score and $u_i$ the 26-dimensional note score.

The selected delta geometry is $\phi_i=[u_i-t_i,\ |u_i-t_i|,\ t_i\odot u_i]$. It has $26+26+26=78$ residual features.

Directional differences encode whether notes move along or against the tabular expectation. Absolute differences encode disagreement magnitude. Elementwise products encode agreement and opposition in the shared clinical manifold.

### Robust residual head

The selected residual head is standardized `SGDRegressor` with Huber loss, elastic-net regularization, `alpha=1e-4`, `l1_ratio=0.05`, `max_iter=4000`, `tol=1e-4`, averaging enabled, and a fixed seed.

Its scaled prediction is $\hat y_i^{PLS}=\hat y_i^{base}+\gamma\hat r_i$, where $\gamma$ is selected on training OOF MAE from a grid between `0` and `1.6`. The selected value is $\gamma=1.55$.

The strict PLS manifold reaches `32.7964177885` train OOF MAE and `31.7702636563` held-out MAE.

## Stage 3: procedure-family calibration

The same PLS correction is not equally reliable for every procedure family. The calibrated predictor uses $\hat y_i^{family}=\hat y_i^{base}+\gamma_{f(i)}(\hat y_i^{PLS}-\hat y_i^{base})$, where $f(i)$ is the procedure family.

Family-specific nonnegative correction scales are learned inside outer patient-group folds. Small groups fall back to a global choice. This stage reaches `32.7692796199` train OOF MAE and `31.6075223696` held-out MAE.

## Stage 4: risk and residual candidate bank

The pipeline then models where the current predictor is likely to be unreliable. Candidate targets include:

- absolute OOF error $|y_i-\hat y_i|$;
- signed OOF residual $y_i-\hat y_i$;
- tail indicators such as $\mathbb{1}(|y_i-\hat y_i|>30)$;
- correction magnitude and candidate disagreement.

Risk models include Extra Trees and histogram gradient boosting. Risk estimates are themselves nested within outer patient-group folds before they are converted into quantile bins. This prevents a row's own target-derived error from defining its validation-time risk category.

Within sufficiently large procedure/risk regions, the algorithm tunes a nonnegative correction scale and optional cap. A generic capped residual candidate is $\hat y_i=\hat y_i^{anchor}+\mathrm{clip}(\gamma_c\hat r_i,-C_c,C_c)$.

The most useful empirical lesson was that error magnitude was more learnable than error direction. The architecture therefore treats risk as a trust-allocation signal rather than assuming it can perfectly predict the sign of every future error.

## Stage 5: selected top-refine meta-gate

The final model blends two already cross-fitted candidates.

Candidate A is `pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00`.

Candidate B is `signed_resid_hgb8_x_family_cap_gamma`.

The gate score is the prediction disagreement $d_i=\hat y_i^{risk\_hgb4}-\hat y_i^{procedure\_family}$.

For each outer fold, quantile edges are learned from the fold-training scores only. The selected model uses six score bins, a minimum local sample size of `80`, and a required local MAE improvement margin of `0.02`.

For candidate predictions $a_i$ and $b_i$, the pair blend is $\hat y_i(w)=(1-w)a_i+wb_i$. The training algorithm searches $w\in\{0,0.01,\ldots,1\}$ and minimizes fold-training MAE.

The global fitted weight is $w=0.37$. Local overrides are accepted only when the region has enough rows and improves training-region MAE beyond the required margin.

The learned score edges are `-16.7446`, `-1.2492`, `-0.3816`, `-0.0305`, `0.0751`, `0.6576`, and `16.2156`.

The selected policy is:

| Score region | Weight on candidate B |
| --- | ---: |
| Lowest disagreement bin | `1.00` |
| Second bin | `0.00` |
| Highest bin | `0.00` |
| Other bins | `0.37` |

For OOF rows, edges and weights are always fitted without the validation fold. For held-out rows, the policy is refitted on all training OOF evidence and then applied unchanged.

## Model-selection rule

The reported model is the candidate with the best training OOF MAE in the targeted top-refine search. The search also recorded a `31.5234` held-out candidate, but that candidate was not selected because its train OOF score was worse. It must not replace the reported model without evaluation on a new lockbox.

## End-to-end algorithm

```text
INPUT:
  tabular case table
  timestamped clinical-note table
  prediction timestamp definition

1. Remove rows with invalid schedule time or target.
2. Sort cases by scheduled_in_room.
3. Construct a later temporal held-out cohort.
4. Remove held-out patients from earlier training rows.
5. For each case, attach only notes with doc_date < scheduled_in_room - 2 days.
6. Reject postoperative document types and cap note count/length/lookback.
7. Build planned scheduling, case, patient, and comorbidity features.
8. In patient-grouped folds:
     fit preprocessing and procedure/surgeon target encodings on fold-train;
     fit paired XGBoost models;
     predict fold-validation.
9. Refit paired XGBoost models on all training rows and predict held-out.
10. Form the 60/40 tabular blend.
11. Set residual target to actual duration minus tabular OOF prediction.
12. Fit note TF-IDF/SVD and tabular SVD using training data.
13. Fit the PLS shared manifold and construct 78 delta features.
14. Fit patient-grouped Huber residual OOF predictions and tune gamma on OOF MAE.
15. Learn procedure-family correction trust inside patient-grouped folds.
16. Build nested risk and signed-residual candidate predictions.
17. Cross-fit the final six-bin pair-blend policy.
18. Select the top-refine variant by training OOF MAE.
19. Refit training-derived transformations and policies on all training rows.
20. Apply once to held-out features and compute held-out MAE.
```

## Why the architecture can work

The model decomposes the task into progressively smaller problems. The tabular base estimates broad procedural duration. The manifold estimates disagreement between structured expectations and narrative clinical context. Family calibration controls heterogeneous trust. Risk heads identify regions where the correction is fragile. The final blend combines candidates rather than making a brittle hard switch everywhere.

This can be viewed as $y_i=m(x_i)+g(x_i,z_i)+\epsilon_i$, where $m$ is the structured base, $z_i$ is eligible note context, $g$ is a conservative residual correction, and $\epsilon_i$ is irreducible or unmodeled variation.

## Safe residual iteration

Additional residual stages are possible, but every stage must consume predictions produced without the row's target.

Let $p_i^{(0)}$ be the current OOF prediction. At stage $k$, define $r_i^{(k)}=y_i-p_i^{(k)}$. Train a new head $h_k$ using nested cross-fitting and update $p_i^{(k+1)}=p_i^{(k)}+\eta_k h_k(x_i,z_i,p_i^{(k)})$.

The shrinkage $\eta_k$ should be selected inside training folds. Continue only if gains are stable across patients, temporal windows, and procedure families. A useful stopping rule is to stop when the lower confidence bound of OOF improvement is not positive or when the gain is operationally negligible.

Recommended next-stage controls:

1. Use an outer forward-time patient-grouped validation loop for model-family selection.
2. Generate every upstream candidate OOF prediction inside that outer loop.
3. Fit risk labels, score bins, blend weights, caps, and residual scalers only on outer-fold training rows.
4. Require minimum region sizes and global fallbacks.
5. Track correction magnitude, calibration drift, and family-level harm.
6. Freeze the entire candidate set before opening a new future lockbox.

Promising residual features include note recency distributions, source-type counts, contradiction between planned procedure and narrative context, candidate spread, calibrated uncertainty, and patient-history summaries that are available before the cutoff. New embeddings or language-model features are acceptable only when generated directly from eligible notes and trained without held-out outcomes or the excluded 11-feature source.

## Validation strengths

- Later held-out cohort.
- Zero patient overlap between training and held-out.
- Exact note timestamps with a two-day buffer.
- Outcome-time columns removed from the feature matrix.
- Fold-local target encoding.
- OOF base predictions used for residual targets.
- Patient-grouped residual and gate evaluation.
- Nested construction of target-derived risk scores.
- OOF-based selection of the reported top-refine candidate.
- Saved prediction-level artifact reproduces the metric.

## Remaining limitations

The same held-out cohort was viewed during multiple research iterations. The selected model was chosen by training OOF performance, which is better than directly selecting the lowest held-out row, but repeated examination can still influence research decisions. The `31.5564` result is therefore a strong retrospective held-out estimate, not publication-grade independent confirmation.

Internal `GroupKFold` is patient-disjoint but not chronological. A stronger estimate for future deployment would use nested forward-time folds with patient purging.

The reported interval is a paired case-level bootstrap. It does not fully account for repeated-patient correlation. A patient-clustered bootstrap is preferred for final inference.

Scheduling-derived durations may encode operational planning expertise and improve accuracy, but they change the prediction task. Their availability must be guaranteed at inference.

Clinical-note coverage differs between train and held-out. Monitoring should stratify performance by note availability and note volume.

The package contains source and predictions rather than fitted estimators. Reproducing training requires the authorized raw data and the pinned software environment.

## Deployment contract

At inference, provide the same tabular schema and planned schedule fields, plus only notes finalized before the two-day cutoff. Preserve training-time category handling and transformations. Do not recompute target encodings, TF-IDF vocabulary, SVD, PLS, score-bin edges, or blend weights using production outcomes during prediction.

For retraining, create a new chronological cutoff, rebuild note bundles, regenerate all OOF candidate predictions, reselect using training-side validation, freeze the model, and then evaluate on a later untouched cohort.

## Reproduction and verification

Install dependencies from `requirements.txt`. Configure the authorized clinical-data repository using `CLINICAL_NOTES_PROJECT`. The original run order is documented in `scripts/run_best_31_5564_pipeline.sh`.

The major stages are:

1. `src/build_raw_strict_high_signal_split.py`
2. `src/validate_strict_raw_run.py`
3. `src/tabular_model_sweep.py`
4. `src/strict_manifold_push.py`
5. `src/risk_category_modulation.py`
6. `src/variance_aware_pls_modulation.py`
7. `src/strict_risk_family_k_sweep.py`
8. `src/strict_meta_gate_push.py`
9. `src/strict_meta_gate_top_refine.py`

To validate the distributed result without raw data, run `python scripts/verify_saved_result.py`. It reads the saved selected held-out predictions, identifies the OOF-selected column from `results/final_result_31_5564.json`, recomputes MAE, and checks agreement with the recorded value.

## File map

| Location | Purpose |
| --- | --- |
| `README.md` | Package entry point |
| `docs/PROCEDURE_DURATION_31_5564_NOTION.md` | This complete document |
| `results/final_result_31_5564.json` | Authoritative result and model identity |
| `results/strict_meta_gate_top_refine/strict_meta_gate_top_refine_summary.json` | Gate weights, edges, metrics, and audit fields |
| `results/strict_meta_gate_top_refine/strict_meta_gate_top_refine_selected_predictions.csv` | Selected and exploratory held-out predictions |
| `results/strict_meta_gate_top_refine/strict_meta_gate_top_refine_candidate_predictions_slim.csv` | Readable held-out candidate bank |
| `src/` | Pipeline source code |
| `vendor/clinical_notes_dependencies/` | Snapshots of imported helper modules |
| `configs/` | Recorded run settings |
| `scripts/run_best_31_5564_pipeline.sh` | Original full run order |
| `scripts/verify_saved_result.py` | Independent saved-result verification |

## Final statement

The selected OOF-ranked model achieved a held-out MAE of `31.5564180731` minutes on `2,839` later cases from patients absent from training. It used structured clinical and planned scheduling data together with clinical notes restricted to more than two days before each case. It did not use the former 11-variable LLM feature table. Its strongest next validation step is not another held-out-guided tweak; it is a fully frozen evaluation on a new later patient-disjoint cohort, ideally preceded by nested forward-time model selection.
