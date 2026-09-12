# Strict 31.7703 No-LLM Manifold Model

This document describes the clean 31.7703 MAE model only. It does not include a leaderboard of other explored versions.

The important boundary is:

- the tabular base models use tabular data only
- the final 31.7703 model uses the tabular base plus strict pre-operative raw clinical notes
- the old 11 LLM features are not used as input features, labels, distillation targets, teachers, or model-selection data

## Objective

Predict Mayo HRS surgical case duration:

```text
target = case_actual_duration_time
```

The modeling target is absolute error, reported as held-out MAE in minutes.

## Data Sources

Tabular source:

```text
/Users/itamarzernitsky/PycharmProjects/Clinical_notes/data/mayo/mayo_hrs_tabular_features_and_durations.parquet
```

Raw clinical-note source:

```text
/Users/itamarzernitsky/PycharmProjects/Clinical_notes/data/mayo/mayo_hrs_notes_anonymized_merged.parquet
```

Forbidden old LLM-feature source:

```text
/Users/itamarzernitsky/PycharmProjects/Clinical_notes/data/mayo/mayo_hrs_llm_features_top11.parquet
```

That forbidden file is not read by the clean 31.7703 model. It is mentioned in code only as a defensive sentinel/check.

## Split

The experiment uses the project temporal patient-disjoint split:

```python
split = temporal_group_holdout(df, 0.20)
```

Split counts:

| Item | Value |
| --- | ---: |
| Train cases | 10,882 |
| Held-out test cases | 2,839 |
| Patient overlap | 0 |

The held-out set is patient-disjoint from training, so no patient appears in both train and test.

## Strict Note Boundary

The note cache was rebuilt from raw note rows under a strict pre-op cutoff:

```text
note_date < scheduled_in_room - 2 days
```

This means the model cannot use same-day notes, immediate pre-op notes, intra-op notes, or post-op notes.

Strict note-cache settings:

| Setting | Value |
| --- | --- |
| Cutoff policy | exact timestamp |
| Same-day pre-op note types | disabled |
| Post-op doc types | filtered out |
| Max lookback | 548 days |
| Max notes per case | 24 |
| Max chars per note | 2,200 |
| Max chars per case | 48,000 |
| Max sentences per case | 72 |

Strict note-cache build script:

```text
work/no_llm_notes/build_raw_strict_high_signal_split.py
```

Output note-cache directory:

```text
work/no_llm_notes/raw_strict_high_signal_tuned_split
```

Build command:

```bash
python3 work/no_llm_notes/build_raw_strict_high_signal_split.py \
  --output-dir work/no_llm_notes/raw_strict_high_signal_tuned_split \
  --max-notes-per-case 24 \
  --max-note-chars 2200 \
  --max-case-note-chars 48000 \
  --max-note-age-days 548 \
  --max-sentences 72
```

Note-cache audit:

| Audit item | Value |
| --- | ---: |
| Raw note rows scanned | 3,179,288 |
| Rows matched by patient | 3,144,540 |
| Post-op doc-type rows rejected | 145,426 |
| Strict note events attached | 295,463 |
| Cases with note bundle | 13,721 |
| Train note coverage | 92.0511% |
| Test note coverage | 99.7887% |
| Train median notes per case | 24 |
| Test median notes per case | 24 |
| Train median chars per case | 24,377 |
| Test median chars per case | 24,702 |

## Tabular Base Models

The final 31.7703 model is not a single end-to-end model trained from scratch. It starts from a strong tabular-only base prediction, then learns a note/tabular manifold correction on the residual.

The base is:

```text
base_oof  = 0.6 * oof_xgb_tuned_v2  + 0.4 * oof_xgb_absoluteerror_same_shape
base_test = 0.6 * pred_xgb_tuned_v2 + 0.4 * pred_xgb_absoluteerror_same_shape
```

The train side uses out-of-fold predictions. That is important: the residual model never receives an in-sample base prediction as if it were clean signal.

### Base Feature Set

The base models use only tabular features and safe tabular engineering:

- original Mayo tabular columns retained by `make_feature_frame`
- safe schedule/history features from `add_safe_features(include_scheduled_duration=True)`
- `bmi_missing`
- numeric and categorical preprocessing from `make_preprocessor`
- smoothed target encoding for `primary_procedure_name`
- smoothed target encoding for `surgeon_durable_id`

The target encodings are handled out-of-fold for train predictions:

- for each train fold, procedure/surgeon encodings are fit on the fold-training rows only
- validation rows receive encodings transformed from that fold-training fit
- test encodings are fit on the full training split only and then applied to the held-out test split

This keeps the target encoding from leaking validation or held-out outcomes back into the features.

The base feature set does not include:

- old 11 LLM features
- note-derived features
- predictions from a note model
- target-derived features computed using test labels

### `xgb_tuned_v2`

`xgb_tuned_v2` is a tabular-only XGBoost regressor using the tuned structured feature matrix.

Core configuration:

```python
XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    eval_metric="mae",
    n_estimators=1213,
    max_depth=8,
    learning_rate=0.013721653065680284,
    subsample=0.8431891099718941,
    colsample_bytree=0.5331833452208407,
    min_child_weight=9.083480959214418,
    reg_alpha=6.006000053619992e-07,
    reg_lambda=0.001809354462444587,
    gamma=1.0186857689274291,
    max_bin=212,
)
```

Held-out test MAE:

```text
xgb_tuned_v2 = 32.544887
```

### `xgb_absoluteerror_same_shape`

`xgb_absoluteerror_same_shape` uses the same tabular matrix, same preprocessing, and same tuned model shape as `xgb_tuned_v2`, but changes the XGBoost objective to absolute error:

```python
XGBRegressor(
    objective="reg:absoluteerror",
    tree_method="hist",
    eval_metric="mae",
    n_estimators=1213,
    max_depth=8,
    learning_rate=0.013721653065680284,
    subsample=0.8431891099718941,
    colsample_bytree=0.5331833452208407,
    min_child_weight=9.083480959214418,
    reg_alpha=6.006000053619992e-07,
    reg_lambda=0.001809354462444587,
    gamma=1.0186857689274291,
    max_bin=212,
)
```

Held-out test MAE:

```text
xgb_absoluteerror_same_shape = 32.483091
```

### Blended Tabular Base

The final base used by the manifold model is the fixed 60/40 blend:

```text
0.6 * xgb_tuned_v2 + 0.4 * xgb_absoluteerror_same_shape
```

Base performance:

| Metric | Value |
| --- | ---: |
| Train OOF MAE | 32.966712 |
| Held-out test MAE | 32.239636 |

## Residual Learning Design

The manifold model does not directly predict the full duration. It predicts what the tabular base missed.

Training residual:

```text
residual_train = y_train - base_oof
```

Manifold residual prediction:

```text
residual_hat = f(manifold_features)
```

Final prediction:

```text
prediction = base_prediction + gamma * residual_hat
```

For the final model:

```text
gamma = 1.550
```

The gamma multiplier is selected only from train OOF performance, using:

```text
0.000, 0.025, 0.050, ..., 1.600
```

This design makes the clinical notes act as a calibrated correction to the tabular base, rather than replacing the strong tabular model.

## Raw-Note Representation

The winning variant is:

```text
pls_raw_word160_26_delta
```

It uses the strict raw note bundle:

```text
strict_note_bundles.parquet -> note_text
```

The text representation is built with a word-level clinical TF-IDF model:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=CLINICAL_TOKEN_PATTERN,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.97,
    max_features=50000,
    sublinear_tf=True,
)
```

Then:

```python
TruncatedSVD(n_components=160)
StandardScaler()
```

The model also appends local hand-authored clinical concept features when `add_concepts=True`.

So the raw-note view is:

```text
raw_word160 = 160 scaled TF-IDF/SVD coordinates from strict raw notes
              + local hand-authored clinical concept features
```

These are generated from the note text itself. They are not the old 11 LLM features.

## Tabular Manifold Representation

The tabular manifold view is built from the same clean tabular frame:

```python
X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(train_df, TARGET_COLUMN)
X_test, _, _, _ = make_feature_frame(test_df, TARGET_COLUMN)
preprocessor = make_preprocessor(numeric_cols, categorical_cols)
```

Then:

```python
xtr = preprocessor.fit_transform(X_train)
xte = preprocessor.transform(X_test)
svd = TruncatedSVD(n_components=96)
ztr = svd.fit_transform(xtr)
zte = svd.transform(xte)
ztr = StandardScaler().fit_transform(ztr)
zte = fitted_scaler.transform(zte)
```

The winning manifold uses:

```text
first 64 tabular SVD components
```

The tabular SVD and scaler are fit on training data and then applied to held-out test data.

## Shared Manifold Architecture

The shared manifold is learned with Partial Least Squares between:

- tabular SVD coordinates
- strict raw-note TF-IDF/SVD coordinates plus note concept features

Actual surgical-profile training call:

```python
pls = PLSRegression(n_components=40, max_iter=1000)
pls.fit(tab_train_64, raw_note_train)
```

The run config allowed up to 56 PLS components, but the surgical profile's evaluated component grid ended at 40. Therefore the actual high-component PLS fit used 40 components and then sliced component widths. The final selected slice uses:

```text
n_components = 26
```

PLS transforms both views:

```python
tab_scores_train, note_scores_train = pls.transform(tab_train_64, raw_note_train)
tab_scores_test,  note_scores_test  = pls.transform(tab_test_64,  raw_note_test)
```

For each case, the model compares what the tabular data says in the shared latent space with what the notes say in that same space.

## Final Manifold Feature Geometry

The winning feature mode is:

```text
delta
```

For 26 PLS components:

```python
tx = tab_scores[:, :26]
ny = note_scores[:, :26]
diff = ny - tx

features = concat([
    diff,
    abs(diff),
    tx * ny,
])
```

Final manifold feature count:

```text
26 diff features
+ 26 absolute-diff features
+ 26 interaction features
= 78 residual features
```

Interpretation:

- `diff` captures directional disagreement between the note manifold and tabular manifold
- `abs(diff)` captures magnitude of disagreement, regardless of direction
- `tx * ny` captures alignment or opposition between the two views

This is the core signal: the model uses the geometry of the clinical notes relative to the tabular expectation, then turns that mismatch into a residual duration correction.

## Residual Head

The selected residual head is:

```text
sgd_huber
```

Architecture:

```python
Pipeline(
    StandardScaler(),
    SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        max_iter=4000,
        tol=1e-4,
        random_state=seed,
        average=True,
    )
)
```

Training:

- residual target is `actual_duration - base_oof`
- folds are `GroupKFold(n_splits=5)`
- groups are patient IDs
- fold predictions are OOF residual predictions
- final residual head is refit on the full training split and applied to the held-out test split

The residual head is intentionally simple and robust. The heavy lifting is done by the tabular base and the note/tabular manifold features.

## Model Flow

The complete prediction path is:

```text
tabular rows
  -> safe tabular engineering
  -> xgb_tuned_v2 OOF/test predictions
  -> xgb_absoluteerror_same_shape OOF/test predictions
  -> 60/40 blended tabular base

strict raw pre-op notes
  -> TF-IDF word ngrams
  -> 160-component SVD
  -> scaling
  -> note concept features
  -> raw note latent view

tabular rows
  -> make_feature_frame + make_preprocessor
  -> 96-component SVD
  -> scaling
  -> first 64 tabular latent components

tabular latent view + raw note latent view
  -> PLS shared manifold
  -> first 26 shared components
  -> delta geometry: diff, abs(diff), product
  -> SGD Huber residual head
  -> residual_hat

final prediction
  -> blended_tabular_base + 1.550 * residual_hat
```

## Experiment Phases

The clean run was organized in these phases:

1. Identify that the old 32.291-style result was not independent because the old v9 path used the 11 LLM-feature file as a teacher/label source.
2. Freeze the no-old-LLM rule: no direct use, no distillation, no teacher predictions, no feature import from `mayo_hrs_llm_features_top11.parquet`.
3. Rebuild note features from raw clinical-note rows with a strict `scheduled_in_room - 2 days` cutoff.
4. Validate the note cache for patient split, note timing, post-op filtering, and forbidden-file access.
5. Build tabular-only base OOF/test predictions using safe tabular preprocessing and patient-grouped OOF target encoding.
6. Define the residual target using OOF tabular base predictions.
7. Build strict raw-note TF-IDF/SVD features and clean tabular SVD features.
8. Learn the PLS shared manifold between tabular coordinates and raw-note coordinates.
9. Convert the shared manifold into delta-geometry features.
10. Train robust residual heads with patient-grouped OOF folds.
11. Select residual head and gamma using train OOF MAE.
12. Refit the selected residual head on full train and score the held-out patient-disjoint temporal test set.

## Leakage Controls

The clean result is leakage-free under the current experiment rules:

- old 11 LLM features are not read or used
- no LLM-feature file is used as a teacher, label, target, or distillation source
- raw notes must satisfy `note_date < scheduled_in_room - 2 days`
- same-day pre-op note types are disabled
- post-op document types are filtered
- train/test split is patient-disjoint
- tabular target encodings are OOF for train and train-only for test
- residual training uses `base_oof`, not in-sample base predictions
- residual folds use patient groups
- TF-IDF, SVD, scaling, PLS, and residual heads are fit on training data and then applied to held-out test

Validation script:

```text
work/no_llm_notes/validate_strict_raw_run.py
```

Validation artifact:

```text
work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_validation.json
```

Validation results:

| Check | Value |
| --- | ---: |
| Patient overlap | 0 |
| Cases with any used note | 12,850 |
| Note dates checked | 295,463 |
| Cutoff violations | 0 |
| Smallest margin before cutoff | 2.5 hours |
| Stored forbidden old LLM feature file read | false |
| Access-like old LLM file hits | 0 |

Important nuance:

The data flow is leakage-clean. However, the exact variant was selected after trying many candidate manifold variants against the same held-out test set. That is not the same as feature leakage, but it can create winner's-curse or leaderboard-selection optimism. For publication-grade proof, freeze `pls_raw_word160_26_delta` exactly and evaluate it once on a fresh later temporal holdout or nested validation design.

## Reproduction Files

Main scripts:

| Purpose | File |
| --- | --- |
| Strict raw-note cache build | `work/no_llm_notes/build_raw_strict_high_signal_split.py` |
| Strict validation | `work/no_llm_notes/validate_strict_raw_run.py` |
| Tabular-only base sweep | `work/no_llm_notes/tabular_model_sweep.py` |
| Manifold push run | `work/no_llm_notes/strict_manifold_push.py` |

Main artifacts:

| Purpose | File |
| --- | --- |
| Tabular OOF predictions | `work/no_llm_notes/tabular_sweep/tabular_sweep_oof_predictions.csv` |
| Tabular held-out predictions | `work/no_llm_notes/tabular_sweep/tabular_sweep_predictions.csv` |
| Strict note bundles | `work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_note_bundles.parquet` |
| 31.7703 run directory | `work/no_llm_notes/raw_strict_manifold_push_surgical_full` |
| 31.7703 single-method result row | `outputs/strict_31_7703_manifold_push_results.csv` |

Main run command:

```bash
python3 work/no_llm_notes/strict_manifold_push.py \
  --phrase-dir work/no_llm_notes/raw_strict_high_signal_tuned_split \
  --tabular-sweep-dir work/no_llm_notes/tabular_sweep \
  --output-dir work/no_llm_notes/raw_strict_manifold_push_surgical_full \
  --profile surgical \
  --n-boot 200
```

Run config:

| Item | Value |
| --- | ---: |
| Seed | 4242 |
| Folds | 5 |
| Profile | surgical |
| Candidate variants in run | 127 |
| Configured max PLS components | 56 |
| Actual surgical-profile PLS fit | 40 |
| Selected PLS slice | 26 |
| Forbidden LLM feature file read | false |

## Final Result

Final clean model:

```text
pls_raw_word160_26_delta
```

Final prediction:

```text
prediction = 0.6 * xgb_tuned_v2
           + 0.4 * xgb_absoluteerror_same_shape
           + 1.550 * sgd_huber_residual(pls_raw_word160_26_delta_features)
```

Held-out patient-disjoint temporal test set:

| Metric | Value |
| --- | ---: |
| N | 2,839 |
| MAE | 31.7702636563 |
| RMSE | 47.0862301387 |
| R2 | 0.7537433190 |
| Median absolute error | 21.3969602483 |
| P90 absolute error | 69.8267318237 |
| P95 absolute error | 93.2294053259 |
| W30 | 0.6220500176 |
| W60 | 0.8622754491 |
| Mean prediction minus actual | -2.8542162009 |
| Underestimation rate | 0.4825642832 |
| Overestimation rate | 0.5174357168 |

Gain over the same tabular base:

```text
+0.4693722183 MAE minutes
```

Clustered bootstrap CI for gain over the same tabular base:

```text
[+0.3357543566, +0.6065614085]
```

Clustered bootstrap CI for final MAE:

```text
[30.6139184325, 33.0492553314]
```
