# Conditional Residual Clinical Notes Learner

**A staged residual architecture for surgical procedure-duration prediction from structured records and pre-operative clinical narrative**

---

| | |
| --- | --- |
| **Author** | Itamar Zernitsky |
| **Affiliation** | Department of Mathematics, Bar-Ilan University, Ramat-Gan, Israel |
| **Research team** | Opmed.ai — informal "Shadows" research team |
| **Prepared for** | Opmed.ai |
| **Document status** | Complete internal method record |
| **Model identity** | `strict_meta_gate_top_refine_oof_selected` |
| **Held-out MAE** | `31.5564180731` minutes |
| **Iterations completed** | **One** |
| **Notion page** | [Conditional Residual Clinical Notes Learner](https://app.notion.com/p/Conditional-Residual-Clinical-Notes-Learner-fc03e1f1817183a8b5a881d5c403d7b4?source=copy_link) |
| **Jira** | DSC-127 — Data Science, Opmed.ai |

---

## 0. How to read this document

This is the complete record of the Conditional Residual Clinical Notes Learner (CRCNL): its motivation, its data handling, every preprocessing decision, the full mathematical formalism of every stage, the risk-residual analysis, the validation and leakage audit, and — importantly — **what has *not* yet been done**.

Three things deserve emphasis before the detail:

1. **This is a residual architecture, not a monolithic model.** The clinical notes never attempt to predict duration. They predict only what the structured record could not.
2. **We have completed exactly one residual iteration.** The architecture is explicitly designed to be iterated to convergence. Section 15 gives the formal iteration operator, the shrinkage schedule, and four candidate stopping rules. Nothing in the current result exhausts the method.
3. **The narrative extraction layer is the weakest link and is scheduled for replacement.** The current concept extraction is five hand-written regular expressions. Section 8 specifies the replacement: a service-based extraction layer with retrieval, a bi-encoder ranker and a cross-encoder reranker, designed so that the pipeline stops depending on hand-authored lexical patterns.

All mathematics uses Notion-compatible inline delimiters, e.g. $y_i$, $\hat y_i$, $\mathrm{MAE}$.

---

## 1. Executive summary

### 1.1 Headline result

| Quantity | Value |
| --- | ---: |
| Selected model | `strict_meta_gate_top_refine_oof_selected` |
| Training cases | `10,882` |
| Held-out cases | `2,839` |
| Train/held-out patient overlap | `0` |
| Training OOF MAE | `32.7156839203` min |
| **Held-out MAE** | **`31.5564180731` min** |
| Gain over the strict PLS manifold | `+0.2138455833` min |
| Paired bootstrap CI for that gain | `[+0.1001, +0.3259]` |
| Gain over the procedure-family gate | `+0.0511042965` min |
| Paired bootstrap CI for that gain | `[−0.0108, +0.1123]` |
| Improved-case rate vs strict PLS | `48.89%` |
| Large wins (> 10 min) | `23` |
| Large losses (< −10 min) | `16` |

The saved prediction table recomputes independently to `31.556418073067334`.

### 1.2 The staged ladder

Each stage is a strict refinement of the one above it. Every number is held-out MAE in minutes.

| Stage | Model | Train OOF MAE | Held-out MAE |
| --- | --- | ---: | ---: |
| 0 | Zero-information (global median) | — | `≈ 81` |
| 1a | `xgb_tuned_v2` (squared error) | — | `32.544887` |
| 1b | `xgb_absoluteerror_same_shape` | — | `32.483091` |
| 1c | **Tabular base** `0.6·(1a) + 0.4·(1b)` | `32.966712` | `32.239636` |
| 2 | **Strict PLS note/tabular manifold** | `32.796418` | `31.770264` |
| 3 | **Procedure-family calibration** | `32.769280` | `31.607522` |
| 4 | Risk/residual candidate bank | *(bank, not a single model)* | — |
| 5 | **Top-refine meta-gate (selected)** | `32.715684` | **`31.556418`** |

Total improvement from tabular base to final: **`0.683218` minutes**, of which **`0.469372` minutes** — roughly 69% — comes from the clinical notes alone (stage 1c → stage 2).

### 1.3 What the notes are worth

The single most important scientific claim in this work:

> Pre-operative clinical narrative, restricted to text finalised more than two days before the case, and used **only** as a residual correction on a strong tabular model, is worth **0.469 minutes of MAE** with a clustered bootstrap interval of `[+0.3358, +0.6066]`.

That interval excludes zero. It is the one result in this document that does not depend on any gate, bin, or selection heuristic.

---

## 2. Motivation

### 2.1 The operational problem

An operating-room schedule is assembled from estimates of case duration. The cost structure is asymmetric in practice but linear in the unit that matters:

- An under-estimate runs the day over. Staff are held past shift; later cases are delayed or cancelled.
- An over-estimate leaves the room idle. Capacity is destroyed silently.

Both are paid for in **minutes of error**, and a minute of overrun on an easy case costs the same as a minute on a hard one. This is the argument for absolute error rather than squared error, and it is made precise in Section 3.2.

### 2.2 Why the procedure code is not enough

The scheduling estimate is normally derived from the procedure code. **Procedure code and procedure difficulty are not the same object.** Two cases carrying identical codes can differ by hours:

- a first-time implant in normal anatomy, versus
- a redo with abandoned leads, venous occlusion, and prior surgical scarring.

That distinction is frequently absent from the structured record. It is frequently present in the pre-operative notes — written in prose, by many hands, over the weeks before the case.

The empirical signature of this is visible in our own family table (Section 13.3): the lead/device extraction family has a base MAE of `49.14` minutes and a mean actual duration of `216` minutes, while the device-implant family has a base MAE of `25.52` minutes at a mean duration of `145` minutes. The hard families are hard in a way the structured record does not see.

### 2.3 Why residual learning rather than joint learning

Three reasons, in order of importance.

**Statistical.** The tabular model already captures procedure identity, schedule, provider, demographics and comorbidity structure. A joint note+tabular model must relearn all of it from a text channel that is noisy, high-dimensional and heterogeneous. With $n \approx 11{,}000$ training cases that is a losing trade. The residual formulation gives the note channel a target that is *already orthogonal* to everything the tabular model explains.

**Operational.** A residual correction degrades gracefully. If the note channel is unavailable for a case, the prediction falls back to the tabular base with no re-architecture. Note coverage is `92.05%` on train and `99.79%` on held-out — not `100%` — so this matters.

**Epistemic.** A residual correction is *auditable as a correction*. We can report its mean magnitude (`3.36` minutes), its alignment rate with the true residual (`56.3%`), and its per-family gain. None of that is legible in a joint model.

### 2.4 Why "conditional"

The correction is not applied uniformly. The architecture learns, at three increasing levels of resolution, *how much to trust the note correction for this case*:

1. by **procedure family** (Section 11);
2. by **learned risk stratum** — where the model expects to be unreliable (Section 12);
3. by **candidate disagreement** — where two already-calibrated predictors disagree (Section 13).

This is the "conditional" in Conditional Residual Clinical Notes Learner. The correction is a function; the *trust* in the correction is a second, separately learned function.

### 2.5 The central empirical lesson

Stated once here because it shaped the whole architecture:

> **Error magnitude is learnable. Error direction is not.**

Across every risk head we trained, out-of-fold correlation with absolute error reached `0.45–0.46`; out-of-fold correlation with signed residual or with final-vs-base gain never exceeded `0.08`. The architecture therefore treats risk as a **trust-allocation signal** — deciding how much of a correction to apply — and never as a direction oracle. Section 12.4 gives the numbers.

---

## 3. Problem formalism

### 3.1 Target and estimand

For case $i$ let

$$y_i = \texttt{case\_actual\_duration\_time}_i \in (0,\infty)$$

measured in minutes. Let $x_i$ denote the structured (tabular) covariates and $z_i$ the eligible pre-operative note bundle. The prediction must be issued at time $\tau_i$, the moment the case is scheduled.

We seek $\hat f$ minimising

$$\mathrm{MAE}(f) = \frac{1}{n}\sum_{i=1}^{n}\bigl|y_i - f(x_i, z_i)\bigr|.$$

### 3.2 Absolute loss targets the conditional median

This is not a stylistic preference; it changes what the model learns. For fixed $x$,

$$\arg\min_a \mathbb{E}\bigl[(y-a)^2 \mid x\bigr] = \mathbb{E}[y\mid x], \qquad
\arg\min_a \mathbb{E}\bigl[|y-a| \mid x\bigr] = \operatorname{med}(y\mid x).$$

**Proof of the second.** Let $F$ be the conditional CDF of $Y$ given $x$ and $g(a)=\mathbb{E}[|Y-a|\mid x]$. Writing

$$g(a)=\int_{-\infty}^{a}(a-y)\,dF(y)+\int_{a}^{\infty}(y-a)\,dF(y)$$

and differentiating gives $g'(a)=F(a)-\bigl(1-F(a)\bigr)=2F(a)-1$, which vanishes exactly at $F(a)=\tfrac12$. $g$ is convex, so this is the global minimum. $\blacksquare$

**Consequence for a skewed target.** The duration distribution is right-skewed (mean exceeds median by roughly 21 minutes on the parent cohort). Optimising squared error would inflate predictions on the ordinary majority in order to hedge against a rare tail. Absolute loss does not; it fits the typical case and lets the tail be wrong. For a schedule, that is the correct behaviour.

This is why the tabular base is a **blend of a squared-error and an absolute-error learner** rather than either alone (Section 9.3): the squared-error member contributes conditional-mean sensitivity, the absolute-error member contributes median-like robustness, and the 60/40 blend was selected on out-of-fold MAE.

### 3.3 The irreducible floor

For any information set $\mathcal{G}=\sigma(X,Z)$,

$$R^\star(\mathcal{G}) = \inf_{f\ \mathcal{G}\text{-measurable}} \mathbb{E}\bigl|Y-f\bigr| = \mathbb{E}\bigl[\operatorname{MAD}(Y\mid X,Z)\bigr],
\qquad \operatorname{MAD}(P):=\min_a \mathbb{E}_P|Y-a|.$$

$R^\star$ is a property of the joint distribution, not of any model. Its zero-information counterpart is $R^\star(\varnothing)=\operatorname{MAD}(Y)\approx 81$ minutes, the reference against which every reported error should be read. A model at `31.56` minutes has removed roughly **61%** of the marginal dispersion.

$R^\star$ is not decoration: it is the quantity that makes the stopping rule in Section 15 well posed. An iteration scheme that cannot say how far it is from the floor cannot say when to stop.

### 3.4 The decomposition the architecture implements

The architecture is a constructive realisation of

$$y_i = \underbrace{m(x_i)}_{\text{structured base}} \;+\; \underbrace{g(x_i, z_i)}_{\text{note residual correction}} \;+\; \underbrace{\varepsilon_i}_{\text{irreducible or unmodelled}}$$

with the crucial refinement that $g$ is not applied raw but **gated**:

$$\hat y_i = m(x_i) + \gamma\bigl(c(x_i)\bigr)\cdot \mathrm{clip}\Bigl(g(x_i,z_i),\,-C_{c},\,+C_{c}\Bigr),$$

where $c(\cdot)$ assigns a case to a trust category and $\gamma(\cdot), C_{(\cdot)}$ are learned, non-negative, category-specific trust and cap parameters.

---

## 4. Data

### 4.1 Sources

| Source | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| Tabular case records | `14,195` | `55` | perioperative fields, scheduling timestamps, comorbidity block, historical aggregates |
| Raw clinical notes | `3,179,288` | `4` | `epic_id`, `doc_date`, `doc_type`, `content_markdown` |

Notes are matched to cases by patient and then filtered against a case-specific timestamp (Section 5.3). Raw note text is never distributed.

### 4.2 Tabular schema

**Identity and split fields**

`case_durable_id`, `patient_durable_id`, `scheduled_in_room`

**Case descriptors**

`case_anesthesia`, `case_urgency_level`, `case_surgery_patient_class`, `case_admission_patient_class`, `surgeon_durable_id`, `procedures`, `case_num_procedures`, `case_num_panels`, `case_num_providers`, `primary_procedure_name`, `is_redo`, `prior_same_proc_duration`, `redo_time_gap_days`

**Patient descriptors**

`patient_sex`, `patient_smoking_status`, `patient_age`, `age_group`, `bmi`

**Comorbidity block**

Coronary artery disease · heart failure · arrhythmia · conduction block · valve disease · cardiac device · vascular disease · cardiac inflammation · ventricular dysfunction · thromboembolism · pulmonary · renal · metabolic · hypertensive · haematologic · neurologic · hepatic · oncologic · psychiatric · musculoskeletal · lifestyle · gastrointestinal · immune · transplant

**Scheduling timestamps**

`scheduled_setup_start`, `scheduled_in_room`, `scheduled_out_of_room`, `scheduled_cleanup_complete`

**Outcome-time fields — present in the raw table, excluded from predictors**

`case_actual_patient_in_room`, `case_actual_duration_time`, `case_actual_prep_time`, `case_actual_procedure_time`, `case_actual_wrapup_time`

Also excluded: `patient_birth_date`, `all_procs_not_performed`, case ID, patient ID.

### 4.3 The explicitly forbidden source

The former 11-variable LLM feature table

```text
data/mayo/mayo_hrs_llm_features_top11.parquet
```

is **not** an input, a teacher, a label, a distillation target, or a selection source anywhere in this work. Its filename appears in the codebase only inside defensive sentinels and audit scans. The selected run records `forbidden_llm_feature_file_read = false`, and a static code scan confirms zero access-like hits.

This constraint exists because an earlier result (`≈ 32.29`) was found to have used that file as a teacher/label source, making it non-independent. Every number in this document is from the post-constraint rebuild.

### 4.4 Where the data lives

No patient data travels with this document or with the code repository. The source files are held in S3 and must be provisioned in an authorised environment.

**Bucket** `opmed-integration-raw-data-mayo`, **prefix** `clinical_notes/anonymized_hrs_notes_shared/`

| File | S3 URI |
| --- | --- |
| Dataset — tabular + lags + target | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_tabular_features_with_lags_and_durations.parquet` |
| Anonymized notes | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_notes_anonymized_merged.parquet` |
| LLM features (top 11) — **not used by this model** | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_llm_features_top11.parquet` |

**How the files join**

```text
dataset.case_durable_id     ->  LLM features.case_durable_id   (case-keyed, left join)
dataset.patient_durable_id  ->  notes.epic_id                  (1:many, patient-keyed)
```

Notes are keyed by **patient**, not by case. A patient with several procedures has all their notes in one bucket, so attaching a case to "its" notes is a **date filter**, not a join key — which is precisely why the cutoff rule of §5.3 is load-bearing rather than cosmetic.

Notes schema: `epic_id`, `doc_date`, `doc_type`, `content_markdown`.

### 4.5 A cutoff discrepancy worth recording

The shipped 11-feature LLM table was built with

```python
cutoff = case["case_actual_patient_in_room"] - pd.Timedelta(days=2)
```

— anchored to the **actual** in-room time. Every note used is still genuinely dated before the surgery, so this is not leakage from the outcome itself. The defect is subtler: the actual surgery date is not fixed at prediction time, so a case rescheduled later inherits a larger note history than a creation-time deployment would ever have seen.

This model uses `scheduled_in_room − 2 days` instead (§5.3), which is known at prediction time. The data guide reaches the same conclusion independently: *if you are extracting new note-derived features of your own, use `scheduled_in_room` — it is known at prediction time, so it will not have this problem.*

Two consequences follow, and they are the reason the two artefacts must not be mixed:

- Reproducing the shipped 11 features **exactly** requires the actual-time cutoff. A "cleaner" cutoff yields different eligible notes and therefore different feature values.
- This model is built on the clean cutoff and **excludes those features entirely**, so the two are not interchangeable and the difference is not a bug in either.

### 4.6 A different cohort design also exists

The shared data guide describes an **expanding-window quarterly CV** on the actual case date: warm-start 2022 Q3–Q4 (lag pools only, never trained or evaluated), train-only 2023 Q1–Q4, eight OOF validation folds over 2024 Q1 – 2025 Q4, holdout 2026 Q1+.

This model instead uses a single temporal cut at the 80th percentile of `scheduled_in_room` with patient disjointness (§5.2). The quarterly design is closer to the nested forward-time selection recommended in §15.7 as the next validation step, and adopting it is a concrete, well-specified upgrade rather than an open research question.

Two dataset caveats from the same guide, neither of which affects the figures here but both of which matter on a rebuild: 6 rows have a null `primary_procedure_name` and 23 have `all_procs_not_performed = True` — the training pipeline filters only on target validity and does **not** drop these. The target is 100% populated and positive as shipped.

---

## 5. The prospective information boundary

### 5.1 Filtration

Let $\mathcal{F}_t$ be the $\sigma$-field of everything knowable at time $t$. A feature is admissible for case $i$ if it is $\mathcal{F}_{\tau_i}$-measurable, where $\tau_i$ is the scheduling moment.

**Leakage is a measurability failure**, not a statistical subtlety. It is a mis-specified $\sigma$-field, and it produces error estimates that cannot be attained in deployment.

The diagnostic value of the exclusion in Section 4.2 is worth stating as a standing rule:

> With the three `case_actual_*` summands included, a routine gradient-boosted model reaches an apparent MAE of `2.56` minutes against a zero-information baseline of `81`. **Any reported error more than an order of magnitude below the marginal dispersion should be treated as a leak until proven otherwise.**

### 5.2 Case filtering and the temporal patient-disjoint split

Rows without `scheduled_in_room`, without the target, or with non-positive duration are removed. Remaining rows are sorted by scheduled time.

Let $t_i$ be the scheduled in-room timestamp and $q_{0.8}$ its 80th percentile. Candidate held-out patients are those with any case at or after $q_{0.8}$. Then:

$$\text{train} = \{i : t_i < q_{0.8}\ \wedge\ \mathrm{patient}(i)\notin \mathcal{P}_{\text{held-out}}\},$$
$$\text{held-out} = \{i : t_i \ge q_{0.8}\ \wedge\ \mathrm{patient}(i)\in \mathcal{P}_{\text{held-out}}\}.$$

| Item | Value |
| --- | ---: |
| Train cases | `10,882` |
| Held-out cases | `2,839` |
| Train patients | `8,948` |
| Held-out patients | `2,499` |
| **Patient overlap** | **`0`** |
| Missing patient IDs | `0` / `0` |

Two separate controls are at work and they answer different questions. The **temporal** cut answers *how would this have performed if fielded on the boundary date*. The **patient disjointness** prevents the model from memorising a patient whose entire note history and comorbidity documentation repeats across cases — without it the evaluation measures recall rather than generalisation.

### 5.3 The note cutoff

A note is eligible for case $i$ only if

$$\texttt{doc\_date} < \texttt{scheduled\_in\_room}_i - 2\ \text{days}.$$

Two design points matter and are easy to get wrong.

**Anchor on the scheduled time, not the actual time.** The actual in-room time is unknown at prediction. Anchoring on it would let a case rescheduled from March to June inherit three months of note history no deployment could have seen.

**Use a two-day margin, not zero.** A note written the morning of the procedure is technically pre-operative but is unavailable at scheduling and may encode the day's operative plan in a form close to the outcome itself. Two days removes that class of near-outcome documentation.

**Note-cache construction settings**

| Setting | Value |
| --- | ---: |
| Timestamp policy | exact timestamp |
| Same-day pre-operative note types | disabled |
| Post-operative document types | rejected |
| Maximum lookback | `548` days |
| Maximum notes per case | `24` |
| Maximum characters per note | `2,200` |
| Maximum characters per case | `48,000` |
| Maximum selected sentences | `72` |

**Audit of the built cache**

| Audit item | Value |
| --- | ---: |
| Raw note rows scanned | `3,179,288` |
| Rows matched by patient | `3,144,540` |
| Post-operative doc-type rows rejected | `145,426` |
| Eligible note events attached | `295,463` |
| Cases with a bundle | `13,721` |
| Cases with at least one used note | `12,850` |
| Note dates checked | `295,463` |
| **Cutoff violations** | **`0`** |
| Smallest margin before cutoff | `2.5` hours |
| Train note coverage | `92.0511%` |
| Held-out note coverage | `99.7887%` |
| Median selected notes per case | `24` (both splits) |
| Median characters per case | `24,377` train / `24,702` held-out |

**A nuance worth recording.** Historical notes can legitimately contain words such as *post* or *discharge*, because they may describe an older event. **Timestamp eligibility, not a word blacklist, is the decisive boundary.** As a secondary check, the high-signal bundles contained zero `actual duration` probe hits on either split and total-time language remained uncommon. Production auditing should retain both the timestamp check and the lexical probe.

### 5.4 The scheduling-feature caveat

The run uses *planned* scheduling information. Derived fields:

- `scheduled_year`, `scheduled_month`, `scheduled_dayofweek`, `scheduled_hour`
- `scheduled_is_weekend`
- `scheduled_room_minutes` $=$ scheduled out-of-room $-$ scheduled in-room
- `scheduled_setup_lead_minutes` $=$ scheduled in-room $-$ scheduled setup start
- `scheduled_cleanup_minutes` $=$ scheduled cleanup complete $-$ scheduled out-of-room

These are **not** actual outcome times, and they are knowable at $\tau_i$ by construction — the booked slot exists *because* the case was scheduled. But they change the prediction task. The `31.5564` result applies to the **schedule-informed** setting. If the deployment question is *predict before a schedule is built*, these fields must be removed and the entire pipeline retrained and re-evaluated.

Two residual risks attach to this and should be audited before deployment:

1. **Revision risk.** If the scheduling system overwrites the booked slot when a case is amended, the stored field is not $\mathcal{F}_{\tau_i}$-measurable. The check is whether the source keeps a booking-version history or an `as_of` timestamp; failing that, an anomalously tight ridge along the diagonal of booked-vs-actual is the diagnostic signature.
2. **Feedback risk.** The booked slot may *causally* influence realised duration — staff pace to the block. Deploying a model to *set* the slot then changes the distribution it learned. This is a deployment-loop concern, not leakage, but it belongs in the limitations.

---

## 6. Tabular preprocessing and data handling

### 6.1 Feature frame construction

The builder removes, in order: identity columns, `patient_birth_date`, the target, every column beginning with `case_actual_`, date-suffixed columns, and the raw schedule timestamps **after** deriving the permitted planned-time features of Section 5.4.

Then:

- Boolean variables are cast to integers.
- Numeric missing values are imputed with the **training median**.
- Categorical missing values are imputed with the **most frequent training category**.
- Categories are one-hot encoded, unknown categories ignored, with a minimum category frequency of `10` where supported.
- An explicit `bmi_missing` indicator is added, because BMI missingness is both common and informative.

Every one of these is a *learned* quantity and every one is estimated on the fitting partition only. Computing them on pooled data is transductive leakage: the evaluation rows influence the representation even though their labels are unused.

### 6.2 Smoothed target encodings

Two high-cardinality categoricals — `primary_procedure_name` and `surgeon_durable_id` — are target-encoded with additive smoothing. For a category $c$ with count $n_c$, category mean $\bar y_c$, global mean $\bar y$, and smoothing constant $k$:

$$\mathrm{TE}(c) = \frac{n_c\,\bar y_c + k\,\bar y}{n_c + k}.$$

| Field | $k$ |
| --- | ---: |
| `primary_procedure_name` | `20` |
| `surgeon_durable_id` | `15` |

Unknown validation or held-out categories receive the training global mean $\bar y$.

**The fold discipline is the whole point.** For each training fold, encodings are fit on the fold-training rows only and *transformed* onto the validation rows. For the held-out split, encodings are fit on the full training split only. Without this, the encoding carries the row's own label and the entire downstream residual construction is contaminated.

---

## 7. Note extraction as currently implemented

This section documents what exists. Section 8 specifies what should replace it.

### 7.1 Bundle assembly

For each case, eligible notes (Section 5.3) are concatenated, capped, and stored as a single `note_text` field per case, together with metadata: note count, character count, document-type counts, recency distribution.

### 7.2 Lexical representation

A word-level clinical TF-IDF model:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=CLINICAL_TOKEN_PATTERN,   # r"(?u)\b(?=[^\W\d_]*[^\W\d_])\w[\w-]{1,}\b"
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.97,
    max_features=50000,
    sublinear_tf=True,
)
```

The clinical token pattern deliberately requires at least two non-digit word characters, which suppresses pure numerics and single characters while retaining hyphenated clinical compounds.

With sublinear term frequency, for term $j$ in bundle $i$ with raw count $c_{ij}>0$:

$$\mathrm{tfidf}_{ij} = \bigl(1+\log c_{ij}\bigr)\cdot\log\frac{N+1}{\mathrm{df}_j+1},$$

subject to the library's $\ell_2$ row normalisation.

### 7.3 Dimensionality reduction

$$\mathrm{TruncatedSVD}(n=160) \;\rightarrow\; \mathrm{StandardScaler}$$

fit on training bundles and applied unchanged to held-out bundles. This yields the note latent view $Z \in \mathbb{R}^{n\times 160}$.

### 7.4 Hand-authored concept indicators — the component to be replaced

Five binary indicators are appended, computed by regular-expression search over the lowered bundle text:

| Concept | Patterns |
| --- | --- |
| `redo_prior_ablation` | `\bredo\b`, `repeat\s+ablation`, `prior\s+ablation` |
| `vt_pvc_instability` | `\bvt\b`, `\bvf\b`, `ventricular\s+tachycardia`, `icd\s+shock` |
| `device_lead_complexity` | `lead\s+(revision\|extraction\|failure\|fracture)`, `generator\s+change` |
| `congenital_complex_anatomy` | `\bfontan\b`, `transposition`, `congenital` |
| `access_support` | `\becmo\b`, `epicardial`, `pericardial\s+access` |

Formally, for concept $k$ with pattern set $\mathcal{P}_k$:

$$\mathrm{flag}_{ik} = \mathbb{1}\Bigl[\exists\, p\in\mathcal{P}_k : p \text{ matches } \mathrm{text}_i\Bigr].$$

These are generated from the eligible note text itself and are **not** the forbidden 11-variable LLM table.

### 7.5 Why this component is the weak link

Four structural problems, each independently sufficient to motivate replacement.

**Binary saturation.** With bundles of median `24,377` characters, the probability that *some* note mentions `congenital` at least once is high regardless of whether the current case is congenital. Prevalence bears this out: `vt_pvc_instability` fires on `40.75%` of cases. A flag that fires on two cases in five carries little conditional information.

**No negation or attribution handling.** `no evidence of VT` and `recurrent VT despite amiodarone` produce the identical feature value. Clinical prose is dense with negation, hypotheticals, family history, and historical resolution; a substring match sees none of it.

**Brittleness to phrasing and site.** The pattern list encodes one institution's dictation habits. `lead revision` matches; `revision of the RV lead` does not. Transfer to another site is unmeasured and probably poor.

**Empirically thin.** The concept-flag ablation table (Section 13.5) shows *negative* gain lift for four of the five flags — cases where the flag is present improve **less** than cases where it is absent. These indicators are not carrying their weight.

Their measured contribution is small enough that the architecture does not depend on them — which is exactly the right moment to replace them with something better.

---

## 8. Proposed replacement: retrieval-based extraction with ranker and reranker

This section is a **design specification, not a completed experiment.** It is written to be implementable and to be evaluated under the same leakage rules as everything else.

### 8.1 Design goals

| Goal | Rationale |
| --- | --- |
| Remove dependence on hand-authored lexical patterns | Section 7.5 |
| Handle negation, attribution and temporality | Clinical prose requires it |
| Produce **graded** evidence, not binary flags | Binary saturates at bundle length |
| Remain auditable | A clinician must be able to see *which sentence* drove a feature |
| Stay inside the note-eligibility boundary | Non-negotiable |
| Degrade gracefully if the service is unavailable | Coverage is not 100% |

### 8.2 Architecture

```text
eligible note bundle (per case)
  │
  ├─ 1. SEGMENT ─────────► sentence / passage units with doc_type + doc_date
  │
  ├─ 2. RETRIEVE ────────► bi-encoder recall: top-M passages per concept query
  │                        (M ≈ 50, cosine over a clinical embedding space)
  │
  ├─ 3. RERANK ──────────► cross-encoder: (concept query, passage) → relevance
  │                        keep top-K (K ≈ 8) per concept
  │
  ├─ 4. EXTRACT ─────────► structured-output API call over the K passages only
  │                        → graded assertion with negation / temporality
  │
  └─ 5. AGGREGATE ───────► per-concept evidence score → feature vector
```

The critical structural property: **the extraction model never sees the whole bundle.** It sees at most $K$ reranked passages per concept. This bounds cost, bounds context dilution, and makes every produced feature traceable to specific sentences.

### 8.3 Stage 1 — segmentation

Split each bundle into passages $\{s_{i1},\dots,s_{iM_i}\}$ of one to three sentences, each carrying its `doc_type`, `doc_date`, and character offset. Passages inherit the eligibility of their parent note; the two-day cutoff is enforced at bundle-construction time and never re-litigated downstream.

### 8.4 Stage 2 — bi-encoder retrieval (the ranker)

Let $\phi(\cdot)\in\mathbb{R}^d$ be a clinical sentence encoder and let $q_k$ be a natural-language query for concept $k$ (for example, *"prior catheter ablation of atrial fibrillation that failed or recurred"*). Rank passages by

$$\mathrm{sim}(q_k, s_{ij}) = \frac{\langle \phi(q_k), \phi(s_{ij})\rangle}{\|\phi(q_k)\|\,\|\phi(s_{ij})\|},$$

and retain the top $M$.

Two properties make this the right first stage. It is **cheap** — passages are embedded once per case and reused across all concepts, so cost is $O(M_i)$ embeddings, not $O(M_i \times K_{\text{concepts}})$. And it is **recall-oriented** — the bi-encoder's job is not to be right, only to not lose the relevant passage before the reranker sees it.

### 8.5 Stage 3 — cross-encoder reranking

The bi-encoder scores query and passage independently, so it cannot model their interaction. The cross-encoder scores them jointly:

$$\rho(q_k, s_{ij}) = \sigma\Bigl(\mathrm{CE}\bigl([q_k;\,s_{ij}]\bigr)\Bigr) \in (0,1),$$

and the top $K$ passages by $\rho$ are passed forward. This is the stage that distinguishes *"history of VT"* from *"no history of VT"* — a distinction invisible to both regex and cosine similarity.

The staged ranker→reranker structure is standard in retrieval for exactly this reason: recall is cheap and precision is expensive, so buy recall broadly and precision narrowly.

### 8.6 Stage 4 — extraction via structured service call

For each concept $k$ and case $i$, issue **one** structured-output call over the $K$ reranked passages, returning a typed object:

```json
{
  "concept": "redo_prior_ablation",
  "present": true,
  "assertion": "affirmed",          // affirmed | negated | hypothetical | family_history
  "temporality": "historical",      // current | historical | planned
  "severity": 0.72,                 // graded, 0-1
  "confidence": 0.88,
  "evidence_passage_ids": [4, 17]
}
```

Three engineering constraints make this safe and reproducible:

1. **Determinism.** Temperature zero, pinned model version, schema-validated output. The model version is recorded in the run config and is part of the artefact's identity.
2. **Caching.** Keyed on `(case_id, concept, passage_id_set, prompt_version, model_version)`. Re-runs cost nothing; a prompt change invalidates cleanly.
3. **Fallback.** If the service is unavailable or the schema validation fails, fall back to the Section 7.4 regex flag and set an `extraction_degraded` indicator. The model must never silently change its input distribution.

**Leakage discipline for this stage.** The extraction call sees *only* eligible passages. It never sees the target, the case duration, any outcome field, or any other case. It is a deterministic function of admissible text, so it is $\mathcal{F}_{\tau_i}$-measurable by construction — the same argument that licenses TF-IDF.

### 8.7 Stage 5 — aggregation into features

For concept $k$, combine the extraction with the reranker evidence:

$$e_{ik} = \underbrace{\mathbb{1}[\text{assertion} = \text{affirmed}]}_{\text{negation-aware}} \cdot \underbrace{\text{severity}}_{\text{graded}} \cdot \underbrace{\text{confidence}}_{\text{calibration}} \cdot \underbrace{w(\Delta t_{ik})}_{\text{recency}},$$

with an exponential recency weight over the age of the supporting evidence,

$$w(\Delta t) = \exp\bigl(-\Delta t / \tau_{\text{half}}\bigr),$$

$\tau_{\text{half}}$ selected inside training folds. Alongside $e_{ik}$, retain:

- $\rho^{\max}_{ik}=\max_j \rho(q_k,s_{ij})$ — peak retrieval evidence;
- $\bar\rho_{ik}$ over the top $K$ — evidence breadth;
- the count of affirmed passages — corroboration;
- an `extraction_degraded` flag.

This yields roughly $4$–$5$ features per concept instead of one binary. With a concept set of 12–15 (extending the current five toward the clinical families that matter), the narrative concept block grows from `5` to `≈ 60` graded features — while becoming *more* auditable, not less, because every value traces to named passages.

### 8.8 How to prove it is better

Replacing a component is a hypothesis, not an improvement. The evaluation must be pre-declared:

1. Freeze everything downstream of the note representation.
2. Rebuild the note latent view with the new concept block in place of the old.
3. Re-run the full stage-2 manifold and report train-OOF MAE.
4. **Accept only if train OOF improves**, with the held-out set untouched until the decision is made.
5. Report per-family and per-coverage-stratum gain, not only the aggregate.

An honest prior from adjacent evidence: broader *learned* text representations have repeatedly **failed** to help on this problem. Adding 48 SVD components of hashed note text cost `0.62` minutes in a parallel study; adding 100 note-text SVD components raised best-OOF MAE from `32.87` to `33.82` in another. Raw note variance is dominated by *which procedure this is* — information the structured record already holds perfectly — so broader text features contribute redundancy and dilution rather than signal.

**This is precisely why the proposed replacement is narrow and targeted rather than broad.** It does not add vocabulary; it adds *assertion-level resolution* to a small, clinically motivated concept set. That is a different axis, and it is the axis where the current pipeline is demonstrably weakest.

### 8.9 Cost and deployment implications

The current pipeline's headline operational virtue is that prediction requires no GPU, no network call, and no embedding service — a few hundred regular-expression evaluations plus tree traversals. **The proposed design gives that up.** The trade must be made explicitly:

| | Current | Proposed |
| --- | --- | --- |
| Prediction-time dependency | none | embedding service + extraction API |
| Per-case cost | negligible | $K_{\text{concepts}}$ cached calls |
| Latency | milliseconds | seconds (cacheable, and pre-computable at scheduling time) |
| Auditability | pattern list | passage-level evidence |
| Negation / temporality | none | explicit |
| Cross-site transfer | untested, likely poor | query-based, expected better |

The mitigating fact is that duration prediction is not a real-time task. The prediction is required when the case is *scheduled*, typically days in advance, so extraction can run asynchronously and be cached long before the prediction is needed.

---

## 9. Stage 1 — the tabular foundation

### 9.1 Role

The tabular base estimates broad procedural duration. It must be strong, because everything above it is a *correction*: a weak base leaves the note channel relearning structured information it should never have to see.

### 9.2 The paired XGBoost models

Two models share an identical tuned tree shape and an identical feature matrix, differing only in objective.

| Parameter | Value |
| --- | ---: |
| `n_estimators` | `1213` |
| `max_depth` | `8` |
| `learning_rate` | `0.013721653065680284` |
| `subsample` | `0.8431891099718941` |
| `colsample_bytree` | `0.5331833452208407` |
| `min_child_weight` | `9.083480959214418` |
| `reg_alpha` | `6.006000053619992e-07` |
| `reg_lambda` | `0.001809354462444587` |
| `gamma` | `1.0186857689274291` |
| `max_bin` | `212` |
| `tree_method` | `hist` |
| `eval_metric` | `mae` |

| Model | Objective | Held-out MAE |
| --- | --- | ---: |
| `xgb_tuned_v2` | `reg:squarederror` | `32.544887` |
| `xgb_absoluteerror_same_shape` | `reg:absoluteerror` | `32.483091` |

### 9.3 The 60/40 blend

$$\hat y_i^{\text{base}} = 0.60\,\hat y_i^{\text{sq}} + 0.40\,\hat y_i^{\text{abs}}.$$

| Metric | Value |
| --- | ---: |
| Train OOF MAE | `32.966712` |
| Held-out MAE | `32.239636` |

Note that the blend beats **both** members on held-out (`32.2396` vs `32.4831` and `32.5449`). The two objectives make genuinely different errors: the squared-error member is sensitive to the conditional mean and therefore to the tail; the absolute-error member is median-like and outlier-resistant. Their convex combination retains some of each.

### 9.4 Cross-fitting

Within the training cohort, predictions use five-fold `GroupKFold` with **patient ID** as the grouping variable. A patient never crosses the fit/validation boundary of the same fold.

This produces $\hat y^{\text{base,OOF}}$ on train — every entry from a model that never saw that row. Separately, both models are refit on the entire training split and applied to held-out, giving $\hat y^{\text{base,test}}$.

**Recorded limitation.** These internal folds are patient-disjoint but **not chronological**. They control repeated-patient dependence, which is the dominant contamination risk, but they do not simulate forward-time deployment. The top-level held-out split is chronological; the internal estimator is not. A stronger design uses nested forward-time folds with patient purging, and this is the first recommendation in Section 15.6.

---

## 10. Stage 2 — the strict clinical-note / tabular shared manifold

This is the stage that produces the largest single gain in the entire system: `+0.469372` minutes.

### 10.1 The residual target

For each training case,

$$r_i = y_i - \hat y_i^{\text{base,OOF}}.$$

Using **OOF** base predictions here is essential and not a technicality. An in-sample base residual would be artificially small and structured by overfit; the note head would then learn to undo the base model's memorisation rather than to add information.

### 10.2 The two latent views

**Note view.** From Section 7: TF-IDF $\rightarrow$ `TruncatedSVD(160)` $\rightarrow$ `StandardScaler`, plus the concept indicators. Call the result $Z\in\mathbb{R}^{n\times d_Z}$.

**Tabular view.** The preprocessed tabular matrix is reduced by `TruncatedSVD(96)`, scaled from training statistics, and the **first 64 coordinates** are supplied to the manifold. Call this $X\in\mathbb{R}^{n\times 64}$.

Both reductions are fit on training rows and applied unchanged to held-out rows.

### 10.3 Partial least squares — the shared manifold

PLS seeks paired directions that maximise **covariance** between the two views. For component $h$, it solves

$$(w_h, c_h) = \arg\max_{\|w\|=\|c\|=1} \operatorname{cov}\bigl(X_{h-1}w,\; Z_{h-1}c\bigr)^2,$$

equivalently the leading singular pair of the cross-covariance matrix $X_{h-1}^\top Z_{h-1}$, followed by deflation:

$$t_h = X_{h-1}w_h, \qquad u_h = Z_{h-1}c_h,$$
$$X_h = X_{h-1} - t_h p_h^\top, \qquad Z_h = Z_{h-1} - u_h q_h^\top,$$

with loadings $p_h = X_{h-1}^\top t_h / (t_h^\top t_h)$ and $q_h = Z_{h-1}^\top u_h / (u_h^\top u_h)$.

**Why covariance and not correlation.** Canonical correlation analysis maximises correlation and is notoriously unstable in high dimension — it will happily find a perfectly correlated pair of near-noise directions. PLS maximises covariance, which is correlation weighted by the variances of both views, and is therefore biased toward directions that actually carry energy. With $d_Z = 160{+}$ and $n \approx 11{,}000$ this stability matters.

**Fitted configuration.** `PLSRegression(n_components=40, max_iter=1000)` is fit on the 64-dimensional tabular view against the note view. The run config permitted up to `56` components; the evaluated grid for the surgical profile ended at `40`. The selected slice retains the **first 26** components.

For each case this gives a tabular score $t_i\in\mathbb{R}^{26}$ and a note score $u_i\in\mathbb{R}^{26}$ — *what the structured record says*, and *what the narrative says*, expressed in the same 26-dimensional coordinate system.

### 10.4 The delta geometry

The residual feature vector is

$$\phi_i = \bigl[\;u_i - t_i,\;\; |u_i - t_i|,\;\; t_i \odot u_i\;\bigr] \in \mathbb{R}^{78},$$

where $\odot$ is the elementwise (Hadamard) product. That is $26 + 26 + 26 = 78$ features.

Each block answers a different question:

| Block | Dimension | Meaning |
| --- | ---: | --- |
| $u_i - t_i$ | `26` | **Directional disagreement** — do the notes push along or against the tabular expectation, and on which latent axis? |
| $\lvert u_i - t_i\rvert$ | `26` | **Magnitude of disagreement**, direction-free — how far apart are the two views at all? |
| $t_i \odot u_i$ | `26` | **Alignment or opposition** — do the two views agree in sign on this axis? |

This is the conceptual core of the whole model. The notes are not used to predict duration. They are used to measure **the geometry of narrative disagreement with the structured expectation**, and that disagreement is turned into a duration correction.

The empirical attribution (Section 13.4) confirms the blocks are not redundant: mean absolute contribution is `8.30` minutes from the directional block, `5.05` from the absolute block, and `2.77` from the product block.

### 10.5 The robust residual head

$$\hat r_i = h(\phi_i), \qquad h = \text{StandardScaler} \circ \text{SGDRegressor}.$$

```python
SGDRegressor(
    loss="huber",
    penalty="elasticnet",
    alpha=1e-4,
    l1_ratio=0.05,
    max_iter=4000,
    tol=1e-4,
    average=True,
    random_state=seed,
)
```

The Huber loss with threshold $\delta$,

$$\ell_\delta(e) = \begin{cases}
\tfrac12 e^2, & |e|\le \delta,\\[2pt]
\delta\bigl(|e| - \tfrac12\delta\bigr), & |e| > \delta,
\end{cases}$$

is quadratic near zero and linear in the tails. Since the residual target $r_i$ inherits the duration distribution's heavy right tail, a squared-error head would be dragged by a handful of extreme cases — precisely the cases where the note signal is least reliable. Elastic-net regularisation with $\ell_1$ ratio `0.05` is mostly ridge, appropriate for `78` correlated latent features.

**The head is deliberately simple.** The representational work is done by the PLS manifold; the head's job is to map `78` geometric coordinates to one scalar without overfitting. A complex head here consistently underperformed.

Residual OOF predictions use `GroupKFold(5)` on patient; the head is then refit on the full training split and applied to held-out.

### 10.6 The trust scalar $\gamma$

$$\hat y_i^{\text{PLS}} = \hat y_i^{\text{base}} + \gamma\,\hat r_i,$$

with $\gamma$ selected **on training OOF MAE only**, over the grid $\{0.000, 0.025, 0.050,\dots, 1.600\}$.

**Selected: $\gamma = 1.550$.**

That $\gamma>1$ is informative. The Huber head, regularised and fit on a heavy-tailed target, systematically *under-shoots* the residual magnitude; $\gamma$ corrects that shrinkage. It is a calibration scalar, not a confidence statement.

### 10.7 Stage 2 result

Model identity: `pls_raw_word160_26_delta__sgd_huber__g1.550`

| Metric | Value |
| --- | ---: |
| Train OOF MAE | `32.7964177885` |
| **Held-out MAE** | **`31.7702636563`** |
| RMSE | `47.0862301387` |
| $R^2$ | `0.7537433190` |
| Median absolute error | `21.3969602483` |
| P90 absolute error | `69.8267318237` |
| P95 absolute error | `93.2294053259` |
| Within 30 min (W30) | `0.6220500176` |
| Within 60 min (W60) | `0.8622754491` |
| Mean prediction − actual | `−2.8542162009` |
| Under-estimation rate | `0.4825642832` |
| Over-estimation rate | `0.5174357168` |
| Gain over tabular base | `+0.4693722183` |
| Clustered bootstrap CI for that gain | `[+0.3357543566, +0.6065614085]` |
| Clustered bootstrap CI for the MAE | `[30.6139184325, 33.0492553314]` |

Run configuration: seed `4242`, `5` folds, surgical profile, `127` candidate variants evaluated, `200` bootstrap replicates, forbidden-file read `false`.

---

## 11. Stage 3 — procedure-family calibration

### 11.1 The gate

The same PLS correction is not equally reliable for every procedure family:

$$\hat y_i^{\text{family}} = \hat y_i^{\text{base}} + \gamma_{f(i)}\bigl(\hat y_i^{\text{PLS}} - \hat y_i^{\text{base}}\bigr),$$

where $f(i)$ is the procedure family and $\gamma_f \ge 0$ is a family-specific trust scalar learned **inside outer patient-group folds**. Small families fall back to the global value.

### 11.2 Fitted trust scalars

| Procedure family | $\gamma_f$ | Reading |
| --- | ---: | --- |
| VT/PVC ablation | `2.500` | correction strongly under-applied |
| AF/PVI ablation | `1.700` | correction under-applied |
| Lead/device extraction–removal | `1.525` | correction under-applied |
| Other ablation | `1.025` | roughly calibrated |
| LAA closure | `1.000` | at the global fallback |
| Device implant | `0.775` | correction over-trusted |
| Device revision–generator–upgrade | `0.775` | correction over-trusted |
| Diagnostic / testing | `0.300` | correction largely distrusted |
| Other | `0.275` | correction largely distrusted |
| *global fallback* | `1.000` | |

**The pattern is clinically coherent, which is the strongest argument for it.** The families where the note correction is *under*-applied are exactly the complex-substrate ablation and extraction families — the ones where the structured record is least descriptive of difficulty and where narrative context (prior failed ablation, scar burden, venous occlusion) carries real information. The families where it is *over*-trusted are short, standardised, high-volume procedures whose duration is close to deterministic given the code.

### 11.3 Held-out effect by family

| Family | Gain (min) | Improved rate |
| --- | ---: | ---: |
| AF/PVI ablation | `+0.5819` | `58.20%` |
| Lead/device extraction–removal | `+0.2409` | `43.42%` |
| Diagnostic / testing | `+0.2296` | `54.94%` |
| Device revision–generator–upgrade | `+0.0816` | `55.88%` |
| VT/PVC ablation | `+0.0262` | `50.85%` |
| Device implant | `−0.0403` | `48.74%` |
| Other | `−0.0663` | `47.83%` |

### 11.4 Result and the VT/PVC warning

| Metric | Value |
| --- | ---: |
| Train OOF MAE | `32.7692796199` |
| **Held-out MAE** | **`31.6075223696`** |
| Gain vs strict PLS | `+0.1627412868` |
| 95% paired bootstrap CI | `[+0.0565, +0.2664]` |
| Improved-case rate | `51.32%` |

**AF/PVI is the clearest family-level win. VT/PVC is the clearest hazard.** Its $\gamma = 2.5$ is the largest in the table and produces both large wins and large losses; its net gain is only `+0.0262`. This family needs either a cap (which Section 12 introduces) or a genuine direction signal before it can be pushed harder.

---

## 12. Stage 4 — the risk and residual candidate bank

### 12.1 The question

Where is the current predictor unreliable, and can that be learned *without* using the row's own outcome?

### 12.2 Risk targets

Four families of learnable target, all computed from OOF predictions:

| Target | Definition | What it asks |
| --- | --- | --- |
| Absolute error | $\lvert y_i - \hat y_i^{\text{OOF}}\rvert$ | how wrong will we be? |
| Signed residual | $y_i - \hat y_i^{\text{OOF}}$ | in which direction? |
| Tail indicator | $\mathbb{1}\bigl(\lvert y_i - \hat y_i\rvert > 30\bigr)$ | will this be a blow-up? |
| Gain | $\lvert y_i - \hat y_i^{\text{base}}\rvert - \lvert y_i - \hat y_i^{\text{final}}\rvert$ | did the correction help? |

Risk learners: Extra Trees and histogram gradient boosting, over safe tabular features, note metadata, the base/final/correction triple, and PLS-disagreement geometry.

### 12.3 Nesting — the part that is easy to get wrong

A risk score is **derived from the target**. If a row's own error contributes to the model that assigns that row a risk category, the category is contaminated and every downstream gate trained on it is optimistic.

The discipline is therefore:

1. Outer patient `GroupKFold`.
2. For each outer validation fold, the risk model is trained **without** that fold.
3. Score-bin quantile edges are computed from the outer-training scores **only**.
4. Category trust scalars and caps are tuned on the outer-training rows **only**.
5. Held-out outcomes are used for final scoring **only**.

An earlier broad risk grid was held-out-safe but not strictly nested at the OOF-bin-selection step. Only the strictly nested results are reported as claims.

### 12.4 What the risk heads actually learned

| Score model | OOF correlation with its target |
| --- | ---: |
| `risk_abs_extra_trees` vs absolute error | `0.4569` |
| `risk_abs_hgb` vs absolute error | `0.4519` |
| *(earlier pass)* risk model vs absolute final error | `0.454` – `0.456` |
| `gain_hgb` vs final-vs-base gain | `0.0823` |
| `signed_resid_hgb` vs signed residual | `0.0793` |
| `tail30_hgb` vs tail indicator | `0.0318` |

**This is the table that determined the architecture.** A correlation of `0.45` with error *magnitude* is a usable trust signal. A correlation of `0.03`–`0.08` with error *direction* is not a usable correction signal. Consequently the risk heads are wired as gates on an existing correction, never as correctors in their own right.

### 12.5 The capped-correction candidate form

The generic candidate in the bank is

$$\hat y_i = \hat y_i^{\text{anchor}} + \mathrm{clip}\bigl(\gamma_c\,\hat r_i,\; -C_c,\; +C_c\bigr),$$

with $\gamma_c \ge 0$ and $C_c > 0$ tuned per category $c$ (procedure family, learned risk stratum, or their interaction) inside outer folds, subject to a minimum region size and a global fallback.

The cap is what makes a large $\gamma$ survivable. A family like VT/PVC can be assigned high trust *and* a bound on how far that trust is allowed to move a single prediction.

### 12.6 Candidate results

| Candidate | Train OOF MAE | Held-out MAE | Gain vs strict PLS | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `crossfit_residual_hgb` | `32.724460` | `31.737202` | `+0.033062` | — |
| `family_cap_gamma` | `32.757110` | `31.628808` | `+0.141456` | `[+0.0317, +0.2491]` |
| `nested_risk_abs3_x_family_cap_gamma` | `32.771967` | `31.607452` | `+0.162811` | `[+0.0269, +0.2985]` |
| `risk3_x_pls_gap_gamma_positive` | `32.768101` | `31.700691` | `+0.069573` | `[−0.0476, +0.1844]` |
| `risk_abs_hgb4_x_family_cap_gamma` | `32.800344` | `31.579011` | `+0.191253` | `[+0.0691, +0.3161]` |
| `procedure_family_gamma_positive` *(Stage 3)* | `32.769280` | `31.607522` | `+0.162741` | `[+0.0565, +0.2664]` |

Two observations of method, both uncomfortable and both worth recording.

**`crossfit_residual_hgb` won on train OOF and lost on held-out.** Best-by-OOF is not a guarantee; it is only the least-biased selector available.

**The best strictly nested risk gate ties the simple family gate**, at `31.6074524935` versus `31.6075223696` — a difference of `0.00007` minutes. The elaborate risk machinery recovered exactly what one categorical variable already gave. That is a negative result and it is reported as one.

### 12.7 PLS-gap gates

| Variant | Held-out MAE | Gain |
| --- | ---: | ---: |
| `pls_gap_bin_gamma_positive` | `31.756141` | `+0.0141` |
| `risk_surface_pls_gamma_positive` | `31.681709` | `+0.0886` |
| `sched_x_pls_gap_gamma_positive` | `31.695244` | `+0.0750` |
| `family_x_pls_gap_gamma_positive` | `31.657513` | `+0.1128` |

The PLS disagreement geometry carries signal, but it destabilises under fine segmentation. The reliable question is not *"how large is the PLS gap?"* but *"how much should the PLS correction be trusted for this clinical family?"*

### 12.8 Where the risk gate helps and where it pays

Stratified by strict-final absolute error:

| Error stratum | Gain |
| --- | ---: |
| `0–15` min | `−0.4138` |
| `30–60` min | `+0.6649` |
| `60–120` min | `+0.3875` |
| `120+` min | `+0.6924` |

Stratified by correction magnitude:

| Absolute PLS correction | Gain | $n$ |
| --- | ---: | ---: |
| `0–5` min | `+0.0359` | `2,176` |
| `5–10` min | `+0.4081` | `618` |
| `10–20` min | `+2.9319` | `45` |

Two clean readings. The gate does what it was built to do on badly-predicted cases, and **pays for it by perturbing already-good ones** — a genuine trade, not a free lunch. And the upside concentrates almost entirely in the small subset where the note correction is non-trivial: `45` cases carry a `+2.93` minute gain.

---

## 13. Stage 5 — the top-refine meta-gate (selected model)

### 13.1 Construction

The final model blends two already cross-fitted candidates.

| | Candidate |
| --- | --- |
| **A** | `pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00` |
| **B** | `signed_resid_hgb8_x_family_cap_gamma` |

The **gate score** is the disagreement between two calibrated predictors:

$$d_i = \hat y_i^{\text{risk\_hgb4}} - \hat y_i^{\text{procedure\_family}}.$$

Using *disagreement between calibrated candidates* as the gate — rather than a raw risk score — is the key structural choice. Disagreement is a model-internal signal that requires no target, so it is clean by construction, and it localises exactly the cases where the trust question is live.

For candidate predictions $a_i, b_i$ the pair blend is

$$\hat y_i(w) = (1-w)\,a_i + w\,b_i,$$

with $w$ searched over $\{0, 0.01, \dots, 1\}$ minimising fold-training MAE. Score-bin edges are quantiles of the fold-training scores only, with `6` bins, a minimum local sample size of `80`, and a required local MAE improvement margin of `0.02` minutes before a local override is accepted.

### 13.2 Fitted policy

Learned score edges:

```text
-16.7446, -1.2492, -0.3816, -0.0305, 0.0751, 0.6576, 16.2156
```

| Score region | Weight on candidate B |
| --- | ---: |
| Bin 0 — lowest disagreement | `1.00` |
| Bin 1 | `0.00` |
| Bins 2, 3, 4 | `0.37` |
| Bin 5 — highest disagreement | `0.00` |
| Global default | `0.37` |

For OOF rows, edges and weights are always fitted without the validation fold. For held-out rows, the policy is refit on all training OOF evidence and applied unchanged.

**Read the policy.** It is not a monotone trust curve. At *extreme* disagreement in either direction (bins 1 and 5) the gate falls entirely back to candidate A — when the two calibrated models disagree violently, the more conservative one wins. At the *lowest* disagreement bin it goes entirely to candidate B. In the middle it blends. That shape is defensible, but with only four distinct fitted weights and a `0.02`-minute acceptance margin it is also the most over-fittable component in the system, which is why its incremental gain has a confidence interval that crosses zero (Section 13.6).

### 13.3 Held-out performance by procedure family

From the stage-2 interpretability pass (base → strict PLS final, $n = 2{,}839$):

| Family | $n$ | Base MAE | Final MAE | Gain | Improved | Mean actual | Mean \|correction\| | Alignment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF/PVI ablation | `677` | `34.409` | `33.102` | `+1.307` | `61.30%` | `217.1` | `4.38` | `64.25%` |
| Lead/device extraction–removal | `76` | `49.141` | `48.415` | `+0.727` | `46.05%` | `216.5` | `2.44` | `48.68%` |
| VT/PVC ablation | `295` | `54.196` | `53.859` | `+0.337` | `53.22%` | `294.9` | `3.47` | `55.59%` |
| Device implant | `712` | `25.520` | `25.188` | `+0.332` | `52.67%` | `145.4` | `2.70` | `55.62%` |
| Other ablation | `365` | `36.598` | `36.295` | `+0.304` | `52.05%` | `200.1` | `3.63` | `54.25%` |
| Other | `115` | `19.070` | `18.907` | `+0.162` | `52.17%` | `64.8` | `2.40` | `53.91%` |
| LAA closure | `47` | `23.793` | `23.699` | `+0.094` | `55.32%` | `160.9` | `3.62` | `59.57%` |
| Device revision–generator–upgrade | `306` | `23.197` | `23.297` | `−0.099` | `47.06%` | `130.3` | `2.82` | `50.98%` |
| Diagnostic / testing | `233` | `23.668` | `23.907` | `−0.239` | `45.92%` | `107.1` | `3.23` | `48.50%` |

**AF/PVI ablation carries the model.** `677` cases, `+1.307` minutes of gain, `61.30%` improved, and an alignment rate of `64.25%` — the only family where the correction points the right way appreciably more often than chance. Note also its mean base residual of `−10.64` minutes: the tabular model systematically *over*-predicts AF/PVI, and the notes know it.

The two families with negative gain — device revision and diagnostic/testing — are exactly the ones Stage 3 subsequently assigned trust scalars of `0.775` and `0.300`. The family calibration is repairing a failure the interpretability pass had already identified.

### 13.4 Latent block attribution

| Manifold block | Mean \|contribution\| (min) | Mean signed contribution (min) |
| --- | ---: | ---: |
| $u - t$ (directional) | `8.302` | `−0.527` |
| $\lvert u-t\rvert$ (magnitude) | `5.046` | `−0.022` |
| $t \odot u$ (product) | `2.768` | `−0.017` |

The directional block dominates, at roughly `1.6×` the magnitude block and `3×` the product block. The signed contributions are all slightly negative, consistent with the global correction mean of `−2.17` minutes: **the note channel's net effect is to shorten predictions.**

### 13.5 Concept-flag diagnostics

| Concept flag | $n$ | Prevalence | Gain when present | Gain when absent | **Lift** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `congenital_complex_anatomy` | `241` | `8.49%` | `+0.4399` | `+0.4721` | `−0.0322` |
| `access_support` | `377` | `13.28%` | `+0.2670` | `+0.5004` | `−0.2333` |
| `vt_pvc_instability` | `1,157` | `40.75%` | `+0.2210` | `+0.6402` | `−0.4192` |
| `redo_prior_ablation` | `442` | `15.57%` | `+0.2180` | `+0.5157` | `−0.2977` |
| `device_lead_complexity` | `345` | `12.15%` | `+0.1435` | `+0.5144` | `−0.3709` |

**Every lift is negative.** Cases where a concept flag fires improve *less* than cases where it does not. `vt_pvc_instability` fires on `40.75%` of cases and has the worst lift of all.

This is the empirical case for Section 8 stated in a table. The flags are not merely weak — they mark the cases the model handles *worse*. Whether that is because the flags are miscalibrated or because flagged cases are intrinsically harder cannot be separated with a binary indicator, and that ambiguity is itself the argument for graded, assertion-aware extraction.

### 13.6 Selected model metrics

| Metric | Value |
| --- | ---: |
| Train OOF MAE | `32.7156839203` |
| **Held-out MAE** | **`31.5564180731`** |
| Gain vs strict PLS | `+0.2138455833` |
| 95% paired bootstrap CI | `[+0.1001, +0.3259]` |
| Gain vs procedure-family gate | `+0.0511042965` |
| 95% paired bootstrap CI | `[−0.0108, +0.1123]` |
| Improved-case rate vs strict PLS | `48.89%` |
| Large wins (> 10 min) | `23` |
| Large losses (< −10 min) | `16` |

**The honest reading.** The gain over the strict PLS manifold is solid: the interval excludes zero. The gain of the meta-gate *over the simple procedure-family gate* is `0.051` minutes with an interval that **crosses zero**. The final two stages of the ladder are not individually established. Note also that the improved-case rate is `48.89%` — below half — meaning the gate wins by making its wins larger, not more frequent.

### 13.7 Model-selection rule and the lockbox candidate

The reported model is the candidate with the best **training OOF** MAE in the targeted top-refine search. The search also recorded a better held-out candidate:

| | Selected | Exploratory |
| --- | --- | --- |
| Variant | `topref_pair__pairblend_proc__…__delta_hgb4_proc6_none_n80_m0.02` | `topref_pair__procedure_family_gamma_positive__…__candidate_spread7_family_n80_m0.02` |
| Train OOF MAE | `32.7156839203` | `32.7806355030` |
| Held-out MAE | `31.5564180731` | `31.5234029457` |
| Selected? | **yes** | no — worse train OOF |

The exploratory candidate is **not** the claim. It is held as a lockbox candidate only. Choosing it because it scored better on held-out would convert the held-out set into a training set.

---

## 14. The complete algorithm

```text
INPUT
  tabular case table
  timestamped clinical-note table
  prediction-timestamp definition

── BOUNDARY ─────────────────────────────────────────────────────────
 1  drop rows with invalid schedule time or non-positive target
 2  sort cases by scheduled_in_room
 3  cut at the 80th percentile -> candidate held-out cohort
 4  remove every held-out patient from the earlier training rows
 5  attach to each case only notes with doc_date < scheduled_in_room - 2 days
 6  reject post-operative doc types; cap note count / length / lookback

── TABULAR FOUNDATION ───────────────────────────────────────────────
 7  build planned-schedule, case, patient and comorbidity features
 8  in patient-grouped folds:
       fit preprocessing + procedure/surgeon target encodings on fold-train
       fit xgb_tuned_v2 and xgb_absoluteerror_same_shape
       predict fold-validation                                  -> base OOF
 9  refit both on all training rows, predict held-out            -> base test
10  base = 0.60 * squared-error member + 0.40 * absolute-error member

── NOTE / TABULAR MANIFOLD ──────────────────────────────────────────
11  residual target  r = y - base_OOF
12  fit note TF-IDF -> SVD(160) -> scale        (train only)
    fit tabular SVD(96) -> scale, keep first 64 (train only)
13  fit PLS(40) between the two views; slice to 26 components
14  build delta geometry  phi = [u-t, |u-t|, t*u]                -> 78 features
15  fit Huber residual head in patient-grouped folds             -> r_hat
16  select gamma on train OOF MAE over {0.000 ... 1.600}         -> 1.550
    y_PLS = base + gamma * r_hat

── CONDITIONAL TRUST ────────────────────────────────────────────────
17  learn procedure-family trust gamma_f inside outer patient folds
       y_family = base + gamma_f * (y_PLS - base)
18  build nested risk / signed-residual / gain candidates
       every score model trained without its own validation fold
       every bin edge and cap from outer-training rows only
19  cross-fit the six-bin pair-blend policy on candidate disagreement

── SELECTION AND SCORING ────────────────────────────────────────────
20  select the top-refine variant by TRAIN OOF MAE
21  refit all training-derived transforms and policies on all training rows
22  apply once to held-out; compute held-out MAE
```

Every learned object in this list — imputation medians, category frequencies, target encodings, TF-IDF vocabulary, SVD bases, scalers, PLS directions, residual head, $\gamma$, family trust scalars, risk models, bin edges, caps, blend weights — is fit on a fitting partition and applied to an evaluation partition. There are no exceptions.

---

## 15. Iteration: one pass completed, convergence not reached

**This is the most important open section of the document.** Everything above describes a *single* residual iteration. The architecture is a truncated instance of a well-defined iterative scheme, and the truncation was a decision about effort, not a statement about convergence.

### 15.1 The iteration operator

Let $p^{(0)}_i$ be the current out-of-fold prediction for case $i$. At stage $k$ define the residual

$$r^{(k)}_i = y_i - p^{(k)}_i,$$

train a new head $h_k$ on $(x_i, z_i, p^{(k)}_i) \mapsto r^{(k)}_i$ under nested cross-fitting, and update

$$\boxed{\;p^{(k+1)}_i = p^{(k)}_i + \eta_k\, h_k\bigl(x_i, z_i, p^{(k)}_i\bigr)\;}$$

with shrinkage $\eta_k \in (0, \eta_{\max}]$ selected **inside training folds**.

What we have executed is exactly two applications of this operator with a conditional trust gate attached:

| $k$ | $p^{(k)}$ | $h_k$ | $\eta_k$ |
| ---: | --- | --- | ---: |
| `0` | tabular base | PLS manifold Huber head | $\gamma = 1.550$ |
| `1` | strict PLS model | family / risk / meta-gate re-weighting | $\gamma_f$, blend $w$ |
| `2` | **not run** | — | — |

Stage $k=1$ is not a full residual head — it is a *re-weighting* of the stage-0 correction rather than a new learner on the stage-1 residual. **A genuine second residual iteration has never been attempted.**

### 15.2 The functional-gradient view

The scheme is gradient boosting in function space under absolute loss. For $\ell(y,f)=|y-f|$,

$$-\frac{\partial \ell}{\partial f} = \operatorname{sign}(y - f),$$

so the exact functional gradient step fits the **sign** of the residual, not its magnitude. Our heads fit the residual itself under a Huber loss, which interpolates: quadratic near zero (magnitude-sensitive, efficient) and linear in the tail (sign-like, robust). The learned scalar $\gamma$ then plays the role of the line-search step length that a pure gradient step would require.

This framing has one immediate and uncomfortable consequence. **Section 12.4 showed that residual direction is essentially unlearnable from our features** — OOF correlation with signed residual is `0.079`. The exact $L_1$ functional gradient is a pure direction object. So the honest statement is:

> The reason further iterations have not obviously succeeded is that the natural gradient direction for absolute loss is the one signal our feature set cannot predict.

This is not a reason to stop. It is a precise specification of what iteration 2 must supply: **a direction signal**, not another magnitude model.

### 15.3 Why naive iteration fails, and the nesting requirement

If $h_k$ is trained on residuals from a prediction that saw the row's own label, then $h_k$ learns the *memorisation* of $p^{(k)}$, not its *error*. Applied out of sample, that correction is noise with the sign of overfit — actively harmful.

The requirement is therefore absolute:

> **Every input to stage $k+1$ must be produced without the row's own target.**

Concretely, at each iteration:

1. All upstream candidate predictions are generated **inside** the outer loop.
2. Risk labels, score bins, blend weights, caps and residual scalers are fit on outer-fold training rows only.
3. Minimum region sizes and global fallbacks are enforced.
4. The full candidate set is frozen before any new lockbox is opened.

The cost is multiplicative: iteration $k$ requires all of iterations $0..k-1$ regenerated within the outer loop. This is the real reason only one iteration has been completed.

### 15.4 Does the iteration converge?

Write $\mathcal{R}(p) = \mathbb{E}|Y - p|$. Under exact line search and a non-degenerate head, each step is non-increasing:

$$\mathcal{R}\bigl(p^{(k+1)}\bigr) \le \mathcal{R}\bigl(p^{(k)}\bigr),$$

so the population risk sequence is monotone and bounded below by $R^\star$, hence convergent. Three caveats stand between that statement and practice.

**Convergence is to a restricted optimum.** The limit is the best predictor in the closure of the span of the head class, not $R^\star$. If the head class cannot represent the remaining structure, the sequence converges above the floor and the gap is approximation error, not noise.

**We observe $\widehat{\mathcal{R}}$, not $\mathcal{R}$.** The empirical OOF risk has standard error of order $\mathrm{sd}(|e|)/\sqrt{n}$; with $n\approx 10{,}900$ and heavy-tailed errors this is roughly `0.3`–`0.4` minutes. **Stage gains of `0.05` minutes are far inside that noise.** The monotonicity that holds in population does not hold in the sample, and our own trace shows it: `crossfit_residual_hgb` won on train OOF and lost on held-out.

**Selection consumes information.** Each iteration chooses among candidates using the same training OOF estimate, so the estimate degrades as a selector with every pass. This is the winner's-curse channel, and it compounds.

### 15.5 Stopping rules

Four rules, to be applied jointly, not alternatively. Iteration stops when **any** fires.

**S1 — Floor gap.** Stop when the gap between the achieved risk and the estimated attainable floor is smaller than the resolution of the evaluation:

$$\widehat{\mathcal{R}}\bigl(p^{(k)}\bigr) - \hat E \;<\; \epsilon,$$

where $\hat E$ estimates the attainable floor. This is the only rule that references the problem rather than the procedure, and it is the one worth investing in.

**S2 — Confidence.** Stop when the lower bound of a patient-clustered bootstrap interval on the OOF improvement is not positive:

$$\mathrm{LB}_{95\%}\Bigl[\widehat{\mathcal{R}}\bigl(p^{(k)}\bigr) - \widehat{\mathcal{R}}\bigl(p^{(k+1)}\bigr)\Bigr] \le 0.$$

**By this rule the current iteration has already stopped at stage 5.** The gain of the meta-gate over the procedure-family gate is `+0.0511` with interval `[−0.0108, +0.1123]`. S2 says: do not add stage 5 without new evidence.

**S3 — Operational negligibility.** Stop when the gain is real but too small to matter. A gain below roughly `0.1` minutes per case is invisible against OR scheduling granularity. The right threshold is a business decision, stated in minutes of OR time per year, not in MAE.

**S4 — No progress / budget.** Stop after $k$ consecutive iterations in which no candidate passes S2, or when the compute budget for nested regeneration is exhausted.

### 15.6 Connecting the stop rule to the irreducible floor

S1 is only meaningful with an estimate of the floor. Parallel work on this same cohort — the Peras irreducible-risk framework, on a closely related but **not identical** pipeline ($n = 12{,}993$, different exclusions and feature arms) — produces:

| Quantity | Value | Interval |
| --- | ---: | --- |
| Certified lower bound $B^\star$ | `28.85` min | $\ge$ `28.27` at the 5% quantile |
| Conservative proved rung | `24.18` min | $\ge$ `23.60` |
| Estimated attainable floor $\hat E$ | `31.12` min | `[30.40, 31.78]` |

Read against the present model at `31.5564`:

- Distance to the **certified** floor: `31.5564 − 28.85 = 2.71` minutes. This is the absolute ceiling on what any further modelling could win on this information set.
- Distance to the **estimated attainable** floor: `31.5564 − 31.12 ≈ 0.44` minutes, and `31.5564` sits inside $\hat E$'s interval `[30.40, 31.78]`.

**The operational implication is blunt.** Under S1 with $\epsilon$ set anywhere near the evaluation's own resolution, **this model is already at the estimated attainable floor for its information set.** Further iteration on the same features is not expected to recover more than a few tenths of a minute.

Two caveats, both important. The floor estimates come from a different pipeline and are not directly transferable without recomputation on this exact feature matrix. And the floor is a property of **the information set**, not of the method — which points at the correct move.

### 15.7 What iteration 2 should actually be

Given S1 and Section 15.2, the conclusion is not *iterate the same scheme harder*. It is:

> **Change the information set, not the estimator.**

$R^\star$ moves only when the representation changes. Everything else — better heads, more bins, finer gates — moves how tightly we can approach a fixed floor, and we are already close to it.

Ranked by expected value:

1. **Replace the narrative extraction layer (Section 8).** This is the only proposed change that alters $R^\star$ itself. Graded, negation-aware, assertion-level concept evidence is a genuinely different information set from five saturated regex flags.
2. **Recompute the floor on this exact pipeline.** Without it, S1 cannot be applied honestly. This is cheap and should come first in wall-clock order.
3. **Attack the direction problem.** Iteration 2 needs a signed-residual signal, not another magnitude model. Candidate sources: note recency distribution, contradiction between planned procedure and narrative context, candidate spread, calibrated uncertainty, pre-cutoff patient-history summaries.
4. **Nested forward-time selection.** Replace patient-only `GroupKFold` with nested forward-time folds with patient purging, so the selection estimate matches the deployment question.
5. **Open a fresh lockbox.** Freeze everything, and evaluate once on a later untouched cohort.

### 15.8 Per-iteration diagnostics to track

Every future iteration should report, before any accept/reject decision:

| Diagnostic | Why |
| --- | --- |
| Train OOF MAE and its clustered bootstrap interval | S2 |
| Correction magnitude: mean, median, mean absolute | drift detection |
| Alignment rate $\Pr[\operatorname{sign}(\hat r) = \operatorname{sign}(r)]$ | direction quality (currently `56.3%`) |
| Correlation of correction with true residual | currently `0.158` |
| Per-family gain, including negative families | harm detection |
| Per-coverage-stratum gain | note availability differs across splits |
| Large-win and large-loss counts | tail behaviour |
| Improved-case rate | currently `48.89%` — below half |
| Distance to $\hat E$ and to $B^\star$ | S1 |

The alignment rate deserves emphasis: at `56.3%` the correction points the right way barely more often than a coin. That number, more than any MAE figure, says how much room the direction channel still has.

---

## 16. Validation and leakage audit

### 16.1 Controls in force

| Control | Status | Evidence |
| --- | :---: | --- |
| Later temporal held-out cohort | ✅ | cut at the 80th percentile of `scheduled_in_room` |
| Zero patient overlap | ✅ | $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{held-out}} = \varnothing$, audited |
| Outcome-time columns excluded | ✅ | all `case_actual_*` removed from the feature frame |
| Exact note timestamps with a 2-day buffer | ✅ | `295,463` dates checked, `0` violations, min margin `2.5` h |
| Post-operative doc types rejected | ✅ | `145,426` rows rejected |
| Same-day pre-operative notes disabled | ✅ | cache setting |
| Fold-local target encoding | ✅ | procedure and surgeon, fit on fold-train only |
| OOF base predictions used for residual targets | ✅ | never in-sample |
| Patient-grouped residual and gate evaluation | ✅ | `GroupKFold(5)` on patient |
| Nested construction of target-derived risk scores | ✅ | score model trained without its own validation fold |
| Bin edges and blend weights from outer-training only | ✅ | meta-gate policy |
| Selection by train OOF, not by held-out | ✅ | `31.5234` candidate rejected |
| Forbidden 11-feature LLM file not read | ✅ | `forbidden_llm_feature_file_read = false`, `0` access-like hits |
| Held-out target used only for final scoring | ✅ | |
| Saved prediction artefact reproduces the metric | ✅ | `31.556418073067334` |

### 16.2 Audit values

```text
train rows:                      10,882
held-out rows:                    2,839
train patients:                   8,948
held-out patients:                2,499
patient overlap:                      0
missing patient ids:            0 / 0
note dates checked:             295,463
cutoff violations:                    0
smallest margin before cutoff:  2.5 hours
forbidden LLM feature file read:  false
access-like hits to that file:        0
```

The forbidden-file scan found the filename in exactly two places, both defensive:

```text
strict_manifold_text_sweep.py:52   FORBIDDEN = PROJECT/"data"/"mayo"/"mayo_hrs_llm_features_top11.parquet"
strict_manifold_text_sweep.py:72   assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"
```

---

## 17. Limitations

Stated plainly, in descending order of how much they should worry a reader.

**The held-out cohort was viewed repeatedly.** The strict PLS run alone evaluated `127` candidate variants, and several subsequent passes examined the same held-out set. Selection was by train OOF, which is materially better than picking the best held-out row, but repeated examination still influences research decisions. **`31.5564` is a strong retrospective held-out estimate, not publication-grade independent confirmation.** The only fix is a fresh lockbox.

**The last two rungs of the ladder are not individually established.** The meta-gate's gain over the simple procedure-family gate is `+0.0511` with a bootstrap interval that crosses zero. The defensible claim stops at the procedure-family gate (`31.6075`, CI `[+0.0565, +0.2664]` over strict PLS). Everything after that is promising, not proven.

**Internal folds are patient-disjoint but not chronological.** They control the dominant contamination risk and do not simulate forward-time deployment. Nested forward-time folds with patient purging would give a better estimate of future performance.

**The reported interval is a paired case-level bootstrap.** It does not fully account for repeated-patient correlation. A patient-clustered bootstrap is preferred and was used for the stage-2 figures; the later-stage intervals should be recomputed on that basis.

**Scheduling-derived features change the task.** They encode operational planning expertise and improve accuracy, but their availability and stability at inference must be guaranteed, and the revision/feedback risks of Section 5.4 must be audited.

**Note coverage differs between splits** (`92.05%` train vs `99.79%` held-out). Monitoring must stratify by note availability and volume; an aggregate gain can hide a coverage-driven artefact.

**The concept-flag block has negative gain lift on every flag** (Section 13.5). The narrative extraction layer is the least defensible component in the system.

**Alignment rate is `56.3%`.** The correction's direction is barely better than chance. The model wins on magnitude allocation, not on knowing which way to move.

**The package contains source and predictions, not fitted estimators.** Reproducing training requires the authorised raw data and the pinned environment.

---

## 18. Deployment contract

**At inference.** Provide the same tabular schema and planned schedule fields, plus only notes finalised before the two-day cutoff. Preserve training-time category handling and transformations. **Do not** recompute target encodings, TF-IDF vocabulary, SVD bases, PLS directions, score-bin edges or blend weights using production outcomes.

**Fallback behaviour.** If the note bundle is empty or the extraction service is unavailable, fall back to the tabular base and record a degradation flag. Never silently change the input distribution.

**For retraining.** Create a new chronological cutoff, rebuild note bundles, regenerate all OOF candidate predictions inside the outer loop, reselect using training-side validation only, freeze the model, and then evaluate once on a later untouched cohort.

**Monitoring.** Stratify by procedure family, note coverage, note volume, and scheduled-duration bin. Track correction magnitude and alignment rate as drift indicators — they move before MAE does.

---

## 19. Reproduction

Install from `requirements.txt`; configure the authorised clinical-data repository via `CLINICAL_NOTES_PROJECT`. The original run order is in `scripts/run_best_31_5564_pipeline.sh`.

| Step | Script |
| ---: | --- |
| 1 | `src/build_raw_strict_high_signal_split.py` |
| 2 | `src/validate_strict_raw_run.py` |
| 3 | `src/tabular_model_sweep.py` |
| 4 | `src/strict_manifold_push.py` |
| 5 | `src/risk_category_modulation.py` |
| 6 | `src/variance_aware_pls_modulation.py` |
| 7 | `src/strict_risk_family_k_sweep.py` |
| 8 | `src/strict_meta_gate_push.py` |
| 9 | `src/strict_meta_gate_top_refine.py` |

To validate the distributed result **without** raw data:

```bash
python scripts/verify_saved_result.py
```

Expected:

```text
heldout_rows=2839
heldout_mae=31.556418073067334
status=PASS
```

The archive excludes raw clinical data, note text, fitted binaries, and credentials.

---

## 20. Notation

| Symbol | Meaning |
| --- | --- |
| $y_i$ | realised procedure duration, minutes |
| $x_i$ | structured (tabular) covariates |
| $z_i$ | eligible pre-operative note bundle |
| $\tau_i$ | prediction timestamp (case scheduling moment) |
| $\mathcal{F}_t$ | $\sigma$-field of everything knowable at time $t$ |
| $\hat y^{\text{base}}$ | 60/40 tabular blend |
| $r_i$ | residual target, $y_i - \hat y_i^{\text{base,OOF}}$ |
| $X, Z$ | tabular and note latent views entering PLS |
| $t_i, u_i$ | PLS scores, tabular and note, $\in\mathbb{R}^{26}$ |
| $\phi_i$ | delta-geometry feature vector, $\in\mathbb{R}^{78}$ |
| $\hat r_i$ | residual head output |
| $\gamma$ | global trust scalar, `1.550` |
| $\gamma_f$ | procedure-family trust scalar |
| $C_c$ | correction cap for category $c$ |
| $d_i$ | meta-gate score (candidate disagreement) |
| $w$ | pair-blend weight on candidate B, `0.37` global |
| $p^{(k)}$ | prediction after $k$ residual iterations |
| $\eta_k$ | iteration shrinkage |
| $R^\star$ | irreducible risk on the information set |
| $B^\star, \hat E$ | certified floor, estimated attainable floor |
| $\odot$ | elementwise (Hadamard) product |

---

## 21. Roadmap

| # | Action | Changes $R^\star$? | Effort | Priority |
| ---: | --- | :---: | --- | --- |
| 1 | Recompute the irreducible floor on this exact pipeline | no | low | **first** |
| 2 | Replace regex concepts with retrieval + rerank + service extraction (§8) | **yes** | medium | **highest value** |
| 3 | Build a direction signal for iteration 2 (§15.7) | partially | medium | high |
| 4 | Nested forward-time selection with patient purging | no | medium | high |
| 5 | Re-run later-stage intervals as patient-clustered | no | low | medium |
| 6 | Audit scheduling-field revision risk | no | low | medium |
| 7 | Freeze and open a fresh lockbox | no | low | **required before any external claim** |
| 8 | Per-case uncertainty via multi-quantile head + conformal calibration | no | medium | medium |

---

## 22. Final statement

The Conditional Residual Clinical Notes Learner reaches a held-out MAE of **`31.5564180731` minutes** on `2,839` later cases from patients absent from training, using structured clinical and planned scheduling data together with clinical notes restricted to text finalised more than two days before each case. It does not use the former 11-variable LLM feature table.

Of the `0.683` minutes gained over the tabular base, `0.469` minutes — the majority, and the only component whose interval comfortably excludes zero — comes from the clinical narrative entering as a residual correction through a PLS shared manifold. The conditional trust machinery above it is coherent, clinically interpretable, and individually unproven at its last two rungs.

**One residual iteration has been completed.** The iteration operator, the shrinkage schedule, the nesting requirement and four stopping rules are specified in Section 15. Under the confidence rule S2 the current ladder has already stopped; under the floor-gap rule S1, and using a parallel estimate of the attainable floor, this model sits roughly `0.44` minutes above the estimated floor for its information set and about `2.71` minutes above the certified one.

That arithmetic is the whole argument for what comes next. Further iteration on the same features is close to exhausted. **The remaining headroom is in the information set, not in the estimator** — which is why the highest-value next step is not another gate, but the replacement of five regular expressions with a retrieval-ranked, reranked, assertion-aware extraction layer, followed by a single frozen evaluation on a cohort no one has seen.

---

*Conditional Residual Clinical Notes Learner — Itamar Zernitsky, Department of Mathematics, Bar-Ilan University; Opmed.ai informal "Shadows" research team. Prepared for Opmed.ai.*
