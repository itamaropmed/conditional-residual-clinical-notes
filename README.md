# Conditional Residual Clinical Notes Learner (CRCNL)

**Surgical procedure-duration prediction from structured records and pre-operative clinical narrative.**

A staged residual model. The clinical notes never predict duration directly — they predict only the residual the tabular model leaves behind, entering through a PLS shared manifold and gated by learned per-family and per-risk trust.

| | |
| --- | --- |
| **Held-out MAE** | **31.5564180731 minutes** |
| Train OOF MAE | 32.7156839203 minutes |
| Held-out cases | 2,839 — temporal split, patient-disjoint, 0 overlap |
| Training cases | 10,882 |
| Zero-information baseline | ≈ 81 min → **~61% of marginal dispersion removed** |
| Model identity | `strict_meta_gate_top_refine_oof_selected` |
| Iterations completed | **one** |
| Notion page | [Conditional Residual Clinical Notes Learner](https://app.notion.com/p/Conditional-Residual-Clinical-Notes-Learner-fc03e1f1817183a8b5a881d5c403d7b4?source=copy_link) |
| Jira | [DSC-127](https://opmed-ai.atlassian.net/browse/DSC-127) |

---

## Author

**Itamar Zernitsky**
Department of Mathematics, Bar-Ilan University, Ramat-Gan, Israel
Opmed.ai — informal "Shadows" research team

- Email: **itamar.zernitsky@opmed.ai**
- Phone: **054-797-1303**
- GitHub: [@itamaropmed](https://github.com/itamaropmed)

Research prepared for Opmed.ai.

---

## Read this first

| Document | What it is |
| --- | --- |
| [`docs/Conditional_Residual_Clinical_Notes_Learner.pdf`](docs/Conditional_Residual_Clinical_Notes_Learner.pdf) | **The full report** — 35 pages: motivation, formalism, data handling, every stage, risk residuals, validation, limitations, iteration and stopping rules |
| [`docs/Conditional Residual Clinical Notes Learner.md`](docs/) | The same document in Markdown (Notion-importable) |
| `docs/methods_and_results_full.md` | Condensed method narrative and results |
| `docs/strict_31_7703_clean_method.md` | The PLS manifold model card |
| `docs/pls_risk_category_modulation_report.md` | Procedure-family and early risk-gate work |
| `docs/variance_aware_pls_modulation_report.md` | Strict nested risk/gain/capped gates |
| `docs/strict_risk_family_k_sweep_report.md` | k-risk-family sweep |
| `docs/strict_meta_gate_top_refine_report.md` | The final selected model |
| `docs/posthoc_interpretability_31_7703_report.md` | Subgroup and case-level diagnostics |

---

## The result ladder

Each stage refines the one above. Held-out MAE, minutes.

| Stage | Model | Held-out MAE | Gain | 95% CI |
| --- | --- | ---: | ---: | --- |
| 1 | Tabular base — 60/40 XGB squared/absolute blend | 32.2396 | — | — |
| 2 | \+ strict PLS note/tabular manifold | 31.7703 | +0.4694 | [+0.336, +0.607] |
| 3 | \+ procedure-family calibration | 31.6075 | +0.1627 | [+0.057, +0.266] |
| 4 | Risk / residual candidate bank | *(bank)* | — | — |
| 5 | \+ top-refine meta-gate **(selected)** | **31.5564** | +0.0511 | [−0.011, +0.112] |

The clinical narrative contributes **0.469 of the 0.683 total minutes gained** — the majority, and the only component whose interval comfortably excludes zero.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                  recorded run settings
├── data/                     EMPTY BY DESIGN — see data/README.md
├── docs/                     full report (PDF + Markdown) and stage reports
├── results/summaries/        de-identified aggregate results and run summaries
├── scripts/
│   ├── run_best_31_5564_pipeline.sh    original full run order
│   ├── verify_saved_result.py          recompute the held-out MAE
│   └── sanitize_predictions.py         build a shareable prediction table
├── src/                      the pipeline, one file per stage
└── vendor/clinical_notes_dependencies/ snapshots of imported helper modules
```

### Source files

| File | Purpose |
| --- | --- |
| `src/build_raw_strict_high_signal_split.py` | Build the strict pre-cutoff raw-note cache |
| `src/validate_strict_raw_run.py` | Audit split, note timing, forbidden-file access |
| `src/tabular_model_sweep.py` | Train the tabular base models |
| `src/strict_manifold_text_sweep.py` | Shared manifold and residual helpers |
| `src/strict_manifold_push.py` | Train and score the strict PLS manifold model |
| `src/risk_category_modulation.py` | First PLS risk/category modulation |
| `src/variance_aware_pls_modulation.py` | Strict nested risk/gain/capped gate tests |
| `src/strict_risk_family_k_sweep.py` | k-risk-family sweep |
| `src/strict_meta_gate_push.py` | Second-level meta-gate |
| `src/strict_meta_gate_refine.py` | Broad saved-prediction refinement |
| `src/strict_meta_gate_top_refine.py` | Final targeted top-refine model |
| `src/posthoc_interpretability_analysis.py` | Subgroup and case-level diagnostics |
| `src/latent_attribution_31_7703.py` | Latent attribution for the PLS model |

---

## Data — not in this repository

**No patient data is committed here, and none should ever be.**

The source datasets are Mayo HRS clinical records held in S3. Nine result files from the original run were also excluded because they carry `patient_durable_id` together with minute-resolution `scheduled_in_room` timestamps, procedure names and realised durations — a combination that is re-identifiable even though the IDs are pseudonymous.

`data/` is git-ignored and stays empty.

### S3 locations

Bucket `opmed-integration-raw-data-mayo`, prefix `clinical_notes/anonymized_hrs_notes_shared/`:

| File | S3 URI |
| --- | --- |
| Dataset — tabular + lags + target | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_tabular_features_with_lags_and_durations.parquet` |
| Anonymized notes | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_notes_anonymized_merged.parquet` |
| LLM features (top 11) — **not used by this model** | `s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/mayo_hrs_llm_features_top11.parquet` |

```bash
export CLINICAL_NOTES_PROJECT=/path/to/clinical_notes_project
mkdir -p "$CLINICAL_NOTES_PROJECT/data/mayo"
aws s3 sync \
  s3://opmed-integration-raw-data-mayo/clinical_notes/anonymized_hrs_notes_shared/ \
  "$CLINICAL_NOTES_PROJECT/data/mayo/" \
  --exclude "*" --include "mayo_hrs_tabular_features_with_lags_and_durations.parquet" \
               --include "mayo_hrs_notes_anonymized_merged.parquet"
```

The notes file is ~3.7 GB. The LLM feature table is deliberately **not** synced — this model
excludes it by design.

**Join keys**

```text
dataset.case_durable_id     ->  LLM features.case_durable_id   (case-keyed, left join)
dataset.patient_durable_id  ->  notes.epic_id                  (1:many, patient-keyed)
```

Notes are keyed by patient, not case, so attaching a case to "its" notes is a **date
filter**, not a join key. See [`data/README.md`](data/README.md) for the cutoff rule and the
files that were held back.

---

## Quick start in a clean environment

### Requirements

- Python **3.10+**
- No GPU
- No network access at training or prediction time
- ~2 GB RAM for verification; considerably more for full retraining

### 1. Clone

```bash
git clone https://github.com/itamaropmed/conditional-residual-clinical-notes.git
cd conditional-residual-clinical-notes
```

### 2. Create an isolated environment

**venv (standard library):**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**conda, if you prefer:**

```bash
conda create -n crcnl python=3.11 -y
conda activate crcnl
pip install -r requirements.txt
```

**uv, fastest:**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Verify the published result

This is the only step that runs without authorised data access. It recomputes the
held-out MAE from a saved prediction table.

```bash
python scripts/verify_saved_result.py
```

Expected:

```text
heldout_rows=2839
heldout_mae=31.556418073067334
status=PASS
```

> **Note.** The prediction table this script reads is one of the nine files excluded
> from the repository. Obtain it from the authorised environment, or generate a
> redacted copy with `scripts/sanitize_predictions.py` (below) and point the script
> at that. Without either, verification will report a missing input rather than fail
> silently.

### 4. Produce a shareable prediction table

Strips the patient identifier and the exact timestamp, and replaces the case ID with
a salted hash, leaving the columns the verification needs:

```bash
python scripts/sanitize_predictions.py \
  --in  /authorised/path/strict_meta_gate_top_refine_selected_predictions.csv \
  --out results/summaries/selected_predictions_redacted.csv \
  --salt "$(openssl rand -hex 16)"
```

Keep the salt out of the repository. Review the output before sharing it — the
redaction is a starting point, not a compliance sign-off.

### 5. Full retraining (authorised environment only)

Requires the two source parquet datasets and `CLINICAL_NOTES_PROJECT` pointing at the
authorised data repository.

```bash
export CLINICAL_NOTES_PROJECT=/path/to/clinical_notes_project
bash scripts/run_best_31_5564_pipeline.sh
```

Stage order, if you prefer to run them individually:

```bash
python src/build_raw_strict_high_signal_split.py \
    --output-dir work/raw_strict_high_signal_tuned_split \
    --max-notes-per-case 24 --max-note-chars 2200 \
    --max-case-note-chars 48000 --max-note-age-days 548 --max-sentences 72
python src/validate_strict_raw_run.py
python src/tabular_model_sweep.py
python src/strict_manifold_push.py --profile surgical --n-boot 200
python src/risk_category_modulation.py
python src/variance_aware_pls_modulation.py
python src/strict_risk_family_k_sweep.py
python src/strict_meta_gate_push.py
python src/strict_meta_gate_top_refine.py
```

Determinism: fixed seeds throughout (seed `4242` for the manifold run). Seed 0
reproduces the reported figures exactly.

---

## Method in brief

**Stage 1 — tabular foundation.** Two XGBoost models on an identical tuned tree shape,
differing only in objective (`reg:squarederror` and `reg:absoluteerror`), blended 60/40.
Patient-grouped 5-fold cross-fitting; smoothed target encodings for procedure and
surgeon fitted fold-locally.

**Stage 2 — the shared manifold.** Residual target `r = y − base_OOF`. Note text →
clinical TF-IDF → SVD(160); tabular → SVD(96), first 64 components. PLS aligns the two
views by maximising covariance; the first 26 components give a tabular score `t` and a
note score `u` in a common space. The residual features are the delta geometry
`[u − t, |u − t|, t ⊙ u]` — 78 features encoding *how the narrative disagrees with the
structured expectation*. A Huber/elastic-net head maps them to a correction, scaled by
`γ = 1.550` chosen on train OOF only.

**Stage 3 — procedure-family calibration.** `ŷ = base + γ_f · (ŷ_PLS − base)`, with
`γ_f` learned inside outer patient folds. Fitted values run from 2.5 for VT/PVC
ablation down to 0.275 for the "Other" family — the note correction is under-applied on
complex-substrate ablations and over-trusted on short standardised procedures.

**Stage 4 — risk bank.** Nested risk heads over absolute error, signed residual, tail
indicator and gain, with capped corrections per category.

**Stage 5 — top-refine meta-gate.** A six-bin pair blend between two calibrated
candidates, gated on their disagreement.

Full derivations, the PLS objective, the Huber formulation, the trust-gate algebra and
the iteration operator are in the PDF.

---

## Leakage controls

| Control | Status |
| --- | :---: |
| Later temporal held-out cohort | ✅ |
| Zero patient overlap between train and held-out | ✅ |
| All `case_actual_*` outcome components excluded | ✅ |
| Note cutoff `doc_date < scheduled_in_room − 2 days`, 295,463 dates checked, 0 violations | ✅ |
| Post-operative document types rejected (145,426 rows) | ✅ |
| Same-day pre-operative notes disabled | ✅ |
| Fold-local target encoding | ✅ |
| OOF base predictions used for residual targets | ✅ |
| Nested construction of target-derived risk scores | ✅ |
| Selection by train OOF, never by held-out | ✅ |
| Former 11-variable LLM feature table not read | ✅ |
| Held-out target used only for final scoring | ✅ |

---

## Known limitations

- The held-out cohort was examined across multiple research iterations. `31.5564` is a
  strong retrospective estimate, **not publication-grade independent confirmation.** A
  fresh lockbox is required before any external claim.
- The gain of stage 5 over stage 3 has a bootstrap interval that crosses zero. The
  defensible claim stops at **31.6075**.
- Internal folds are patient-disjoint but not chronological.
- The result applies to the **schedule-informed** setting — planned scheduling
  timestamps are used as features.
- The five regex concept flags show negative gain lift on every flag; the narrative
  extraction layer is the weakest component and is the first thing scheduled for
  replacement.

---

## Roadmap

1. Replace regex concept extraction with retrieval + bi-encoder ranker + cross-encoder
   reranker + structured extraction API — the only proposed change that moves the
   irreducible floor.
2. Recompute the irreducible floor on this exact pipeline.
3. Build a direction signal for iteration 2 — error *magnitude* is learnable
   (OOF corr ≈ 0.45), error *direction* is not (≈ 0.08).
4. Nested forward-time selection with patient purging.
5. Freeze and open a fresh lockbox cohort.

---

## Tracking

Jira: **DSC-127** — Data Science project, Opmed.ai.

---

## Licence and use

Internal Opmed.ai research. Not for redistribution. Any use of the underlying clinical
data is governed by the applicable data use agreement.
