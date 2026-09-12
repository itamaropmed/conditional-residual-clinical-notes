#!/usr/bin/env bash
set -euo pipefail

# This script documents the intended run order. Paths may need adjustment if the
# original Clinical_notes repository or Codex workspace is moved.

export CLINICAL_NOTES_PROJECT="${CLINICAL_NOTES_PROJECT:-/Users/itamarzernitsky/PycharmProjects/Clinical_notes}"

python src/build_raw_strict_high_signal_split.py \
  --output-dir work/no_llm_notes/raw_strict_high_signal_tuned_split \
  --max-notes-per-case 24 \
  --max-note-chars 2200 \
  --max-case-note-chars 48000 \
  --max-note-age-days 548 \
  --max-sentences 72

python src/validate_strict_raw_run.py \
  --bundle-dir work/no_llm_notes/raw_strict_high_signal_tuned_split \
  --code-path src/build_raw_strict_high_signal_split.py \
  --code-path src/tabular_model_sweep.py \
  --code-path src/strict_manifold_text_sweep.py \
  --code-path src/strict_manifold_push.py \
  --output-json work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_validation.json

python src/tabular_model_sweep.py \
  --output-dir work/no_llm_notes/tabular_sweep \
  --n-folds 5 \
  --seed 42 \
  --n-boot 500

python src/strict_manifold_push.py \
  --phrase-dir work/no_llm_notes/raw_strict_high_signal_tuned_split \
  --tabular-sweep-dir work/no_llm_notes/tabular_sweep \
  --output-dir work/no_llm_notes/raw_strict_manifold_push_surgical_full \
  --profile surgical \
  --n-boot 200

python src/risk_category_modulation.py
python src/variance_aware_pls_modulation.py
python src/strict_risk_family_k_sweep.py
python src/strict_meta_gate_push.py
python src/strict_meta_gate_top_refine.py

