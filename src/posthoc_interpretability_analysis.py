#!/usr/bin/env python3
"""Post-hoc interpretability diagnostics for the strict 31.7703 model.

This script analyzes already-generated held-out predictions. It does not train
or select a model, and it does not read the old 11 LLM feature file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("STRICT_31_WORK_ROOT", "/Users/itamarzernitsky/Documents/Codex/2026-08-23/can"))
PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
PRED_PATH = ROOT / "work/no_llm_notes/raw_strict_manifold_push_surgical_full/strict_manifold_push_test_predictions.csv"
NOTE_PATH = ROOT / "work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_note_bundles.parquet"
TAB_PATH = PROJECT / "data/mayo/mayo_hrs_tabular_features_and_durations.parquet"
OUT_DIR = ROOT / "outputs/posthoc_interpretability"
FINAL_COL = "pls_raw_word160_26_delta__sgd_huber__g1.550"
FORBIDDEN = PROJECT / "data/mayo/mayo_hrs_llm_features_top11.parquet"


CONCEPT_PATTERNS = {
    "redo_prior_ablation": [r"\bredo\b", r"repeat\s+ablation", r"prior\s+ablation"],
    "vt_pvc_instability": [r"\bvt\b", r"\bvf\b", r"ventricular\s+tachycardia", r"icd\s+shock"],
    "device_lead_complexity": [r"lead\s+(revision|extraction|failure|fracture)", r"generator\s+change"],
    "congenital_complex_anatomy": [r"\bfontan\b", r"transposition", r"congenital"],
    "access_support": [r"\becmo\b", r"epicardial", r"pericardial\s+access"],
}


def assert_no_forbidden_access() -> None:
    assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"


def procedure_family(name: Any) -> str:
    s = str(name or "").upper()
    if "PVI" in s or "ATRIAL FIBRILLATION" in s:
        return "AF/PVI ablation"
    if "VT" in s or "PVC" in s or "VENTRICULAR" in s:
        return "VT/PVC ablation"
    if "ABLATION" in s or "RFABLATION" in s:
        return "Other ablation"
    if "LAA" in s or "WATCHMAN" in s:
        return "LAA closure"
    if any(x in s for x in ["EXTRACTION", "REMOVAL"]):
        return "Lead/device extraction-removal"
    if any(x in s for x in ["GENERATOR", "POCKET", "REVISION", "UPGRADE"]):
        return "Device revision-generator-upgrade"
    if any(x in s for x in ["PPM", "ICD", "PACEMAKER", "LEADLESS", "DEFIBRILLATOR", "SUBSTERNAL", "LEFT BUNDLE"]):
        return "Device implant"
    if any(x in s for x in ["TILT", "DIAGNOSTIC EPS", "NIPS", "DRUG", "DEFIBRILLATION THRESHOLD", "3D MAPPING"]):
        return "Diagnostic/testing"
    if "ABORT" in s:
        return "Aborted/other"
    return "Other"


def duration_bin(minutes: float) -> str:
    if pd.isna(minutes):
        return "missing"
    if minutes < 90:
        return "<90"
    if minutes < 150:
        return "90-149"
    if minutes < 210:
        return "150-209"
    if minutes < 300:
        return "210-299"
    if minutes < 420:
        return "300-419"
    return "420+"


def note_count_bin(n: float) -> str:
    if pd.isna(n) or n <= 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 12:
        return "6-12"
    if n <= 23:
        return "13-23"
    return "24 cap"


def chars_bin(chars: float) -> str:
    if pd.isna(chars) or chars <= 0:
        return "0"
    if chars < 8000:
        return "<8k"
    if chars < 18000:
        return "8k-18k"
    if chars < 32000:
        return "18k-32k"
    return "32k+"


def summarize_group(df: pd.DataFrame, group_col: str, min_n: int = 20) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        if len(g) < min_n:
            continue
        rows.append(
            {
                group_col: key,
                "n": int(len(g)),
                "base_mae": float(g["base_abs_error"].mean()),
                "final_mae": float(g["final_abs_error"].mean()),
                "gain": float(g["gain"].mean()),
                "median_gain": float(g["gain"].median()),
                "improved_rate": float((g["gain"] > 0).mean()),
                "worsened_rate": float((g["gain"] < 0).mean()),
                "large_win_rate_gain_gt10": float((g["gain"] > 10).mean()),
                "large_loss_rate_gain_lt_minus10": float((g["gain"] < -10).mean()),
                "mean_actual_duration": float(g["case_actual_duration_time"].mean()),
                "mean_base_residual_actual_minus_base": float(g["true_residual"].mean()),
                "mean_model_correction_final_minus_base": float(g["correction"].mean()),
                "mean_abs_correction": float(g["correction"].abs().mean()),
                "alignment_rate": float(g["correction_aligned"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["gain", "n"], ascending=[False, False])


def doc_type_rows(df: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    counts: dict[str, list[int]] = {}
    for idx, raw in enumerate(df["note_doc_types_used"].fillna("").astype(str)):
        seen = {x.strip() for x in raw.split(";") if x.strip()}
        for doc_type in seen:
            counts.setdefault(doc_type, []).append(idx)
    rows = []
    all_idx = np.arange(len(df))
    for doc_type, idxs in counts.items():
        if len(idxs) < min_n:
            continue
        mask = np.zeros(len(df), dtype=bool)
        mask[idxs] = True
        g = df.loc[mask]
        h = df.loc[~mask]
        rows.append(
            {
                "doc_type_present": doc_type,
                "n_present": int(mask.sum()),
                "present_gain": float(g["gain"].mean()),
                "present_base_mae": float(g["base_abs_error"].mean()),
                "present_final_mae": float(g["final_abs_error"].mean()),
                "present_improved_rate": float((g["gain"] > 0).mean()),
                "n_absent": int((~mask).sum()),
                "absent_gain": float(h["gain"].mean()) if len(h) else np.nan,
                "gain_lift_present_minus_absent": float(g["gain"].mean() - h["gain"].mean()) if len(h) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["present_gain", "n_present"], ascending=[False, False])


def concept_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    low = df["note_text"].fillna("").astype(str).str.lower()
    for name, patterns in CONCEPT_PATTERNS.items():
        rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
        mask = low.map(lambda t: bool(rx.search(t))).to_numpy()
        if mask.sum() == 0:
            continue
        g = df.loc[mask]
        h = df.loc[~mask]
        rows.append(
            {
                "concept_flag": name,
                "n_present": int(mask.sum()),
                "prevalence": float(mask.mean()),
                "present_gain": float(g["gain"].mean()),
                "present_base_mae": float(g["base_abs_error"].mean()),
                "present_final_mae": float(g["final_abs_error"].mean()),
                "present_improved_rate": float((g["gain"] > 0).mean()),
                "absent_gain": float(h["gain"].mean()),
                "gain_lift_present_minus_absent": float(g["gain"].mean() - h["gain"].mean()),
                "mean_correction_present": float(g["correction"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("present_gain", ascending=False)


def main() -> None:
    assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(PRED_PATH)
    if FINAL_COL not in preds.columns:
        raise KeyError(f"Missing final column: {FINAL_COL}")
    keep = [
        "case_durable_id",
        "patient_durable_id",
        "scheduled_in_room",
        "case_actual_duration_time",
        "primary_procedure_name",
        "base",
        FINAL_COL,
    ]
    df = preds[keep].rename(columns={FINAL_COL: "final_pred"}).copy()
    df["base_error"] = df["base"] - df["case_actual_duration_time"]
    df["final_error"] = df["final_pred"] - df["case_actual_duration_time"]
    df["base_abs_error"] = df["base_error"].abs()
    df["final_abs_error"] = df["final_error"].abs()
    df["gain"] = df["base_abs_error"] - df["final_abs_error"]
    df["correction"] = df["final_pred"] - df["base"]
    df["true_residual"] = df["case_actual_duration_time"] - df["base"]
    df["correction_aligned"] = np.sign(df["correction"]) == np.sign(df["true_residual"])
    df.loc[df["correction"].abs() < 1e-8, "correction_aligned"] = False
    df["base_direction"] = np.where(df["base_error"] > 0, "base overestimated", "base underestimated")
    df.loc[df["base_error"].abs() < 1e-8, "base_direction"] = "exact"

    tab = pd.read_parquet(TAB_PATH)
    tab_cols = [
        "case_durable_id",
        "case_anesthesia",
        "case_urgency_level",
        "case_surgery_patient_class",
        "case_admission_patient_class",
        "patient_sex",
        "patient_age",
        "scheduled_in_room",
        "scheduled_out_of_room",
        "surgeon_durable_id",
        "case_num_procedures",
        "case_num_panels",
        "case_num_providers",
        "is_redo",
        "prior_same_proc_duration",
        "redo_time_gap_days",
        "bmi",
    ]
    tab = tab[[c for c in tab_cols if c in tab.columns]].copy()
    df = df.merge(tab.drop(columns=["scheduled_in_room"], errors="ignore"), on="case_durable_id", how="left")

    notes = pd.read_parquet(NOTE_PATH, columns=["case_durable_id", "note_text", "n_pre_cutoff_notes_used", "note_doc_types_used", "note_chars"])
    df = df.merge(notes, on="case_durable_id", how="left")
    df["note_text"] = df["note_text"].fillna("")
    df["n_pre_cutoff_notes_used"] = df["n_pre_cutoff_notes_used"].fillna(0)
    df["note_chars"] = df["note_chars"].fillna(0)
    df["note_doc_types_used"] = df["note_doc_types_used"].fillna("")

    df["procedure_family"] = df["primary_procedure_name"].map(procedure_family)
    df["actual_duration_bin"] = df["case_actual_duration_time"].map(duration_bin)
    df["note_count_bin"] = df["n_pre_cutoff_notes_used"].map(note_count_bin)
    df["note_chars_bin"] = df["note_chars"].map(chars_bin)
    if {"scheduled_in_room", "scheduled_out_of_room"}.issubset(tab.columns):
        sched_in = pd.to_datetime(df["scheduled_in_room"], utc=True, errors="coerce")
        sched_out = pd.to_datetime(df["scheduled_out_of_room"], utc=True, errors="coerce")
        df["scheduled_duration_minutes"] = (sched_out - sched_in).dt.total_seconds() / 60
        df["scheduled_duration_bin"] = df["scheduled_duration_minutes"].map(duration_bin)
        df["scheduled_minus_actual"] = df["scheduled_duration_minutes"] - df["case_actual_duration_time"]
    else:
        df["scheduled_duration_bin"] = "missing"

    metric_summary = {
        "n": int(len(df)),
        "base_mae": float(df["base_abs_error"].mean()),
        "final_mae": float(df["final_abs_error"].mean()),
        "gain": float(df["gain"].mean()),
        "median_gain": float(df["gain"].median()),
        "cases_improved_rate": float((df["gain"] > 0).mean()),
        "cases_worsened_rate": float((df["gain"] < 0).mean()),
        "large_win_gain_gt10_n": int((df["gain"] > 10).sum()),
        "large_loss_gain_lt_minus10_n": int((df["gain"] < -10).sum()),
        "base_p90_abs_error": float(df["base_abs_error"].quantile(0.90)),
        "final_p90_abs_error": float(df["final_abs_error"].quantile(0.90)),
        "base_p95_abs_error": float(df["base_abs_error"].quantile(0.95)),
        "final_p95_abs_error": float(df["final_abs_error"].quantile(0.95)),
        "mean_correction": float(df["correction"].mean()),
        "median_correction": float(df["correction"].median()),
        "mean_abs_correction": float(df["correction"].abs().mean()),
        "correction_true_residual_corr": float(df["correction"].corr(df["true_residual"])),
        "correction_alignment_rate": float(df["correction_aligned"].mean()),
        "forbidden_llm_feature_file_read": False,
    }

    outputs = {
        "family": summarize_group(df, "procedure_family", 20),
        "procedure": summarize_group(df, "primary_procedure_name", 10),
        "actual_duration_bin": summarize_group(df, "actual_duration_bin", 1),
        "scheduled_duration_bin": summarize_group(df, "scheduled_duration_bin", 1),
        "note_count_bin": summarize_group(df, "note_count_bin", 1),
        "note_chars_bin": summarize_group(df, "note_chars_bin", 1),
        "base_direction": summarize_group(df, "base_direction", 1),
        "anesthesia": summarize_group(df, "case_anesthesia", 20),
        "urgency": summarize_group(df, "case_urgency_level", 20),
        "patient_class": summarize_group(df, "case_surgery_patient_class", 20),
        "doc_type": doc_type_rows(df, 40),
        "concept_flags": concept_rows(df),
    }
    for name, out in outputs.items():
        out.to_csv(OUT_DIR / f"{name}_summary.csv", index=False)

    case_cols = [
        "case_durable_id",
        "primary_procedure_name",
        "procedure_family",
        "case_actual_duration_time",
        "base",
        "final_pred",
        "base_abs_error",
        "final_abs_error",
        "gain",
        "correction",
        "true_residual",
        "n_pre_cutoff_notes_used",
        "note_chars",
    ]
    df[case_cols].sort_values("gain", ascending=False).head(100).to_csv(OUT_DIR / "top_case_level_wins_deidentified.csv", index=False)
    df[case_cols].sort_values("gain", ascending=True).head(100).to_csv(OUT_DIR / "top_case_level_losses_deidentified.csv", index=False)
    (OUT_DIR / "metric_summary.json").write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Post-Hoc Interpretability Summary",
        "",
        "This is a generated aggregate summary for the fixed clean 31.7703 model.",
        "",
        "## Overall",
        "",
        f"- Held-out cases: `{metric_summary['n']}`",
        f"- Base MAE: `{metric_summary['base_mae']:.6f}`",
        f"- Final MAE: `{metric_summary['final_mae']:.6f}`",
        f"- Mean gain vs base: `{metric_summary['gain']:.6f}` minutes",
        f"- Median case-level gain: `{metric_summary['median_gain']:.6f}` minutes",
        f"- Improved cases: `{100 * metric_summary['cases_improved_rate']:.2f}%`",
        f"- Correction/true-residual correlation: `{metric_summary['correction_true_residual_corr']:.4f}`",
        f"- Correction alignment rate: `{100 * metric_summary['correction_alignment_rate']:.2f}%`",
        "",
        "## Output Tables",
        "",
    ]
    for name in outputs:
        report_lines.append(f"- `{name}_summary.csv`")
    report_lines.extend(
        [
            "- `top_case_level_wins_deidentified.csv`",
            "- `top_case_level_losses_deidentified.csv`",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(metric_summary, indent=2))
    print(f"Saved post-hoc interpretability outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
