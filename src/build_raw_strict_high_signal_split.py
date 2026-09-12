#!/usr/bin/env python3
"""Build high-signal split from raw notes with exact scheduled_in_room - 2d cutoff.

This is the higher-recall version of build_strict_high_signal_split.py. It does
not recut an already capped note bundle; it streams raw notes and builds a fresh
strict bundle, so older legal notes are not displaced by same-day notes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "mayo_residual_distiller_study"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_mayo_cure_residual_embedding import (  # noqa: E402
    build_case_note_bundles,
    high_signal_source_aware_note_text_for_case,
    source_aware_marker_summary,
)
from train_mayo_tabular_xgboost_baseline import (  # noqa: E402
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    read_table,
    temporal_group_holdout,
)


FORBIDDEN = PROJECT / "data" / "mayo" / "mayo_hrs_llm_features_top11.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default=str(PROJECT / "data" / "mayo" / "mayo_hrs_tabular_features_and_durations.parquet"),
    )
    parser.add_argument(
        "--notes-data",
        default=str(PROJECT / "data" / "mayo" / "mayo_hrs_notes_anonymized_merged.parquet"),
    )
    parser.add_argument("--output-dir", default="work/no_llm_notes/raw_strict_high_signal_tuned_split")
    parser.add_argument("--max-notes-per-case", type=int, default=24)
    parser.add_argument("--max-note-chars", type=int, default=2200)
    parser.add_argument("--max-case-note-chars", type=int, default=48000)
    parser.add_argument("--max-note-age-days", type=float, default=548.0)
    parser.add_argument("--max-note-rows", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=72)
    return parser.parse_args()


def assert_no_forbidden_access() -> None:
    assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"


def strip_procedure_only_markers(text: str) -> str:
    text = re.sub(r"\bproc_family_[A-Za-z0-9_]+\b", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def red_flag_rates(texts: pd.Series) -> dict[str, float]:
    low = texts.fillna("").astype(str).str.lower()
    patterns = {
        "post": r"\bpost(?:-| )?(?:op|procedure|operative)?\b",
        "discharge": r"\bdischarge\b",
        "tolerated": r"\btolerated\b",
        "procedure_well": r"procedure (?:was )?well|tolerated (?:the )?procedure well",
        "out_of_room": r"out of room",
        "actual_duration": r"actual duration",
        "total_time": r"total .{0,20}time",
    }
    return {name: float(low.str.contains(pattern, regex=True).mean()) for name, pattern in patterns.items()}


def lexical_probes(texts: pd.Series) -> dict[str, float]:
    low = texts.fillna("").astype(str).str.lower()
    probes = {
        "total_time": r"total .{0,20}time",
        "post_op": r"post-op",
        "tolerated_procedure_well": r"tolerated the procedure well",
        "discharge_summary": r"discharge summary",
        "case_ended_or_completed": r"case (ended|completed)",
        "out_of_room": r"out of room",
        "actual_duration": r"actual duration",
        "procedure_time": r"procedure time",
    }
    return {name: float(low.str.contains(pattern, regex=True).mean()) for name, pattern in probes.items()}


def high_signal_text(note_text: Any, procedure: Any, max_sentences: int) -> str:
    text = high_signal_source_aware_note_text_for_case(note_text, procedure, max_sentences=max_sentences)
    return strip_procedure_only_markers(text)


def split_note_frame(cases: pd.DataFrame, bundles: pd.DataFrame, max_sentences: int) -> pd.DataFrame:
    merged = cases[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    merged["note_text"] = merged["note_text"].fillna("")
    out = pd.DataFrame({CASE_ID_COLUMN: merged[CASE_ID_COLUMN]})
    out["high_signal_text"] = [
        high_signal_text(note_text, proc, max_sentences)
        for note_text, proc in zip(merged["note_text"], merged["primary_procedure_name"])
    ]
    return out


def main() -> None:
    args = parse_args()
    assert_no_forbidden_access()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.tabular_data), TARGET_COLUMN, None)
    df = add_safe_features(df, include_scheduled_duration=True)
    split = temporal_group_holdout(df, 0.20)
    train_df, test_df = split.train_df.copy(), split.test_df.copy()
    all_cases = pd.concat([train_df, test_df], ignore_index=True)

    shifted_cases = all_cases.copy()
    shifted_cases[DATE_COLUMN] = pd.to_datetime(shifted_cases[DATE_COLUMN], utc=True) - pd.Timedelta(days=2)
    print(
        f"Streaming raw notes for {len(shifted_cases):,} cases with exact cutoff "
        f"{DATE_COLUMN} shifted by -2d...",
        flush=True,
    )
    bundles = build_case_note_bundles(
        notes_path=Path(args.notes_data),
        case_df=shifted_cases,
        max_notes_per_case=args.max_notes_per_case,
        max_note_chars=args.max_note_chars,
        max_case_note_chars=args.max_case_note_chars,
        max_note_age_days=args.max_note_age_days,
        max_note_rows=args.max_note_rows,
        note_cutoff_policy="timestamp",
        include_same_day_preop_note_types=False,
        keep_postop_doc_types=False,
    )
    bundles.to_parquet(output_dir / "strict_note_bundles.parquet", index=False)

    train_text = split_note_frame(train_df, bundles, args.max_sentences)
    test_text = split_note_frame(test_df, bundles, args.max_sentences)
    train_text.to_parquet(output_dir / "train_high_signal_text.parquet", index=False)
    test_text.to_parquet(output_dir / "test_high_signal_text.parquet", index=False)

    train_join = train_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    test_join = test_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    for frame in (train_join, test_join):
        frame["note_text"] = frame["note_text"].fillna("")
        frame["n_pre_cutoff_notes_used"] = frame["n_pre_cutoff_notes_used"].fillna(0)
        frame["note_chars"] = frame["note_chars"].fillna(0)

    audit = {
        "forbidden_llm_feature_file_read": False,
        "forbidden_llm_feature_file": str(FORBIDDEN),
        "notes_data": str(Path(args.notes_data)),
        "cutoff_rule": "raw doc_date < scheduled_in_room - 2d, timestamp exact, max age 548d",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "overlap_patients": int(
            len(set(train_df[PATIENT_ID_COLUMN].astype(str)) & set(test_df[PATIENT_ID_COLUMN].astype(str)))
        ),
        "bundle_attrs": {k: str(v) for k, v in bundles.attrs.items()},
        "bundles_out": int(len(bundles)),
        "coverage_all": float(len(bundles) / len(all_cases)),
        "train_note_coverage": float((train_join["note_text"].str.len() > 0).mean()),
        "test_note_coverage": float((test_join["note_text"].str.len() > 0).mean()),
        "train_median_notes": float(train_join["n_pre_cutoff_notes_used"].median()),
        "test_median_notes": float(test_join["n_pre_cutoff_notes_used"].median()),
        "train_median_chars": float(train_join["note_chars"].median()),
        "test_median_chars": float(test_join["note_chars"].median()),
        "raw_note_lexical_probes_train": lexical_probes(train_join["note_text"]),
        "raw_note_lexical_probes_test": lexical_probes(test_join["note_text"]),
        "high_signal_red_flags_train": red_flag_rates(train_text["high_signal_text"]),
        "high_signal_red_flags_test": red_flag_rates(test_text["high_signal_text"]),
        "top_markers_train": source_aware_marker_summary(train_text["high_signal_text"])[:40],
        "top_markers_test": source_aware_marker_summary(test_text["high_signal_text"])[:40],
    }
    (output_dir / "raw_strict_high_signal_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in [
        "bundles_out",
        "coverage_all",
        "train_note_coverage",
        "test_note_coverage",
        "train_median_notes",
        "test_median_notes",
        "train_median_chars",
        "test_median_chars",
    ]}, indent=2))
    print(f"Saved raw strict high-signal split to {output_dir}")


if __name__ == "__main__":
    main()
