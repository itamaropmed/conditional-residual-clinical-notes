#!/usr/bin/env python3
"""Validate the strict raw-note no-LLM run without printing row-level data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from train_mayo_tabular_xgboost_baseline import (  # noqa: E402
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    read_table,
    temporal_group_holdout,
)


FORBIDDEN_PATTERNS = (
    "mayo_hrs_llm_features_top11.parquet",
    "llm_features_top11",
)
ACCESS_RE = re.compile(r"\b(read_parquet|read_csv|open|load|from_parquet|scan_parquet)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default="/Users/itamarzernitsky/PycharmProjects/Clinical_notes/data/mayo/mayo_hrs_tabular_features_and_durations.parquet",
    )
    parser.add_argument("--bundle-dir", default="work/no_llm_notes/raw_strict_high_signal_tuned_split")
    parser.add_argument("--code-path", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=20000)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--cutoff-days", type=float, default=2.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_note_bundles(bundle_dir: Path) -> pd.DataFrame:
    path = bundle_dir / "note_bundles.parquet"
    if not path.exists():
        path = bundle_dir / "strict_note_bundles.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def scan_forbidden_paths(paths: list[str]) -> dict[str, dict[str, list[str]]]:
    hits: dict[str, dict[str, list[str]]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = [p for p in path.rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md", ".csv"}]
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                lines = candidate.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            mentions: list[str] = []
            access_like: list[str] = []
            for lineno, line in enumerate(lines, start=1):
                if any(pat in line for pat in FORBIDDEN_PATTERNS):
                    mentions.append(f"{lineno}: {line.strip()}")
                    if ACCESS_RE.search(line):
                        access_like.append(f"{lineno}: {line.strip()}")
            if mentions or access_like:
                hits[str(candidate)] = {
                    "mentions": mentions,
                    "access_like_hits": access_like,
                }
    return hits


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    df = read_table(Path(args.tabular_data), TARGET_COLUMN, args.max_cases)
    df = add_safe_features(df, include_scheduled_duration=True)
    split = temporal_group_holdout(df, args.test_fraction)
    all_cases = pd.concat([split.train_df, split.test_df], ignore_index=True)
    patient_overlap = set(split.train_df[PATIENT_ID_COLUMN].astype(str)) & set(split.test_df[PATIENT_ID_COLUMN].astype(str))

    bundles = load_note_bundles(bundle_dir)
    merged = all_cases[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN]].merge(
        bundles[[CASE_ID_COLUMN, "note_dates_used", "n_pre_cutoff_notes_used"]],
        on=CASE_ID_COLUMN,
        how="left",
    )
    scheduled = pd.to_datetime(merged[DATE_COLUMN], errors="coerce", utc=True)
    cutoffs = scheduled - pd.to_timedelta(args.cutoff_days, unit="D")

    exploded = pd.DataFrame(
        {
            "_cutoff": cutoffs,
            "note_date_token": merged["note_dates_used"].fillna("").astype(str).str.split(r";\s*", regex=True),
        }
    ).explode("note_date_token", ignore_index=True)
    exploded["note_date_token"] = exploded["note_date_token"].fillna("").astype(str).str.strip()
    exploded = exploded[exploded["note_date_token"] != ""].copy()
    exploded["note_date"] = pd.to_datetime(exploded["note_date_token"], errors="coerce", utc=True)
    exploded = exploded[pd.notna(exploded["note_date"]) & pd.notna(exploded["_cutoff"])].copy()
    note_dates_checked = int(len(exploded))
    margins = (exploded["_cutoff"] - exploded["note_date"]).dt.total_seconds() / 3600.0
    violation_count = int((exploded["note_date"] >= exploded["_cutoff"]).sum())
    latest_margin_hours = None if margins.empty else float(margins.min())

    notes_used = pd.to_numeric(merged["n_pre_cutoff_notes_used"], errors="coerce").fillna(0)
    audit_path = bundle_dir / "raw_strict_high_signal_audit.json"
    stored_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    result = {
        "bundle_dir": str(bundle_dir),
        "train_rows": int(len(split.train_df)),
        "test_rows": int(len(split.test_df)),
        "patient_overlap": int(len(patient_overlap)),
        "cases_with_bundle": int(len(bundles)),
        "cases_with_any_used_note": int((notes_used > 0).sum()),
        "note_dates_checked": int(note_dates_checked),
        "cutoff_rule": f"each note_date < scheduled_in_room - {args.cutoff_days:g} days",
        "cutoff_violations": int(violation_count),
        "smallest_margin_hours_before_cutoff": None if latest_margin_hours is None else float(latest_margin_hours),
        "stored_forbidden_llm_feature_file_read": bool(stored_audit.get("forbidden_llm_feature_file_read", False)),
        "forbidden_code_scan": scan_forbidden_paths(args.code_path),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
