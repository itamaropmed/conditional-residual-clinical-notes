"""Produce a shareable prediction table from an authorised one.

Removes the patient identifier and the exact scheduled timestamp, and replaces the
case identifier with a salted hash. Keeps the target, the procedure family and every
prediction column, so downstream verification still works.

    python scripts/sanitize_predictions.py \
        --in  /authorised/path/strict_meta_gate_top_refine_selected_predictions.csv \
        --out results/summaries/selected_predictions_redacted.csv \
        --salt "$(openssl rand -hex 16)"

Keep the salt out of version control. Review the output before sharing: this is a
starting point for de-identification, not a compliance sign-off.
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd

DROP = ["patient_durable_id", "scheduled_in_room"]
HASH = ["case_durable_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--salt", required=True, help="random hex string; do not commit it")
    ap.add_argument("--keep-procedure-name", action="store_true",
                    help="retain primary_procedure_name (kept out by default)")
    a = ap.parse_args()

    df = pd.read_csv(a.src)
    before = list(df.columns)

    for c in DROP:
        if c in df.columns:
            df = df.drop(columns=c)
    if not a.keep_procedure_name and "primary_procedure_name" in df.columns:
        df = df.drop(columns="primary_procedure_name")
    for c in HASH:
        if c in df.columns:
            df[c] = df[c].astype(str).map(
                lambda v: hashlib.sha256((a.salt + v).encode()).hexdigest()[:16]
            )

    df.to_csv(a.dst, index=False)
    removed = [c for c in before if c not in df.columns]
    print(f"rows      : {len(df):,}")
    print(f"removed   : {removed}")
    print(f"hashed    : {[c for c in HASH if c in df.columns]}")
    print(f"written   : {a.dst}")
    print("\nReview the output before sharing it.")


if __name__ == "__main__":
    main()
