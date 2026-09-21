"""Feature-level audit: is every column knowable at prediction time?

Prediction time is when the case is booked -- `scheduled_in_room`. A column is
disqualified if its value could only be established during or after the case.

Two tests:
  1. Reconstruct history-derived columns from strictly-past rows. If the file's
     values match a strictly-past reconstruction, they are time-honest; if they
     match a row-inclusive one better, they leak.
  2. For count columns, compare against the booked list, which is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import crcnl_data as D
from causal_features import parse_procs


def main():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True)).copy()
    df = df.sort_values(D.TIME).reset_index(drop=True)
    y = df[D.TARGET].to_numpy(dtype="float64")

    print("=" * 70)
    print("1. COUNT COLUMNS — booked or actual?")
    print("=" * 70)
    nproc = df["procedures"].map(lambda v: len(parse_procs(v)))
    for c in ["case_num_procedures", "case_num_panels", "case_num_providers"]:
        v = df[c]
        print(f"  {c:22s} corr(y)={np.corrcoef(v.fillna(v.median()), y)[0,1]:+.3f}  "
              f"nunique={v.nunique():3d}  == len(procedures) in "
              f"{(v == nproc).mean():.3f}")
    print("  case_num_procedures matches the booked list exactly -> booked.")
    print("  case_num_panels / case_num_providers: staffing. If these are the")
    print("  ACTUAL people who scrubbed in, they are post-hoc. Checking spread")
    print("  against duration within a procedure, which is what a post-hoc")
    print("  staffing count would track:")
    for c in ["case_num_panels", "case_num_providers"]:
        g = df.groupby("primary_procedure_name", observed=True)
        within = g.apply(lambda d: d[c].corr(d[D.TARGET]) if d[c].nunique() > 1 else np.nan,
                         include_groups=False)
        print(f"    {c:22s} median within-procedure corr with duration "
              f"{within.median():+.3f}")

    print()
    print("=" * 70)
    print("2. HISTORY COLUMNS — strictly past?")
    print("=" * 70)
    df["_surg"] = df["surgeon_durable_id"].astype(str)
    df["_proc"] = df["primary_procedure_name"].astype(str)
    df["_pat"] = df[D.PATIENT].astype(str)

    # prior_same_proc_duration should equal this patient's previous duration on
    # the same procedure, from strictly earlier rows.
    key = df["_pat"] + "||" + df["_proc"]
    past = df.groupby(key, observed=True)[D.TARGET].shift(1)
    incl = df.groupby(key, observed=True)[D.TARGET].transform("last")
    col = df["prior_same_proc_duration"]
    m = col.notna() & past.notna()
    print(f"  prior_same_proc_duration: {int(col.notna().sum())} non-null")
    if m.sum() > 20:
        print(f"    corr with strictly-past reconstruction  {col[m].corr(past[m]):.3f}")
        print(f"    corr with row-inclusive reconstruction  {col[m].corr(incl[m]):.3f}")
        print(f"    exact match to strictly-past            {(col[m] == past[m]).mean():.3f}")
    print(f"    corr with target                        "
          f"{col[col.notna()].corr(df[D.TARGET][col.notna()]):.3f}")

    for c in ["is_redo", "redo_time_gap_days"]:
        v = df[c]
        print(f"  {c:24s} non-null {int(v.notna().sum()):5d}  "
              f"corr(y) {v[v.notna()].corr(df[D.TARGET][v.notna()]):+.3f}")

    print()
    print("=" * 70)
    print("3. RULE 10 — burn-in quarters")
    print("=" * 70)
    q = df[D.TIME].dt.tz_convert(None).dt.to_period("Q")
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    for period in sorted(q.unique())[:4]:
        sel = q == period
        print(f"  {period}  n={int(sel.sum()):4d}  lag_sp_mean null "
              f"{df.loc[sel, 'lag_sp_mean'].isna().mean():.2f}")
    pre2023 = df[D.TIME] < pd.Timestamp("2023-01-01", tz="UTC")
    print(f"  rows before 2023-01-01: {int(pre2023.sum())} ({pre2023.mean()*100:.1f}%)")

    print()
    print("=" * 70)
    print("4. RULE 11 — repeat patients across the split")
    print("=" * 70)
    tr, te, te_new = D.temporal_split(df, strict_disjoint=False)
    pid = df[D.PATIENT].astype(str).to_numpy()
    shared = set(pid[tr]) & set(pid[te])
    print(f"  plain forward split (no patient purge):")
    print(f"    patients on both sides       {len(shared)}")
    print(f"    holdout cases with a prior   {len(te) - len(te_new)}")
    print(f"    new-patient holdout cases    {len(te_new)}")


if __name__ == "__main__":
    main()
