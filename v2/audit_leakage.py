"""Leakage and validity audit of the v2 result.

Checks, in order of how badly each would invalidate the 30.6866:
  A. split direction and patient disjointness
  B. is `procedures` the BOOKED list or the PERFORMED list
  C. exclusion rules 3/4/5 from the data guide, which v2 did not apply
  D. vocabulary built using holdout rows (multi-hot + rare bucketing)
  E. whether the model leans on bmi_missing as a time index (Rule 6)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import crcnl_data as D

DATA = Path("/mnt/user-data/uploads/Clinical_notes/data/mayo")


def main():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    tr, te, te_new = D.temporal_split(df)
    t = df[D.TIME]

    print("=" * 66)
    print("A. SPLIT")
    print("=" * 66)
    print(f"  train  n={len(tr):>6}  {t.iloc[tr].min().date()} -> {t.iloc[tr].max().date()}")
    print(f"  hold   n={len(te):>6}  {t.iloc[te].min().date()} -> {t.iloc[te].max().date()}")
    overlap_days = (t.iloc[tr].max() - t.iloc[te].min()).total_seconds() / 86400
    print(f"  train max is {overlap_days:+.2f} days vs holdout min "
          f"({'FORWARD-ONLY OK' if overlap_days <= 0 else 'OVERLAP'})")
    pid = df[D.PATIENT].astype(str).to_numpy()
    print(f"  shared patients: {len(set(pid[tr]) & set(pid[te]))}")
    print(f"  dropped from train for patient overlap: {int(round(len(df)*0.8)) - len(tr)}")

    print()
    print("=" * 66)
    print("B. IS `procedures` BOOKED OR PERFORMED?")
    print("=" * 66)
    # If the list were the performed set, cases flagged as 'no procedure
    # performed' would have an empty or shorter list than booked peers.
    from causal_features import parse_procs
    n_proc = df["procedures"].map(lambda v: len(parse_procs(v)))
    flag = df["all_procs_not_performed"].fillna(0).astype(float) > 0
    print(f"  cases flagged all_procs_not_performed: {int(flag.sum())}")
    print(f"  mean #procs listed, flagged   : {n_proc[flag].mean():.2f}")
    print(f"  mean #procs listed, not flagged: {n_proc[~flag].mean():.2f}")
    print("  -> a PERFORMED list would be ~0 for flagged cases;")
    print("     a BOOKED list stays populated.")
    print(f"  corr(#procs, duration) = {np.corrcoef(n_proc, df[D.TARGET])[0,1]:.3f}")
    print(f"  case_num_procedures vs len(procedures) equal in "
          f"{(df['case_num_procedures'] == n_proc).mean():.3f} of rows")

    print()
    print("=" * 66)
    print("C. EXCLUSION RULES v2 DID NOT APPLY")
    print("=" * 66)
    sched = df["sched_room_minutes"]
    backfilled = (df[D.TARGET] - sched).abs() < (1.0 / 60.0)
    print(f"  Rule 3 backfilled (|actual-sched| < 1s): {int(backfilled.sum())} rows "
          f"({backfilled.mean()*100:.2f}%)")
    print(f"     of which in holdout: {int(backfilled.to_numpy()[te].sum())}")
    print(f"  Rule 4 all_procs_not_performed        : {int(flag.sum())} rows")
    print(f"     of which in holdout: {int(flag.to_numpy()[te].sum())}")
    nullproc = df["primary_procedure_name"].isna()
    print(f"  Rule 4b null primary_procedure_name   : {int(nullproc.sum())} rows")
    print("  NOTE: v2 kept all of these, and kept all_procs_not_performed as a FEATURE.")

    print()
    print("=" * 66)
    print("D. VOCABULARY BUILT USING HOLDOUT ROWS")
    print("=" * 66)
    lists_all = df["procedures"].map(parse_procs)
    cnt_all, cnt_tr = {}, {}
    for i, L in enumerate(lists_all):
        for p in L:
            cnt_all[p] = cnt_all.get(p, 0) + 1
            if i in set(tr.tolist()) if False else False:
                pass
    tr_set = set(tr.tolist())
    for i, L in enumerate(lists_all):
        if i in tr_set:
            for p in L:
                cnt_tr[p] = cnt_tr.get(p, 0) + 1
    keep_all = {p for p, c in cnt_all.items() if c >= 30}
    keep_tr = {p for p, c in cnt_tr.items() if c >= 30}
    print(f"  multi-hot columns, vocab from ALL rows  : {len(keep_all)}")
    print(f"  multi-hot columns, vocab from TRAIN only: {len(keep_tr)}")
    print(f"  columns present only because of holdout : "
          f"{len(keep_all - keep_tr)}  {sorted(keep_all - keep_tr)}")

    print()
    print("=" * 66)
    print("E. bmi_missing AS A TIME INDEX (Rule 6)")
    print("=" * 66)
    q = t.dt.to_period("Q")
    bm = df["bmi"].isna()
    tab = pd.DataFrame({"q": q, "bmi_missing": bm}).groupby("q").bmi_missing.mean()
    print("  bmi null rate by quarter (first/last 3):")
    for k, v in list(tab.items())[:3] + list(tab.items())[-3:]:
        print(f"    {k}  {v:.2f}")
    print(f"  train bmi-null rate {bm.to_numpy()[tr].mean():.3f}   "
          f"holdout {bm.to_numpy()[te].mean():.3f}")


if __name__ == "__main__":
    main()
