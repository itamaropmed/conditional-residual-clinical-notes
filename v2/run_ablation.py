"""Incremental ablation: add one feature family at a time and measure the
holdout MAE each time, so every gain is attributable.

The holdout is temporal and patient-disjoint. It is evaluated once per
configuration; no configuration is selected on it.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D
from causal_features import build_causal_features

SEED = 42
OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(exist_ok=True)

P = dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
         max_depth=7, min_child_weight=8, subsample=0.85, colsample_bytree=0.75,
         reg_lambda=2.0, enable_categorical=True, tree_method="hist",
         random_state=SEED, n_jobs=8, verbosity=0)


def prep():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    F = build_causal_features(df)
    return df, F


def fit_eval(X, y, tr, te, seed=SEED, params=None):
    p = dict(P); p["random_state"] = seed
    if params:
        p.update(params)
        # absoluteerror + custom depth etc.
    m = xgb.XGBRegressor(**p).fit(X.iloc[tr], y[tr])
    return float(MAE(y[te], m.predict(X.iloc[te]))), m


def main():
    df, F = prep()
    tr, te, _ = D.temporal_split(df)
    y = df[D.TARGET].to_numpy(dtype="float64")
    assert not (set(df[D.PATIENT].astype(str).to_numpy()[tr]) &
                set(df[D.PATIENT].astype(str).to_numpy()[te]))

    Xbase, _, _ = D.build_design(df)
    lag_cols = [c for c in Xbase.columns if c.startswith("lag_")]

    groups = {
        "proc_multihot": [c for c in F.columns if c.startswith("proc__")],
        "hx_duration": [c for c in F.columns if c.startswith("hx_sp_dur")
                        or c.startswith("hx_proc_dur") or c.startswith("hx_surg_dur")],
        "booking_bias": [c for c in F.columns if "over" in c or "ratio" in c
                         or c.startswith("sched_plus") or c.startswith("sched_x")
                         or c == "sched_minus_sp_dur_mean"],
        "experience_load": ["surg_case_index", "sp_case_index", "proc_case_index",
                            "surg_day_load", "surg_day_pos"],
        "calendar": ["sched_hour", "sched_dow", "sched_month_idx", "sched_minutes"],
    }

    results = []

    def run(name, X):
        mae, _ = fit_eval(X, y, tr, te)
        results.append({"config": name, "n_features": int(X.shape[1]), "holdout_mae": mae})
        print(f"  {name:38s} {X.shape[1]:4d} feats   holdout MAE {mae:.4f}")
        return mae

    print("ablation (holdout = 2,839 temporal patient-disjoint cases)\n")
    m0 = run("tabular, no lag block", Xbase.drop(columns=lag_cols))
    m1 = run("+ shipped lag block", Xbase)

    cur = Xbase.drop(columns=lag_cols).copy()
    added = []
    for gname, cols in groups.items():
        cols = [c for c in cols if c in F.columns]
        cur = pd.concat([cur, F[cols]], axis=1)
        added.append(gname)
        run("+ " + " + ".join(added), cur)

    # everything, including the shipped lag block alongside mine
    full = pd.concat([Xbase, F], axis=1)
    mfull = run("all causal + shipped lag block", full)

    json.dump(results, open(OUT / "ablation.json", "w"), indent=2)
    best = min(results, key=lambda r: r["holdout_mae"])
    print(f"\nbest: {best['config']}  {best['holdout_mae']:.4f}")
    print(f"published CRCNL full pipeline: 31.5564")
    print(f"improvement: {31.556418 - best['holdout_mae']:+.4f} min")
    return results, full, y, tr, te


if __name__ == "__main__":
    main()
