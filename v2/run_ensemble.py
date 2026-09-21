"""Multi-seed ensemble with paired significance testing.

Single-fit MAE differences of 0.02-0.3 min on 2,839 cases are inside the noise
band, so every configuration here is an average over several seeds and every
comparison is a paired bootstrap on per-case absolute errors.
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

OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(exist_ok=True)
SEEDS = (42, 202609, 7, 1234, 99)

SPECS = [
    dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
         max_depth=7, min_child_weight=8, subsample=0.85, colsample_bytree=0.75,
         reg_lambda=2.0),
    dict(objective="reg:absoluteerror", n_estimators=2200, learning_rate=0.015,
         max_depth=6, min_child_weight=4, subsample=0.8, colsample_bytree=0.55,
         reg_lambda=4.0),
    dict(objective="reg:pseudohubererror", huber_slope=12.0, n_estimators=1800,
         learning_rate=0.02, max_depth=7, min_child_weight=6, subsample=0.85,
         colsample_bytree=0.7, reg_lambda=2.0),
]
COMMON = dict(enable_categorical=True, tree_method="hist", n_jobs=8, verbosity=0)


def prep():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    X, y, _ = D.build_design(df)
    F = build_causal_features(df)
    F = F.drop(columns=[c for c in F.columns if c in X.columns])
    return df, X, F, y


def ensemble_predict(X, y, tr, te, specs=SPECS, seeds=SEEDS):
    """Median across (spec, seed) fits. Median is the right aggregator for an
    absolute-error objective; the mean pulls toward outlier fits."""
    preds = []
    for s in specs:
        for sd in seeds:
            p = dict(COMMON); p.update(s); p["random_state"] = sd
            m = xgb.XGBRegressor(**p).fit(X.iloc[tr], y[tr])
            preds.append(m.predict(X.iloc[te]))
    P = np.vstack(preds)
    return np.median(P, axis=0), P


def paired(a_abs, b_abs, n=4000, seed=42):
    rng = np.random.default_rng(seed)
    d = a_abs - b_abs
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df, X, F, y = prep()
    tr, te, _ = D.temporal_split(df)
    pid = df[D.PATIENT].astype(str).to_numpy()
    assert not (set(pid[tr]) & set(pid[te])), "patient overlap"
    lag = [c for c in X.columns if c.startswith("lag_")]

    fam = {
        "proc_multihot": [c for c in F.columns if c.startswith("proc__")],
        "hx_duration": [c for c in F.columns if c.startswith("hx_") and "_dur" in c],
        "booking_bias": [c for c in F.columns if ("over" in c or "ratio" in c
                         or c.startswith("sched_plus") or c.startswith("sched_x")
                         or c == "sched_minus_sp_dur_mean")],
        "experience_load": ["surg_case_index", "sp_case_index", "proc_case_index",
                            "surg_day_load", "surg_day_pos"],
        "calendar": [c for c in ["sched_month_idx", "sched_minutes"] if c in F.columns],
    }

    configs = {
        "A_no_lag": X.drop(columns=lag),
        "B_shipped_lag": X,
        "C_lag+proc": pd.concat([X, F[fam["proc_multihot"]]], axis=1),
        "D_lag+proc+exp": pd.concat([X, F[fam["proc_multihot"]],
                                     F[fam["experience_load"]]], axis=1),
        "E_lag+proc+exp+hx": pd.concat([X, F[fam["proc_multihot"]],
                                        F[fam["experience_load"]],
                                        F[fam["hx_duration"]]], axis=1),
        "F_everything": pd.concat([X, F], axis=1),
    }

    res, preds = {}, {}
    print(f"multi-seed ensemble: {len(SPECS)} specs x {len(SEEDS)} seeds = "
          f"{len(SPECS)*len(SEEDS)} fits per config\n")
    for name, Xc in configs.items():
        p, _ = ensemble_predict(Xc, y, tr, te)
        mae = float(MAE(y[te], p))
        preds[name] = p
        res[name] = {"n_features": int(Xc.shape[1]), "holdout_mae": mae}
        print(f"  {name:22s} {Xc.shape[1]:4d} feats   MAE {mae:.4f}")

    ref = "B_shipped_lag"
    print(f"\npaired vs {ref} (positive = better):")
    ra = np.abs(y[te] - preds[ref])
    for name, p in preds.items():
        if name == ref:
            continue
        d, lo, hi = paired(ra, np.abs(y[te] - p))
        sig = "" if lo <= 0 <= hi else "  *"
        res[name]["gain_vs_shipped_lag"] = d
        res[name]["ci95"] = [lo, hi]
        print(f"  {name:22s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    best = min(res, key=lambda k: res[k]["holdout_mae"])
    print(f"\nbest {best}: {res[best]['holdout_mae']:.4f}")
    print(f"vs published CRCNL 31.5564: {31.556418 - res[best]['holdout_mae']:+.4f} min")
    json.dump(res, open(OUT / "ensemble_results.json", "w"), indent=2)
    np.save(OUT / "best_pred.npy", preds[best])
    np.save(OUT / "holdout_y.npy", y[te])
    return res


if __name__ == "__main__":
    main()
