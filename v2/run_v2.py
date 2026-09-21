"""CRCNL v2 — the real experiment.

Base is the tuned tabular model WITH the audit-cleared lag block, which is a
much stronger starting point than the one the published 31.5564 was measured
against. The note channel is then asked to beat that.

Guards applied from the data-rules audit:
  * lag_* dropped for the first two quarters rather than imputed
  * an explicit bmi-missing indicator, so XGBoost cannot use NaN-direction
    as a free time index
  * surgeon id is categorical, not numeric
Residual iterations run to a stopping rule instead of stopping at one.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D

SEED = 42
N_FOLDS = 5
OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(exist_ok=True)

BASE_PARAMS = dict(
    objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
    max_depth=7, min_child_weight=8, subsample=0.85, colsample_bytree=0.75,
    reg_lambda=2.0, enable_categorical=True, tree_method="hist",
    random_state=SEED, n_jobs=8, verbosity=0,
)


def prepare():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)

    # bmi missingness was a free time index -- make it explicit and neutralise
    df["bmi_missing"] = df["bmi"].isna().astype("float32")

    # lag_* is not populated in the first two quarters; blank it rather than
    # let the model read "lag is missing" as "this case is old"
    q = pd.PeriodIndex(df[D.TIME].dt.tz_convert(None), freq="Q")
    early = q < (q.min() + 2)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df.loc[early, lag_cols] = np.nan
    df["lag_unavailable"] = early.astype("float32")

    llm = D.load_llm()
    llm_cols = D.llm_feature_columns(llm)
    df[D.CASE] = df[D.CASE].astype(str)
    df = df.merge(llm[[D.CASE] + llm_cols], on=D.CASE, how="left")
    return df, llm_cols


def base_fit(X_tr, y_tr, X_ap, seed=SEED):
    p = dict(BASE_PARAMS); p["random_state"] = seed
    return xgb.XGBRegressor(**p).fit(X_tr, y_tr).predict(X_ap)


def oof_predict(X, y, groups, idx, seed=SEED):
    oof = np.zeros(len(idx))
    Xi, yi, gi = X.iloc[idx], y[idx], groups[idx]
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Xi, yi, gi):
        oof[va] = base_fit(Xi.iloc[tr], yi[tr], Xi.iloc[va], seed)
    return oof


def residual_oof(Z, r, groups, seed=SEED):
    """Grouped OOF residual predictions, so gamma is not chosen on fitted values."""
    oof = np.zeros(len(r))
    pick = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(Z, r, groups):
        sc = StandardScaler().fit(Z[tr])
        A, B = sc.transform(Z[tr]), sc.transform(Z[va])
        best, bm, bn = None, np.inf, None
        for nm, m in (("huber", HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800)),
                      ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))):
            try:
                m.fit(A, r[tr])
                v = MAE(r[va], m.predict(B))
                if v < bm:
                    best, bm, bn = m, v, nm
            except Exception:
                continue
        oof[va] = best.predict(B); pick.append(bn)
    return oof, max(set(pick), key=pick.count)


def residual_full(Z_tr, r_tr, Z_ap, kind):
    sc = StandardScaler().fit(Z_tr)
    m = (HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800) if kind == "huber"
         else RidgeCV(alphas=np.logspace(-3, 3, 25)))
    m.fit(sc.transform(Z_tr), r_tr)
    return m.predict(sc.transform(Z_ap))


def pick_gamma(pred, r, grid=np.linspace(0.0, 1.6, 33)):
    v = [np.abs(r - g * pred).mean() for g in grid]
    return float(grid[int(np.argmin(v))])


def boot(a, b, n=4000, seed=SEED):
    rng = np.random.default_rng(seed); d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df, llm_cols = prepare()
    tr, te, te_new = D.temporal_split(df)
    X, y, _ = D.build_design(df.drop(columns=llm_cols))
    groups = df[D.PATIENT].astype(str).to_numpy()
    assert not (set(groups[tr]) & set(groups[te])), "patient overlap"
    print(f"train {len(tr)}  holdout {len(te)}  features {X.shape[1]}")

    # ---------------------------------------------------------------- base
    oof = oof_predict(X, y, groups, tr)
    hold = base_fit(X.iloc[tr], y[tr], X.iloc[te])
    res = {"n_train": int(len(tr)), "n_holdout": int(len(te)),
           "n_llm_features": len(llm_cols),
           "base": {"oof_mae": float(MAE(y[tr], oof)),
                    "holdout_mae": float(MAE(y[te], hold))},
           "iterations": []}
    print(f"base           OOF {res['base']['oof_mae']:.4f}  holdout {res['base']['holdout_mae']:.4f}")

    # ------------------------------------------------- the note channel
    Lraw = df[llm_cols].to_numpy(dtype="float64")
    seen = (~np.isnan(Lraw)).astype("float64")
    Z = np.hstack([np.nan_to_num(Lraw, nan=0.0), seen])

    base_abs = np.abs(y[te] - hold)
    cur_oof, cur_hold = oof.copy(), hold.copy()

    # ------------------------------------- residual iterations to convergence
    MAX_ITER, TOL = 5, 0.02
    prev = MAE(y[tr], cur_oof)
    for it in range(1, MAX_ITER + 1):
        r = y[tr] - cur_oof
        pr_oof, kind = residual_oof(Z[tr], r, groups[tr])
        g = pick_gamma(pr_oof, r)
        new_oof = cur_oof + g * pr_oof
        oof_mae = MAE(y[tr], new_oof)

        pr_te = residual_full(Z[tr], r, Z[te], kind)
        new_hold = cur_hold + g * pr_te
        hold_mae = MAE(y[te], new_hold)

        d, lo, hi = boot(base_abs, np.abs(y[te] - new_hold))
        rec = {"iter": it, "learner": kind, "gamma": g,
               "oof_mae": float(oof_mae), "holdout_mae": float(hold_mae),
               "cum_gain_vs_base": d, "cum_gain_ci95": [lo, hi],
               "oof_improvement": float(prev - oof_mae)}
        res["iterations"].append(rec)
        print(f"  iter {it}  {kind:5s} g={g:.3f}  OOF {oof_mae:.4f} "
              f"(-{prev-oof_mae:+.4f})  holdout {hold_mae:.4f}  "
              f"cum gain {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

        cur_oof, cur_hold = new_oof, new_hold
        if prev - oof_mae < TOL:
            rec["stopped"] = f"OOF improvement {prev-oof_mae:.4f} < tol {TOL}"
            print(f"  stopping: OOF improvement below {TOL} min")
            prev = oof_mae
            break
        prev = oof_mae

    res["final"] = {"oof_mae": float(MAE(y[tr], cur_oof)),
                    "holdout_mae": float(MAE(y[te], cur_hold)),
                    "published_crcnl_holdout_mae": 31.556418073067334}
    (OUT / "v2_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nfinal holdout {res['final']['holdout_mae']:.4f} "
          f"vs published CRCNL 31.5564")
    return res


if __name__ == "__main__":
    main()
