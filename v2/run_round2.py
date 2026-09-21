"""Round two: objective choice, target parameterisation, and calibration.

Round one showed the procedure multi-hot is worth real minutes. This round
attacks three remaining levers:

  1. Objective. Duration is right-skewed, so a squared or pseudo-Huber loss
     targets the conditional mean while MAE wants the conditional median.
     Round one's median-of-specs ensemble included pseudo-Huber, which drags
     predictions upward. Tested here in isolation.
  2. Target parameterisation. Predicting the overrun (actual - booked slot)
     instead of the duration lets the tree spend its capacity on the
     correction rather than re-learning the booking.
  3. Calibration. A monotone recalibration fitted on the last slice of
     training time, applied to the holdout.

Selection happens on a time-based validation tail carved out of training.
The holdout is scored once per finalist.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D
from causal_features import build_causal_features

OUT = Path("/home/claude/crcnl2/out")
SEEDS = (42, 202609, 7, 1234, 99, 555, 8080)
COMMON = dict(enable_categorical=True, tree_method="hist", n_jobs=8, verbosity=0)

OBJ = {
    "mae_d7": dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
                   max_depth=7, min_child_weight=8, subsample=0.85,
                   colsample_bytree=0.75, reg_lambda=2.0),
    "mae_d6": dict(objective="reg:absoluteerror", n_estimators=2200, learning_rate=0.015,
                   max_depth=6, min_child_weight=4, subsample=0.8,
                   colsample_bytree=0.55, reg_lambda=4.0),
    "q50": dict(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=1600,
                learning_rate=0.02, max_depth=7, min_child_weight=6,
                subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0),
}


def prep():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    X, y, _ = D.build_design(df)
    F = build_causal_features(df)
    F = F.drop(columns=[c for c in F.columns if c in X.columns])
    proc = [c for c in F.columns if c.startswith("proc__")]
    # Round one: lag block + procedure multi-hot was the significant winner
    # (+0.5422 [+0.3266, +0.7519] vs the lag block alone). Adding expanding
    # history or booking-bias families on top made it worse -- they are
    # redundant with the lag block and add variance.
    Xc = pd.concat([X, F[proc]], axis=1)
    sched = F["sched_minutes"].to_numpy(dtype="float64")
    return df, Xc, y, sched


def fit_many(X, y, tr, ap, spec, seeds=SEEDS):
    preds = []
    for sd in seeds:
        p = dict(COMMON); p.update(OBJ[spec]); p["random_state"] = sd
        preds.append(xgb.XGBRegressor(**p).fit(X.iloc[tr], y[tr]).predict(X.iloc[ap]))
    return np.median(np.vstack(preds), axis=0)


def main():
    df, X, y, sched = prep()
    tr, te, _ = D.temporal_split(df)
    # time-ordered validation tail inside training, for selection only
    vcut = int(len(tr) * 0.85)
    tr_in, tr_val = tr[:vcut], tr[vcut:]
    print(f"train-inner {len(tr_in)}  val {len(tr_val)}  holdout {len(te)}\n")

    sched_f = np.where(np.isfinite(sched), sched, np.nanmedian(sched))
    results = {}

    print("selection on the validation tail (never the holdout):")
    for spec in OBJ:
        # direct target
        pv = fit_many(X, y, tr_in, tr_val, spec)
        m_direct = MAE(y[tr_val], pv)
        # overrun target
        ov = y - sched_f
        pv2 = fit_many(X, ov, tr_in, tr_val, spec) + sched_f[tr_val]
        m_over = MAE(y[tr_val], pv2)
        results[spec] = {"val_direct": float(m_direct), "val_overrun": float(m_over)}
        print(f"  {spec:8s}  direct {m_direct:.4f}   overrun-target {m_over:.4f}")

    # pick the best (spec, parameterisation) pairs on validation
    ranked = sorted(
        [(s, k, v[f"val_{k}"]) for s, v in results.items() for k in ("direct", "overrun")],
        key=lambda t: t[2])
    print("\nranked:", [(s, k, round(m, 4)) for s, k, m in ranked[:4]])

    # blend the top three on validation with non-negative weights
    top = ranked[:3]
    val_preds, te_preds = [], []
    for spec, kind, _ in top:
        if kind == "direct":
            val_preds.append(fit_many(X, y, tr_in, tr_val, spec))
            te_preds.append(fit_many(X, y, tr, te, spec))
        else:
            ov = y - sched_f
            val_preds.append(fit_many(X, ov, tr_in, tr_val, spec) + sched_f[tr_val])
            te_preds.append(fit_many(X, ov, tr, te, spec) + sched_f[te])
    V = np.vstack(val_preds); T = np.vstack(te_preds)

    best_w, best_m = None, np.inf
    for w in np.round(np.linspace(0, 1, 21), 2):
        for w2 in np.round(np.linspace(0, 1 - w, int((1 - w) * 20) + 1), 2):
            ws = np.array([w, w2, 1 - w - w2])
            m = MAE(y[tr_val], ws @ V)
            if m < best_m:
                best_m, best_w = m, ws
    print(f"\nblend weights {best_w} -> val MAE {best_m:.4f}")

    blend_te = best_w @ T
    mae_blend = float(MAE(y[te], blend_te))

    # monotone recalibration fitted on validation
    iso = IsotonicRegression(out_of_bounds="clip").fit(best_w @ V, y[tr_val])
    mae_iso = float(MAE(y[te], iso.predict(blend_te)))

    single = float(MAE(y[te], T[0]))
    print(f"\nholdout:")
    print(f"  best single ({top[0][0]}/{top[0][1]})  {single:.4f}")
    print(f"  blended                        {mae_blend:.4f}")
    print(f"  blended + isotonic             {mae_iso:.4f}")

    out = {"val": results, "blend_weights": best_w.tolist(),
           "holdout_single": single, "holdout_blend": mae_blend,
           "holdout_blend_isotonic": mae_iso,
           "published_crcnl": 31.556418}
    json.dump(out, open(OUT / "round2.json", "w"), indent=2)
    np.save(OUT / "round2_pred.npy", blend_te)
    return out


if __name__ == "__main__":
    main()
