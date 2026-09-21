"""CRCNL v2 final model.

    tabular + shipped lag block + procedure multi-hot
    -> two MAE-objective XGBoost specs x 5 seeds, median aggregated

Holdout MAE 30.6855 on the 2,839-case temporal patient-disjoint holdout,
against 31.5564 for the published CRCNL pipeline.

Everything that did NOT work is in docs/experiments_that_failed.md -- the
expanding-history block, booking-bias features, the overrun target
parameterisation, quantile regression, recency weighting, and both note
channels. Each was measured, not assumed.
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
COMMON = dict(enable_categorical=True, tree_method="hist", n_jobs=8, verbosity=0)

# Both specs use an absolute-error objective. Duration is right-skewed, so a
# squared or pseudo-Huber loss targets the conditional mean while MAE wants
# the median -- measured at 0.115 min worse when pseudo-Huber was in the mix.
SPECS = [
    dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
         max_depth=7, min_child_weight=8, subsample=0.85,
         colsample_bytree=0.75, reg_lambda=2.0),
    dict(objective="reg:absoluteerror", n_estimators=2200, learning_rate=0.015,
         max_depth=6, min_child_weight=4, subsample=0.8,
         colsample_bytree=0.55, reg_lambda=4.0),
]


def build():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    X, y, _ = D.build_design(df)
    F = build_causal_features(df)
    proc = [c for c in F.columns if c.startswith("proc__") and c not in X.columns]
    return df, pd.concat([X, F[proc]], axis=1), y


def predict(X, y, tr, ap):
    preds = []
    for s in SPECS:
        for sd in SEEDS:
            p = dict(COMMON); p.update(s); p["random_state"] = sd
            preds.append(xgb.XGBRegressor(**p).fit(X.iloc[tr], y[tr]).predict(X.iloc[ap]))
    return np.median(np.vstack(preds), axis=0)


def paired(a, b, n=4000, seed=42):
    rng = np.random.default_rng(seed); d = a - b
    i = rng.integers(0, len(d), size=(n, len(d)))
    m = d[i].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df, X, y = build()
    tr, te, te_new = D.temporal_split(df)
    pid = df[D.PATIENT].astype(str).to_numpy()
    assert not (set(pid[tr]) & set(pid[te])), "patient overlap"
    print(f"train {len(tr)}  holdout {len(te)}  features {X.shape[1]}")

    # baseline: same model without the procedure multi-hot
    base_cols = [c for c in X.columns if not c.startswith("proc__")]
    p_base = predict(X[base_cols], y, tr, te)
    p_full = predict(X, y, tr, te)

    m_base, m_full = float(MAE(y[te], p_base)), float(MAE(y[te], p_full))
    d, lo, hi = paired(np.abs(y[te] - p_base), np.abs(y[te] - p_full))

    res = {
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
        "n_features": int(X.shape[1]),
        "holdout_mae_without_procedure_multihot": m_base,
        "holdout_mae_final": m_full,
        "procedure_multihot_gain": d, "gain_ci95": [lo, hi],
        "published_crcnl_holdout_mae": 31.556418073067334,
        "improvement_vs_published": 31.556418073067334 - m_full,
        "zero_information_mad": float(np.abs(y[te] - np.median(y[tr])).mean()),
    }
    print(f"  without procedure multi-hot   {m_base:.4f}")
    print(f"  FINAL                         {m_full:.4f}")
    print(f"  multi-hot gain                {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  vs published 31.5564          {res['improvement_vs_published']:+.4f} min")

    json.dump(res, open(OUT / "FINAL.json", "w"), indent=2)
    np.save(OUT / "final_pred.npy", p_full)
    return res


if __name__ == "__main__":
    main()
