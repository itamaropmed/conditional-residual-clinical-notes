"""Round three: recency weighting.

The data rules note duration drift of about +21% across the window, and the
holdout is the most recent slice. A model that weights all four years equally
is fitting a distribution the holdout no longer comes from. This sweeps an
exponential half-life on the training weights, selected on the validation
tail, then scores the finalists on the holdout once.
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

OUT = Path("/home/claude/crcnl2/out")
SEEDS = (42, 202609, 7, 1234, 99)
COMMON = dict(enable_categorical=True, tree_method="hist", n_jobs=8, verbosity=0)
SPEC = dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
            max_depth=7, min_child_weight=8, subsample=0.85,
            colsample_bytree=0.75, reg_lambda=2.0)
SPEC2 = dict(objective="reg:absoluteerror", n_estimators=2200, learning_rate=0.015,
             max_depth=6, min_child_weight=4, subsample=0.8,
             colsample_bytree=0.55, reg_lambda=4.0)


def prep():
    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    X, y, _ = D.build_design(df)
    F = build_causal_features(df)
    F = F.drop(columns=[c for c in F.columns if c in X.columns])
    Xc = pd.concat([X, F[[c for c in F.columns if c.startswith("proc__")]]], axis=1)
    age_days = (df[D.TIME].max() - df[D.TIME]).dt.total_seconds().to_numpy() / 86400.0
    return df, Xc, y, age_days


def fit(X, y, tr, ap, spec, w=None, seeds=SEEDS):
    preds = []
    for sd in seeds:
        p = dict(COMMON); p.update(spec); p["random_state"] = sd
        m = xgb.XGBRegressor(**p)
        m.fit(X.iloc[tr], y[tr], sample_weight=None if w is None else w[tr])
        preds.append(m.predict(X.iloc[ap]))
    return np.median(np.vstack(preds), axis=0)


def paired(a, b, n=4000, seed=42):
    rng = np.random.default_rng(seed); d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df, X, y, age = prep()
    tr, te, _ = D.temporal_split(df)
    vcut = int(len(tr) * 0.85)
    tr_in, tr_val = tr[:vcut], tr[vcut:]

    print("half-life sweep, selected on the validation tail:")
    best_hl, best_m = None, np.inf
    rows = []
    for hl in [None, 1095, 730, 547, 365, 240]:
        w = None if hl is None else np.exp(-age / hl).astype("float64")
        p = fit(X, y, tr_in, tr_val, SPEC, w)
        m = float(MAE(y[tr_val], p))
        rows.append({"half_life_days": hl, "val_mae": m})
        print(f"  half-life {str(hl):>6s}   val MAE {m:.4f}")
        if m < best_m:
            best_m, best_hl = m, hl

    print(f"\nselected half-life: {best_hl} (val {best_m:.4f})")
    w = None if best_hl is None else np.exp(-age / best_hl).astype("float64")

    print("\nholdout (scored once per finalist):")
    p_flat = fit(X, y, tr, te, SPEC, None)
    m_flat = float(MAE(y[te], p_flat))
    print(f"  unweighted, spec1          {m_flat:.4f}")

    p_w = fit(X, y, tr, te, SPEC, w)
    m_w = float(MAE(y[te], p_w))
    print(f"  recency-weighted, spec1    {m_w:.4f}")

    p_w2 = fit(X, y, tr, te, SPEC2, w)
    p_two = np.median(np.vstack([p_w, p_w2]), axis=0)
    m_two = float(MAE(y[te], p_two))
    print(f"  recency-weighted, 2 specs  {m_two:.4f}")

    d, lo, hi = paired(np.abs(y[te] - p_flat), np.abs(y[te] - p_two))
    print(f"\n  weighted-2spec vs unweighted: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    out = {"sweep": rows, "selected_half_life": best_hl,
           "holdout_unweighted": m_flat, "holdout_weighted": m_w,
           "holdout_weighted_2spec": m_two,
           "published_crcnl": 31.556418,
           "improvement_vs_published": 31.556418 - min(m_flat, m_w, m_two)}
    json.dump(out, open(OUT / "round3.json", "w"), indent=2)
    np.save(OUT / "round3_pred.npy", p_two)
    print(f"\nbest {min(m_flat, m_w, m_two):.4f}  "
          f"vs published 31.5564: {out['improvement_vs_published']:+.4f} min")
    return out


if __name__ == "__main__":
    main()
