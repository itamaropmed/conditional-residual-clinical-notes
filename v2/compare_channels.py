"""Does the graded extraction beat the binary regex flags at predicting the
residual the tabular model leaves behind?

Measured on the 2,352 sampled TRAINING cases only, with patient-grouped
cross-validation. The holdout is not touched anywhere in this file.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D
from stage_b_extract import extract_frame
from eval_extractor import old_flags
from gold_labels import FIELDS

OUT = Path("/home/claude/crcnl2/out")
SEED = 42


def main():
    ev = pd.read_parquet(OUT / "evidence_ordered.parquet")
    ev["case_durable_id"] = ev.case_durable_id.astype(str)

    df = D.add_schedule_derived(D.load_tabular(with_lags=True))
    df["surgeon_durable_id"] = df["surgeon_durable_id"].astype(str)
    df["bmi_missing"] = df["bmi"].isna().astype("float32")
    df[D.CASE] = df[D.CASE].astype(str)

    tr, te, _ = D.temporal_split(df)
    train = df.iloc[tr].copy()
    sub = train[train[D.CASE].isin(set(ev.case_durable_id))].copy()
    print(f"{len(sub)} sampled training cases with evidence")

    X, y, _ = D.build_design(df)
    g = df[D.PATIENT].astype(str).to_numpy()

    # base OOF over the full training set, then restrict to the sampled cases
    oof = np.zeros(len(tr))
    Xt, yt, gt = X.iloc[tr], y[tr], g[tr]
    for a, b in GroupKFold(n_splits=5).split(Xt, yt, gt):
        m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1500,
                             learning_rate=0.02, max_depth=7, min_child_weight=8,
                             subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0,
                             enable_categorical=True, tree_method="hist",
                             random_state=SEED, n_jobs=8, verbosity=0)
        m.fit(Xt.iloc[a], yt[a])
        oof[b] = m.predict(Xt.iloc[b])
    train["_oof"] = oof
    sub = sub.merge(train[[D.CASE, "_oof"]], on=D.CASE, how="left")
    r = (sub[D.TARGET] - sub["_oof"]).to_numpy()
    grp = sub[D.PATIENT].astype(str).to_numpy()
    print(f"residual: MAE {np.abs(r).mean():.4f}  sd {r.std():.4f}")

    e = ev.set_index("case_durable_id").reindex(sub[D.CASE])

    # ---- channel 1: the incumbent binary regex flags
    Zold = pd.DataFrame([old_flags(t) for t in e.evidence]).to_numpy(dtype="float64")

    # ---- channel 2: graded extraction, with an explicit seen-mask
    gx = extract_frame(e.reset_index()).set_index("case_durable_id").reindex(sub[D.CASE])
    V = gx[FIELDS].to_numpy(dtype="float64")
    seen = (V >= 0).astype("float64")
    Znew = np.hstack([np.where(V < 0, 0.0, V), seen])

    def cv_mae(Z, name):
        pred = np.zeros(len(r))
        for a, b in GroupKFold(n_splits=5).split(Z, r, grp):
            sc = StandardScaler().fit(Z[a])
            m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(Z[a]), r[a])
            pred[b] = m.predict(sc.transform(Z[b]))
        # shrinkage chosen inside CV folds would be cleaner; use 1.0 to stay honest
        base = np.abs(r).mean()
        got = np.abs(r - pred).mean()
        rng = np.random.default_rng(SEED)
        d = np.abs(r) - np.abs(r - pred)
        idx = rng.integers(0, len(d), size=(4000, len(d)))
        bs = d[idx].mean(axis=1)
        print(f"  {name:22s} residual MAE {got:.4f}  gain {base-got:+.4f} "
              f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  "
              f"corr {np.corrcoef(pred, r)[0,1]:.3f}")
        return {"name": name, "residual_mae": float(got), "gain": float(base - got),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "corr": float(np.corrcoef(pred, r)[0, 1])}

    print("\npatient-grouped CV on the residual:")
    res = [cv_mae(Zold, "regex binary flags"), cv_mae(Znew, "graded extraction")]

    json.dump({"n_cases": int(len(sub)), "channels": res},
              open(OUT / "channel_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
