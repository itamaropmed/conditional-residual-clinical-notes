"""v4: the leak-safe model.

Keeps v3's four fixes and adds the rest of the documented rules, pricing each
one so the cost of safety is explicit.

New in v4:
  * Rule 10 -- training starts 2023-01-01. 2022Q3 has 100% null lag features
    and 2022Q4 is 38% null; training on them teaches a state that never recurs.
  * Staffing columns dropped. `case_num_providers` correlates +0.604 with the
    target and +0.358 within procedure, and matches the booked procedure count
    in only 88% of rows -- so 12% of its value comes from somewhere other than
    the booking. It may be the providers who ACTUALLY scrubbed in, which is
    post-hoc. The schema does not say. Dropped by default; the cost is measured
    so the decision can be revisited if Mayo confirms it is booked staffing.
  * Rule 6 -- bmi_missing is tested as a time index rather than assumed benign.
  * Rule 11 -- both numbers reported: patient-purged and new-patient-only.

Verified clean and kept: `prior_same_proc_duration` matches a strictly-past
reconstruction in 100.0% of rows (corr 1.000 vs 0.741 row-inclusive).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D
from causal_features import parse_procs
from v3_corrected import clean, predict, paired, SPECS, COMMON, SEEDS

OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(exist_ok=True)

POST_HOC = ["all_procs_not_performed"]
STAFFING_RISK = ["case_num_providers", "case_num_panels"]
TRAIN_START = pd.Timestamp("2023-01-01", tz="UTC")


def design(df, tr, drop_extra=()):
    drop = set(D.POST_HOC + POST_HOC + list(drop_extra) +
               [D.CASE, D.PATIENT, "patient_birth_date", D.TIME,
                "scheduled_setup_start", "scheduled_out_of_room",
                "scheduled_cleanup_complete", "procedures"])
    X = df[[c for c in df.columns if c not in drop]].copy()
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            X[c] = X[c].astype("int64") // 10**9
        elif pd.api.types.is_bool_dtype(dt) or pd.api.types.is_numeric_dtype(dt):
            X[c] = X[c].astype("float64")
        else:
            s = X[c].astype(str).fillna("__na__")
            vc = s.iloc[tr].value_counts()                      # train-only
            keep = set(vc[vc >= 25].index)                      # Rules 7 & 8
            X[c] = pd.Categorical(s.where(s.isin(keep), "__rare__"),
                                  categories=sorted(keep | {"__rare__"}))
    lists = df["procedures"].map(parse_procs)
    cnt: dict[str, int] = {}
    for i in tr:
        for p in lists.iloc[i]:
            cnt[p] = cnt.get(p, 0) + 1
    vocab = sorted(p for p, c in cnt.items() if c >= 30)         # train-only
    M = np.zeros((len(df), len(vocab)), dtype="float32")
    pos = {p: i for i, p in enumerate(vocab)}
    for r, L in enumerate(lists):
        for p in L:
            if p in pos:
                M[r, pos[p]] = 1.0
    P = pd.DataFrame(M, columns=[f"proc__{p}" for p in vocab], index=X.index)
    P["proc__n_unique"] = lists.map(len).to_numpy(dtype="float32")
    return pd.concat([X, P], axis=1)


def main():
    raw = D.add_schedule_derived(D.load_tabular(with_lags=True))
    raw["surgeon_durable_id"] = raw["surgeon_durable_id"].astype(str)
    raw["bmi_missing"] = raw["bmi"].isna().astype("float32")
    df, stats = clean(raw)
    print(f"exclusions: {stats}")

    tr, te, te_new = D.temporal_split(df)
    y = df[D.TARGET].to_numpy(dtype="float64")
    t = df[D.TIME]

    # Rule 10: restrict training to 2023-01-01 onward (holdout untouched)
    tr10 = tr[t.iloc[tr].to_numpy() >= TRAIN_START]
    print(f"train {len(tr)} -> {len(tr10)} after Rule 10 burn-in "
          f"(dropped {len(tr)-len(tr10)})")
    print(f"holdout {len(te)}, of which new-patient {len(te_new)}")

    res, preds = {}, {}

    def run(name, tr_idx, drop_extra=()):
        X = design(df, tr_idx, drop_extra)
        p = predict(X, y, tr_idx, te)
        m = float(MAE(y[te], p))
        mn = float(MAE(y[te_new], p[np.isin(te, te_new)]))
        preds[name] = p
        res[name] = {"n_train": int(len(tr_idx)), "n_features": int(X.shape[1]),
                     "holdout_mae": m, "new_patient_mae": mn}
        print(f"  {name:28s} n_tr={len(tr_idx):5d} f={X.shape[1]:4d}  "
              f"MAE {m:.4f}   new-patient {mn:.4f}")
        return m

    print("\nconfigurations:")
    run("v3 (all rows from 2022)", tr)
    run("v4a +Rule10 burn-in", tr10)
    run("v4b -staffing (LEAK-SAFE)", tr10, STAFFING_RISK)
    run("v4c -staffing -bmi", tr10, STAFFING_RISK + ["bmi", "bmi_missing"])

    print("\npaired vs the leak-safe v4b (positive = v4b better):")
    rb = np.abs(y[te] - preds["v4b -staffing (LEAK-SAFE)"])
    for k, p in preds.items():
        if k.startswith("v4b"):
            continue
        d, lo, hi = paired(np.abs(y[te] - p), rb)
        res[k]["vs_v4b"] = {"gain": d, "ci95": [lo, hi]}
        tag = "" if lo <= 0 <= hi else "  *"
        print(f"  {k:28s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{tag}")

    # Rule 6: does the model lean on bmi_missing?
    Xs = design(df, tr10, STAFFING_RISK)
    mdl = xgb.XGBRegressor(**{**COMMON, **SPECS[0], "random_state": 42})
    mdl.fit(Xs.iloc[tr10], y[tr10])
    imp = pd.Series(mdl.feature_importances_, index=Xs.columns).sort_values(ascending=False)
    rank = list(imp.index).index("bmi_missing") + 1 if "bmi_missing" in imp.index else None
    print(f"\nRule 6: bmi_missing importance rank {rank}/{len(imp)} "
          f"(gain share {imp.get('bmi_missing', 0):.4f})")
    print("  top 8 features:", ", ".join(imp.head(8).index))
    res["rule6"] = {"bmi_missing_rank": rank,
                    "bmi_missing_importance": float(imp.get("bmi_missing", 0)),
                    "top8": list(imp.head(8).index)}

    res["exclusions"] = stats
    res["n_holdout"] = int(len(te))
    res["n_holdout_new_patient"] = int(len(te_new))
    json.dump(res, open(OUT / "V4_LEAKSAFE.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
