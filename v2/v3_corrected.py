"""v3: the corrected model.

The audit found four problems with v2. Fixed here, and re-measured.

  1. `all_procs_not_performed` was used as a FEATURE. It is a post-hoc fact --
     you only know no procedure was performed after the case. Removed.
  2. Rule 3 not applied: 105 rows whose actual duration equals the scheduled
     duration to the second are backfill artefacts, and they are free MAE for
     any model holding the scheduled duration. 23 of them sat in the holdout.
     Dropped.
  3. Rules 4/4b not applied: 23 aborted cases (19 in holdout) and 6 rows with
     a null primary procedure. Dropped.
  4. Vocabulary leakage: the procedure multi-hot kept codes appearing >= 30
     times ACROSS ALL ROWS, and the categorical rare-bucketing counted levels
     the same way. Ten multi-hot columns existed only because holdout rows
     pushed them over the threshold. Both vocabularies are now built from
     training rows only.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error as MAE
import xgboost as xgb

import crcnl_data as D
from causal_features import parse_procs, build_causal_features

OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(exist_ok=True)
SEEDS = (42, 202609, 7, 1234, 99)
COMMON = dict(enable_categorical=True, tree_method="hist", n_jobs=8, verbosity=0)
SPECS = [
    dict(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.02,
         max_depth=7, min_child_weight=8, subsample=0.85,
         colsample_bytree=0.75, reg_lambda=2.0),
    dict(objective="reg:absoluteerror", n_estimators=2200, learning_rate=0.015,
         max_depth=6, min_child_weight=4, subsample=0.8,
         colsample_bytree=0.55, reg_lambda=4.0),
]

POST_HOC_FEATURES = ["all_procs_not_performed"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    sched = df["sched_room_minutes"]
    drop_backfill = (df[D.TARGET] - sched).abs() < (1.0 / 60.0)      # Rule 3
    drop_abort = df["all_procs_not_performed"].fillna(0).astype(float) > 0  # Rule 4
    drop_null = df["primary_procedure_name"].isna()                   # Rule 4b
    keep = ~(drop_backfill | drop_abort | drop_null)
    stats = {"start": n0, "backfilled": int(drop_backfill.sum()),
             "aborted": int(drop_abort.sum()), "null_procedure": int(drop_null.sum()),
             "kept": int(keep.sum())}
    return df[keep].reset_index(drop=True), stats


def design_train_vocab(df: pd.DataFrame, tr: np.ndarray):
    """Build the feature frame with every vocabulary fitted on TRAIN rows only."""
    drop = set(D.POST_HOC + POST_HOC_FEATURES +
               [D.CASE, D.PATIENT, "patient_birth_date", D.TIME,
                "scheduled_setup_start", "scheduled_out_of_room",
                "scheduled_cleanup_complete", "procedures"])
    feat = [c for c in df.columns if c not in drop]
    X = df[feat].copy()

    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            X[c] = X[c].astype("int64") // 10**9
        elif pd.api.types.is_bool_dtype(dt):
            X[c] = X[c].astype("float64")
        elif pd.api.types.is_numeric_dtype(dt):
            X[c] = X[c].astype("float64")
        else:
            s = X[c].astype(str).fillna("__na__")
            vc = s.iloc[tr].value_counts()                 # TRAIN ONLY
            keep_lv = set(vc[vc >= 25].index)
            s = s.where(s.isin(keep_lv), "__rare__")
            X[c] = pd.Categorical(s, categories=sorted(keep_lv | {"__rare__"}))

    # procedure multi-hot, vocabulary from TRAIN rows only
    lists = df["procedures"].map(parse_procs)
    cnt: dict[str, int] = {}
    for i in tr:
        for p in lists.iloc[i]:
            cnt[p] = cnt.get(p, 0) + 1
    vocab = sorted(p for p, c in cnt.items() if c >= 30)
    M = np.zeros((len(df), len(vocab)), dtype="float32")
    pos = {p: i for i, p in enumerate(vocab)}
    for r, L in enumerate(lists):
        for p in L:
            j = pos.get(p)
            if j is not None:
                M[r, j] = 1.0
    P = pd.DataFrame(M, columns=[f"proc__{p}" for p in vocab], index=X.index)
    P["proc__n_unique"] = lists.map(len).to_numpy(dtype="float32")

    return pd.concat([X, P], axis=1), len(vocab)


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
    raw = D.add_schedule_derived(D.load_tabular(with_lags=True))
    raw["surgeon_durable_id"] = raw["surgeon_durable_id"].astype(str)
    raw["bmi_missing"] = raw["bmi"].isna().astype("float32")
    df, stats = clean(raw)
    print(f"exclusions: {stats}")

    tr, te, _ = D.temporal_split(df)
    y = df[D.TARGET].to_numpy(dtype="float64")
    pid = df[D.PATIENT].astype(str).to_numpy()
    assert not (set(pid[tr]) & set(pid[te])), "patient overlap"
    print(f"train {len(tr)}  holdout {len(te)}")

    X, nvocab = design_train_vocab(df, tr)
    print(f"features {X.shape[1]}  (procedure vocab {nvocab}, train-only)")

    base_cols = [c for c in X.columns if not c.startswith("proc__")]
    p_base = predict(X[base_cols], y, tr, te)
    p_full = predict(X, y, tr, te)
    m_base, m_full = float(MAE(y[te], p_base)), float(MAE(y[te], p_full))
    d, lo, hi = paired(np.abs(y[te] - p_base), np.abs(y[te] - p_full))

    # tail-stratified report (Rule 5)
    resid = np.abs(y[te] - p_full)
    q = np.quantile(y[te], [0.01, 0.99])
    mid = (y[te] >= q[0]) & (y[te] <= q[1])
    print(f"\n  v3 without procedure multi-hot  {m_base:.4f}")
    print(f"  v3 FINAL                        {m_full:.4f}")
    print(f"  multi-hot gain                  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  middle 98% MAE                  {resid[mid].mean():.4f}")
    print(f"  tails (1%/99%) MAE              {resid[~mid].mean():.4f}")
    print(f"\n  v2 (uncorrected) was            30.6866")
    print(f"  published CRCNL                 31.5564")

    res = {"exclusions": stats, "n_train": int(len(tr)), "n_holdout": int(len(te)),
           "n_features": int(X.shape[1]), "procedure_vocab_train_only": nvocab,
           "holdout_mae_without_multihot": m_base, "holdout_mae_v3": m_full,
           "multihot_gain": d, "multihot_ci95": [lo, hi],
           "mae_middle_98pct": float(resid[mid].mean()),
           "mae_tails": float(resid[~mid].mean()),
           "v2_uncorrected": 30.6866, "published_crcnl": 31.556418,
           "note": "v2 and v3 holdouts differ (v3 drops 42 excluded rows), "
                   "so the two MAEs are not strictly comparable"}
    json.dump(res, open(OUT / "V3_CORRECTED.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
