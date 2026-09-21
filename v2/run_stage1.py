"""CRCNL rebuild: tabular base -> residual corrector, comparing the regex
concept channel against the LLM-extracted structured fields.

Everything is fit on training folds only. The holdout is touched exactly once
per variant, at the end, to report a number.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import crcnl_data as D

SEED = 42
N_FOLDS = 5
OUT = Path("/home/claude/crcnl2/out")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- base model

def fit_base(X_tr, y_tr, X_ap, objective: str, seed: int = SEED):
    m = xgb.XGBRegressor(
        objective=objective, n_estimators=600, learning_rate=0.045,
        max_depth=6, min_child_weight=4, subsample=0.85, colsample_bytree=0.75,
        reg_lambda=2.0, reg_alpha=0.1, enable_categorical=True,
        tree_method="hist", random_state=seed, n_jobs=8, verbosity=0,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_ap)


def base_blend(X_tr, y_tr, X_ap, w_squared: float = 0.60):
    """The CRCNL 60/40 squared/absolute blend."""
    a = fit_base(X_tr, y_tr, X_ap, "reg:squarederror")
    b = fit_base(X_tr, y_tr, X_ap, "reg:absoluteerror")
    return w_squared * a + (1.0 - w_squared) * b


def oof_base(X, y, groups, idx, n_folds=N_FOLDS):
    """Patient-grouped out-of-fold base predictions over the training rows."""
    oof = np.zeros(len(idx))
    gkf = GroupKFold(n_splits=n_folds)
    Xi, yi, gi = X.iloc[idx], y[idx], groups[idx]
    for tr, va in gkf.split(Xi, yi, gi):
        oof[va] = base_blend(Xi.iloc[tr], yi[tr], Xi.iloc[va])
    return oof


# ----------------------------------------------------------- residual stage

def fit_residual(Z_tr, r_tr, Z_ap):
    """Pick between Huber and RidgeCV by grouped OOF MAE on the residual."""
    sc = StandardScaler().fit(Z_tr)
    A, B = sc.transform(Z_tr), sc.transform(Z_ap)
    cands = {
        "huber": HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
    }
    best, best_mae, best_name = None, np.inf, None
    for name, m in cands.items():
        try:
            m.fit(A, r_tr)
            mae = mean_absolute_error(r_tr, m.predict(A))
            if mae < best_mae:
                best, best_mae, best_name = m, mae, name
        except Exception:
            continue
    return best.predict(B), best_name


def shrink(pred_resid, r_true, grid=np.linspace(0.0, 2.0, 41)):
    """Global shrinkage gamma chosen on training residuals only."""
    best_g, best = 0.0, np.inf
    for g in grid:
        m = np.abs(r_true - g * pred_resid).mean()
        if m < best:
            best, best_g = m, g
    return best_g


def boot_ci(a, b, n=2000, seed=SEED):
    """Paired bootstrap on the per-case absolute-error difference (a - b)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------- driver

def main():
    df = D.add_schedule_derived(D.load_tabular())
    llm = D.load_llm()
    llm_cols = D.llm_feature_columns(llm)

    df[D.CASE] = df[D.CASE].astype(str)
    df = df.merge(llm[[D.CASE] + llm_cols], on=D.CASE, how="left")

    tr, te, te_new = D.temporal_split(df)
    X, y, _ = D.build_design(df.drop(columns=llm_cols))
    groups = df[D.PATIENT].astype(str).to_numpy()

    print(f"train {len(tr)}  holdout {len(te)}  new-patient holdout {len(te_new)}")
    assert not (set(groups[tr]) & set(groups[te])), "patient overlap"

    # ---- base
    print("fitting base (patient-grouped OOF + holdout) ...")
    oof = oof_base(X, y, groups, tr)
    hold = base_blend(X.iloc[tr], y[tr], X.iloc[te])
    base_oof_mae = mean_absolute_error(y[tr], oof)
    base_hold_mae = mean_absolute_error(y[te], hold)
    print(f"  base   OOF {base_oof_mae:.4f}   holdout {base_hold_mae:.4f}")

    r_tr = y[tr] - oof                       # the residual the notes must explain

    # ---- residual channels
    L = df[llm_cols].to_numpy(dtype="float64")
    seen = ~np.isnan(L)
    L = np.nan_to_num(L, nan=0.0)
    # Missingness is informative: a field the extractor never saw is a
    # different state from one it saw and scored zero.
    Lf = np.hstack([L, seen.astype("float64")])

    channels = {
        "llm_structured": Lf,
    }

    results = {
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
        "n_holdout_new_patient": int(len(te_new)),
        "n_llm_features": len(llm_cols),
        "base": {"oof_mae": base_oof_mae, "holdout_mae": base_hold_mae},
        "variants": {},
    }

    base_abs = np.abs(y[te] - hold)

    for name, Z in channels.items():
        pr_tr, which = fit_residual(Z[tr], r_tr, Z[tr])
        pr_te, _ = fit_residual(Z[tr], r_tr, Z[te])
        g = shrink(pr_tr, r_tr)
        corr = hold + g * pr_te
        mae = mean_absolute_error(y[te], corr)
        oof_corr = mean_absolute_error(y[tr], oof + g * pr_tr)
        d, lo, hi = boot_ci(base_abs, np.abs(y[te] - corr))
        results["variants"][name] = {
            "learner": which, "gamma": float(g),
            "oof_mae": float(oof_corr), "holdout_mae": float(mae),
            "gain_vs_base": d, "gain_ci95": [lo, hi],
        }
        print(f"  {name:16s} gamma={g:.3f}  OOF {oof_corr:.4f}  "
              f"holdout {mae:.4f}  gain {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    (OUT / "stage1_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'stage1_results.json'}")
    return results


if __name__ == "__main__":
    main()
