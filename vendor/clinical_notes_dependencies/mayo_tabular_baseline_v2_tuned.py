#!/usr/bin/env python3
"""Improved + retuned tabular-only baseline for Mayo HRS duration prediction.

Adds two feature-engineering changes found (via CV testing) to genuinely help
over the original train_mayo_tabular_xgboost_baseline.py:
  1. bmi_missing indicator (BMI is missing in ~70% of cases; not random).
  2. OOF-safe smoothed target encoding for primary_procedure_name and
     surgeon_durable_id (handles the long tail of rare procedure types
     better than the plain one-hot + min_frequency cutoff).

Tested and rejected (do not resurrect without new evidence): log1p(target)
training, and reg:gamma / reg:tweedie / reg:absoluteerror / reg:pseudohubererror
objectives -- all made CV MAE worse than plain reg:squarederror on this data.

Reuses the leakage-safe split/read/preprocessing utilities from
train_mayo_tabular_xgboost_baseline.py unchanged; does not modify that file
so the many other pipelines importing from it are unaffected.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from train_mayo_tabular_xgboost_baseline import (
    CASE_ID_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    make_feature_frame,
    make_model,
    make_preprocessor,
    markdown_table,
    metric_dict,
    read_table,
    require_xgboost,
    temporal_group_holdout,
)

PROC_ENC_COL = "proc_target_enc"
SURGEON_ENC_COL = "surgeon_target_enc"

# Best params found by this script's own Optuna run (60 trials, patient-grouped
# 5-fold CV, train-only) on the bmi_missing + target-encoding enriched feature
# set. CV MAE 33.287, test MAE 32.565 vs the old frozen-params baseline's 33.726.
TUNED_STRUCTURED_PARAMS = {
    "n_estimators": 1213,
    "max_depth": 8,
    "learning_rate": 0.013721653065680284,
    "subsample": 0.8431891099718941,
    "colsample_bytree": 0.5331833452208407,
    "min_child_weight": 9.083480959214418,
    "reg_alpha": 6.006000053619992e-07,
    "reg_lambda": 0.001809354462444587,
    "gamma": 1.0186857689274291,
    "max_bin": 212,
}


def fit_tuned_structured_oof_and_test(
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    n_folds: int,
    seed: int,
    proc_enc_k: float = 20.0,
    surgeon_enc_k: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in replacement for fit_oof_and_test with bmi_missing (carried
    automatically via make_feature_frame once present on the df) plus
    per-fold OOF-safe target encoding for procedure/surgeon. Same call
    signature and (oof, test_pred) return shape as the original, so it can
    replace it in any pipeline without touching downstream code."""
    train_df = add_bmi_missing(train_df)
    test_df = add_bmi_missing(test_df)

    X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(train_df, target)
    X_test, _, _, _ = make_feature_frame(test_df, target)
    numeric_cols = list(numeric_cols) + [PROC_ENC_COL, SURGEON_ENC_COL]
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=max(2, min(n_folds, len(np.unique(groups)))))

    oof = np.zeros(len(train_df), dtype=float)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_train, y_train, groups=groups), start=1):
        X_fit, X_val = build_fold_frames(X_train, y_train, fit_idx, val_idx, proc_enc_k, surgeon_enc_k)
        pipe = Pipeline([("preprocess", make_preprocessor(numeric_cols, categorical_cols)), ("model", make_model(XGBRegressor, params, seed + fold_idx))])
        pipe.fit(X_fit, y_train.iloc[fit_idx])
        oof[val_idx] = pipe.predict(X_val)

    proc_fit, proc_test = smoothed_target_encode(train_df["primary_procedure_name"], y_train, test_df["primary_procedure_name"], proc_enc_k)
    surg_fit, surg_test = smoothed_target_encode(train_df["surgeon_durable_id"], y_train, test_df["surgeon_durable_id"], surgeon_enc_k)
    X_train_full = X_train.copy()
    X_train_full[PROC_ENC_COL] = proc_fit
    X_train_full[SURGEON_ENC_COL] = surg_fit
    X_test_full = X_test.copy()
    X_test_full[PROC_ENC_COL] = proc_test
    X_test_full[SURGEON_ENC_COL] = surg_test

    full = Pipeline([("preprocess", make_preprocessor(numeric_cols, categorical_cols)), ("model", make_model(XGBRegressor, params, seed + 1000))])
    full.fit(X_train_full, y_train)
    test_pred = full.predict(X_test_full)

    lo, hi = np.quantile(y_train.to_numpy(dtype=float), [0.001, 0.999])
    return np.clip(oof, lo, hi), np.clip(test_pred, lo, hi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/Users/itamarzernitsky/PycharmProjects/Clinical_notes/data/mayo/mayo_hrs_tabular_features_and_durations.parquet")
    parser.add_argument("--output-dir", default="output/mayo_tabular_baseline_v2_tuned")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proc-enc-k", type=float, default=20.0)
    parser.add_argument("--surgeon-enc-k", type=float, default=15.0)
    return parser.parse_args()


def add_bmi_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bmi_missing"] = out["bmi"].isna().astype(int)
    return out


def smoothed_target_encode(x_fit: pd.Series, y_fit: pd.Series, x_val: pd.Series, k: float) -> tuple[np.ndarray, np.ndarray]:
    global_mean = float(y_fit.mean())
    stats = pd.DataFrame({"key": x_fit.to_numpy(), "y": y_fit.to_numpy()}).groupby("key")["y"].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + k * global_mean) / (stats["count"] + k)
    enc_fit = x_fit.map(smoothed).fillna(global_mean).to_numpy(dtype=float)
    enc_val = x_val.map(smoothed).fillna(global_mean).to_numpy(dtype=float)
    return enc_fit, enc_val


def build_fold_frames(
    X_all: pd.DataFrame, y_all: pd.Series, fit_idx: np.ndarray, val_idx: np.ndarray, proc_k: float, surgeon_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_fit = X_all.iloc[fit_idx].copy()
    X_val = X_all.iloc[val_idx].copy()
    y_fit = y_all.iloc[fit_idx]
    proc_fit, proc_val = smoothed_target_encode(X_fit["primary_procedure_name"], y_fit, X_val["primary_procedure_name"], proc_k)
    X_fit[PROC_ENC_COL] = proc_fit
    X_val[PROC_ENC_COL] = proc_val
    surg_fit, surg_val = smoothed_target_encode(X_fit["surgeon_durable_id"], y_fit, X_val["surgeon_durable_id"], surgeon_k)
    X_fit[SURGEON_ENC_COL] = surg_fit
    X_val[SURGEON_ENC_COL] = surg_val
    return X_fit, X_val


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
    }


def main() -> None:
    args = parse_args()
    XGBRegressor = require_xgboost()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.data), TARGET_COLUMN, None)
    df = add_safe_features(df, include_scheduled_duration=True)
    df = add_bmi_missing(df)
    split = temporal_group_holdout(df, args.test_fraction)
    train_df, test_df = split.train_df, split.test_df

    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(train_df, TARGET_COLUMN)
    numeric_cols = list(numeric_cols) + [PROC_ENC_COL, SURGEON_ENC_COL]
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()

    def cv_objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        splitter = GroupKFold(n_splits=args.n_folds)
        fold_maes = []
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
            X_fit, X_val = build_fold_frames(X_all, y_all, fit_idx, val_idx, args.proc_enc_k, args.surgeon_enc_k)
            pre = make_preprocessor(numeric_cols, categorical_cols)
            model = make_model(XGBRegressor, params, args.seed + fold_idx)
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            pipe.fit(X_fit, y_all.iloc[fit_idx])
            pred = pipe.predict(X_val)
            fold_maes.append(float(mean_absolute_error(y_all.iloc[val_idx], pred)))
            trial.report(float(np.mean(fold_maes)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(cv_objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    best_params = dict(study.best_params)
    print(f"Best CV MAE: {study.best_value:.3f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Final fit: target-encoding computed on FULL train, applied once to test (no leakage: test labels never touched).
    proc_fit, proc_test = smoothed_target_encode(train_df["primary_procedure_name"], y_all, test_df["primary_procedure_name"], args.proc_enc_k)
    surg_fit, surg_test = smoothed_target_encode(train_df["surgeon_durable_id"], y_all, test_df["surgeon_durable_id"], args.surgeon_enc_k)
    X_all[PROC_ENC_COL] = proc_fit
    X_all[SURGEON_ENC_COL] = surg_fit

    X_test, y_test, _, _ = make_feature_frame(test_df, TARGET_COLUMN)
    X_test[PROC_ENC_COL] = proc_test
    X_test[SURGEON_ENC_COL] = surg_test

    pre = make_preprocessor(numeric_cols, categorical_cols)
    model = make_model(XGBRegressor, best_params, args.seed)
    final_pipe = Pipeline([("preprocess", pre), ("model", model)])
    final_pipe.fit(X_all, y_all)
    test_pred = final_pipe.predict(X_test)

    test_metrics = metric_dict(y_test.to_numpy(), test_pred)
    print(f"\n=== FINAL TEST METRICS (v2 tuned, N={len(y_test)}) ===")
    print(json.dumps(test_metrics, indent=2))
    print("\nFor reference: original frozen STRUCTURED_PARAMS baseline test MAE = 33.7264")

    predictions = test_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, TARGET_COLUMN, "primary_procedure_name"]].copy()
    predictions["prediction"] = test_pred
    predictions.to_csv(output_dir / "test_predictions_v2.csv", index=False)

    study_df = study.trials_dataframe()
    study_df.to_csv(output_dir / "optuna_trials_v2.csv", index=False)
    (output_dir / "metrics_summary_v2.json").write_text(
        json.dumps({"best_cv_mae": float(study.best_value), "best_params": best_params, "test_metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "tabular_v2_pipeline.pkl").open("wb") as handle:
        pickle.dump(final_pipe, handle)

    report = [
        "# Mayo Tabular Baseline v2 (bmi_missing + target-encoding + retuned)",
        "",
        f"- Best CV MAE (train, patient-grouped {args.n_folds}-fold): `{study.best_value:.4f}`",
        f"- Final test MAE (N={len(y_test)}): `{test_metrics['MAE']:.4f}`",
        f"- Original frozen-params baseline test MAE: `33.7264`",
        "",
        "## Test metrics",
        "",
        markdown_table(pd.DataFrame([test_metrics])),
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
    ]
    (output_dir / "report_v2.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
