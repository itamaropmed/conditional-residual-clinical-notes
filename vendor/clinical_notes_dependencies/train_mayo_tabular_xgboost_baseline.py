#!/usr/bin/env python3
"""Leakage-safe tabular XGBoost baseline for Mayo HRS duration prediction.

This script intentionally uses only tabular/scheduling/procedure metadata.
It does not read clinical notes.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "case_actual_duration_time"
PATIENT_ID_COLUMN = "patient_durable_id"
CASE_ID_COLUMN = "case_durable_id"
DATE_COLUMN = "scheduled_in_room"

LEAKAGE_COLUMNS = {
    CASE_ID_COLUMN,
    PATIENT_ID_COLUMN,
    "patient_birth_date",
    TARGET_COLUMN,
    "scheduled_setup_start",
    "scheduled_in_room",
    "scheduled_out_of_room",
    "scheduled_cleanup_complete",
    "case_actual_patient_in_room",
    "case_actual_prep_time",
    "case_actual_procedure_time",
    "case_actual_wrapup_time",
    "all_procs_not_performed",
}


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cutoff: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/mayo/mayo_hrs_tabular_features_and_durations.parquet",
        help="Path to the Mayo tabular parquet.",
    )
    parser.add_argument("--output-dir", default="output/mayo_tabular_xgb_baseline")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test cap. Uses earliest rows after sorting by date.",
    )
    parser.add_argument(
        "--include-scheduled-duration",
        action="store_true",
        default=True,
        help="Include duration implied by scheduled in/out timestamps. This is usually pre-op available.",
    )
    parser.add_argument(
        "--drop-scheduled-duration",
        action="store_false",
        dest="include_scheduled_duration",
        help="Do not include scheduled-duration derived features.",
    )
    return parser.parse_args()


def require_xgboost() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit(
            "xgboost is not installed in this environment.\n"
            "Install it with:\n"
            "  pip install xgboost\n"
            "Then rerun this script."
        ) from exc
    return XGBRegressor


def read_table(path: Path, target: str, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Required date column {DATE_COLUMN!r} not found.")
    if PATIENT_ID_COLUMN not in df.columns:
        raise ValueError(f"Required patient ID column {PATIENT_ID_COLUMN!r} not found.")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df[df[DATE_COLUMN].notna()]
    df = df[df[target].notna()]
    df = df[df[target].astype(float) > 0]
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df


def add_safe_features(df: pd.DataFrame, include_scheduled_duration: bool) -> pd.DataFrame:
    out = df.copy()
    scheduled_in = pd.to_datetime(out.get("scheduled_in_room"), errors="coerce")
    out["scheduled_year"] = scheduled_in.dt.year
    out["scheduled_month"] = scheduled_in.dt.month
    out["scheduled_dayofweek"] = scheduled_in.dt.dayofweek
    out["scheduled_hour"] = scheduled_in.dt.hour
    out["scheduled_is_weekend"] = scheduled_in.dt.dayofweek.isin([5, 6]).astype("int64")

    if include_scheduled_duration:
        scheduled_out = pd.to_datetime(out.get("scheduled_out_of_room"), errors="coerce")
        setup_start = pd.to_datetime(out.get("scheduled_setup_start"), errors="coerce")
        cleanup_complete = pd.to_datetime(out.get("scheduled_cleanup_complete"), errors="coerce")
        out["scheduled_room_minutes"] = (scheduled_out - scheduled_in).dt.total_seconds() / 60.0
        out["scheduled_setup_lead_minutes"] = (scheduled_in - setup_start).dt.total_seconds() / 60.0
        out["scheduled_cleanup_minutes"] = (cleanup_complete - scheduled_out).dt.total_seconds() / 60.0

    return out


def temporal_group_holdout(df: pd.DataFrame, test_fraction: float) -> SplitResult:
    if not 0.05 <= test_fraction <= 0.45:
        raise ValueError("--test-fraction should be between 0.05 and 0.45")

    cutoff = df[DATE_COLUMN].quantile(1.0 - test_fraction)
    candidate_test = df[df[DATE_COLUMN] >= cutoff].copy()
    test_patients = set(candidate_test[PATIENT_ID_COLUMN].astype(str))

    test_df = df[df[PATIENT_ID_COLUMN].astype(str).isin(test_patients)].copy()
    test_df = test_df[test_df[DATE_COLUMN] >= cutoff].copy()
    train_df = df[
        (df[DATE_COLUMN] < cutoff) & (~df[PATIENT_ID_COLUMN].astype(str).isin(test_patients))
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal group split produced an empty train or test set.")
    return SplitResult(train_df=train_df.reset_index(drop=True), test_df=test_df.reset_index(drop=True), cutoff=cutoff)


def make_feature_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    drop_cols = set(LEAKAGE_COLUMNS)
    drop_cols.add(target)
    drop_cols.update(
        c
        for c in df.columns
        if c.startswith("case_actual_") or c.endswith("_date")
    )

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target].astype(float)

    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("int64")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    try:
        numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        numeric_imputer = SimpleImputer(strategy="median")

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_imputer, numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Return a simple markdown table without requiring pandas[tabulate]."""
    if df.empty:
        return "_No rows._"
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        else:
            display_df[col] = display_df[col].astype(str)

    columns = [str(c) for c in display_df.columns]
    rows = display_df.astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def make_model(XGBRegressor: Any, params: dict[str, Any], seed: int) -> Any:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="mae",
        **params,
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAbsError": float(median_absolute_error(y_true, y_pred)),
        "P90AbsError": float(np.quantile(abs_err, 0.90)),
        "P95AbsError": float(np.quantile(abs_err, 0.95)),
        "W30": float(np.mean(abs_err <= 30.0)),
        "W60": float(np.mean(abs_err <= 60.0)),
        "MeanError_pred_minus_actual": float(np.mean(err)),
        "UnderestimationRate": float(np.mean(err < 0)),
        "OverestimationRate": float(np.mean(err > 0)),
        "N": int(len(y_true)),
    }


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


def run_cv_objective(
    trial: optuna.Trial,
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    target: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
    n_folds: int,
) -> float:
    X_all, y_all, _, _ = make_feature_frame(train_df, target)
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    actual_folds = min(n_folds, len(unique_groups))
    if actual_folds < 2:
        raise ValueError("Not enough unique patients for grouped CV.")

    splitter = GroupKFold(n_splits=actual_folds)
    fold_maes: list[float] = []
    params = suggest_params(trial)

    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        fit_groups = set(groups[fit_idx])
        val_groups = set(groups[val_idx])
        if fit_groups & val_groups:
            raise RuntimeError("Patient leakage detected inside CV fold.")

        preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        model = make_model(XGBRegressor, params, seed + fold_idx)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
        pred = pipe.predict(X_all.iloc[val_idx])
        fold_mae = mean_absolute_error(y_all.iloc[val_idx], pred)
        fold_maes.append(float(fold_mae))
        trial.report(float(np.mean(fold_maes)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_maes))


def evaluate_cv_best(
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    target: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
    n_folds: int,
) -> pd.DataFrame:
    X_all, y_all, _, _ = make_feature_frame(train_df, target)
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    rows: list[dict[str, float | int]] = []

    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        model = make_model(XGBRegressor, params, seed + 100 + fold_idx)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
        pred = pipe.predict(X_all.iloc[val_idx])
        row = {"fold": fold_idx}
        row.update(metric_dict(y_all.iloc[val_idx].to_numpy(), pred))
        row["train_patients"] = len(set(groups[fit_idx]))
        row["val_patients"] = len(set(groups[val_idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def procedure_slice_metrics(test_df: pd.DataFrame, predictions: np.ndarray, target: str) -> pd.DataFrame:
    frame = test_df[[target, "primary_procedure_name"]].copy()
    frame["prediction"] = predictions
    frame["abs_error"] = (frame["prediction"] - frame[target]).abs()
    rows = []
    for proc, group in frame.groupby("primary_procedure_name", dropna=False):
        if len(group) < 10:
            continue
        rows.append(
            {
                "primary_procedure_name": proc,
                "N": len(group),
                "MAE": float(group["abs_error"].mean()),
                "MedianAbsError": float(group["abs_error"].median()),
                "P90AbsError": float(group["abs_error"].quantile(0.90)),
                "W30": float((group["abs_error"] <= 30.0).mean()),
                "ActualMean": float(group[target].mean()),
                "PredMean": float(group["prediction"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "primary_procedure_name",
                "N",
                "MAE",
                "MedianAbsError",
                "P90AbsError",
                "W30",
                "ActualMean",
                "PredMean",
            ]
        )
    return pd.DataFrame(rows).sort_values(["N", "MAE"], ascending=[False, True])


def feature_importance(pipe: Pipeline, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocess"]
    names: list[str] = []
    names.extend(numeric_cols)
    if categorical_cols:
        try:
            cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
            names.extend([str(x) for x in cat_names])
        except Exception:
            names.extend(categorical_cols)

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame()
    n = min(len(names), len(importance))
    return (
        pd.DataFrame({"feature": names[:n], "importance": importance[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    XGBRegressor = require_xgboost()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(data_path, args.target, args.max_rows)
    df = add_safe_features(df, args.include_scheduled_duration)
    split = temporal_group_holdout(df, args.test_fraction)

    train_patients = set(split.train_df[PATIENT_ID_COLUMN].astype(str))
    test_patients = set(split.test_df[PATIENT_ID_COLUMN].astype(str))
    leakage_patients = train_patients & test_patients
    if leakage_patients:
        raise RuntimeError(f"Patient leakage detected between train and test: {len(leakage_patients)} patients")

    X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(split.train_df, args.target)
    X_test, y_test, _, _ = make_feature_frame(split.test_df, args.target)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: run_cv_objective(
            trial=trial,
            XGBRegressor=XGBRegressor,
            train_df=split.train_df,
            target=args.target,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=args.seed,
            n_folds=args.n_folds,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)
    cv_metrics = evaluate_cv_best(
        XGBRegressor,
        split.train_df,
        args.target,
        numeric_cols,
        categorical_cols,
        best_params,
        args.seed,
        args.n_folds,
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    model = make_model(XGBRegressor, best_params, args.seed)
    final_pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    final_pipe.fit(X_train, y_train)
    test_pred = final_pipe.predict(X_test)

    median_pred = np.full_like(y_test.to_numpy(dtype=float), float(y_train.median()), dtype=float)
    proc_medians = split.train_df.groupby("primary_procedure_name")[args.target].median().to_dict()
    global_median = float(y_train.median())
    proc_pred = split.test_df["primary_procedure_name"].map(proc_medians).fillna(global_median).to_numpy(dtype=float)

    test_metrics = metric_dict(y_test.to_numpy(), test_pred)
    median_metrics = metric_dict(y_test.to_numpy(), median_pred)
    procedure_median_metrics = metric_dict(y_test.to_numpy(), proc_pred)

    predictions = split.test_df[
        [CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, "primary_procedure_name", args.target]
    ].copy()
    predictions["prediction"] = test_pred
    predictions["error_pred_minus_actual"] = predictions["prediction"] - predictions[args.target]
    predictions["abs_error"] = predictions["error_pred_minus_actual"].abs()

    slice_metrics = procedure_slice_metrics(split.test_df, test_pred, args.target)
    importances = feature_importance(final_pipe, numeric_cols, categorical_cols)

    study_df = study.trials_dataframe()
    summary = {
        "data_path": str(data_path),
        "target": args.target,
        "rows_after_filtering": int(len(df)),
        "train_rows": int(len(split.train_df)),
        "test_rows": int(len(split.test_df)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "patient_overlap_train_test": int(len(leakage_patients)),
        "temporal_cutoff": str(split.cutoff),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "best_cv_mae": float(study.best_value),
        "best_params": best_params,
        "test_metrics_xgboost": test_metrics,
        "test_metrics_global_median": median_metrics,
        "test_metrics_procedure_median": procedure_median_metrics,
    }

    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    cv_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    slice_metrics.to_csv(output_dir / "procedure_slice_metrics.csv", index=False)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)
    study_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "xgboost_tabular_pipeline.pkl").open("wb") as handle:
        pickle.dump(final_pipe, handle)

    report = [
        "# Mayo HRS Tabular XGBoost Baseline",
        "",
        "This run uses structured/tabular fields only. Clinical notes are not read.",
        "",
        "## Leakage Controls",
        "",
        f"- Target: `{args.target}`",
        "- Excluded all `case_actual_*` columns from features.",
        f"- Temporal cutoff: `{split.cutoff}`",
        "- Train/test split is patient-disjoint.",
        f"- Train/test patient overlap: `{len(leakage_patients)}`",
        "",
        "## Test Metrics",
        "",
        markdown_table(pd.DataFrame(
            [
                {"model": "XGBoost_tabular", **test_metrics},
                {"model": "Procedure_median_baseline", **procedure_median_metrics},
                {"model": "Global_median_baseline", **median_metrics},
            ]
        )),
        "",
        "## Cross-Validation Metrics",
        "",
        markdown_table(cv_metrics),
        "",
        "## Best Optuna Parameters",
        "",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
        "",
        "## Top Feature Importances",
        "",
        markdown_table(importances.head(30)) if not importances.empty else "No feature importance available.",
        "",
        "## Saved Files",
        "",
        "- `test_predictions.csv`",
        "- `cv_fold_metrics.csv`",
        "- `procedure_slice_metrics.csv`",
        "- `feature_importance.csv`",
        "- `optuna_trials.csv`",
        "- `metrics_summary.json`",
        "- `xgboost_tabular_pipeline.pkl`",
    ]
    (output_dir / "tabular_baseline_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {output_dir}")
    print(f"Open report: {output_dir / 'tabular_baseline_report.md'}")


if __name__ == "__main__":
    main()
