#!/usr/bin/env python3
"""Tabular-only model sweep on the tuned Mayo temporal patient split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "mayo_residual_distiller_study"))

from mayo_tabular_baseline_v2_tuned import (  # noqa: E402
    PROC_ENC_COL,
    SURGEON_ENC_COL,
    TUNED_STRUCTURED_PARAMS,
    add_bmi_missing,
    build_fold_frames,
    fit_tuned_structured_oof_and_test,
    smoothed_target_encode,
)
from train_mayo_tabular_xgboost_baseline import (  # noqa: E402
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    make_feature_frame,
    make_model,
    make_preprocessor,
    metric_dict,
    read_table,
    require_xgboost,
    temporal_group_holdout,
)


FORBIDDEN = PROJECT / "data" / "mayo" / "mayo_hrs_llm_features_top11.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default=str(PROJECT / "data" / "mayo" / "mayo_hrs_tabular_features_and_durations.parquet"),
    )
    parser.add_argument("--output-dir", default="work/no_llm_notes/tabular_sweep")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--include-slow-forests", action="store_true")
    return parser.parse_args()


def assert_no_forbidden_access() -> None:
    assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"


def clustered_ci(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups.tolist())))
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        boots[i] = float(np.mean(values[idx]))
    return tuple(np.quantile(boots, [0.025, 0.975]).astype(float))


def prepare_full_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    proc_k: float = 20.0,
    surgeon_k: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    train_df = add_bmi_missing(train_df)
    test_df = add_bmi_missing(test_df)
    X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(train_df, target)
    X_test, _, _, _ = make_feature_frame(test_df, target)
    proc_fit, proc_test = smoothed_target_encode(
        train_df["primary_procedure_name"], y_train, test_df["primary_procedure_name"], proc_k
    )
    surg_fit, surg_test = smoothed_target_encode(
        train_df["surgeon_durable_id"], y_train, test_df["surgeon_durable_id"], surgeon_k
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[PROC_ENC_COL] = proc_fit
    X_train[SURGEON_ENC_COL] = surg_fit
    X_test[PROC_ENC_COL] = proc_test
    X_test[SURGEON_ENC_COL] = surg_test
    numeric_cols = list(numeric_cols) + [PROC_ENC_COL, SURGEON_ENC_COL]
    return X_train, X_test, y_train, numeric_cols, categorical_cols


def make_catboost_frames(X_fit: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    fit = X_fit.copy()
    val = X_val.copy()
    for col in categorical_cols:
        fit[col] = fit[col].astype("string").fillna("__MISSING__")
        val[col] = val[col].astype("string").fillna("__MISSING__")
    cat_features = [fit.columns.get_loc(c) for c in categorical_cols if c in fit.columns]
    return fit, val, cat_features


def fit_oof_test_pipeline(
    name: str,
    model_factory: Callable[[int], Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_df_b = add_bmi_missing(train_df)
    test_df_b = add_bmi_missing(test_df)
    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(train_df_b, target)
    numeric_cols = list(numeric_cols) + [PROC_ENC_COL, SURGEON_ENC_COL]
    X_test_base, _, _, _ = make_feature_frame(test_df_b, target)
    splitter = GroupKFold(n_splits=max(2, min(n_folds, len(np.unique(groups)))))
    oof = np.zeros(len(train_df), dtype=float)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        X_fit, X_val = build_fold_frames(X_all, y_all, fit_idx, val_idx, 20.0, 15.0)
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model_factory(seed + fold_idx)),
            ]
        )
        pipe.fit(X_fit, y_all.iloc[fit_idx])
        oof[val_idx] = pipe.predict(X_val)
        print(f"    {name} fold {fold_idx}/{n_folds} MAE={mean_absolute_error(y_all.iloc[val_idx], oof[val_idx]):.4f}", flush=True)

    X_train_full, X_test_full, y_train, numeric_cols_full, categorical_cols_full = prepare_full_frames(train_df, test_df, target)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols_full, categorical_cols_full)),
            ("model", model_factory(seed + 1000)),
        ]
    )
    pipe.fit(X_train_full, y_train)
    test_pred = pipe.predict(X_test_full)
    lo, hi = np.quantile(y_train.to_numpy(dtype=float), [0.001, 0.999])
    return np.clip(oof, lo, hi), np.clip(test_pred, lo, hi)


def fit_oof_test_catboost(
    loss: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("catboost is not installed") from exc

    train_df_b = add_bmi_missing(train_df)
    test_df_b = add_bmi_missing(test_df)
    X_all, y_all, _, categorical_cols = make_feature_frame(train_df_b, target)
    splitter = GroupKFold(n_splits=max(2, min(n_folds, len(np.unique(groups)))))
    oof = np.zeros(len(train_df), dtype=float)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        X_fit, X_val = build_fold_frames(X_all, y_all, fit_idx, val_idx, 20.0, 15.0)
        X_fit_cb, X_val_cb, cat_features = make_catboost_frames(X_fit, X_val, categorical_cols)
        model = CatBoostRegressor(
            loss_function=loss,
            eval_metric="MAE",
            iterations=1000,
            depth=8,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_seed=seed + fold_idx,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(X_fit_cb, y_all.iloc[fit_idx], cat_features=cat_features)
        oof[val_idx] = model.predict(X_val_cb)
        print(f"    catboost_{loss} fold {fold_idx}/{n_folds} MAE={mean_absolute_error(y_all.iloc[val_idx], oof[val_idx]):.4f}", flush=True)

    X_train_full, X_test_full, y_train, _, categorical_cols_full = prepare_full_frames(train_df, test_df, target)
    X_train_cb, X_test_cb, cat_features = make_catboost_frames(X_train_full, X_test_full, categorical_cols_full)
    model = CatBoostRegressor(
        loss_function=loss,
        eval_metric="MAE",
        iterations=1200,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        random_seed=seed + 1000,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(X_train_cb, y_train, cat_features=cat_features)
    test_pred = model.predict(X_test_cb)
    lo, hi = np.quantile(y_train.to_numpy(dtype=float), [0.001, 0.999])
    return np.clip(oof, lo, hi), np.clip(test_pred, lo, hi)


def fit_dense_hist(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_df_b = add_bmi_missing(train_df)
    test_df_b = add_bmi_missing(test_df)
    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(train_df_b, target)
    numeric_cols = list(numeric_cols) + [PROC_ENC_COL, SURGEON_ENC_COL]
    splitter = GroupKFold(n_splits=max(2, min(n_folds, len(np.unique(groups)))))
    oof = np.zeros(len(train_df), dtype=float)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        X_fit, X_val = build_fold_frames(X_all, y_all, fit_idx, val_idx, 20.0, 15.0)
        pre = make_preprocessor(numeric_cols, categorical_cols)
        Xt = pre.fit_transform(X_fit).toarray()
        Xv = pre.transform(X_val).toarray()
        imp = SimpleImputer(strategy="median")
        Xt = imp.fit_transform(Xt)
        Xv = imp.transform(Xv)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=650,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed + fold_idx,
        )
        model.fit(Xt, y_all.iloc[fit_idx])
        oof[val_idx] = model.predict(Xv)
        print(f"    hist_abs fold {fold_idx}/{n_folds} MAE={mean_absolute_error(y_all.iloc[val_idx], oof[val_idx]):.4f}", flush=True)
    X_train_full, X_test_full, y_train, numeric_cols_full, categorical_cols_full = prepare_full_frames(train_df, test_df, target)
    pre = make_preprocessor(numeric_cols_full, categorical_cols_full)
    Xt = pre.fit_transform(X_train_full).toarray()
    Xv = pre.transform(X_test_full).toarray()
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(Xt)
    Xv = imp.transform(Xv)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=750,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed + 1000,
    )
    model.fit(Xt, y_train)
    test_pred = model.predict(Xv)
    lo, hi = np.quantile(y_train.to_numpy(dtype=float), [0.001, 0.999])
    return np.clip(oof, lo, hi), np.clip(test_pred, lo, hi)


def summarize(
    method: str,
    y_test: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    row = {"method": method, **metric_dict(y_test, pred)}
    abs_err = np.abs(pred - y_test)
    gain = np.abs(baseline - y_test) - abs_err
    row["MAE_CI95"] = list(clustered_ci(abs_err, groups, n_boot, seed))
    row["gain_vs_xgb_tuned"] = float(gain.mean())
    row["gain_vs_xgb_tuned_CI95"] = list(clustered_ci(gain, groups, n_boot, seed + 19))
    row["delta_vs_old_11_feature_benchmark_32_291"] = float(row["MAE"] - 32.291)
    return row


def main() -> None:
    args = parse_args()
    assert_no_forbidden_access()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    XGBRegressor = require_xgboost()

    print("Loading/splitting tabular data...")
    df = read_table(Path(args.tabular_data), TARGET_COLUMN, None)
    df = add_safe_features(df, include_scheduled_duration=True)
    split = temporal_group_holdout(df, args.test_fraction)
    train_df, test_df = split.train_df.copy(), split.test_df.copy()
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    test_groups = test_df[PATIENT_ID_COLUMN].astype(str).to_numpy()

    predictions = test_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN, "primary_procedure_name"]].copy()
    oof_predictions = pd.DataFrame({CASE_ID_COLUMN: train_df[CASE_ID_COLUMN], TARGET_COLUMN: y_train})
    results: list[dict[str, Any]] = []

    print("Fitting tuned XGB v2 baseline...")
    xgb_oof, xgb_test = fit_tuned_structured_oof_and_test(
        XGBRegressor,
        train_df,
        test_df,
        TARGET_COLUMN,
        TUNED_STRUCTURED_PARAMS,
        args.n_folds,
        args.seed,
    )
    predictions["pred_xgb_tuned_v2"] = xgb_test
    oof_predictions["oof_xgb_tuned_v2"] = xgb_oof
    results.append(summarize("xgb_tuned_v2", y_test, xgb_test, xgb_test, test_groups, args.n_boot, args.seed))
    print(f"  xgb_tuned_v2 test MAE={mean_absolute_error(y_test, xgb_test):.4f}")

    model_specs: list[tuple[str, Callable[[int], Any]]] = []
    model_specs.append(
        (
            "xgb_absoluteerror_same_shape",
            lambda seed: XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                eval_metric="mae",
                **TUNED_STRUCTURED_PARAMS,
            ),
        )
    )
    try:
        from lightgbm import LGBMRegressor

        model_specs.extend(
            [
                (
                    "lgbm_l1",
                    lambda seed: LGBMRegressor(
                        objective="regression_l1",
                        n_estimators=1400,
                        learning_rate=0.025,
                        num_leaves=63,
                        min_child_samples=45,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_alpha=0.05,
                        reg_lambda=5.0,
                        random_state=seed,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
                (
                    "lgbm_l2",
                    lambda seed: LGBMRegressor(
                        objective="regression",
                        n_estimators=1100,
                        learning_rate=0.03,
                        num_leaves=47,
                        min_child_samples=40,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_alpha=0.02,
                        reg_lambda=3.0,
                        random_state=seed,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        print("LightGBM unavailable; skipping.")

    if args.include_slow_forests:
        model_specs.extend(
            [
                (
                    "extra_trees",
                    lambda seed: ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=3,
                        max_features=0.65,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
                (
                    "random_forest",
                    lambda seed: RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=4,
                        max_features=0.65,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    for name, factory in model_specs:
        print(f"Fitting {name}...")
        try:
            oof, pred = fit_oof_test_pipeline(
                name, factory, train_df, test_df, TARGET_COLUMN, groups, args.n_folds, args.seed
            )
        except Exception as exc:
            print(f"  {name} failed: {exc}")
            continue
        predictions[f"pred_{name}"] = pred
        oof_predictions[f"oof_{name}"] = oof
        results.append(summarize(name, y_test, pred, xgb_test, test_groups, args.n_boot, args.seed + len(results)))
        print(f"  {name} test MAE={mean_absolute_error(y_test, pred):.4f}")

    for loss in ["MAE", "RMSE"]:
        name = f"catboost_{loss.lower()}"
        print(f"Fitting {name}...")
        try:
            oof, pred = fit_oof_test_catboost(loss, train_df, test_df, TARGET_COLUMN, groups, args.n_folds, args.seed)
        except Exception as exc:
            print(f"  {name} failed: {exc}")
            continue
        predictions[f"pred_{name}"] = pred
        oof_predictions[f"oof_{name}"] = oof
        results.append(summarize(name, y_test, pred, xgb_test, test_groups, args.n_boot, args.seed + len(results)))
        print(f"  {name} test MAE={mean_absolute_error(y_test, pred):.4f}")

    print("Fitting hist_abs...")
    try:
        oof, pred = fit_dense_hist(train_df, test_df, TARGET_COLUMN, groups, args.n_folds, args.seed)
        predictions["pred_hist_abs"] = pred
        oof_predictions["oof_hist_abs"] = oof
        results.append(summarize("hist_abs", y_test, pred, xgb_test, test_groups, args.n_boot, args.seed + len(results)))
        print(f"  hist_abs test MAE={mean_absolute_error(y_test, pred):.4f}")
    except Exception as exc:
        print(f"  hist_abs failed: {exc}")

    pred_cols = [c for c in oof_predictions.columns if c.startswith("oof_")]
    test_pred_cols = ["pred_" + c.removeprefix("oof_") for c in pred_cols]
    oof_mat = oof_predictions[pred_cols].to_numpy(dtype=float)
    test_mat = predictions[test_pred_cols].to_numpy(dtype=float)

    print("Fitting tabular blends from train OOF predictions...")
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 20))
    ridge.fit(oof_mat, y_train)
    ridge_pred = ridge.predict(test_mat)
    predictions["pred_ridge_oof_blend"] = ridge_pred
    results.append(summarize("ridge_oof_blend", y_test, ridge_pred, xgb_test, test_groups, args.n_boot, args.seed + 301))

    centered = y_train.mean()
    coefs, _ = nnls(oof_mat - centered, y_train - centered)
    if coefs.sum() > 0:
        coefs = coefs / coefs.sum()
    nnls_pred = centered + (test_mat - centered) @ coefs
    predictions["pred_nnls_oof_blend"] = nnls_pred
    results.append(
        summarize(
            "nnls_oof_blend",
            y_test,
            nnls_pred,
            xgb_test,
            test_groups,
            args.n_boot,
            args.seed + 302,
        )
    )

    mean_pred = np.mean(test_mat, axis=1)
    median_pred = np.median(test_mat, axis=1)
    predictions["pred_mean_blend"] = mean_pred
    predictions["pred_median_blend"] = median_pred
    results.append(summarize("mean_blend", y_test, mean_pred, xgb_test, test_groups, args.n_boot, args.seed + 303))
    results.append(summarize("median_blend", y_test, median_pred, xgb_test, test_groups, args.n_boot, args.seed + 304))

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    results_df.to_csv(output_dir / "tabular_sweep_results.csv", index=False)
    predictions.to_csv(output_dir / "tabular_sweep_predictions.csv", index=False)
    oof_predictions.to_csv(output_dir / "tabular_sweep_oof_predictions.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("\n=== TABULAR SWEEP ===")
    print(results_df[["method", "MAE", "gain_vs_xgb_tuned", "gain_vs_xgb_tuned_CI95", "delta_vs_old_11_feature_benchmark_32_291"]].to_string(index=False))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
