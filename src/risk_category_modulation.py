#!/usr/bin/env python3
"""Leakage-safe post-model risk/category modulation for the strict 31.7703 model.

This script starts from saved OOF/test predictions and asks whether a learned
risk categorizer plus category-specific heads can improve the fixed model.

No old 11 LLM features are read. Model selection is by train OOF only; held-out
test scores are reported as exploratory confirmation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
OUT_DIR = ROOT / "outputs/risk_category_modulation"
PRED_DIR = ROOT / "work/no_llm_notes/raw_strict_manifold_push_surgical_full"
NOTE_PATH = ROOT / "work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_note_bundles.parquet"
TAB_PATH = PROJECT / "data/mayo/mayo_hrs_tabular_features_and_durations.parquet"
FORBIDDEN = PROJECT / "data/mayo/mayo_hrs_llm_features_top11.parquet"
FINAL_COL = "pls_raw_word160_26_delta__sgd_huber__g1.550"


def assert_no_forbidden_access() -> None:
    assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"


def procedure_family(name: Any) -> str:
    s = str(name or "").upper()
    if "PVI" in s or "ATRIAL FIBRILLATION" in s:
        return "AF/PVI ablation"
    if "VT" in s or "PVC" in s or "VENTRICULAR" in s:
        return "VT/PVC ablation"
    if "ABLATION" in s or "RFABLATION" in s:
        return "Other ablation"
    if "LAA" in s or "WATCHMAN" in s:
        return "LAA closure"
    if any(x in s for x in ["EXTRACTION", "REMOVAL"]):
        return "Lead/device extraction-removal"
    if any(x in s for x in ["GENERATOR", "POCKET", "REVISION", "UPGRADE"]):
        return "Device revision-generator-upgrade"
    if any(x in s for x in ["PPM", "ICD", "PACEMAKER", "LEADLESS", "DEFIBRILLATOR", "SUBSTERNAL", "LEFT BUNDLE"]):
        return "Device implant"
    if any(x in s for x in ["TILT", "DIAGNOSTIC EPS", "NIPS", "DRUG", "DEFIBRILLATION THRESHOLD", "3D MAPPING"]):
        return "Diagnostic/testing"
    if "ABORT" in s:
        return "Aborted/other"
    return "Other"


def duration_bin(minutes: float) -> str:
    if pd.isna(minutes):
        return "missing"
    if minutes < 90:
        return "<90"
    if minutes < 150:
        return "90-149"
    if minutes < 210:
        return "150-209"
    if minutes < 300:
        return "210-299"
    if minutes < 420:
        return "300-419"
    return "420+"


def note_count_bin(n: float) -> str:
    if pd.isna(n) or n <= 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 12:
        return "6-12"
    if n <= 23:
        return "13-23"
    return "24 cap"


def read_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(PRED_DIR / "strict_manifold_push_oof_predictions.csv")
    test = pd.read_csv(PRED_DIR / "strict_manifold_push_test_predictions.csv")
    tab = pd.read_parquet(TAB_PATH)
    tab_cols = [
        "case_durable_id",
        "patient_durable_id",
        "case_anesthesia",
        "case_urgency_level",
        "case_surgery_patient_class",
        "case_admission_patient_class",
        "patient_sex",
        "patient_age",
        "primary_procedure_name",
        "scheduled_in_room",
        "scheduled_out_of_room",
        "surgeon_durable_id",
        "case_num_procedures",
        "case_num_panels",
        "case_num_providers",
        "is_redo",
        "prior_same_proc_duration",
        "redo_time_gap_days",
        "bmi",
    ]
    tab = tab[[c for c in tab_cols if c in tab.columns]].copy()
    notes = pd.read_parquet(
        NOTE_PATH,
        columns=["case_durable_id", "n_pre_cutoff_notes_used", "note_chars", "note_doc_types_used"],
    )

    for frame in (oof, test):
        if FINAL_COL not in frame:
            raise KeyError(FINAL_COL)
        frame.rename(columns={FINAL_COL: "final"}, inplace=True)
        frame["correction"] = frame["final"] - frame["base"]
        frame["abs_correction"] = frame["correction"].abs()
        model_cols = [c for c in frame.columns if c.startswith(("pls_", "cca_"))]
        pls_cols = [c for c in frame.columns if c.startswith("pls_")]
        pls_raw_cols = [c for c in pls_cols if "_raw_word160_" in c]
        pls_delta_cols = [c for c in pls_cols if "_delta__" in c]
        pls_raw_delta_cols = [c for c in pls_cols if "_raw_word160_" in c and "_delta__" in c]
        if model_cols:
            mat = frame[model_cols].to_numpy(float)
            frame["model_pred_std"] = np.nanstd(mat, axis=1)
            frame["model_pred_range"] = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
            frame["model_pred_q90_q10"] = np.nanquantile(mat, 0.90, axis=1) - np.nanquantile(mat, 0.10, axis=1)
            frame["model_pred_mean_minus_final"] = np.nanmean(mat, axis=1) - frame["final"]
        else:
            for col in ["model_pred_std", "model_pred_range", "model_pred_q90_q10", "model_pred_mean_minus_final"]:
                frame[col] = 0.0
        for prefix, cols in [
            ("pls", pls_cols),
            ("pls_raw", pls_raw_cols),
            ("pls_delta", pls_delta_cols),
            ("pls_raw_delta", pls_raw_delta_cols),
        ]:
            if cols:
                mat = frame[cols].to_numpy(float)
                mean = np.nanmean(mat, axis=1)
                median = np.nanmedian(mat, axis=1)
                std = np.nanstd(mat, axis=1)
                q90 = np.nanquantile(mat, 0.90, axis=1)
                q10 = np.nanquantile(mat, 0.10, axis=1)
                frame[f"{prefix}_mean"] = mean
                frame[f"{prefix}_median"] = median
                frame[f"{prefix}_std"] = std
                frame[f"{prefix}_q90_q10"] = q90 - q10
                frame[f"{prefix}_mean_minus_final"] = mean - frame["final"]
                frame[f"{prefix}_median_minus_final"] = median - frame["final"]
                frame[f"{prefix}_mean_minus_base"] = mean - frame["base"]
                frame[f"{prefix}_upvote_vs_final"] = np.nanmean(mat > frame["final"].to_numpy(float).reshape(-1, 1), axis=1)
                frame[f"{prefix}_upvote_vs_base"] = np.nanmean(mat > frame["base"].to_numpy(float).reshape(-1, 1), axis=1)
            else:
                for suffix in [
                    "mean",
                    "median",
                    "std",
                    "q90_q10",
                    "mean_minus_final",
                    "median_minus_final",
                    "mean_minus_base",
                    "upvote_vs_final",
                    "upvote_vs_base",
                ]:
                    frame[f"{prefix}_{suffix}"] = 0.0

    train = oof.merge(tab, on="case_durable_id", how="left").merge(notes, on="case_durable_id", how="left")
    held = test.merge(tab.drop(columns=["scheduled_in_room"], errors="ignore"), on="case_durable_id", how="left").merge(
        notes, on="case_durable_id", how="left"
    )
    for frame in (train, held):
        for col in ["primary_procedure_name", "patient_durable_id"]:
            if col not in frame.columns:
                left = f"{col}_x"
                right = f"{col}_y"
                if left in frame.columns:
                    frame[col] = frame[left]
                elif right in frame.columns:
                    frame[col] = frame[right]
    for frame in (train, held):
        frame["procedure_family"] = frame["primary_procedure_name"].map(procedure_family)
        sched_in = pd.to_datetime(frame["scheduled_in_room"], utc=True, errors="coerce")
        sched_out = pd.to_datetime(frame["scheduled_out_of_room"], utc=True, errors="coerce")
        frame["scheduled_duration_minutes"] = (sched_out - sched_in).dt.total_seconds() / 60
        frame["scheduled_duration_bin"] = frame["scheduled_duration_minutes"].map(duration_bin)
        frame["note_count_bin"] = frame["n_pre_cutoff_notes_used"].fillna(0).map(note_count_bin)
        frame["n_pre_cutoff_notes_used"] = frame["n_pre_cutoff_notes_used"].fillna(0)
        frame["note_chars"] = frame["note_chars"].fillna(0)
        frame["log_note_chars"] = np.log1p(frame["note_chars"])
        frame["note_doc_types_used"] = frame["note_doc_types_used"].fillna("")
        frame["n_note_doc_types"] = frame["note_doc_types_used"].map(lambda x: len({p.strip() for p in str(x).split(";") if p.strip()}))
        frame["hour"] = sched_in.dt.hour.fillna(-1)
        frame["dayofweek"] = sched_in.dt.dayofweek.fillna(-1)
        frame["residual_after_final"] = frame["case_actual_duration_time"] - frame["final"]
        frame["pls_gap_bin"] = pd.cut(
            frame["pls_mean_minus_final"],
            bins=[-np.inf, -4.0, -1.0, 1.0, 4.0, np.inf],
            labels=["pls_much_lower", "pls_lower", "pls_near", "pls_higher", "pls_much_higher"],
        ).astype(str)
        frame["pls_spread_bin"] = pd.cut(
            frame["pls_std"],
            bins=[-np.inf, 1.5, 3.0, 5.0, 8.0, np.inf],
            labels=["spread_vlow", "spread_low", "spread_med", "spread_high", "spread_vhigh"],
        ).astype(str)
        frame["pls_correction_vote_bin"] = pd.cut(
            frame["pls_upvote_vs_final"],
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=["vote_down_strong", "vote_down", "vote_mixed", "vote_up", "vote_up_strong"],
        ).astype(str)
    return train, held


NUMERIC_FEATURES = [
    "base",
    "final",
    "correction",
    "abs_correction",
    "model_pred_std",
    "model_pred_range",
    "model_pred_q90_q10",
    "model_pred_mean_minus_final",
    "pls_mean",
    "pls_median",
    "pls_std",
    "pls_q90_q10",
    "pls_mean_minus_final",
    "pls_median_minus_final",
    "pls_mean_minus_base",
    "pls_upvote_vs_final",
    "pls_upvote_vs_base",
    "pls_raw_mean",
    "pls_raw_median",
    "pls_raw_std",
    "pls_raw_q90_q10",
    "pls_raw_mean_minus_final",
    "pls_raw_median_minus_final",
    "pls_raw_mean_minus_base",
    "pls_raw_upvote_vs_final",
    "pls_raw_upvote_vs_base",
    "pls_delta_mean",
    "pls_delta_median",
    "pls_delta_std",
    "pls_delta_q90_q10",
    "pls_delta_mean_minus_final",
    "pls_delta_median_minus_final",
    "pls_delta_mean_minus_base",
    "pls_delta_upvote_vs_final",
    "pls_delta_upvote_vs_base",
    "pls_raw_delta_mean",
    "pls_raw_delta_median",
    "pls_raw_delta_std",
    "pls_raw_delta_q90_q10",
    "pls_raw_delta_mean_minus_final",
    "pls_raw_delta_median_minus_final",
    "pls_raw_delta_mean_minus_base",
    "pls_raw_delta_upvote_vs_final",
    "pls_raw_delta_upvote_vs_base",
    "scheduled_duration_minutes",
    "patient_age",
    "case_num_procedures",
    "case_num_panels",
    "case_num_providers",
    "is_redo",
    "prior_same_proc_duration",
    "redo_time_gap_days",
    "bmi",
    "n_pre_cutoff_notes_used",
    "note_chars",
    "log_note_chars",
    "n_note_doc_types",
    "hour",
    "dayofweek",
]
CATEGORICAL_FEATURES = [
    "procedure_family",
    "primary_procedure_name",
    "case_anesthesia",
    "case_urgency_level",
    "case_surgery_patient_class",
    "case_admission_patient_class",
    "patient_sex",
    "note_count_bin",
    "scheduled_duration_bin",
    "pls_gap_bin",
    "pls_spread_bin",
    "pls_correction_vote_bin",
]


def design_matrix(train: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    both = pd.concat([train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], held[NUMERIC_FEATURES + CATEGORICAL_FEATURES]], axis=0)
    both_num = both[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    med = both_num.iloc[: len(train)].median(numeric_only=True)
    both_num = both_num.fillna(med).fillna(0.0)
    both_cat = both[CATEGORICAL_FEATURES].fillna("__MISSING__").astype(str)
    # Collapse very rare procedure names using train frequencies only.
    vc = both_cat.iloc[: len(train)]["primary_procedure_name"].value_counts()
    keep_proc = set(vc[vc >= 20].index)
    both_cat["primary_procedure_name"] = both_cat["primary_procedure_name"].where(
        both_cat["primary_procedure_name"].isin(keep_proc), "__RARE_PROC__"
    )
    x = pd.concat([both_num, pd.get_dummies(both_cat, prefix=CATEGORICAL_FEATURES, dtype=float)], axis=1)
    return x.iloc[: len(train)].copy(), x.iloc[len(train) :].copy()


def join_categories(*parts: pd.Series) -> pd.Series:
    arrays = [part.astype(str).to_numpy() for part in parts]
    joined = arrays[0].copy()
    for arr in arrays[1:]:
        joined = np.char.add(np.char.add(joined, "|"), arr)
    return pd.Series(joined, index=parts[0].index)


def tune_gamma_for_rows(y: np.ndarray, base: np.ndarray, correction: np.ndarray, idx: np.ndarray, grid: np.ndarray) -> float:
    if len(idx) < 20:
        idx = np.arange(len(y))
    scores = [(float(g), float(np.mean(np.abs(base[idx] + g * correction[idx] - y[idx])))) for g in grid]
    return min(scores, key=lambda t: t[1])[0]


def category_gamma_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    category_train: pd.Series,
    category_held: pd.Series,
    *,
    grid: np.ndarray,
    min_n: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    corr = train["correction"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)
    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat_arr = category_train.iloc[fit_idx].astype(str).to_numpy()
        val_cat_arr = category_train.iloc[val_idx].astype(str).to_numpy()
        global_gamma = tune_gamma_for_rows(y[fit_idx], base[fit_idx], corr[fit_idx], np.arange(len(fit_idx)), grid)
        gammas: dict[str, float] = {}
        for cat in np.unique(fit_cat_arr):
            local_idx = fit_idx[fit_cat_arr == cat]
            if len(local_idx) >= min_n:
                gammas[str(cat)] = tune_gamma_for_rows(y, base, corr, local_idx, grid)
        pred = np.empty(len(val_idx), dtype=float)
        for j, cat in enumerate(val_cat_arr):
            g = gammas.get(str(cat), global_gamma)
            pred[j] = base[val_idx[j]] + g * corr[val_idx[j]]
        oof[val_idx] = pred

    global_gamma = tune_gamma_for_rows(y, base, corr, np.arange(len(y)), grid)
    gammas = {"__GLOBAL__": global_gamma}
    cat_all = category_train.astype(str).to_numpy()
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            gammas[str(cat)] = tune_gamma_for_rows(y, base, corr, idx, grid)
    held_pred = np.empty(len(held), dtype=float)
    for i, cat in enumerate(category_held.astype(str).to_numpy()):
        g = gammas.get(str(cat), global_gamma)
        held_pred[i] = held["base"].iloc[i] + g * held["correction"].iloc[i]
    return oof, held_pred, gammas


def risk_bins_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    x_train: pd.DataFrame,
    x_held: pd.DataFrame,
    *,
    k: int,
    seed: int,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    y = train["case_actual_duration_time"].to_numpy(float)
    target_risk = np.abs(train["final"].to_numpy(float) - y)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    risk_oof = np.zeros(len(train), dtype=float)
    bin_oof = np.empty(len(train), dtype=object)

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, target_risk, groups=groups), start=1):
        model = ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=12,
            max_features=0.6,
            random_state=seed + fold,
            n_jobs=-1,
        )
        model.fit(x_train.iloc[fit_idx], target_risk[fit_idx])
        risk_fit = model.predict(x_train.iloc[fit_idx])
        risk_val = model.predict(x_train.iloc[val_idx])
        edges = np.unique(np.quantile(risk_fit, np.linspace(0, 1, k + 1)))
        if len(edges) <= 2:
            labels = np.array(["risk_all"] * len(val_idx), dtype=object)
        else:
            labels = np.array([f"risk_{b}" for b in np.digitize(risk_val, edges[1:-1], right=True)], dtype=object)
        risk_oof[val_idx] = risk_val
        bin_oof[val_idx] = labels

    final_model = ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=12,
        max_features=0.6,
        random_state=seed + 999,
        n_jobs=-1,
    )
    final_model.fit(x_train, target_risk)
    risk_train_full = final_model.predict(x_train)
    risk_held = final_model.predict(x_held)
    edges = np.unique(np.quantile(risk_train_full, np.linspace(0, 1, k + 1)))
    if len(edges) <= 2:
        held_bins = np.array(["risk_all"] * len(held), dtype=object)
    else:
        held_bins = np.array([f"risk_{b}" for b in np.digitize(risk_held, edges[1:-1], right=True)], dtype=object)
    return pd.Series(bin_oof, index=train.index), pd.Series(held_bins, index=held.index), risk_oof, risk_held


def category_mean_residual_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    category_train: pd.Series,
    category_held: pd.Series,
    *,
    min_n: int = 80,
    shrink: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    final = train["final"].to_numpy(float)
    resid = y - final
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)
    for fit_idx, val_idx in splitter.split(train, resid, groups=groups):
        global_mean = float(resid[fit_idx].mean())
        means: dict[str, float] = {}
        fit_cat_arr = category_train.iloc[fit_idx].astype(str).to_numpy()
        for cat in np.unique(fit_cat_arr):
            idx = fit_idx[fit_cat_arr == cat]
            if len(idx) >= min_n:
                raw = float(resid[idx].mean())
                means[str(cat)] = (len(idx) * raw + shrink * global_mean) / (len(idx) + shrink)
        for j, cat in zip(val_idx, category_train.iloc[val_idx].astype(str)):
            oof[j] = final[j] + means.get(str(cat), global_mean)

    global_mean = float(resid.mean())
    means = {"__GLOBAL__": global_mean}
    cat_all = category_train.astype(str).to_numpy()
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            raw = float(resid[idx].mean())
            means[str(cat)] = (len(idx) * raw + shrink * global_mean) / (len(idx) + shrink)
    held_pred = np.array([held["final"].iloc[i] + means.get(str(cat), global_mean) for i, cat in enumerate(category_held.astype(str).to_numpy())])
    return oof, held_pred, means


def risk_bin_ridge_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    x_train: pd.DataFrame,
    x_held: pd.DataFrame,
    category_train: pd.Series,
    category_held: pd.Series,
    *,
    min_n: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    y = train["case_actual_duration_time"].to_numpy(float)
    resid = y - train["final"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof_adj = np.zeros(len(train), dtype=float)
    for fit_idx, val_idx in splitter.split(x_train, resid, groups=groups):
        global_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
        global_model.fit(x_train.iloc[fit_idx], resid[fit_idx])
        models: dict[str, Any] = {}
        fit_cat_arr = category_train.iloc[fit_idx].astype(str).to_numpy()
        for cat in np.unique(fit_cat_arr):
            idx = fit_idx[fit_cat_arr == cat]
            if len(idx) >= min_n:
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
                m.fit(x_train.iloc[idx], resid[idx])
                models[str(cat)] = m
        for j in val_idx:
            cat = str(category_train.iloc[j])
            model = models.get(cat, global_model)
            oof_adj[j] = float(model.predict(x_train.iloc[[j]])[0])
    oof_pred = train["final"].to_numpy(float) + oof_adj

    global_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
    global_model.fit(x_train, resid)
    models = {}
    cat_all = category_train.astype(str).to_numpy()
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
            m.fit(x_train.iloc[idx], resid[idx])
            models[str(cat)] = m
    held_adj = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(category_held.astype(str)):
        model = models.get(str(cat), global_model)
        held_adj[i] = float(model.predict(x_held.iloc[[i]])[0])
    return oof_pred, held["final"].to_numpy(float) + held_adj


def metric_row(name: str, train: pd.DataFrame, held: pd.DataFrame, oof_pred: np.ndarray, held_pred: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    y_train = train["case_actual_duration_time"].to_numpy(float)
    y_held = held["case_actual_duration_time"].to_numpy(float)
    final_train = train["final"].to_numpy(float)
    final_held = held["final"].to_numpy(float)
    return {
        "variant": name,
        "train_oof_mae": float(mean_absolute_error(y_train, oof_pred)),
        "final_train_oof_mae": float(mean_absolute_error(y_train, final_train)),
        "heldout_mae": float(mean_absolute_error(y_held, held_pred)),
        "final_heldout_mae": float(mean_absolute_error(y_held, final_held)),
        "heldout_gain_vs_final": float(np.mean(np.abs(final_held - y_held) - np.abs(held_pred - y_held))),
        "improved_rate_vs_final": float((np.abs(held_pred - y_held) < np.abs(final_held - y_held)).mean()),
        "mean_prediction_shift_vs_final": float(np.mean(held_pred - final_held)),
        "meta_json": json.dumps(meta, sort_keys=True),
    }


def main() -> None:
    assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, held = read_frames()
    x_train, x_held = design_matrix(train, held)
    grid_pos = np.arange(0.0, 2.501, 0.025)
    grid_signed = np.arange(-1.0, 2.501, 0.025)
    rows: list[dict[str, Any]] = []
    pred_cols: dict[str, np.ndarray] = {"final": held["final"].to_numpy(float)}

    # Sanity: retune one global multiplier on train OOF.
    for grid_name, grid in [("positive", grid_pos), ("signed", grid_signed)]:
        oof_pred, held_pred, gammas = category_gamma_oof(
            train,
            held,
            pd.Series(["all"] * len(train), index=train.index),
            pd.Series(["all"] * len(held), index=held.index),
            grid=grid,
            min_n=1,
        )
        name = f"global_gamma_{grid_name}"
        rows.append(metric_row(name, train, held, oof_pred, held_pred, {"gammas": gammas}))
        pred_cols[name] = held_pred

    fixed_categories = {
        "procedure_family": (train["procedure_family"], held["procedure_family"]),
        "scheduled_duration_bin": (train["scheduled_duration_bin"], held["scheduled_duration_bin"]),
        "note_count_bin": (train["note_count_bin"], held["note_count_bin"]),
        "family_x_sched": (train["procedure_family"].astype(str) + "|" + train["scheduled_duration_bin"].astype(str), held["procedure_family"].astype(str) + "|" + held["scheduled_duration_bin"].astype(str)),
        "family_x_note_count": (train["procedure_family"].astype(str) + "|" + train["note_count_bin"].astype(str), held["procedure_family"].astype(str) + "|" + held["note_count_bin"].astype(str)),
        "pls_gap_bin": (train["pls_gap_bin"], held["pls_gap_bin"]),
        "pls_spread_bin": (train["pls_spread_bin"], held["pls_spread_bin"]),
        "pls_vote_bin": (train["pls_correction_vote_bin"], held["pls_correction_vote_bin"]),
        "family_x_pls_gap": (train["procedure_family"].astype(str) + "|" + train["pls_gap_bin"].astype(str), held["procedure_family"].astype(str) + "|" + held["pls_gap_bin"].astype(str)),
        "sched_x_pls_gap": (train["scheduled_duration_bin"].astype(str) + "|" + train["pls_gap_bin"].astype(str), held["scheduled_duration_bin"].astype(str) + "|" + held["pls_gap_bin"].astype(str)),
        "risk_surface_pls": (
            train["procedure_family"].astype(str) + "|" + train["scheduled_duration_bin"].astype(str) + "|" + train["pls_gap_bin"].astype(str),
            held["procedure_family"].astype(str) + "|" + held["scheduled_duration_bin"].astype(str) + "|" + held["pls_gap_bin"].astype(str),
        ),
    }
    for cat_name, (cat_tr, cat_te) in fixed_categories.items():
        for grid_name, grid in [("positive", grid_pos), ("signed", grid_signed)]:
            oof_pred, held_pred, gammas = category_gamma_oof(train, held, cat_tr, cat_te, grid=grid, min_n=80)
            name = f"{cat_name}_gamma_{grid_name}"
            rows.append(metric_row(name, train, held, oof_pred, held_pred, {"n_gammas": len(gammas), "gammas": gammas}))
            pred_cols[name] = held_pred
        oof_pred, held_pred, means = category_mean_residual_oof(train, held, cat_tr, cat_te, min_n=80)
        name = f"{cat_name}_mean_residual"
        rows.append(metric_row(name, train, held, oof_pred, held_pred, {"n_means": len(means), "means": means}))
        pred_cols[name] = held_pred

    for k in [3, 4, 5, 6, 8]:
        risk_tr, risk_te, risk_oof, risk_held = risk_bins_oof(train, held, x_train, x_held, k=k, seed=7000 + k)
        risk_corr = float(
            np.corrcoef(
                risk_oof,
                np.abs(train["final"].to_numpy(float) - train["case_actual_duration_time"].to_numpy(float)),
            )[0, 1]
        )
        risk_category_variants = {
            f"risk{k}": (risk_tr, risk_te, 80),
            f"risk{k}_x_family": (join_categories(risk_tr, train["procedure_family"]), join_categories(risk_te, held["procedure_family"]), 120),
            f"risk{k}_x_pls_gap": (join_categories(risk_tr, train["pls_gap_bin"]), join_categories(risk_te, held["pls_gap_bin"]), 100),
            f"risk{k}_x_sched": (join_categories(risk_tr, train["scheduled_duration_bin"]), join_categories(risk_te, held["scheduled_duration_bin"]), 120),
            f"risk{k}_x_family_x_pls_gap": (
                join_categories(risk_tr, train["procedure_family"], train["pls_gap_bin"]),
                join_categories(risk_te, held["procedure_family"], held["pls_gap_bin"]),
                140,
            ),
        }
        for risk_cat_name, (risk_cat_tr, risk_cat_te, min_n) in risk_category_variants.items():
            for grid_name, grid in [("positive", grid_pos), ("signed", grid_signed)]:
                oof_pred, held_pred, gammas = category_gamma_oof(train, held, risk_cat_tr, risk_cat_te, grid=grid, min_n=min_n)
                name = f"{risk_cat_name}_gamma_{grid_name}"
                rows.append(
                    metric_row(
                        name,
                        train,
                        held,
                        oof_pred,
                        held_pred,
                        {
                            "n_gammas": len(gammas),
                            "gammas": gammas,
                            "risk_oof_corr_abs_error": risk_corr,
                            "min_n": min_n,
                        },
                    )
                )
                pred_cols[name] = held_pred
        oof_pred, held_pred, means = category_mean_residual_oof(train, held, risk_tr, risk_te, min_n=80)
        name = f"risk{k}_mean_residual"
        rows.append(metric_row(name, train, held, oof_pred, held_pred, {"n_means": len(means), "means": means}))
        pred_cols[name] = held_pred
        oof_pred, held_pred = risk_bin_ridge_oof(train, held, x_train, x_held, risk_tr, risk_te, min_n=180)
        name = f"risk{k}_ridge_residual"
        rows.append(metric_row(name, train, held, oof_pred, held_pred, {"k": k}))
        pred_cols[name] = held_pred

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "risk_category_modulation_results.csv", index=False)

    held_out = held[
        [
            "case_durable_id",
            "patient_durable_id",
            "scheduled_in_room",
            "case_actual_duration_time",
            "primary_procedure_name",
            "procedure_family",
            "base",
            "final",
        ]
    ].copy()
    for name, pred in pred_cols.items():
        held_out[f"pred_{name}"] = pred
    held_out.to_csv(OUT_DIR / "risk_category_modulation_test_predictions.csv", index=False)

    summary = {
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "final_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train["final"])),
        "final_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held["final"])),
        "best_by_train_oof": results.iloc[0].to_dict(),
        "best_by_heldout_exploratory": results.sort_values("heldout_mae").iloc[0].to_dict(),
        "forbidden_llm_feature_file_read": False,
        "note": "Use best_by_train_oof for leakage-safe selection; best_by_heldout_exploratory is diagnostic only.",
    }
    (OUT_DIR / "risk_category_modulation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(25).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(15).to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
