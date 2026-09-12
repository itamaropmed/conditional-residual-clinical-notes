#!/usr/bin/env python3
"""Stricter variance-aware gating for the strict PLS manifold model.

This explores whether high-variance cases can be handled by:
  * capping the PLS correction by group,
  * nested risk/gain/residual-score gates,
  * direct cross-fit residual correction heads.

The old 11 LLM feature file is not read. Score-derived categories are nested
inside the outer patient GroupKFold to avoid target bleed into OOF selection.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
OUT_DIR = ROOT / "outputs/variance_aware_pls_modulation"
BASE_SCRIPT = ROOT / "work/no_llm_notes/risk_category_modulation.py"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("risk_category_modulation", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tune_gamma_cap(
    y: np.ndarray,
    base: np.ndarray,
    corr: np.ndarray,
    idx: np.ndarray,
    gammas: np.ndarray,
    caps: list[float],
) -> tuple[float, float]:
    if len(idx) < 20:
        idx = np.arange(len(y))
    best = (math.inf, 1.0, math.inf)
    corr_idx = corr[idx]
    for cap in caps:
        clipped = corr_idx if math.isinf(cap) else np.clip(corr_idx, -cap, cap)
        for gamma in gammas:
            pred = base[idx] + gamma * clipped
            score = float(np.mean(np.abs(pred - y[idx])))
            if score < best[0]:
                best = (score, float(gamma), float(cap))
    return best[1], best[2]


def apply_gamma_cap(base: np.ndarray, corr: np.ndarray, gamma: float, cap: float) -> np.ndarray:
    clipped = corr if math.isinf(cap) else np.clip(corr, -cap, cap)
    return base + gamma * clipped


def category_gamma_cap_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    category_train: pd.Series,
    category_held: pd.Series,
    *,
    gammas: np.ndarray,
    caps: list[float],
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    corr = train["correction"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)

    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat = category_train.iloc[fit_idx].astype(str).to_numpy()
        val_cat = category_train.iloc[val_idx].astype(str).to_numpy()
        global_gamma, global_cap = tune_gamma_cap(y, base, corr, fit_idx, gammas, caps)
        params: dict[str, tuple[float, float]] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) >= min_n:
                params[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)
        for pos, cat in enumerate(val_cat):
            gamma, cap = params.get(str(cat), (global_gamma, global_cap))
            oof[val_idx[pos]] = apply_gamma_cap(base[val_idx[pos] : val_idx[pos] + 1], corr[val_idx[pos] : val_idx[pos] + 1], gamma, cap)[0]

    cat_all = category_train.astype(str).to_numpy()
    global_gamma, global_cap = tune_gamma_cap(y, base, corr, np.arange(len(y)), gammas, caps)
    params_full: dict[str, tuple[float, float]] = {"__GLOBAL__": (global_gamma, global_cap)}
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            params_full[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)

    held_base = held["base"].to_numpy(float)
    held_corr = held["correction"].to_numpy(float)
    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(category_held.astype(str).to_numpy()):
        gamma, cap = params_full.get(str(cat), (global_gamma, global_cap))
        held_pred[i] = apply_gamma_cap(held_base[i : i + 1], held_corr[i : i + 1], gamma, cap)[0]

    meta = {k: {"gamma": v[0], "cap": v[1]} for k, v in params_full.items()}
    return oof, held_pred, meta


def join_categories(index: pd.Index, *parts: np.ndarray | pd.Series) -> pd.Series:
    arrays = [pd.Series(part, index=index).astype(str).to_numpy() for part in parts]
    out = arrays[0].copy()
    for arr in arrays[1:]:
        out = np.char.add(np.char.add(out, "|"), arr)
    return pd.Series(out, index=index)


def score_model(seed: int, kind: str = "extra_trees") -> Any:
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=24,
            max_features=0.55,
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.2,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=24,
            max_features=0.55,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def inner_oof_score(
    x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    kind: str,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2:
        model = score_model(seed, kind)
        model.fit(x, target)
        return model.predict(x)
    splitter = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(target), dtype=float)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x, target, groups=groups), start=1):
        model = score_model(seed + fold, kind)
        model.fit(x.iloc[fit_idx], target[fit_idx])
        pred[val_idx] = model.predict(x.iloc[val_idx])
    return pred


def bins_from_scores(train_scores: np.ndarray, apply_scores: np.ndarray, k: int, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(train_scores, np.linspace(0, 1, k + 1)))
    if len(edges) <= 2:
        return np.array([f"{prefix}_all"] * len(train_scores), dtype=object), np.array([f"{prefix}_all"] * len(apply_scores), dtype=object), edges
    train_bins = np.array([f"{prefix}_{b}" for b in np.digitize(train_scores, edges[1:-1], right=True)], dtype=object)
    apply_bins = np.array([f"{prefix}_{b}" for b in np.digitize(apply_scores, edges[1:-1], right=True)], dtype=object)
    return train_bins, apply_bins, edges


def nested_score_gate_oof(
    train: pd.DataFrame,
    held: pd.DataFrame,
    x_train: pd.DataFrame,
    x_held: pd.DataFrame,
    target: np.ndarray,
    *,
    score_name: str,
    k: int,
    extra_specs: dict[str, tuple[pd.Series, pd.Series, int]],
    gammas: np.ndarray,
    caps: list[float],
    kind: str,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    corr = train["correction"].to_numpy(float)
    held_base = held["base"].to_numpy(float)
    held_corr = held["correction"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)

    oof_by_name = {name: np.zeros(len(train), dtype=float) for name in extra_specs}
    outer_score_oof = np.zeros(len(train), dtype=float)

    for outer_fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, target, groups=groups), start=1):
        x_fit = x_train.iloc[fit_idx]
        target_fit = target[fit_idx]
        groups_fit = groups[fit_idx]
        fit_score = inner_oof_score(x_fit, target_fit, groups_fit, seed=seed + outer_fold * 100, kind=kind)
        model = score_model(seed + outer_fold * 1000, kind)
        model.fit(x_fit, target_fit)
        val_score = model.predict(x_train.iloc[val_idx])
        outer_score_oof[val_idx] = val_score
        fit_bins, val_bins, _ = bins_from_scores(fit_score, val_score, k, score_name)

        for name, (extra_train, _extra_held, min_n) in extra_specs.items():
            if name == score_name:
                fit_cat = pd.Series(fit_bins, index=train.index[fit_idx])
                val_cat = pd.Series(val_bins, index=train.index[val_idx])
            else:
                fit_cat = join_categories(train.index[fit_idx], fit_bins, extra_train.iloc[fit_idx].to_numpy())
                val_cat = join_categories(train.index[val_idx], val_bins, extra_train.iloc[val_idx].to_numpy())
            global_gamma, global_cap = tune_gamma_cap(y, base, corr, fit_idx, gammas, caps)
            params: dict[str, tuple[float, float]] = {}
            fit_cat_arr = fit_cat.astype(str).to_numpy()
            val_cat_arr = val_cat.astype(str).to_numpy()
            for cat in np.unique(fit_cat_arr):
                idx = fit_idx[fit_cat_arr == cat]
                if len(idx) >= min_n:
                    params[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)
            for pos, cat in enumerate(val_cat_arr):
                gamma, cap = params.get(str(cat), (global_gamma, global_cap))
                row = val_idx[pos]
                oof_by_name[name][row] = apply_gamma_cap(base[row : row + 1], corr[row : row + 1], gamma, cap)[0]

    train_score_full = inner_oof_score(x_train, target, groups, seed=seed + 9999, kind=kind)
    final_model = score_model(seed + 19999, kind)
    final_model.fit(x_train, target)
    held_score = final_model.predict(x_held)
    train_bins, held_bins, edges = bins_from_scores(train_score_full, held_score, k, score_name)

    out: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for name, (extra_train, extra_held, min_n) in extra_specs.items():
        if name == score_name:
            train_cat = pd.Series(train_bins, index=train.index)
            held_cat = pd.Series(held_bins, index=held.index)
        else:
            train_cat = join_categories(train.index, train_bins, extra_train.to_numpy())
            held_cat = join_categories(held.index, held_bins, extra_held.to_numpy())

        cat_all = train_cat.astype(str).to_numpy()
        global_gamma, global_cap = tune_gamma_cap(y, base, corr, np.arange(len(y)), gammas, caps)
        params_full: dict[str, tuple[float, float]] = {"__GLOBAL__": (global_gamma, global_cap)}
        for cat in np.unique(cat_all):
            idx = np.flatnonzero(cat_all == cat)
            if len(idx) >= min_n:
                params_full[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)

        held_pred = np.zeros(len(held), dtype=float)
        for i, cat in enumerate(held_cat.astype(str).to_numpy()):
            gamma, cap = params_full.get(str(cat), (global_gamma, global_cap))
            held_pred[i] = apply_gamma_cap(held_base[i : i + 1], held_corr[i : i + 1], gamma, cap)[0]

        meta = {
            "score_name": score_name,
            "k": k,
            "kind": kind,
            "min_n": min_n,
            "score_oof_corr_target": float(np.corrcoef(outer_score_oof, target)[0, 1]),
            "score_full_oof_corr_target": float(np.corrcoef(train_score_full, target)[0, 1]),
            "edges": [float(x) for x in edges],
            "params": {key: {"gamma": val[0], "cap": val[1]} for key, val in params_full.items()},
        }
        out[name] = (oof_by_name[name], held_pred, meta)
    return out


def crossfit_residual_head(
    train: pd.DataFrame,
    held: pd.DataFrame,
    x_train: pd.DataFrame,
    x_held: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    final = train["final"].to_numpy(float)
    resid = y - final
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    resid_oof = np.zeros(len(train), dtype=float)

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, resid, groups=groups), start=1):
        if model_name == "ridge":
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 18)))
        elif model_name == "huber":
            model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=400))
        else:
            model = score_model(seed + fold, model_name)
        model.fit(x_train.iloc[fit_idx], resid[fit_idx])
        resid_oof[val_idx] = model.predict(x_train.iloc[val_idx])

    grid = np.arange(-0.5, 0.501, 0.025)
    scores = [(float(a), float(mean_absolute_error(y, final + a * resid_oof))) for a in grid]
    alpha = min(scores, key=lambda t: t[1])[0]

    if model_name == "ridge":
        final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 18)))
    elif model_name == "huber":
        final_model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=400))
    else:
        final_model = score_model(seed + 999, model_name)
    final_model.fit(x_train, resid)
    held_resid = final_model.predict(x_held)
    return final + alpha * resid_oof, held["final"].to_numpy(float) + alpha * held_resid, {
        "model": model_name,
        "alpha": alpha,
        "resid_oof_corr_target": float(np.corrcoef(resid_oof, resid)[0, 1]),
    }


def metric_row(
    name: str,
    train: pd.DataFrame,
    held: pd.DataFrame,
    oof_pred: np.ndarray,
    held_pred: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
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


def paired_ci(diff: np.ndarray, seed: int = 20260823) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    mod = load_base_module()
    mod.assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, held = mod.read_frames()
    x_train, x_held = mod.design_matrix(train, held)

    gammas = np.arange(0.0, 2.501, 0.05)
    caps = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 45.0, math.inf]
    rows: list[dict[str, Any]] = []
    pred_cols: dict[str, np.ndarray] = {"final": held["final"].to_numpy(float)}

    fixed_specs = {
        "family_cap_gamma": (train["procedure_family"], held["procedure_family"], 80),
        "family_x_pls_gap_cap_gamma": (
            train["procedure_family"].astype(str) + "|" + train["pls_gap_bin"].astype(str),
            held["procedure_family"].astype(str) + "|" + held["pls_gap_bin"].astype(str),
            100,
        ),
        "family_x_pls_spread_cap_gamma": (
            train["procedure_family"].astype(str) + "|" + train["pls_spread_bin"].astype(str),
            held["procedure_family"].astype(str) + "|" + held["pls_spread_bin"].astype(str),
            100,
        ),
        "family_x_note_count_cap_gamma": (
            train["procedure_family"].astype(str) + "|" + train["note_count_bin"].astype(str),
            held["procedure_family"].astype(str) + "|" + held["note_count_bin"].astype(str),
            100,
        ),
    }
    for name, (cat_tr, cat_te, min_n) in fixed_specs.items():
        oof_pred, held_pred, params = category_gamma_cap_oof(train, held, cat_tr, cat_te, gammas=gammas, caps=caps, min_n=min_n)
        rows.append(metric_row(name, train, held, oof_pred, held_pred, {"n_params": len(params), "params": params, "min_n": min_n}))
        pred_cols[name] = held_pred

    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    final = train["final"].to_numpy(float)
    targets = {
        "risk_abs": np.abs(final - y),
        "gain_vs_base": np.abs(base - y) - np.abs(final - y),
    }
    for score_name, target in targets.items():
        for k in [3]:
            specs = {
                f"{score_name}{k}": (pd.Series([""] * len(train), index=train.index), pd.Series([""] * len(held), index=held.index), 80),
                f"{score_name}{k}_x_family": (train["procedure_family"], held["procedure_family"], 120),
                f"{score_name}{k}_x_pls_gap": (train["pls_gap_bin"], held["pls_gap_bin"], 100),
                f"{score_name}{k}_x_family_x_pls_gap": (
                    train["procedure_family"].astype(str) + "|" + train["pls_gap_bin"].astype(str),
                    held["procedure_family"].astype(str) + "|" + held["pls_gap_bin"].astype(str),
                    140,
                ),
            }
            nested = nested_score_gate_oof(
                train,
                held,
                x_train,
                x_held,
                target,
                score_name=f"{score_name}{k}",
                k=k,
                extra_specs=specs,
                gammas=gammas,
                caps=caps,
                kind="extra_trees",
                seed=9000 + k + 100 * list(targets).index(score_name),
            )
            for name, (oof_pred, held_pred, meta) in nested.items():
                rows.append(metric_row(f"nested_{name}_cap_gamma", train, held, oof_pred, held_pred, meta))
                pred_cols[f"nested_{name}_cap_gamma"] = held_pred

    for model_name in ["ridge", "hgb", "extra_trees"]:
        oof_pred, held_pred, meta = crossfit_residual_head(train, held, x_train, x_held, model_name=model_name, seed=12000)
        name = f"crossfit_residual_{model_name}"
        rows.append(metric_row(name, train, held, oof_pred, held_pred, meta))
        pred_cols[name] = held_pred

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "variance_aware_results.csv", index=False)

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
    held_out.to_csv(OUT_DIR / "variance_aware_test_predictions.csv", index=False)

    y_held = held["case_actual_duration_time"].to_numpy(float)
    final_held = held["final"].to_numpy(float)
    best_train = results.iloc[0].to_dict()
    best_held = results.sort_values("heldout_mae").iloc[0].to_dict()
    for label, row in [("best_by_train_oof", best_train), ("best_by_heldout_exploratory", best_held)]:
        pred = held_out[f"pred_{row['variant']}"].to_numpy(float)
        diff = np.abs(final_held - y_held) - np.abs(pred - y_held)
        row["heldout_gain_ci95"] = paired_ci(diff)
        row["large_wins_gt10"] = int((diff > 10).sum())
        row["large_losses_ltminus10"] = int((diff < -10).sum())

    summary = {
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "final_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train["final"])),
        "final_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held["final"])),
        "best_by_train_oof": best_train,
        "best_by_heldout_exploratory": best_held,
        "patient_overlap": int(
            len(set(train["patient_durable_id"].astype(str)).intersection(set(held["patient_durable_id"].astype(str))))
        ),
        "forbidden_llm_feature_file_read": False,
        "note": "Nested score gates are the stricter leakage-safe versions. best_by_heldout_exploratory is diagnostic only.",
    }
    (OUT_DIR / "variance_aware_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(30).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(20).to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
