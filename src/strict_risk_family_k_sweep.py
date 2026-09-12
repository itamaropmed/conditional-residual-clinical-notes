#!/usr/bin/env python3
"""Strict nested k-risk-family sweep for the clean PLS manifold model.

This answers whether more than 3 risk families, tail-risk scores, signed
residual scores, and conservative gate blends can improve the 31.607 family
calibration without leakage.

No old 11 LLM feature file is read.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
BASE_SCRIPT = ROOT / "work/no_llm_notes/risk_category_modulation.py"
OUT_DIR = ROOT / "outputs/strict_risk_family_k_sweep"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("risk_category_modulation", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_model(kind: str, seed: int) -> Any:
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=24,
            max_features=0.55,
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "hgb_abs":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.2,
            random_state=seed,
        )
    if kind == "hgb_sq":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.2,
            random_state=seed,
        )
    raise ValueError(kind)


def inner_oof_score(x: pd.DataFrame, target: np.ndarray, groups: np.ndarray, *, kind: str, seed: int) -> np.ndarray:
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        model = score_model(kind, seed)
        model.fit(x, target)
        return model.predict(x)
    out = np.zeros(len(target), dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x, target, groups=groups), start=1):
        model = score_model(kind, seed + fold)
        model.fit(x.iloc[fit_idx], target[fit_idx])
        out[val_idx] = model.predict(x.iloc[val_idx])
    return out


def compute_nested_scores(
    train: pd.DataFrame,
    held: pd.DataFrame,
    x_train: pd.DataFrame,
    x_held: pd.DataFrame,
    target: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> dict[str, Any]:
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    folds: list[dict[str, Any]] = []
    outer_oof = np.zeros(len(train), dtype=float)

    for outer_fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, target, groups=groups), start=1):
        fit_score = inner_oof_score(
            x_train.iloc[fit_idx],
            target[fit_idx],
            groups[fit_idx],
            kind=kind,
            seed=seed + 100 * outer_fold,
        )
        model = score_model(kind, seed + 1000 * outer_fold)
        model.fit(x_train.iloc[fit_idx], target[fit_idx])
        val_score = model.predict(x_train.iloc[val_idx])
        outer_oof[val_idx] = val_score
        folds.append({"fit_idx": fit_idx, "val_idx": val_idx, "fit_score": fit_score, "val_score": val_score})

    train_score_full = inner_oof_score(x_train, target, groups, kind=kind, seed=seed + 7000)
    final_model = score_model(kind, seed + 9000)
    final_model.fit(x_train, target)
    held_score = final_model.predict(x_held)
    return {
        "folds": folds,
        "outer_oof": outer_oof,
        "train_score_full": train_score_full,
        "held_score": held_score,
        "outer_corr_target": float(np.corrcoef(outer_oof, target)[0, 1]),
        "full_oof_corr_target": float(np.corrcoef(train_score_full, target)[0, 1]),
    }


def bins_from_fit(fit_score: np.ndarray, apply_score: np.ndarray, k: int, prefix: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    edges = np.unique(np.quantile(fit_score, np.linspace(0, 1, k + 1)))
    if len(edges) <= 2:
        fit_bins = np.array([f"{prefix}_all"] * len(fit_score), dtype=object)
        apply_bins = np.array([f"{prefix}_all"] * len(apply_score), dtype=object)
    else:
        fit_bins = np.array([f"{prefix}_{b}" for b in np.digitize(fit_score, edges[1:-1], right=True)], dtype=object)
        apply_bins = np.array([f"{prefix}_{b}" for b in np.digitize(apply_score, edges[1:-1], right=True)], dtype=object)
    return fit_bins, apply_bins, [float(x) for x in edges]


def join_arrays(*arrays: np.ndarray) -> np.ndarray:
    out = arrays[0].astype(str)
    for arr in arrays[1:]:
        out = np.array([f"{a}|{b}" for a, b in zip(out, arr.astype(str))], dtype=object)
    return out


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
    best_score = math.inf
    best_gamma = 1.0
    best_cap = math.inf
    for cap in caps:
        c = corr[idx] if math.isinf(cap) else np.clip(corr[idx], -cap, cap)
        for gamma in gammas:
            score = float(np.mean(np.abs(base[idx] + gamma * c - y[idx])))
            if score < best_score:
                best_score = score
                best_gamma = float(gamma)
                best_cap = float(cap)
    return best_gamma, best_cap


def pred_gamma_cap(base: np.ndarray, corr: np.ndarray, gamma: float, cap: float) -> np.ndarray:
    c = corr if math.isinf(cap) else np.clip(corr, -cap, cap)
    return base + gamma * c


def eval_static_category(
    train: pd.DataFrame,
    held: pd.DataFrame,
    cat_train: pd.Series,
    cat_held: pd.Series,
    *,
    name: str,
    gammas: np.ndarray,
    caps: list[float],
    min_n: int,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    corr = train["correction"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)

    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat = cat_train.iloc[fit_idx].astype(str).to_numpy()
        val_cat = cat_train.iloc[val_idx].astype(str).to_numpy()
        global_gamma, global_cap = tune_gamma_cap(y, base, corr, fit_idx, gammas, caps)
        params: dict[str, tuple[float, float]] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) >= min_n:
                params[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)
        for pos, cat in enumerate(val_cat):
            gamma, cap = params.get(str(cat), (global_gamma, global_cap))
            row = val_idx[pos]
            oof[row] = pred_gamma_cap(base[row : row + 1], corr[row : row + 1], gamma, cap)[0]

    cat_all = cat_train.astype(str).to_numpy()
    global_gamma, global_cap = tune_gamma_cap(y, base, corr, np.arange(len(y)), gammas, caps)
    params_full: dict[str, tuple[float, float]] = {"__GLOBAL__": (global_gamma, global_cap)}
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            params_full[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)

    held_base = held["base"].to_numpy(float)
    held_corr = held["correction"].to_numpy(float)
    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(cat_held.astype(str).to_numpy()):
        gamma, cap = params_full.get(str(cat), (global_gamma, global_cap))
        held_pred[i] = pred_gamma_cap(held_base[i : i + 1], held_corr[i : i + 1], gamma, cap)[0]

    return name, oof, held_pred, {
        "selection": "outer GroupKFold category tuning",
        "min_n": min_n,
        "n_params": len(params_full),
        "params": {key: {"gamma": val[0], "cap": val[1]} for key, val in params_full.items()},
    }


def eval_nested_score_category(
    train: pd.DataFrame,
    held: pd.DataFrame,
    score_pack: dict[str, Any],
    k: int,
    extra_train: pd.Series | None,
    extra_held: pd.Series | None,
    *,
    name: str,
    prefix: str,
    gammas: np.ndarray,
    caps: list[float],
    min_n: int,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    corr = train["correction"].to_numpy(float)
    oof = np.zeros(len(train), dtype=float)
    edges_full: list[float] = []

    for fold_info in score_pack["folds"]:
        fit_idx = fold_info["fit_idx"]
        val_idx = fold_info["val_idx"]
        fit_bins, val_bins, _ = bins_from_fit(fold_info["fit_score"], fold_info["val_score"], k, prefix)
        if extra_train is not None:
            fit_cat = join_arrays(fit_bins, extra_train.iloc[fit_idx].astype(str).to_numpy())
            val_cat = join_arrays(val_bins, extra_train.iloc[val_idx].astype(str).to_numpy())
        else:
            fit_cat = fit_bins
            val_cat = val_bins

        global_gamma, global_cap = tune_gamma_cap(y, base, corr, fit_idx, gammas, caps)
        params: dict[str, tuple[float, float]] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) >= min_n:
                params[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)
        for pos, cat in enumerate(val_cat):
            gamma, cap = params.get(str(cat), (global_gamma, global_cap))
            row = val_idx[pos]
            oof[row] = pred_gamma_cap(base[row : row + 1], corr[row : row + 1], gamma, cap)[0]

    train_bins, held_bins, edges_full = bins_from_fit(score_pack["train_score_full"], score_pack["held_score"], k, prefix)
    if extra_train is not None and extra_held is not None:
        train_cat = join_arrays(train_bins, extra_train.astype(str).to_numpy())
        held_cat = join_arrays(held_bins, extra_held.astype(str).to_numpy())
    else:
        train_cat = train_bins
        held_cat = held_bins

    cat_all = train_cat.astype(str)
    global_gamma, global_cap = tune_gamma_cap(y, base, corr, np.arange(len(y)), gammas, caps)
    params_full: dict[str, tuple[float, float]] = {"__GLOBAL__": (global_gamma, global_cap)}
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) >= min_n:
            params_full[str(cat)] = tune_gamma_cap(y, base, corr, idx, gammas, caps)

    held_base = held["base"].to_numpy(float)
    held_corr = held["correction"].to_numpy(float)
    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(held_cat.astype(str)):
        gamma, cap = params_full.get(str(cat), (global_gamma, global_cap))
        held_pred[i] = pred_gamma_cap(held_base[i : i + 1], held_corr[i : i + 1], gamma, cap)[0]

    return name, oof, held_pred, {
        "selection": "strict nested score bins + outer GroupKFold category tuning",
        "k": k,
        "prefix": prefix,
        "min_n": min_n,
        "n_params": len(params_full),
        "score_outer_corr_target": score_pack["outer_corr_target"],
        "score_full_oof_corr_target": score_pack["full_oof_corr_target"],
        "edges": edges_full,
        "params": {key: {"gamma": val[0], "cap": val[1]} for key, val in params_full.items()},
    }


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


def tune_two_way_blend(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, np.ndarray]:
    grid = np.linspace(0, 1, 101)
    scores = []
    for w in grid:
        pred = (1 - w) * pred_a + w * pred_b
        scores.append((float(w), float(mean_absolute_error(y, pred))))
    best_w = min(scores, key=lambda t: t[1])[0]
    return best_w, (1 - best_w) * pred_a + best_w * pred_b


def fit_linear_stack(y: np.ndarray, train_preds: pd.DataFrame, held_preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = LinearRegression(positive=True)
    model.fit(train_preds, y)
    oof_pred = model.predict(train_preds)
    held_pred = model.predict(held_preds)
    return oof_pred, held_pred, {"selection": "positive linear stack on OOF predictions", "columns": list(train_preds.columns), "coef": model.coef_.tolist(), "intercept": float(model.intercept_)}


def paired_ci(diff: np.ndarray, seed: int = 20260823) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    mod = load_base_module()
    mod.assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, held = mod.read_frames()
    x_train, x_held = mod.design_matrix(train, held)
    for frame in (train, held):
        frame["abs_pls_correction"] = frame["correction"].abs()
        frame["corr_size_bin"] = pd.cut(
            frame["abs_pls_correction"],
            bins=[-np.inf, 5.0, 10.0, 20.0, np.inf],
            labels=["corr_0_5", "corr_5_10", "corr_10_20", "corr_20_plus"],
        ).astype(str)

    y = train["case_actual_duration_time"].to_numpy(float)
    base = train["base"].to_numpy(float)
    final = train["final"].to_numpy(float)
    gammas_pos = np.arange(0.0, 2.501, 0.05)
    caps = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, math.inf]

    rows: list[dict[str, Any]] = []
    train_preds: dict[str, np.ndarray] = {"final": train["final"].to_numpy(float)}
    held_preds: dict[str, np.ndarray] = {"final": held["final"].to_numpy(float)}

    def add_variant(name: str, oof_pred: np.ndarray, held_pred: np.ndarray, meta: dict[str, Any]) -> None:
        rows.append(metric_row(name, train, held, oof_pred, held_pred, meta))
        train_preds[name] = oof_pred
        held_preds[name] = held_pred

    # Prior clean uncapped family gate for reference.
    oof_pred, held_pred, gammas = mod.category_gamma_oof(
        train,
        held,
        train["procedure_family"],
        held["procedure_family"],
        grid=np.arange(0.0, 2.501, 0.025),
        min_n=80,
    )
    add_variant("procedure_family_gamma_positive", oof_pred, held_pred, {"selection": "outer GroupKFold category tuning", "gammas": gammas})

    for name, oof_pred, held_pred, meta in [
        eval_static_category(train, held, train["procedure_family"], held["procedure_family"], name="family_cap_gamma", gammas=gammas_pos, caps=caps, min_n=80),
        eval_static_category(
            train,
            held,
            train["procedure_family"].astype(str) + "|" + train["corr_size_bin"].astype(str),
            held["procedure_family"].astype(str) + "|" + held["corr_size_bin"].astype(str),
            name="family_x_corr_size_cap_gamma",
            gammas=gammas_pos,
            caps=caps,
            min_n=100,
        ),
    ]:
        add_variant(name, oof_pred, held_pred, meta)

    targets = {
        "risk_abs_et": (np.abs(final - y), "extra_trees", [3, 4, 5, 6, 8]),
        "risk_abs_hgb": (np.abs(final - y), "hgb_abs", [4, 5, 6, 8]),
        "tail30_hgb": (np.maximum(np.abs(final - y) - 30.0, 0.0), "hgb_abs", [4, 6, 8]),
        "gain_hgb": (np.abs(base - y) - np.abs(final - y), "hgb_sq", [4, 6, 8]),
        "signed_resid_hgb": (y - final, "hgb_sq", [4, 6, 8]),
    }

    score_packs: dict[str, dict[str, Any]] = {}
    for score_name, (target, kind, k_values) in targets.items():
        score_packs[score_name] = compute_nested_scores(train, held, x_train, x_held, target, kind=kind, seed=13000 + len(score_packs) * 1000)
        for k in k_values:
            specs = [
                (None, None, f"{score_name}{k}_cap_gamma", 80),
                (train["procedure_family"], held["procedure_family"], f"{score_name}{k}_x_family_cap_gamma", 120),
                (
                    train["procedure_family"].astype(str) + "|" + train["corr_size_bin"].astype(str),
                    held["procedure_family"].astype(str) + "|" + held["corr_size_bin"].astype(str),
                    f"{score_name}{k}_x_family_x_corr_size_cap_gamma",
                    140,
                ),
            ]
            if score_name in {"risk_abs_et", "tail30_hgb", "signed_resid_hgb"} and k in {4, 6}:
                specs.append((train["procedure_family"].astype(str) + "|" + train["pls_gap_bin"].astype(str), held["procedure_family"].astype(str) + "|" + held["pls_gap_bin"].astype(str), f"{score_name}{k}_x_family_x_pls_gap_cap_gamma", 160))
            for extra_tr, extra_te, variant_name, min_n in specs:
                name, oof_pred, held_pred, meta = eval_nested_score_category(
                    train,
                    held,
                    score_packs[score_name],
                    k,
                    extra_tr,
                    extra_te,
                    name=variant_name,
                    prefix=f"{score_name}{k}",
                    gammas=gammas_pos,
                    caps=caps,
                    min_n=min_n,
                )
                add_variant(name, oof_pred, held_pred, meta)

    # Conservative OOF-selected blends among the safest interpretable candidates.
    y_train = train["case_actual_duration_time"].to_numpy(float)
    blend_pairs = [
        ("procedure_family_gamma_positive", "risk_abs_et4_x_family_cap_gamma"),
        ("procedure_family_gamma_positive", "risk_abs_et6_x_family_cap_gamma"),
        ("procedure_family_gamma_positive", "tail30_hgb6_x_family_cap_gamma"),
        ("procedure_family_gamma_positive", "signed_resid_hgb6_x_family_cap_gamma"),
    ]
    for a, b in blend_pairs:
        if a in train_preds and b in train_preds:
            w, oof_blend = tune_two_way_blend(y_train, train_preds[a], train_preds[b])
            held_blend = (1 - w) * held_preds[a] + w * held_preds[b]
            add_variant(f"blend_{a}__{b}", oof_blend, held_blend, {"selection": "OOF-tuned two-way convex blend", "weight_on_second": w, "first": a, "second": b})

    stack_cols = [c for c in ["final", "procedure_family_gamma_positive", "family_cap_gamma", "risk_abs_et4_x_family_cap_gamma", "risk_abs_et6_x_family_cap_gamma", "tail30_hgb6_x_family_cap_gamma", "signed_resid_hgb6_x_family_cap_gamma"] if c in train_preds]
    if len(stack_cols) >= 3:
        tr_mat = pd.DataFrame({c: train_preds[c] for c in stack_cols})
        te_mat = pd.DataFrame({c: held_preds[c] for c in stack_cols})
        oof_stack, held_stack, meta = fit_linear_stack(y_train, tr_mat, te_mat)
        add_variant("positive_linear_stack_selected_gates", oof_stack, held_stack, meta)

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "strict_k_sweep_results.csv", index=False)

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
            "corr_size_bin",
        ]
    ].copy()
    for name, pred in held_preds.items():
        held_out[f"pred_{name}"] = pred
    held_out.to_csv(OUT_DIR / "strict_k_sweep_test_predictions.csv", index=False)

    final_held = held["final"].to_numpy(float)
    y_held = held["case_actual_duration_time"].to_numpy(float)
    best_by_train = results.iloc[0].to_dict()
    best_by_held = results.sort_values("heldout_mae").iloc[0].to_dict()
    for row in (best_by_train, best_by_held):
        pred = held_out[f"pred_{row['variant']}"].to_numpy(float)
        diff = np.abs(final_held - y_held) - np.abs(pred - y_held)
        row["heldout_gain_ci95"] = paired_ci(diff)
        row["large_wins_gt10"] = int((diff > 10).sum())
        row["large_losses_ltminus10"] = int((diff < -10).sum())

    near = results[results["train_oof_mae"] <= results["train_oof_mae"].iloc[0] + 0.01].copy()
    near["n_params"] = near["meta_json"].map(lambda s: json.loads(s).get("n_params", 9999) if isinstance(s, str) else 9999)
    complexity_selected = near.sort_values(["n_params", "train_oof_mae"]).iloc[0].to_dict()
    pred = held_out[f"pred_{complexity_selected['variant']}"].to_numpy(float)
    diff = np.abs(final_held - y_held) - np.abs(pred - y_held)
    complexity_selected["heldout_gain_ci95"] = paired_ci(diff)

    summary = {
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "final_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train["final"])),
        "final_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held["final"])),
        "patient_overlap": int(len(set(train["patient_durable_id"].astype(str)) & set(held["patient_durable_id"].astype(str)))),
        "best_by_train_oof": best_by_train,
        "best_by_heldout_exploratory": best_by_held,
        "complexity_selected_within_0_01_oof": complexity_selected,
        "score_pack_correlations": {k: {"outer_corr_target": v["outer_corr_target"], "full_oof_corr_target": v["full_oof_corr_target"]} for k, v in score_packs.items()},
        "forbidden_llm_feature_file_read": False,
        "note": "Nested score categories are fit without each outer validation fold. Held-out target is scoring only.",
    }
    (OUT_DIR / "strict_k_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(35).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(20).to_string(index=False))
    print("\nScore correlations:")
    print(json.dumps(summary["score_pack_correlations"], indent=2))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

