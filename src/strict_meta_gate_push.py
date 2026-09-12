#!/usr/bin/env python3
"""Second-level strict meta-gates for the clean PLS manifold candidates.

Goal: try to improve both train OOF and held-out by letting a nested meta-gate
choose or blend between the safe procedure-family calibration and selected
risk-family candidates.

No old 11 LLM feature file is read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
KSWEEP_SCRIPT = ROOT / "work/no_llm_notes/strict_risk_family_k_sweep.py"
OUT_DIR = ROOT / "outputs/strict_meta_gate_push"


def load_ksweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("strict_risk_family_k_sweep", KSWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {KSWEEP_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bin_scores(fit_score: np.ndarray, apply_score: np.ndarray, k: int, prefix: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
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


def get_categories(
    spec: dict[str, Any],
    train: pd.DataFrame,
    held: pd.DataFrame,
    fit_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if spec["kind"] == "static":
        train_arr = spec["train"].astype(str).to_numpy()
        held_arr = spec["held"].astype(str).to_numpy()
        if fit_idx is None or val_idx is None:
            return train_arr, held_arr, []
        return train_arr[fit_idx], train_arr[val_idx], []

    if spec["kind"] != "score":
        raise ValueError(spec["kind"])

    score_train = spec["score_train"]
    score_apply = spec["score_held"] if fit_idx is None or val_idx is None else score_train[val_idx]
    score_fit = score_train if fit_idx is None or val_idx is None else score_train[fit_idx]
    base_bins, apply_bins, edges = bin_scores(score_fit, score_apply, spec["k"], spec["prefix"])

    extras_fit: list[np.ndarray] = []
    extras_apply: list[np.ndarray] = []
    for extra_name in spec.get("extras", []):
        if extra_name == "family":
            train_extra = train["procedure_family"].astype(str).to_numpy()
            held_extra = held["procedure_family"].astype(str).to_numpy()
        elif extra_name == "corr_size":
            train_extra = train["corr_size_bin"].astype(str).to_numpy()
            held_extra = held["corr_size_bin"].astype(str).to_numpy()
        elif extra_name == "pls_gap":
            train_extra = train["pls_gap_bin"].astype(str).to_numpy()
            held_extra = held["pls_gap_bin"].astype(str).to_numpy()
        else:
            raise ValueError(extra_name)
        if fit_idx is None or val_idx is None:
            extras_fit.append(train_extra)
            extras_apply.append(held_extra)
        else:
            extras_fit.append(train_extra[fit_idx])
            extras_apply.append(train_extra[val_idx])

    if extras_fit:
        return join_arrays(base_bins, *extras_fit), join_arrays(apply_bins, *extras_apply), edges
    return base_bins, apply_bins, edges


def crossfit_candidate_selector(
    train: pd.DataFrame,
    held: pd.DataFrame,
    train_preds: dict[str, np.ndarray],
    held_preds: dict[str, np.ndarray],
    candidates: list[str],
    cat_spec: dict[str, Any],
    *,
    min_n: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)

    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat, val_cat, _ = get_categories(cat_spec, train, held, fit_idx, val_idx)
        global_scores = {c: float(mean_absolute_error(y[fit_idx], train_preds[c][fit_idx])) for c in candidates}
        global_best = min(global_scores, key=global_scores.get)
        choices: dict[str, str] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) < min_n:
                continue
            local_scores = {c: float(mean_absolute_error(y[idx], train_preds[c][idx])) for c in candidates}
            local_best = min(local_scores, key=local_scores.get)
            if local_scores[local_best] + margin < local_scores.get(global_best, global_scores[global_best]):
                choices[str(cat)] = local_best
        for pos, cat in enumerate(val_cat):
            chosen = choices.get(str(cat), global_best)
            oof[val_idx[pos]] = train_preds[chosen][val_idx[pos]]

    train_cat, held_cat, edges = get_categories(cat_spec, train, held)
    global_scores = {c: float(mean_absolute_error(y, train_preds[c])) for c in candidates}
    global_best = min(global_scores, key=global_scores.get)
    choices = {"__GLOBAL__": global_best}
    cat_all = train_cat.astype(str)
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) < min_n:
            continue
        local_scores = {c: float(mean_absolute_error(y[idx], train_preds[c][idx])) for c in candidates}
        local_best = min(local_scores, key=local_scores.get)
        if local_scores[local_best] + margin < local_scores.get(global_best, global_scores[global_best]):
            choices[str(cat)] = local_best
    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(held_cat.astype(str)):
        chosen = choices.get(str(cat), global_best)
        held_pred[i] = held_preds[chosen][i]

    return oof, held_pred, {
        "selection": "strict second-level GroupKFold candidate selector",
        "category": cat_spec["name"],
        "min_n": min_n,
        "margin": margin,
        "candidates": candidates,
        "global_scores": global_scores,
        "global_best": global_best,
        "n_choices": len(choices),
        "edges": edges,
        "choices": choices,
    }


def tune_pair(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, idx: np.ndarray, grid: np.ndarray) -> float:
    best_w = 0.0
    best_score = float("inf")
    for w in grid:
        pred = (1.0 - w) * pred_a[idx] + w * pred_b[idx]
        score = float(np.mean(np.abs(pred - y[idx])))
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w


def crossfit_pair_blend(
    train: pd.DataFrame,
    held: pd.DataFrame,
    train_preds: dict[str, np.ndarray],
    held_preds: dict[str, np.ndarray],
    first: str,
    second: str,
    cat_spec: dict[str, Any],
    *,
    min_n: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    grid = np.linspace(0, 1, 21)
    oof = np.zeros(len(train), dtype=float)

    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat, val_cat, _ = get_categories(cat_spec, train, held, fit_idx, val_idx)
        global_w = tune_pair(y, train_preds[first], train_preds[second], fit_idx, grid)
        global_score = float(np.mean(np.abs((1.0 - global_w) * train_preds[first][fit_idx] + global_w * train_preds[second][fit_idx] - y[fit_idx])))
        weights: dict[str, float] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) < min_n:
                continue
            w = tune_pair(y, train_preds[first], train_preds[second], idx, grid)
            local_score = float(np.mean(np.abs((1.0 - w) * train_preds[first][idx] + w * train_preds[second][idx] - y[idx])))
            fallback_score = float(np.mean(np.abs((1.0 - global_w) * train_preds[first][idx] + global_w * train_preds[second][idx] - y[idx])))
            if local_score + margin < fallback_score:
                weights[str(cat)] = w
        for pos, cat in enumerate(val_cat):
            w = weights.get(str(cat), global_w)
            row = val_idx[pos]
            oof[row] = (1.0 - w) * train_preds[first][row] + w * train_preds[second][row]

    train_cat, held_cat, edges = get_categories(cat_spec, train, held)
    global_w = tune_pair(y, train_preds[first], train_preds[second], np.arange(len(y)), grid)
    weights = {"__GLOBAL__": global_w}
    cat_all = train_cat.astype(str)
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) < min_n:
            continue
        w = tune_pair(y, train_preds[first], train_preds[second], idx, grid)
        local_score = float(np.mean(np.abs((1.0 - w) * train_preds[first][idx] + w * train_preds[second][idx] - y[idx])))
        fallback_score = float(np.mean(np.abs((1.0 - global_w) * train_preds[first][idx] + global_w * train_preds[second][idx] - y[idx])))
        if local_score + margin < fallback_score:
            weights[str(cat)] = w

    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(held_cat.astype(str)):
        w = weights.get(str(cat), global_w)
        held_pred[i] = (1.0 - w) * held_preds[first][i] + w * held_preds[second][i]

    return oof, held_pred, {
        "selection": "strict second-level GroupKFold pair blend",
        "category": cat_spec["name"],
        "first": first,
        "second": second,
        "min_n": min_n,
        "margin": margin,
        "global_weight_on_second": global_w,
        "n_weights": len(weights),
        "edges": edges,
        "weights": weights,
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


def paired_ci(diff: np.ndarray, seed: int = 20260823) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    ks = load_ksweep_module()
    base_mod = ks.load_base_module()
    base_mod.assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, held = base_mod.read_frames()
    x_train, x_held = base_mod.design_matrix(train, held)
    for frame in (train, held):
        frame["abs_pls_correction"] = frame["correction"].abs()
        frame["corr_size_bin"] = pd.cut(
            frame["abs_pls_correction"],
            bins=[-np.inf, 5.0, 10.0, 20.0, np.inf],
            labels=["corr_0_5", "corr_5_10", "corr_10_20", "corr_20_plus"],
        ).astype(str)

    y = train["case_actual_duration_time"].to_numpy(float)
    final = train["final"].to_numpy(float)
    base = train["base"].to_numpy(float)
    train_preds: dict[str, np.ndarray] = {"final": train["final"].to_numpy(float)}
    held_preds: dict[str, np.ndarray] = {"final": held["final"].to_numpy(float)}
    rows: list[dict[str, Any]] = []

    def add(name: str, oof_pred: np.ndarray, held_pred: np.ndarray, meta: dict[str, Any]) -> None:
        train_preds[name] = oof_pred
        held_preds[name] = held_pred
        rows.append(metric_row(name, train, held, oof_pred, held_pred, meta))

    oof_pred, held_pred, gammas = base_mod.category_gamma_oof(
        train,
        held,
        train["procedure_family"],
        held["procedure_family"],
        grid=np.arange(0.0, 2.501, 0.025),
        min_n=80,
    )
    add("procedure_family_gamma_positive", oof_pred, held_pred, {"selection": "outer GroupKFold category tuning", "gammas": gammas})

    for name, oof_pred, held_pred, meta in [
        ks.eval_static_category(
            train,
            held,
            train["procedure_family"],
            held["procedure_family"],
            name="family_cap_gamma",
            gammas=np.arange(0.0, 2.501, 0.05),
            caps=[4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, np.inf],
            min_n=80,
        )
    ]:
        add(name, oof_pred, held_pred, meta)

    target_risk = np.abs(final - y)
    target_signed = y - final
    risk_hgb_pack = ks.compute_nested_scores(train, held, x_train, x_held, target_risk, kind="hgb_abs", seed=21000)
    risk_et_pack = ks.compute_nested_scores(train, held, x_train, x_held, target_risk, kind="extra_trees", seed=22000)
    signed_pack = ks.compute_nested_scores(train, held, x_train, x_held, target_signed, kind="hgb_sq", seed=23000)

    candidate_specs = [
        ("risk_abs_hgb4_x_family_cap_gamma", risk_hgb_pack, 4, "risk_abs_hgb4", train["procedure_family"], held["procedure_family"], 120),
        ("risk_abs_hgb6_x_family_cap_gamma", risk_hgb_pack, 6, "risk_abs_hgb6", train["procedure_family"], held["procedure_family"], 120),
        ("risk_abs_et3_x_family_cap_gamma", risk_et_pack, 3, "risk_abs_et3", train["procedure_family"], held["procedure_family"], 120),
        ("risk_abs_et4_x_family_cap_gamma", risk_et_pack, 4, "risk_abs_et4", train["procedure_family"], held["procedure_family"], 120),
        ("signed_resid_hgb8_x_family_cap_gamma", signed_pack, 8, "signed_resid_hgb8", train["procedure_family"], held["procedure_family"], 120),
    ]
    for name, pack, k, prefix, extra_tr, extra_te, min_n in candidate_specs:
        vname, oof_pred, held_pred, meta = ks.eval_nested_score_category(
            train,
            held,
            pack,
            k,
            extra_tr,
            extra_te,
            name=name,
            prefix=prefix,
            gammas=np.arange(0.0, 2.501, 0.05),
            caps=[4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, np.inf],
            min_n=min_n,
        )
        add(vname, oof_pred, held_pred, meta)

    disagree_hgb4_train = np.abs(train_preds["risk_abs_hgb4_x_family_cap_gamma"] - train_preds["procedure_family_gamma_positive"])
    disagree_hgb4_held = np.abs(held_preds["risk_abs_hgb4_x_family_cap_gamma"] - held_preds["procedure_family_gamma_positive"])
    cat_specs = [
        {"name": "family", "kind": "static", "train": train["procedure_family"], "held": held["procedure_family"]},
        {"name": "risk_hgb4", "kind": "score", "score_train": risk_hgb_pack["outer_oof"], "score_held": risk_hgb_pack["held_score"], "k": 4, "prefix": "risk_hgb4"},
        {"name": "risk_hgb4_x_family", "kind": "score", "score_train": risk_hgb_pack["outer_oof"], "score_held": risk_hgb_pack["held_score"], "k": 4, "prefix": "risk_hgb4", "extras": ["family"]},
        {"name": "risk_hgb6_x_family", "kind": "score", "score_train": risk_hgb_pack["outer_oof"], "score_held": risk_hgb_pack["held_score"], "k": 6, "prefix": "risk_hgb6", "extras": ["family"]},
        {"name": "risk_et3_x_family", "kind": "score", "score_train": risk_et_pack["outer_oof"], "score_held": risk_et_pack["held_score"], "k": 3, "prefix": "risk_et3", "extras": ["family"]},
        {"name": "disagree_hgb4_x_family", "kind": "score", "score_train": disagree_hgb4_train, "score_held": disagree_hgb4_held, "k": 4, "prefix": "disagree_hgb4", "extras": ["family"]},
    ]
    candidates = [
        "final",
        "procedure_family_gamma_positive",
        "family_cap_gamma",
        "risk_abs_hgb4_x_family_cap_gamma",
        "risk_abs_hgb6_x_family_cap_gamma",
        "risk_abs_et3_x_family_cap_gamma",
        "risk_abs_et4_x_family_cap_gamma",
        "signed_resid_hgb8_x_family_cap_gamma",
    ]
    for cat_spec in cat_specs:
        for margin in [0.0, 0.05, 0.10]:
            oof_pred, held_pred, meta = crossfit_candidate_selector(
                train,
                held,
                train_preds,
                held_preds,
                candidates,
                cat_spec,
                min_n=120,
                margin=margin,
            )
            add(f"selector_{cat_spec['name']}_m{margin:.2f}", oof_pred, held_pred, meta)
        for second in ["risk_abs_hgb4_x_family_cap_gamma", "risk_abs_et3_x_family_cap_gamma", "signed_resid_hgb8_x_family_cap_gamma"]:
            for margin in [0.0, 0.05]:
                oof_pred, held_pred, meta = crossfit_pair_blend(
                    train,
                    held,
                    train_preds,
                    held_preds,
                    "procedure_family_gamma_positive",
                    second,
                    cat_spec,
                    min_n=120,
                    margin=margin,
                )
                add(f"pairblend_proc__{second}__{cat_spec['name']}_m{margin:.2f}", oof_pred, held_pred, meta)

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "strict_meta_gate_results.csv", index=False)

    train_out = train[["case_durable_id", "patient_durable_id", "case_actual_duration_time", "primary_procedure_name", "procedure_family", "base", "final", "corr_size_bin"]].copy()
    held_out = held[["case_durable_id", "patient_durable_id", "scheduled_in_room", "case_actual_duration_time", "primary_procedure_name", "procedure_family", "base", "final", "corr_size_bin"]].copy()
    for name, pred in train_preds.items():
        train_out[f"pred_{name}"] = pred
    for name, pred in held_preds.items():
        held_out[f"pred_{name}"] = pred
    train_out.to_csv(OUT_DIR / "strict_meta_gate_train_oof_predictions.csv", index=False)
    held_out.to_csv(OUT_DIR / "strict_meta_gate_test_predictions.csv", index=False)

    final_held = held["final"].to_numpy(float)
    y_held = held["case_actual_duration_time"].to_numpy(float)
    best_train = results.iloc[0].to_dict()
    best_held = results.sort_values("heldout_mae").iloc[0].to_dict()
    for row in (best_train, best_held):
        pred = held_out[f"pred_{row['variant']}"].to_numpy(float)
        diff = np.abs(final_held - y_held) - np.abs(pred - y_held)
        row["heldout_gain_ci95"] = paired_ci(diff)
        row["large_wins_gt10"] = int((diff > 10).sum())
        row["large_losses_ltminus10"] = int((diff < -10).sum())

    summary = {
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "patient_overlap": int(len(set(train["patient_durable_id"].astype(str)) & set(held["patient_durable_id"].astype(str)))),
        "final_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train["final"])),
        "final_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held["final"])),
        "best_by_train_oof": best_train,
        "best_by_heldout_exploratory": best_held,
        "score_correlations": {
            "risk_hgb": {"outer": risk_hgb_pack["outer_corr_target"], "full_oof": risk_hgb_pack["full_oof_corr_target"]},
            "risk_et": {"outer": risk_et_pack["outer_corr_target"], "full_oof": risk_et_pack["full_oof_corr_target"]},
            "signed_hgb": {"outer": signed_pack["outer_corr_target"], "full_oof": signed_pack["full_oof_corr_target"]},
        },
        "forbidden_llm_feature_file_read": False,
    }
    (OUT_DIR / "strict_meta_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(40).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(25).to_string(index=False))
    print(json.dumps(summary["score_correlations"], indent=2))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

