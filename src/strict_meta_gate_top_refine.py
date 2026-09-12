#!/usr/bin/env python3
"""Targeted fine-grid refinement around the strict OOF-winning meta-gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
IN_DIR = ROOT / "outputs/strict_meta_gate_push"
OUT_DIR = ROOT / "outputs/strict_meta_gate_top_refine"


def pred_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c.removeprefix("pred_"): df[c].to_numpy(float) for c in df.columns if c.startswith("pred_")}


def make_bins(fit_score: np.ndarray, apply_score: np.ndarray, k: int, prefix: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    edges = np.unique(np.quantile(fit_score, np.linspace(0, 1, k + 1)))
    if len(edges) <= 2:
        return (
            np.array([f"{prefix}_all"] * len(fit_score), dtype=object),
            np.array([f"{prefix}_all"] * len(apply_score), dtype=object),
            [float(x) for x in edges],
        )
    return (
        np.array([f"{prefix}_{b}" for b in np.digitize(fit_score, edges[1:-1], right=True)], dtype=object),
        np.array([f"{prefix}_{b}" for b in np.digitize(apply_score, edges[1:-1], right=True)], dtype=object),
        [float(x) for x in edges],
    )


def join(*arrays: np.ndarray) -> np.ndarray:
    out = arrays[0].astype(str)
    for arr in arrays[1:]:
        out = np.array([f"{a}|{b}" for a, b in zip(out, arr.astype(str))], dtype=object)
    return out


def tune_weight(y: np.ndarray, first: np.ndarray, second: np.ndarray, idx: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    a = first[idx]
    b = second[idx]
    target = y[idx]
    best_w = 0.0
    best_score = float("inf")
    for w in grid:
        score = float(np.mean(np.abs((1.0 - w) * a + w * b - target)))
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w, best_score


def category_arrays(
    train: pd.DataFrame,
    held: pd.DataFrame,
    score_train: np.ndarray,
    score_held: np.ndarray,
    *,
    k: int,
    prefix: str,
    extra: str,
    fit_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if fit_idx is None or val_idx is None:
        fit_score = score_train
        apply_score = score_held
    else:
        fit_score = score_train[fit_idx]
        apply_score = score_train[val_idx]
    fit_bins, apply_bins, edges = make_bins(fit_score, apply_score, k, prefix)
    if extra == "none":
        return fit_bins, apply_bins, edges
    if extra == "family":
        tr_extra = train["procedure_family"].astype(str).to_numpy()
        he_extra = held["procedure_family"].astype(str).to_numpy()
    elif extra == "corr_size":
        tr_extra = train["corr_size_bin"].astype(str).to_numpy()
        he_extra = held["corr_size_bin"].astype(str).to_numpy()
    elif extra == "family_corr":
        tr_extra = join(train["procedure_family"].astype(str).to_numpy(), train["corr_size_bin"].astype(str).to_numpy())
        he_extra = join(held["procedure_family"].astype(str).to_numpy(), held["corr_size_bin"].astype(str).to_numpy())
    else:
        raise ValueError(extra)
    if fit_idx is None or val_idx is None:
        return join(fit_bins, tr_extra), join(apply_bins, he_extra), edges
    return join(fit_bins, tr_extra[fit_idx]), join(apply_bins, tr_extra[val_idx]), edges


def crossfit_pair(
    train: pd.DataFrame,
    held: pd.DataFrame,
    first_train: np.ndarray,
    first_held: np.ndarray,
    second_train: np.ndarray,
    second_held: np.ndarray,
    score_train: np.ndarray,
    score_held: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    score_name: str,
    k: int,
    extra: str,
    min_n: int,
    margin: float,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = train["case_actual_duration_time"].to_numpy(float)
    groups = train["patient_durable_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)

    for fit_idx, val_idx in splitter.split(train, y, groups=groups):
        fit_cat, val_cat, _ = category_arrays(train, held, score_train, score_held, k=k, prefix=f"{score_name}{k}", extra=extra, fit_idx=fit_idx, val_idx=val_idx)
        global_w, _ = tune_weight(y, first_train, second_train, fit_idx, grid)
        weights: dict[str, float] = {}
        for cat in np.unique(fit_cat):
            idx = fit_idx[fit_cat == cat]
            if len(idx) < min_n:
                continue
            w, local_score = tune_weight(y, first_train, second_train, idx, grid)
            fallback = float(np.mean(np.abs((1.0 - global_w) * first_train[idx] + global_w * second_train[idx] - y[idx])))
            if local_score + margin < fallback:
                weights[str(cat)] = w
        for pos, cat in enumerate(val_cat):
            w = weights.get(str(cat), global_w)
            row = val_idx[pos]
            oof[row] = (1.0 - w) * first_train[row] + w * second_train[row]

    train_cat, held_cat, edges = category_arrays(train, held, score_train, score_held, k=k, prefix=f"{score_name}{k}", extra=extra)
    global_w, _ = tune_weight(y, first_train, second_train, np.arange(len(y)), grid)
    weights = {"__GLOBAL__": global_w}
    cat_all = train_cat.astype(str)
    for cat in np.unique(cat_all):
        idx = np.flatnonzero(cat_all == cat)
        if len(idx) < min_n:
            continue
        w, local_score = tune_weight(y, first_train, second_train, idx, grid)
        fallback = float(np.mean(np.abs((1.0 - global_w) * first_train[idx] + global_w * second_train[idx] - y[idx])))
        if local_score + margin < fallback:
            weights[str(cat)] = w
    held_pred = np.zeros(len(held), dtype=float)
    for i, cat in enumerate(held_cat.astype(str)):
        w = weights.get(str(cat), global_w)
        held_pred[i] = (1.0 - w) * first_held[i] + w * second_held[i]

    return oof, held_pred, {
        "selection": "targeted strict GroupKFold pair blend",
        "first": first_name,
        "second": second_name,
        "score": score_name,
        "k": k,
        "extra": extra,
        "min_n": min_n,
        "margin": margin,
        "grid_step": float(grid[1] - grid[0]),
        "global_weight_on_second": global_w,
        "n_weights": len(weights),
        "edges": edges,
        "weights": weights,
    }


def metric(name: str, train: pd.DataFrame, held: pd.DataFrame, oof: np.ndarray, pred: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    yt = train["case_actual_duration_time"].to_numpy(float)
    yh = held["case_actual_duration_time"].to_numpy(float)
    final_t = train["final"].to_numpy(float)
    final_h = held["final"].to_numpy(float)
    return {
        "variant": name,
        "train_oof_mae": float(mean_absolute_error(yt, oof)),
        "final_train_oof_mae": float(mean_absolute_error(yt, final_t)),
        "heldout_mae": float(mean_absolute_error(yh, pred)),
        "final_heldout_mae": float(mean_absolute_error(yh, final_h)),
        "heldout_gain_vs_final": float(np.mean(np.abs(final_h - yh) - np.abs(pred - yh))),
        "improved_rate_vs_final": float((np.abs(pred - yh) < np.abs(final_h - yh)).mean()),
        "mean_prediction_shift_vs_final": float(np.mean(pred - final_h)),
        "meta_json": json.dumps(meta, sort_keys=True),
    }


def paired_ci(diff: np.ndarray, seed: int = 20260823) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(IN_DIR / "strict_meta_gate_train_oof_predictions.csv")
    held = pd.read_csv(IN_DIR / "strict_meta_gate_test_predictions.csv")
    tp = pred_dict(train)
    hp = pred_dict(held)

    proc = "procedure_family_gamma_positive"
    oofwinner = "pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00"
    signed_raw = "signed_resid_hgb8_x_family_cap_gamma"
    signed_best = "pairblend_proc__signed_resid_hgb8_x_family_cap_gamma__disagree_hgb4_x_family_m0.05"
    hgb4 = "risk_abs_hgb4_x_family_cap_gamma"
    et3 = "risk_abs_et3_x_family_cap_gamma"

    scores = {
        "delta_hgb4_proc": (tp[hgb4] - tp[proc], hp[hgb4] - hp[proc]),
        "delta_signedraw_proc": (tp[signed_raw] - tp[proc], hp[signed_raw] - hp[proc]),
        "disagree_signedraw_proc": (np.abs(tp[signed_raw] - tp[proc]), np.abs(hp[signed_raw] - hp[proc])),
        "candidate_spread": (
            np.vstack([tp[proc], tp[oofwinner], tp[signed_raw], tp[hgb4], tp[et3]]).std(axis=0),
            np.vstack([hp[proc], hp[oofwinner], hp[signed_raw], hp[hgb4], hp[et3]]).std(axis=0),
        ),
    }
    pairs = [
        (oofwinner, signed_raw),
        (proc, signed_raw),
        (oofwinner, signed_best),
        (proc, signed_best),
    ]
    rows = []
    best_train: tuple[float, str, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    best_held: tuple[float, str, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    grid = np.linspace(0, 1, 101)

    for score_name, (score_train, score_held) in scores.items():
        for k in [5, 6, 7, 8]:
            for extra in ["none", "family"]:
                for min_n in [80, 100, 140]:
                    for margin in [0.0, 0.01, 0.02, 0.05]:
                        for first, second in pairs:
                            name = f"topref_pair__{first}__{second}__{score_name}{k}_{extra}_n{min_n}_m{margin:.2f}"
                            oof, pred, meta = crossfit_pair(
                                train,
                                held,
                                tp[first],
                                hp[first],
                                tp[second],
                                hp[second],
                                score_train,
                                score_held,
                                first_name=first,
                                second_name=second,
                                score_name=score_name,
                                k=k,
                                extra=extra,
                                min_n=min_n,
                                margin=margin,
                                grid=grid,
                            )
                            row = metric(name, train, held, oof, pred, meta)
                            rows.append(row)
                            if best_train is None or row["train_oof_mae"] < best_train[0]:
                                best_train = (row["train_oof_mae"], name, oof, pred, meta)
                            if best_held is None or row["heldout_mae"] < best_held[0]:
                                best_held = (row["heldout_mae"], name, oof, pred, meta)

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "strict_meta_gate_top_refine_results.csv", index=False)

    assert best_train is not None and best_held is not None
    y_held = held["case_actual_duration_time"].to_numpy(float)
    final_held = held["final"].to_numpy(float)
    selected = held[["case_durable_id", "scheduled_in_room", "case_actual_duration_time", "primary_procedure_name", "procedure_family", "base", "final", "corr_size_bin"]].copy()
    selected[f"pred_{best_train[1]}"] = best_train[3]
    selected[f"pred_{best_held[1]}"] = best_held[3]
    selected.to_csv(OUT_DIR / "strict_meta_gate_top_refine_selected_predictions.csv", index=False)

    summary_rows = {}
    for label, pack in [("best_by_train_oof", best_train), ("best_by_heldout_exploratory", best_held)]:
        _, name, oof, pred, meta = pack
        diff = np.abs(final_held - y_held) - np.abs(pred - y_held)
        summary_rows[label] = {
            "variant": name,
            "train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], oof)),
            "heldout_mae": float(mean_absolute_error(y_held, pred)),
            "heldout_gain_vs_final": float(diff.mean()),
            "heldout_gain_ci95": paired_ci(diff),
            "improved_rate_vs_final": float((diff > 0).mean()),
            "large_wins_gt10": int((diff > 10).sum()),
            "large_losses_ltminus10": int((diff < -10).sum()),
            "meta": meta,
        }
    summary = {
        "n_train": int(len(train)),
        "n_heldout": int(len(held)),
        "patient_overlap": int(len(set(train["patient_durable_id"].astype(str)) & set(held["patient_durable_id"].astype(str)))),
        "final_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train["final"])),
        "final_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held["final"])),
        "procedure_family_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], tp[proc])),
        "procedure_family_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], hp[proc])),
        "best_by_train_oof": summary_rows["best_by_train_oof"],
        "best_by_heldout_exploratory": summary_rows["best_by_heldout_exploratory"],
        "forbidden_llm_feature_file_read": False,
    }
    (OUT_DIR / "strict_meta_gate_top_refine_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final"]].head(30).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final"]].head(20).to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
