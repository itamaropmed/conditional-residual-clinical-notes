#!/usr/bin/env python3
"""Lightweight refinement over saved strict meta-gate OOF predictions.

Uses only saved OOF/test predictions from strict_meta_gate_push and builds a
third-level crossfit selector/blender over prediction disagreement categories.

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


ROOT = Path("/Users/itamarzernitsky/Documents/Codex/2026-08-23/can")
META_SCRIPT = ROOT / "work/no_llm_notes/strict_meta_gate_push.py"
IN_DIR = ROOT / "outputs/strict_meta_gate_push"
OUT_DIR = ROOT / "outputs/strict_meta_gate_refine"


def load_meta_module() -> Any:
    spec = importlib.util.spec_from_file_location("strict_meta_gate_push", META_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {META_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for col in df.columns:
        if col.startswith("pred_"):
            out[col.removeprefix("pred_")] = df[col].to_numpy(float)
    return out


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
    meta_mod = load_meta_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(IN_DIR / "strict_meta_gate_train_oof_predictions.csv")
    held = pd.read_csv(IN_DIR / "strict_meta_gate_test_predictions.csv")
    train_preds = pred_dict(train)
    held_preds = pred_dict(held)

    rows: list[dict[str, Any]] = []
    added_train: dict[str, np.ndarray] = {}
    added_held: dict[str, np.ndarray] = {}

    def add(name: str, oof_pred: np.ndarray, held_pred: np.ndarray, info: dict[str, Any]) -> None:
        rows.append(metric_row(name, train, held, oof_pred, held_pred, info))
        train_preds[name] = oof_pred
        held_preds[name] = held_pred
        added_train[name] = oof_pred
        added_held[name] = held_pred

    core = [
        "final",
        "procedure_family_gamma_positive",
        "family_cap_gamma",
        "risk_abs_hgb4_x_family_cap_gamma",
        "risk_abs_et3_x_family_cap_gamma",
        "signed_resid_hgb8_x_family_cap_gamma",
        "pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00",
        "pairblend_proc__signed_resid_hgb8_x_family_cap_gamma__disagree_hgb4_x_family_m0.05",
        "pairblend_proc__signed_resid_hgb8_x_family_cap_gamma__risk_hgb6_x_family_m0.05",
    ]
    for name in core:
        add(name, train_preds[name], held_preds[name], {"selection": "input saved OOF candidate"})

    proc = "procedure_family_gamma_positive"
    oofwinner = "pairblend_proc__risk_abs_et3_x_family_cap_gamma__risk_hgb4_m0.00"
    signed_best = "pairblend_proc__signed_resid_hgb8_x_family_cap_gamma__disagree_hgb4_x_family_m0.05"
    signed_raw = "signed_resid_hgb8_x_family_cap_gamma"
    hgb4 = "risk_abs_hgb4_x_family_cap_gamma"
    et3 = "risk_abs_et3_x_family_cap_gamma"

    score_arrays = {
        "disagree_signedbest_proc": (
            np.abs(train_preds[signed_best] - train_preds[proc]),
            np.abs(held_preds[signed_best] - held_preds[proc]),
        ),
        "delta_signedbest_proc": (
            train_preds[signed_best] - train_preds[proc],
            held_preds[signed_best] - held_preds[proc],
        ),
        "disagree_signedraw_proc": (
            np.abs(train_preds[signed_raw] - train_preds[proc]),
            np.abs(held_preds[signed_raw] - held_preds[proc]),
        ),
        "delta_signedraw_proc": (
            train_preds[signed_raw] - train_preds[proc],
            held_preds[signed_raw] - held_preds[proc],
        ),
        "disagree_hgb4_proc": (
            np.abs(train_preds[hgb4] - train_preds[proc]),
            np.abs(held_preds[hgb4] - held_preds[proc]),
        ),
        "delta_hgb4_proc": (
            train_preds[hgb4] - train_preds[proc],
            held_preds[hgb4] - held_preds[proc],
        ),
        "candidate_spread": (
            np.vstack([train_preds[proc], train_preds[hgb4], train_preds[et3], train_preds[signed_raw]]).std(axis=0),
            np.vstack([held_preds[proc], held_preds[hgb4], held_preds[et3], held_preds[signed_raw]]).std(axis=0),
        ),
        "abs_correction": (
            np.abs(train["final"].to_numpy(float) - train["base"].to_numpy(float)),
            np.abs(held["final"].to_numpy(float) - held["base"].to_numpy(float)),
        ),
    }

    cat_specs: list[dict[str, Any]] = [
        {"name": "family", "kind": "static", "train": train["procedure_family"], "held": held["procedure_family"]},
    ]
    for score_name, (score_train, score_held) in score_arrays.items():
        for k in [3, 4, 5, 6]:
            cat_specs.append({"name": f"{score_name}{k}", "kind": "score", "score_train": score_train, "score_held": score_held, "k": k, "prefix": f"{score_name}{k}"})
            cat_specs.append({"name": f"{score_name}{k}_x_family", "kind": "score", "score_train": score_train, "score_held": score_held, "k": k, "prefix": f"{score_name}{k}", "extras": ["family"]})

    pair_jobs = [
        (proc, signed_best),
        (proc, signed_raw),
        (proc, hgb4),
        (proc, et3),
        (oofwinner, signed_best),
        (oofwinner, signed_raw),
        (oofwinner, hgb4),
        (oofwinner, et3),
    ]
    for cat_spec in cat_specs:
        # Keep the grid broad, but this is cheap because it uses saved predictions.
        min_n = 100 if "family" in cat_spec["name"] else 80
        for first, second in pair_jobs:
            for margin in [0.0, 0.02, 0.05, 0.10, 0.20]:
                oof_pred, held_pred, info = meta_mod.crossfit_pair_blend(
                    train,
                    held,
                    train_preds,
                    held_preds,
                    first,
                    second,
                    cat_spec,
                    min_n=min_n,
                    margin=margin,
                )
                add(f"refine_pair__{first}__{second}__{cat_spec['name']}_m{margin:.2f}", oof_pred, held_pred, info)

    selector_candidates = [proc, "family_cap_gamma", hgb4, et3, signed_raw, oofwinner, signed_best]
    for cat_spec in cat_specs:
        min_n = 120 if "family" in cat_spec["name"] else 100
        for margin in [0.0, 0.02, 0.05, 0.10, 0.20]:
            oof_pred, held_pred, info = meta_mod.crossfit_candidate_selector(
                train,
                held,
                train_preds,
                held_preds,
                selector_candidates,
                cat_spec,
                min_n=min_n,
                margin=margin,
            )
            add(f"refine_selector__{cat_spec['name']}_m{margin:.2f}", oof_pred, held_pred, info)

    results = pd.DataFrame(rows).sort_values("train_oof_mae").reset_index(drop=True)
    results.to_csv(OUT_DIR / "strict_meta_gate_refine_results.csv", index=False)

    train_out = train[["case_durable_id", "patient_durable_id", "case_actual_duration_time", "primary_procedure_name", "procedure_family", "base", "final", "corr_size_bin"]].copy()
    held_out = held[["case_durable_id", "patient_durable_id", "scheduled_in_room", "case_actual_duration_time", "primary_procedure_name", "procedure_family", "base", "final", "corr_size_bin"]].copy()
    for name, pred in added_train.items():
        train_out[f"pred_{name}"] = pred
    for name, pred in added_held.items():
        held_out[f"pred_{name}"] = pred
    train_out.to_csv(OUT_DIR / "strict_meta_gate_refine_train_oof_predictions.csv", index=False)
    held_out.to_csv(OUT_DIR / "strict_meta_gate_refine_test_predictions.csv", index=False)

    y_held = held["case_actual_duration_time"].to_numpy(float)
    final_held = held["final"].to_numpy(float)
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
        "procedure_family_train_oof_mae": float(mean_absolute_error(train["case_actual_duration_time"], train_preds[proc])),
        "procedure_family_heldout_mae": float(mean_absolute_error(held["case_actual_duration_time"], held_preds[proc])),
        "best_by_train_oof": best_train,
        "best_by_heldout_exploratory": best_held,
        "forbidden_llm_feature_file_read": False,
        "note": "Uses saved OOF predictions only; meta gate is crossfit by patient groups.",
    }
    (OUT_DIR / "strict_meta_gate_refine_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(results[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(40).to_string(index=False))
    print("\nBest heldout exploratory:")
    print(results.sort_values("heldout_mae")[["variant", "train_oof_mae", "heldout_mae", "heldout_gain_vs_final", "improved_rate_vs_final", "mean_prediction_shift_vs_final"]].head(25).to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

