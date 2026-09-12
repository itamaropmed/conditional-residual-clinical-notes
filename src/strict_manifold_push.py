#!/usr/bin/env python3
"""Push the strict no-LLM manifold direction.

This expands the raw strict note/tabular manifold search while preserving the
same data boundary:
  * raw strict notes only
  * note_date < scheduled_in_room - 2 days
  * no old 11 LLM features
  * train-OOF selection for residual heads, gammas, and blends
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cross_decomposition import CCA, PLSRegression

from strict_manifold_text_sweep import (  # noqa: E402
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    clean_spaces,
    clustered_ci,
    dense,
    fit_oof_residual,
    metric_dict,
    read_table,
    tabular_svd_features,
    temporal_group_holdout,
    tfidf_svd,
    tune_gamma,
)

DEFAULT_PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default=str(DEFAULT_PROJECT / "data" / "mayo" / "mayo_hrs_tabular_features_and_durations.parquet"),
    )
    parser.add_argument("--phrase-dir", default="work/no_llm_notes/raw_strict_high_signal_tuned_split")
    parser.add_argument("--tabular-sweep-dir", default="work/no_llm_notes/tabular_sweep")
    parser.add_argument("--output-dir", default="work/no_llm_notes/raw_strict_manifold_push")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--max-pls-components", type=int, default=56)
    parser.add_argument("--profile", choices=["surgical", "broad"], default="surgical")
    return parser.parse_args()


def assert_no_forbidden_access() -> None:
    # The filename is allowed here only as a sentinel. This script never reads it.
    forbidden_name = "mayo_hrs_llm_features_top11.parquet"
    assert forbidden_name.endswith("_top11.parquet")


def make_latent_features(tab_scores: np.ndarray, note_scores: np.ndarray, n_comp: int, mode: str) -> np.ndarray:
    tx = tab_scores[:, :n_comp]
    ny = note_scores[:, :n_comp]
    diff = ny - tx
    if mode == "basic":
        return np.concatenate([tx, ny, diff], axis=1).astype("float32")
    if mode == "geom":
        return np.concatenate([tx, ny, diff, np.abs(diff), tx * ny], axis=1).astype("float32")
    if mode == "delta":
        return np.concatenate([diff, np.abs(diff), tx * ny], axis=1).astype("float32")
    raise ValueError(mode)


def evaluate_variant_with_predictions(
    name: str,
    xtr: np.ndarray,
    xte: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    n_folds: int,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    residual = y_train - base_oof
    best: dict[str, Any] | None = None
    best_train_pred: np.ndarray | None = None
    best_test_pred: np.ndarray | None = None
    for model_name in ["huber", "ridge", "sgd_huber"]:
        try:
            oof_resid, test_resid, meta = fit_oof_residual(
                dense(xtr), dense(xte), residual, train_groups, n_folds, seed, model_name
            )
        except Exception as exc:
            print(f"    {name}/{model_name} failed: {exc}", flush=True)
            continue
        gamma, train_mae = tune_gamma(y_train, base_oof, oof_resid)
        train_pred = base_oof + gamma * oof_resid
        test_pred = base_test + gamma * test_resid
        row = {
            "variant": name,
            "model": model_name,
            "selected_gamma": gamma,
            "selected_train_oof_mae": train_mae,
            "base_train_oof_mae": float(np.mean(np.abs(base_oof - y_train))),
            **metric_dict(y_test, test_pred),
        }
        row["gain_vs_base"] = float(np.mean(np.abs(base_test - y_test) - np.abs(test_pred - y_test)))
        row["gain_vs_base_CI95"] = list(
            clustered_ci(np.abs(base_test - y_test) - np.abs(test_pred - y_test), test_groups, n_boot, seed + 19)
        )
        row["MAE_CI95"] = list(clustered_ci(np.abs(test_pred - y_test), test_groups, n_boot, seed + 23))
        row["delta_vs_31_794666"] = float(row["MAE"] - 31.794666)
        row["delta_vs_32_028831"] = float(row["MAE"] - 32.028831)
        row["meta_json"] = json.dumps(meta, sort_keys=True)
        if best is None or train_mae < best["selected_train_oof_mae"]:
            best = row
            best_train_pred = train_pred
            best_test_pred = test_pred
    if best is None or best_train_pred is None or best_test_pred is None:
        raise RuntimeError(f"No model succeeded for {name}")
    return best, best_train_pred, best_test_pred


def optimize_convex_blend(y: np.ndarray, pred_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    n = pred_matrix.shape[1]
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def objective(w: np.ndarray) -> float:
        return float(np.mean(np.abs(pred_matrix @ w - y)))

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-9})
    if not result.success:
        return x0, objective(x0)
    w = np.asarray(result.x, dtype=float)
    w[w < 1e-8] = 0.0
    w = w / w.sum()
    return w, objective(w)


def main() -> None:
    args = parse_args()
    assert_no_forbidden_access()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.tabular_data), TARGET_COLUMN, None)
    df = add_safe_features(df, include_scheduled_duration=True)
    split = temporal_group_holdout(df, 0.20)
    train_df, test_df = split.train_df.copy(), split.test_df.copy()
    y_train = train_df[TARGET_COLUMN].to_numpy(float)
    y_test = test_df[TARGET_COLUMN].to_numpy(float)
    train_groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    test_groups = test_df[PATIENT_ID_COLUMN].astype(str).to_numpy()

    tab_oof = pd.read_csv(Path(args.tabular_sweep_dir) / "tabular_sweep_oof_predictions.csv")
    tab_test = pd.read_csv(Path(args.tabular_sweep_dir) / "tabular_sweep_predictions.csv")
    base_oof = 0.6 * tab_oof["oof_xgb_tuned_v2"].to_numpy(float) + 0.4 * tab_oof["oof_xgb_absoluteerror_same_shape"].to_numpy(float)
    base_test = 0.6 * tab_test["pred_xgb_tuned_v2"].to_numpy(float) + 0.4 * tab_test["pred_xgb_absoluteerror_same_shape"].to_numpy(float)
    print(
        f"Base alpha=0.600: train OOF MAE={np.mean(np.abs(base_oof-y_train)):.4f} "
        f"test MAE={np.mean(np.abs(base_test-y_test)):.4f}",
        flush=True,
    )

    phrase_dir = Path(args.phrase_dir)
    hs_train = clean_spaces(pd.read_parquet(phrase_dir / "train_high_signal_text.parquet")["high_signal_text"])
    hs_test = clean_spaces(pd.read_parquet(phrase_dir / "test_high_signal_text.parquet")["high_signal_text"])
    bundles = pd.read_parquet(phrase_dir / "strict_note_bundles.parquet")
    raw_train = train_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    raw_test = test_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    raw_train_text = clean_spaces(raw_train["note_text"])
    raw_test_text = clean_spaces(raw_test["note_text"])

    print(f"Building strict note/tabular SVD views for profile={args.profile}...", flush=True)
    if args.profile == "surgical":
        tab_svd = tabular_svd_features(train_df, test_df, 96, args.seed + 1)
        hs_word = tfidf_svd(hs_train, hs_test, analyzer="word", ngram_range=(1, 3), max_features=50000, min_df=2, svd_dim=160, seed=args.seed + 2, add_concepts=True)
        raw_word = tfidf_svd(raw_train_text, raw_test_text, analyzer="word", ngram_range=(1, 2), max_features=50000, min_df=3, svd_dim=160, seed=args.seed + 3, add_concepts=True)
        note_views = {
            "hs_word160": hs_word,
            "raw_word160": raw_word,
            "hs_raw_word160": (
                np.concatenate([hs_word[0], raw_word[0]], axis=1),
                np.concatenate([hs_word[1], raw_word[1]], axis=1),
            ),
        }
        tab_train = tab_svd[0][:, :64]
        tab_test_arr = tab_svd[1][:, :64]
        dims = [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]
        cca_view_names = ["hs_word160", "hs_raw_word160"]
    else:
        tab128 = tabular_svd_features(train_df, test_df, 128, args.seed + 1)
        hs_word256 = tfidf_svd(hs_train, hs_test, analyzer="word", ngram_range=(1, 3), max_features=90000, min_df=2, svd_dim=256, seed=args.seed + 2, add_concepts=True)
        raw_word256 = tfidf_svd(raw_train_text, raw_test_text, analyzer="word", ngram_range=(1, 2), max_features=90000, min_df=3, svd_dim=256, seed=args.seed + 3, add_concepts=True)
        hs_char160 = tfidf_svd(hs_train, hs_test, analyzer="char_wb", ngram_range=(3, 5), max_features=100000, min_df=3, svd_dim=160, seed=args.seed + 4)
        raw_char160 = tfidf_svd(raw_train_text, raw_test_text, analyzer="char_wb", ngram_range=(3, 5), max_features=110000, min_df=3, svd_dim=160, seed=args.seed + 5)
        note_views = {
            "hs_word256": hs_word256,
            "raw_word256": raw_word256,
            "hs_raw_word": (
                np.concatenate([hs_word256[0], raw_word256[0]], axis=1),
                np.concatenate([hs_word256[1], raw_word256[1]], axis=1),
            ),
            "hs_word_char": (
                np.concatenate([hs_word256[0], hs_char160[0]], axis=1),
                np.concatenate([hs_word256[1], hs_char160[1]], axis=1),
            ),
            "raw_word_char": (
                np.concatenate([raw_word256[0], raw_char160[0]], axis=1),
                np.concatenate([raw_word256[1], raw_char160[1]], axis=1),
            ),
            "all_note_views": (
                np.concatenate([hs_word256[0], raw_word256[0], hs_char160[0], raw_char160[0]], axis=1),
                np.concatenate([hs_word256[1], raw_word256[1], hs_char160[1], raw_char160[1]], axis=1),
            ),
        }
        tab_train = tab128[0][:, :96]
        tab_test_arr = tab128[1][:, :96]
        dims = [12, 16, 20, 24, 28, 32, 36, 40, 48, 56]
        cca_view_names = ["hs_word256", "hs_raw_word", "all_note_views"]
    dims = [d for d in dims if d <= args.max_pls_components and d <= tab_train.shape[1]]
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    print("Building PLS manifold variants by slicing high-component fits...", flush=True)
    for view_name, (note_train, note_test) in note_views.items():
        max_comp = min(max(dims), tab_train.shape[1], note_train.shape[1], args.max_pls_components)
        pls = PLSRegression(n_components=max_comp, max_iter=1000)
        pls.fit(tab_train, note_train)
        tr_tab, tr_note = pls.transform(tab_train, note_train)
        te_tab, te_note = pls.transform(tab_test_arr, note_test)
        for d in dims:
            if d > max_comp:
                continue
            for mode in ["basic", "geom", "delta"]:
                variants.append(
                    (
                        f"pls_{view_name}_{d}_{mode}",
                        make_latent_features(tr_tab, tr_note, d, mode),
                        make_latent_features(te_tab, te_note, d, mode),
                    )
                )

    print("Building targeted CCA variants for the strongest note views...", flush=True)
    for view_name in cca_view_names:
        note_train, note_test = note_views[view_name]
        max_comp = min(40, tab_train.shape[1], note_train.shape[1])
        cca = CCA(n_components=max_comp, max_iter=1500)
        try:
            cca.fit(tab_train, note_train)
            tr_tab, tr_note = cca.transform(tab_train, note_train)
            te_tab, te_note = cca.transform(tab_test_arr, note_test)
        except Exception as exc:
            print(f"  CCA {view_name} failed: {exc}", flush=True)
            continue
        for d in [12, 16, 20, 24, 28, 32, 40]:
            if d > max_comp:
                continue
            for mode in ["basic", "geom"]:
                variants.append(
                    (
                        f"cca_{view_name}_{d}_{mode}",
                        make_latent_features(tr_tab, tr_note, d, mode),
                        make_latent_features(te_tab, te_note, d, mode),
                    )
                )

    if args.max_variants:
        variants = variants[: args.max_variants]
    print(f"Evaluating {len(variants)} expanded manifold variants...", flush=True)

    rows: list[dict[str, Any]] = []
    train_pred_cols: dict[str, np.ndarray] = {"base": base_oof}
    test_pred_cols: dict[str, np.ndarray] = {"base": base_test}
    for i, (name, xtr, xte) in enumerate(variants, start=1):
        print(f"{i:03d}/{len(variants)} evaluating {name} dim={xtr.shape[1]}", flush=True)
        row, train_pred, test_pred = evaluate_variant_with_predictions(
            name,
            xtr,
            xte,
            y_train,
            y_test,
            base_oof,
            base_test,
            train_groups,
            test_groups,
            args.n_folds,
            args.n_boot,
            args.seed + i * 11,
        )
        rows.append(row)
        col = f"{row['variant']}__{row['model']}__g{row['selected_gamma']:.3f}"
        train_pred_cols[col] = train_pred
        test_pred_cols[col] = test_pred
        print(
            f"    selected {row['model']} gamma={row['selected_gamma']:.3f} "
            f"train={row['selected_train_oof_mae']:.4f} test={row['MAE']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    results.to_csv(out / "strict_manifold_push_results.csv", index=False)

    train_pred_df = pd.DataFrame({CASE_ID_COLUMN: train_df[CASE_ID_COLUMN].values, TARGET_COLUMN: y_train, **train_pred_cols})
    test_pred_df = test_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN, "primary_procedure_name"]].copy()
    for key, val in test_pred_cols.items():
        test_pred_df[key] = val
    train_pred_df.to_csv(out / "strict_manifold_push_oof_predictions.csv", index=False)
    test_pred_df.to_csv(out / "strict_manifold_push_test_predictions.csv", index=False)

    # Blend candidates are selected by train OOF MAE only. The base prediction is
    # included so the optimizer can shrink toward no note correction.
    blend_rows: list[dict[str, Any]] = []
    candidate_rank = (
        pd.DataFrame(
            [
                {"name": name, "train_mae": float(np.mean(np.abs(pred - y_train)))}
                for name, pred in train_pred_cols.items()
            ]
        )
        .sort_values("train_mae")
        .reset_index(drop=True)
    )
    for k in [2, 3, 5, 8, 12, 16, 24, 32]:
        names = ["base"] + [n for n in candidate_rank["name"].tolist() if n != "base"][:k]
        names = list(dict.fromkeys(names))
        tr_mat = np.column_stack([train_pred_cols[n] for n in names])
        te_mat = np.column_stack([test_pred_cols[n] for n in names])
        weights, train_mae = optimize_convex_blend(y_train, tr_mat)
        test_pred = te_mat @ weights
        row = {
            "blend": f"convex_top{k}_by_train_oof",
            "n_inputs": len(names),
            "selected_train_oof_mae": float(train_mae),
            "input_names_json": json.dumps(names),
            "weights_json": json.dumps([float(w) for w in weights]),
            **metric_dict(y_test, test_pred),
        }
        row["gain_vs_base"] = float(np.mean(np.abs(base_test - y_test) - np.abs(test_pred - y_test)))
        row["gain_vs_base_CI95"] = list(
            clustered_ci(np.abs(base_test - y_test) - np.abs(test_pred - y_test), test_groups, args.n_boot, args.seed + k)
        )
        row["delta_vs_31_794666"] = float(row["MAE"] - 31.794666)
        blend_rows.append(row)
        test_pred_df[f"pred_{row['blend']}"] = test_pred
    blends = pd.DataFrame(blend_rows).sort_values("selected_train_oof_mae").reset_index(drop=True)
    blends.to_csv(out / "strict_manifold_push_blends.csv", index=False)
    test_pred_df.to_csv(out / "strict_manifold_push_test_predictions.csv", index=False)

    run_config = vars(args) | {
        "n_variants": len(variants),
        "forbidden_llm_feature_file_read": False,
        "blend_selection": "convex weights fit on train OOF predictions only",
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("\n=== EXPANDED MANIFOLD SINGLE VARIANTS ===")
    print(results[["variant", "model", "selected_gamma", "selected_train_oof_mae", "MAE", "gain_vs_base", "delta_vs_31_794666"]].head(20).to_string(index=False))
    print("\n=== TRAIN-OOF CONVEX BLENDS ===")
    print(blends[["blend", "n_inputs", "selected_train_oof_mae", "MAE", "gain_vs_base", "delta_vs_31_794666"]].to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
