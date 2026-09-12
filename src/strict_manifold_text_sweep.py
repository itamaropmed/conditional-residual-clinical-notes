#!/usr/bin/env python3
"""Strict note/text manifold and fusion sweep.

No old 11 LLM features. Uses strict -2d note text plus tabular OOF baselines.
Feature families include richer note SVDs, tabular SVD manifolds, CCA/PLS
shared manifolds, and RBF/Nystroem fused coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import HuberRegressor, RidgeCV, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "mayo_residual_distiller_study"))

from mayo_llm_flag_residual_eval import CLINICAL_TOKEN_PATTERN, hand_authored_features  # noqa: E402
from train_mayo_cure_residual_embedding import source_aware_note_text_for_case  # noqa: E402
from train_mayo_tabular_xgboost_baseline import (  # noqa: E402
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    make_feature_frame,
    make_preprocessor,
    metric_dict,
    read_table,
    temporal_group_holdout,
)


FORBIDDEN = PROJECT / "data" / "mayo" / "mayo_hrs_llm_features_top11.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default=str(PROJECT / "data" / "mayo" / "mayo_hrs_tabular_features_and_durations.parquet"),
    )
    parser.add_argument("--phrase-dir", default="work/no_llm_notes/strict_high_signal_tuned_split")
    parser.add_argument("--tabular-sweep-dir", default="work/no_llm_notes/tabular_sweep")
    parser.add_argument("--output-dir", default="work/no_llm_notes/strict_manifold_text_sweep")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--max-variants", type=int, default=0, help="0 = all variants; useful for smoke tests.")
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


def dense(x: Any) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def clean_spaces(text: pd.Series) -> pd.Series:
    return text.fillna("").astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())


def tfidf_svd(
    train_text: pd.Series,
    test_text: pd.Series,
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    min_df: int,
    svd_dim: int,
    seed: int,
    add_concepts: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if analyzer == "word":
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=CLINICAL_TOKEN_PATTERN,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.97,
            max_features=max_features,
            sublinear_tf=True,
        )
    else:
        vec = TfidfVectorizer(
            lowercase=True,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.995,
            max_features=max_features,
            sublinear_tf=True,
        )
    xtr = vec.fit_transform(train_text.fillna(""))
    xte = vec.transform(test_text.fillna(""))
    n_comp = max(2, min(svd_dim, xtr.shape[0] - 2, xtr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    ztr = svd.fit_transform(xtr)
    zte = svd.transform(xte)
    scaler = StandardScaler()
    ztr = scaler.fit_transform(ztr)
    zte = scaler.transform(zte)
    if add_concepts:
        ztr = np.concatenate([ztr, hand_authored_features(train_text)], axis=1)
        zte = np.concatenate([zte, hand_authored_features(test_text)], axis=1)
    return ztr.astype("float32"), zte.astype("float32")


def tabular_svd_features(train_df: pd.DataFrame, test_df: pd.DataFrame, svd_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X_train, _, numeric_cols, categorical_cols = make_feature_frame(train_df, TARGET_COLUMN)
    X_test, _, _, _ = make_feature_frame(test_df, TARGET_COLUMN)
    pre = make_preprocessor(numeric_cols, categorical_cols)
    xtr = pre.fit_transform(X_train)
    xte = pre.transform(X_test)
    n_comp = max(2, min(svd_dim, xtr.shape[0] - 2, xtr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    ztr = svd.fit_transform(xtr)
    zte = svd.transform(xte)
    scaler = StandardScaler()
    return scaler.fit_transform(ztr).astype("float32"), scaler.transform(zte).astype("float32")


def fit_oof_residual(
    x_train: np.ndarray,
    x_test: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(x_train)
    xtr = scaler.transform(x_train)
    xte = scaler.transform(x_test)
    splitter = GroupKFold(n_splits=max(2, min(n_folds, len(np.unique(groups)))))
    oof = np.zeros(len(xtr), dtype=float)

    def make_model() -> Any:
        if model_name == "ridge":
            return RidgeCV(alphas=np.logspace(-3, 4, 24))
        if model_name == "huber":
            return HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1200)
        if model_name == "sgd_huber":
            return make_pipeline(
                StandardScaler(),
                SGDRegressor(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=1e-4,
                    l1_ratio=0.05,
                    max_iter=4000,
                    tol=1e-4,
                    random_state=seed,
                    average=True,
                ),
            )
        raise ValueError(model_name)

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(xtr, residual, groups=groups), start=1):
        model = make_model()
        model.fit(xtr[fit_idx], residual[fit_idx])
        oof[val_idx] = model.predict(xtr[val_idx])
    final = make_model()
    final.fit(xtr, residual)
    pred = final.predict(xte)
    return oof, pred, {"model": model_name, "residual_oof_mae": float(mean_absolute_error(residual, oof))}


def tune_gamma(y: np.ndarray, base: np.ndarray, resid_oof: np.ndarray) -> tuple[float, float]:
    gamma_grid = np.arange(0.0, 1.601, 0.025)
    rows = [(float(g), float(np.mean(np.abs(base + g * resid_oof - y)))) for g in gamma_grid]
    return min(rows, key=lambda x: x[1])


def evaluate_variant(
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
) -> tuple[dict[str, Any], np.ndarray]:
    residual = y_train - base_oof
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for model_name in ["huber", "ridge", "sgd_huber"]:
        try:
            oof_resid, test_resid, meta = fit_oof_residual(
                xtr, xte, residual, train_groups, n_folds, seed, model_name
            )
        except Exception as exc:
            print(f"    {name}/{model_name} failed: {exc}", flush=True)
            continue
        gamma, train_mae = tune_gamma(y_train, base_oof, oof_resid)
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
        row["delta_vs_32_0288"] = float(row["MAE"] - 32.028831)
        row["delta_vs_old_32_291"] = float(row["MAE"] - 32.291)
        row["meta_json"] = json.dumps(meta, sort_keys=True)
        if best is None or row["selected_train_oof_mae"] < best["selected_train_oof_mae"]:
            best = row
            best_pred = test_pred
    if best is None or best_pred is None:
        raise RuntimeError(f"No model succeeded for {name}")
    return best, best_pred


def main() -> None:
    args = parse_args()
    assert_no_forbidden_access()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    xgb_oof = tab_oof["oof_xgb_tuned_v2"].to_numpy(float)
    abs_oof = tab_oof["oof_xgb_absoluteerror_same_shape"].to_numpy(float)
    xgb_test = tab_test["pred_xgb_tuned_v2"].to_numpy(float)
    abs_test = tab_test["pred_xgb_absoluteerror_same_shape"].to_numpy(float)
    base_oof = 0.6 * xgb_oof + 0.4 * abs_oof
    base_test = 0.6 * xgb_test + 0.4 * abs_test
    print(f"Base alpha=0.600: train OOF MAE={np.mean(np.abs(base_oof-y_train)):.4f} test MAE={np.mean(np.abs(base_test-y_test)):.4f}")

    phrase_dir = Path(args.phrase_dir)
    hs_train = clean_spaces(pd.read_parquet(phrase_dir / "train_high_signal_text.parquet")["high_signal_text"])
    hs_test = clean_spaces(pd.read_parquet(phrase_dir / "test_high_signal_text.parquet")["high_signal_text"])
    strict_notes = pd.read_parquet(phrase_dir / "strict_note_bundles.parquet")
    raw_train = train_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(strict_notes, on=CASE_ID_COLUMN, how="left")
    raw_test = test_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(strict_notes, on=CASE_ID_COLUMN, how="left")
    raw_train_text = clean_spaces(raw_train["note_text"])
    raw_test_text = clean_spaces(raw_test["note_text"])

    print("Building base note/tabular manifolds...")
    hs_word96 = tfidf_svd(hs_train, hs_test, analyzer="word", ngram_range=(1, 2), max_features=12000, min_df=3, svd_dim=96, seed=args.seed, add_concepts=True)
    hs_word160 = tfidf_svd(hs_train, hs_test, analyzer="word", ngram_range=(1, 3), max_features=50000, min_df=2, svd_dim=160, seed=args.seed + 1, add_concepts=True)
    hs_char128 = tfidf_svd(hs_train, hs_test, analyzer="char_wb", ngram_range=(3, 5), max_features=80000, min_df=3, svd_dim=128, seed=args.seed + 2)
    raw_word160 = tfidf_svd(raw_train_text, raw_test_text, analyzer="word", ngram_range=(1, 2), max_features=50000, min_df=3, svd_dim=160, seed=args.seed + 3, add_concepts=True)
    raw_char128 = tfidf_svd(raw_train_text, raw_test_text, analyzer="char_wb", ngram_range=(3, 5), max_features=90000, min_df=3, svd_dim=128, seed=args.seed + 4)
    tab_svd96 = tabular_svd_features(train_df, test_df, 96, args.seed + 5)

    variants: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("hs_word96_currentish", *hs_word96),
        ("hs_word160_word13", *hs_word160),
        ("hs_char128_charwb35", *hs_char128),
        ("raw_word160_word12", *raw_word160),
        ("raw_char128_charwb35", *raw_char128),
        ("hs_word160_plus_hs_char128", np.concatenate([hs_word160[0], hs_char128[0]], axis=1), np.concatenate([hs_word160[1], hs_char128[1]], axis=1)),
        ("hs_word160_plus_raw_word160", np.concatenate([hs_word160[0], raw_word160[0]], axis=1), np.concatenate([hs_word160[1], raw_word160[1]], axis=1)),
        ("hs_word160_plus_tab_svd96", np.concatenate([hs_word160[0], tab_svd96[0]], axis=1), np.concatenate([hs_word160[1], tab_svd96[1]], axis=1)),
    ]

    print("Building shared manifold mappings...")
    tab64 = tab_svd96[0][:, :64], tab_svd96[1][:, :64]
    hs64 = hs_word160[0][:, :64], hs_word160[1][:, :64]
    for n_comp in [12, 24, 40]:
        cca = CCA(n_components=n_comp, max_iter=1000)
        cca.fit(tab64[0], hs64[0])
        tr_tab, tr_note = cca.transform(tab64[0], hs64[0])
        te_tab, te_note = cca.transform(tab64[1], hs64[1])
        variants.append((f"cca_tab_note_{n_comp}", np.concatenate([tr_tab, tr_note, tr_note - tr_tab], axis=1), np.concatenate([te_tab, te_note, te_note - te_tab], axis=1)))

        pls = PLSRegression(n_components=n_comp, max_iter=1000)
        pls.fit(tab64[0], hs64[0])
        tr_tab, tr_note = pls.transform(tab64[0], hs64[0])
        te_tab, te_note = pls.transform(tab64[1], hs64[1])
        variants.append((f"pls_tab_note_{n_comp}", np.concatenate([tr_tab, tr_note, tr_note - tr_tab], axis=1), np.concatenate([te_tab, te_note, te_note - te_tab], axis=1)))

    fused_tr = np.concatenate([hs_word160[0], hs_char128[0], tab_svd96[0]], axis=1)
    fused_te = np.concatenate([hs_word160[1], hs_char128[1], tab_svd96[1]], axis=1)
    fused_scaler = StandardScaler()
    fused_tr_s = fused_scaler.fit_transform(fused_tr)
    fused_te_s = fused_scaler.transform(fused_te)
    for gamma in [0.01, 0.03, 0.08]:
        nys = Nystroem(kernel="rbf", gamma=gamma, n_components=256, random_state=args.seed + int(gamma * 1000))
        variants.append((f"nystroem_rbf_fused_gamma_{gamma:g}", nys.fit_transform(fused_tr_s).astype("float32"), nys.transform(fused_te_s).astype("float32")))

    if args.max_variants:
        variants = variants[: args.max_variants]

    rows = []
    preds = test_df[[CASE_ID_COLUMN, PATIENT_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN, "primary_procedure_name"]].copy()
    preds["pred_base_xgb2_alpha_0_600"] = base_test
    for i, (name, xtr, xte) in enumerate(variants, start=1):
        print(f"{i:02d}/{len(variants)} evaluating {name} dim={xtr.shape[1]}", flush=True)
        row, pred = evaluate_variant(
            name,
            dense(xtr),
            dense(xte),
            y_train,
            y_test,
            base_oof,
            base_test,
            train_groups,
            test_groups,
            args.n_folds,
            args.n_boot,
            args.seed + i * 7,
        )
        rows.append(row)
        preds[f"pred_{name}"] = pred
        print(
            f"    selected {row['model']} gamma={row['selected_gamma']:.3f} "
            f"train={row['selected_train_oof_mae']:.4f} test={row['MAE']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    results.to_csv(output_dir / "strict_manifold_text_sweep_results.csv", index=False)
    preds.to_csv(output_dir / "strict_manifold_text_sweep_predictions.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("\n=== STRICT MANIFOLD/TEXT SWEEP ===")
    print(results[["variant", "model", "selected_gamma", "selected_train_oof_mae", "MAE", "gain_vs_base", "gain_vs_base_CI95", "delta_vs_32_0288", "delta_vs_old_32_291"]].to_string(index=False))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
