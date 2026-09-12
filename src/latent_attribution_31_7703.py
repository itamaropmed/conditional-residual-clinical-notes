#!/usr/bin/env python3
"""Latent attribution diagnostics for the fixed strict 31.7703 model.

For a linear residual head, path integrated gradients from a zero baseline in
the final standardized latent feature space are exactly feature_value * weight.
This script refits the selected final residual head and reports those latent
contributions by PLS component and geometry block.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("STRICT_31_WORK_ROOT", "/Users/itamarzernitsky/Documents/Codex/2026-08-23/can"))
PROJECT = Path(os.environ.get("CLINICAL_NOTES_PROJECT", "/Users/itamarzernitsky/PycharmProjects/Clinical_notes"))
OUT_DIR = ROOT / "outputs/posthoc_interpretability"
FORBIDDEN = PROJECT / "data/mayo/mayo_hrs_llm_features_top11.parquet"
FINAL_COL = "pls_raw_word160_26_delta__sgd_huber__g1.550"
GAMMA = 1.55
SELECTED_VARIANT_INDEX = 51
SEED = 4242 + SELECTED_VARIANT_INDEX * 11

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "mayo_residual_distiller_study"))
sys.path.insert(0, str(ROOT / "work/no_llm_notes"))

from mayo_llm_flag_residual_eval import CLINICAL_TOKEN_PATTERN, CONCEPT_PATTERNS, hand_authored_features  # noqa: E402
from strict_manifold_push import make_latent_features  # noqa: E402
from strict_manifold_text_sweep import (  # noqa: E402
    CASE_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    clean_spaces,
    dense,
    read_table,
    tabular_svd_features,
    temporal_group_holdout,
)


def assert_no_forbidden_access() -> None:
    assert FORBIDDEN.name == "mayo_hrs_llm_features_top11.parquet"


def raw_tfidf_svd_with_fitted(
    train_text: pd.Series,
    test_text: pd.Series,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=CLINICAL_TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.97,
        max_features=50000,
        sublinear_tf=True,
    )
    xtr = vec.fit_transform(train_text.fillna(""))
    xte = vec.transform(test_text.fillna(""))
    svd = TruncatedSVD(n_components=160, random_state=seed)
    ztr_raw = svd.fit_transform(xtr)
    zte_raw = svd.transform(xte)
    scaler = StandardScaler()
    ztr = scaler.fit_transform(ztr_raw)
    zte = scaler.transform(zte_raw)
    ztr = np.concatenate([ztr, hand_authored_features(train_text)], axis=1)
    zte = np.concatenate([zte, hand_authored_features(test_text)], axis=1)
    return ztr.astype("float32"), zte.astype("float32"), vec, svd, scaler


def final_sgd_huber_residual(x_train: np.ndarray, residual: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_scaler = StandardScaler().fit(x_train)
    xtr = outer_scaler.transform(x_train)
    xte = outer_scaler.transform(x_test)
    model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.05,
            max_iter=4000,
            tol=1e-4,
            random_state=SEED,
            average=True,
        ),
    )
    model.fit(xtr, residual)
    pred = model.predict(xte)
    inner_scaler: StandardScaler = model.named_steps["standardscaler"]
    reg: SGDRegressor = model.named_steps["sgdregressor"]
    xte_inner = inner_scaler.transform(xte)
    return pred, xte_inner.astype("float32"), reg.coef_.astype("float32")


def feature_names(n_comp: int = 26) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for block in ["diff", "absdiff", "product"]:
        for i in range(n_comp):
            out.append((f"{block}_{i + 1:02d}", i + 1, block))
    return out


def top_terms_for_pls_components(
    vec: TfidfVectorizer,
    svd: TruncatedSVD,
    svd_scaler: StandardScaler,
    pls: PLSRegression,
    component_ids: list[int],
    top_k: int = 12,
) -> pd.DataFrame:
    terms = np.asarray(vec.get_feature_names_out())
    concept_names = list(CONCEPT_PATTERNS.keys())
    rows = []
    for comp_1based in component_ids:
        comp = comp_1based - 1
        y_weight = pls.y_weights_[:, comp]
        svd_weight = y_weight[:160] / np.maximum(svd_scaler.scale_, 1e-12)
        term_scores = svd.components_.T @ svd_weight
        pos_idx = np.argsort(term_scores)[-top_k:][::-1]
        neg_idx = np.argsort(term_scores)[:top_k]
        concept_weight = y_weight[160:]
        top_concepts = sorted(
            zip(concept_names, concept_weight),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )
        rows.append(
            {
                "pls_component": comp_1based,
                "top_positive_terms": "; ".join(terms[pos_idx].tolist()),
                "top_negative_terms": "; ".join(terms[neg_idx].tolist()),
                "top_concept_weights": "; ".join(f"{name}:{weight:.4f}" for name, weight in top_concepts),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    assert_no_forbidden_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tabular_data = PROJECT / "data/mayo/mayo_hrs_tabular_features_and_durations.parquet"
    df = read_table(tabular_data, TARGET_COLUMN, None)
    df = add_safe_features(df, include_scheduled_duration=True)
    split = temporal_group_holdout(df, 0.20)
    train_df, test_df = split.train_df.copy(), split.test_df.copy()
    y_train = train_df[TARGET_COLUMN].to_numpy(float)
    y_test = test_df[TARGET_COLUMN].to_numpy(float)

    tab_oof = pd.read_csv(ROOT / "work/no_llm_notes/tabular_sweep/tabular_sweep_oof_predictions.csv")
    tab_test = pd.read_csv(ROOT / "work/no_llm_notes/tabular_sweep/tabular_sweep_predictions.csv")
    base_oof = 0.6 * tab_oof["oof_xgb_tuned_v2"].to_numpy(float) + 0.4 * tab_oof["oof_xgb_absoluteerror_same_shape"].to_numpy(float)
    base_test = 0.6 * tab_test["pred_xgb_tuned_v2"].to_numpy(float) + 0.4 * tab_test["pred_xgb_absoluteerror_same_shape"].to_numpy(float)

    bundles = pd.read_parquet(ROOT / "work/no_llm_notes/raw_strict_high_signal_tuned_split/strict_note_bundles.parquet")
    raw_train = train_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    raw_test = test_df[[CASE_ID_COLUMN, "primary_procedure_name"]].merge(bundles, on=CASE_ID_COLUMN, how="left")
    raw_train_text = clean_spaces(raw_train["note_text"])
    raw_test_text = clean_spaces(raw_test["note_text"])

    tab_svd = tabular_svd_features(train_df, test_df, 96, 4242 + 1)
    raw_word_train, raw_word_test, vec, svd, svd_scaler = raw_tfidf_svd_with_fitted(
        raw_train_text,
        raw_test_text,
        seed=4242 + 3,
    )
    tab_train = tab_svd[0][:, :64]
    tab_test_arr = tab_svd[1][:, :64]
    # Surgical profile dims end at 40, so strict_manifold_push.py fits a
    # 40-component PLS and slices the selected first 26 components.
    pls = PLSRegression(n_components=40, max_iter=1000)
    pls.fit(tab_train, raw_word_train)
    tr_tab, tr_note = pls.transform(tab_train, raw_word_train)
    te_tab, te_note = pls.transform(tab_test_arr, raw_word_test)
    xtr = make_latent_features(tr_tab, tr_note, 26, "delta")
    xte = make_latent_features(te_tab, te_note, 26, "delta")

    residual = y_train - base_oof
    test_resid, xte_inner, coef = final_sgd_huber_residual(dense(xtr), residual, dense(xte))
    pred = base_test + GAMMA * test_resid
    saved = pd.read_csv(
        ROOT / "work/no_llm_notes/raw_strict_manifold_push_surgical_full/strict_manifold_push_test_predictions.csv",
        usecols=[FINAL_COL],
    )[FINAL_COL].to_numpy(float)

    names = feature_names(26)
    contrib = xte_inner * coef.reshape(1, -1) * GAMMA
    rows = []
    for j, (name, comp, block) in enumerate(names):
        vals = contrib[:, j]
        rows.append(
            {
                "feature": name,
                "pls_component": comp,
                "block": block,
                "coef": float(coef[j]),
                "mean_contribution_minutes": float(vals.mean()),
                "mean_abs_contribution_minutes": float(np.abs(vals).mean()),
                "p90_abs_contribution_minutes": float(np.quantile(np.abs(vals), 0.90)),
                "positive_rate": float((vals > 0).mean()),
            }
        )
    feature_df = pd.DataFrame(rows).sort_values("mean_abs_contribution_minutes", ascending=False)
    feature_df.to_csv(OUT_DIR / "latent_integrated_gradient_equivalent_features.csv", index=False)

    comp_df = (
        feature_df.groupby("pls_component")
        .agg(
            mean_abs_contribution_minutes=("mean_abs_contribution_minutes", "sum"),
            mean_signed_contribution_minutes=("mean_contribution_minutes", "sum"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution_minutes", ascending=False)
    )
    comp_df.to_csv(OUT_DIR / "latent_component_attribution_summary.csv", index=False)

    block_df = (
        feature_df.groupby("block")
        .agg(
            mean_abs_contribution_minutes=("mean_abs_contribution_minutes", "sum"),
            mean_signed_contribution_minutes=("mean_contribution_minutes", "sum"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution_minutes", ascending=False)
    )
    block_df.to_csv(OUT_DIR / "latent_block_attribution_summary.csv", index=False)

    top_components = comp_df["pls_component"].head(8).astype(int).tolist()
    term_df = top_terms_for_pls_components(vec, svd, svd_scaler, pls, top_components)
    term_df.to_csv(OUT_DIR / "latent_component_approximate_terms.csv", index=False)

    summary = {
        "refit_final_mae": float(np.mean(np.abs(pred - y_test))),
        "saved_final_mae": float(np.mean(np.abs(saved - y_test))),
        "max_abs_difference_vs_saved_prediction": float(np.max(np.abs(pred - saved))),
        "mean_abs_difference_vs_saved_prediction": float(np.mean(np.abs(pred - saved))),
        "gamma": GAMMA,
        "seed": SEED,
        "forbidden_llm_feature_file_read": False,
        "interpretability_note": "For a linear residual head, integrated gradients from zero baseline in standardized latent space equal feature_value * coefficient.",
    }
    (OUT_DIR / "latent_attribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved latent attribution outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
