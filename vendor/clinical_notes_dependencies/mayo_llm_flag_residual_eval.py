#!/usr/bin/env python3
"""Evaluate the Ollama-extracted exception-path flags as a residual-correction
feature source, reusing the proven tuned structured baseline (no re-tuning,
fast) so this is an honest apples-to-apples comparison against the existing
best result (~31.70-31.84 MAE) and the tabular-only baseline (~32.63 MAE).

Three candidate residual correctors are compared:
  A) llm_flags_only    -- just the 6 LLM-extracted exception flags.
  B) tfidf_concept_only -- the existing TF-IDF+SVD+regex-concept features
                            (replicates the prior best approach's inputs).
  C) llm_plus_tfidf     -- both combined.
All fit via GroupKFold OOF on train (never touching test), Huber + RidgeCV
candidates, best picked by OOF MAE.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from train_mayo_tabular_xgboost_baseline import (
    CASE_ID_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    metric_dict,
    read_table,
    require_xgboost,
    temporal_group_holdout,
)

CLINICAL_TOKEN_PATTERN = r"(?u)\b(?=[^\W\d_]*[^\W\d_])\w[\w-]{1,}\b"
CONCEPT_PATTERNS = {
    "redo_prior_ablation": [r"\bredo\b", r"repeat\s+ablation", r"prior\s+ablation"],
    "vt_pvc_instability": [r"\bvt\b", r"\bvf\b", r"ventricular\s+tachycardia", r"icd\s+shock"],
    "device_lead_complexity": [r"lead\s+(revision|extraction|failure|fracture)", r"generator\s+change"],
    "congenital_complex_anatomy": [r"\bfontan\b", r"transposition", r"congenital"],
    "access_support": [r"\becmo\b", r"epicardial", r"pericardial\s+access"],
}
FLAG_KEYS = [
    "redo_complex_substrate", "device_lead_complexity", "vt_instability",
    "hidden_multi_procedure_burden", "congenital_complex_anatomy", "access_support_difficulty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-data", default="data/mayo/mayo_hrs_tabular_features_and_durations.parquet")
    parser.add_argument(
        "--phrase-dir",
        default="/Users/itamarzernitsky/Documents/clinical notes/output/mayo_sparse_phrase_residual_miner_full",
    )
    parser.add_argument(
        "--llm-flags-dir",
        default="output/mayo_ollama_exception_signals",
    )
    parser.add_argument("--output-dir", default="output/mayo_llm_flag_residual_eval")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=12000)
    parser.add_argument("--svd-dim", type=int, default=96)
    return parser.parse_args()


def hand_authored_features(texts: pd.Series) -> np.ndarray:
    lowered = texts.fillna("").str.lower()
    feats = np.zeros((len(texts), len(CONCEPT_PATTERNS)), dtype="float32")
    for i, patterns in enumerate(CONCEPT_PATTERNS.values()):
        combined = re.compile("|".join(patterns), flags=re.IGNORECASE)
        feats[:, i] = lowered.map(lambda t: 1.0 if combined.search(t) else 0.0)
    return feats


def make_text_matrix(
    train_text: pd.Series, test_text: pd.Series, max_features: int, svd_dim: int, seed: int,
    return_fitted: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
    """return_fitted=True additionally returns the fitted (vec, svd, scaler)
    so callers can transform ad-hoc counterfactual text later (e.g. a case's
    joined-note text with one note removed) through the exact same fitted
    pipeline used at train time, without refitting. Default behavior/return
    arity unchanged for existing callers."""
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", token_pattern=CLINICAL_TOKEN_PATTERN,
        ngram_range=(1, 2), min_df=3, max_df=0.97, max_features=max_features, sublinear_tf=True,
    )
    x_train = vec.fit_transform(train_text.fillna(""))
    x_test = vec.transform(test_text.fillna(""))
    n_comp = max(2, min(svd_dim, x_train.shape[0] - 2, x_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    z_train = svd.fit_transform(x_train)
    z_test = svd.transform(x_test)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    concept_train = hand_authored_features(train_text)
    concept_test = hand_authored_features(test_text)
    out_train = np.concatenate([z_train, concept_train], axis=1).astype("float32")
    out_test = np.concatenate([z_test, concept_test], axis=1).astype("float32")
    if return_fitted:
        return out_train, out_test, vec, svd, scaler
    return out_train, out_test


def transform_text_matrix(text: pd.Series, vec: TfidfVectorizer, svd: TruncatedSVD, scaler: StandardScaler) -> np.ndarray:
    """Apply an already-fitted make_text_matrix(..., return_fitted=True) pipeline
    to new text (e.g. a note-ablation counterfactual)."""
    x = vec.transform(text.fillna(""))
    z = scaler.transform(svd.transform(x))
    concept = hand_authored_features(text)
    return np.concatenate([z, concept], axis=1).astype("float32")


def fit_residual_corrector(
    name: str, x_train: np.ndarray, residual_train: np.ndarray, groups: np.ndarray,
    x_test: np.ndarray, n_folds: int, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)

    candidates = {
        "huber": lambda s: HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000),
        "ridge": lambda s: RidgeCV(alphas=np.logspace(-3, 4, 20)),
    }
    rows = []
    for cand_name, make in candidates.items():
        oof = np.zeros(len(x_train_s), dtype=float)
        for fit_idx, val_idx in splitter.split(x_train_s, residual_train, groups=groups):
            m = make(seed)
            m.fit(x_train_s[fit_idx], residual_train[fit_idx])
            oof[val_idx] = m.predict(x_train_s[val_idx])
        rows.append({"candidate": cand_name, "oof_mae": float(mean_absolute_error(residual_train, oof)), "oof_pred": oof})

    best = sorted(rows, key=lambda r: r["oof_mae"])[0]
    best_model = candidates[best["candidate"]](seed)
    best_model.fit(x_train_s, residual_train)
    test_pred = best_model.predict(x_test_s)
    return test_pred, {"name": name, "best_candidate": best["candidate"], "oof_mae": best["oof_mae"], "oof_pred": best["oof_pred"]}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    XGBRegressor = require_xgboost()

    print("Loading tabular data and fitting tuned structured baseline...")
    df = read_table(Path(args.tabular_data), TARGET_COLUMN, None)
    df = add_safe_features(df, True)
    split = temporal_group_holdout(df, args.test_fraction)
    train_df, test_df = split.train_df.copy(), split.test_df.copy()
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    from mayo_tabular_baseline_v2_tuned import TUNED_STRUCTURED_PARAMS, fit_tuned_structured_oof_and_test

    y0_oof, y0_test = fit_tuned_structured_oof_and_test(
        XGBRegressor, train_df, test_df, TARGET_COLUMN, TUNED_STRUCTURED_PARAMS, args.n_folds, args.seed
    )
    structured_metrics = metric_dict(y_test, y0_test)
    print(f"  Structured baseline: train-OOF MAE={mean_absolute_error(y_train, y0_oof):.4f}  test MAE={structured_metrics['MAE']:.4f}")
    residual_train = y_train - y0_oof

    print("Loading LLM exception flags...")
    llm_dir = Path(args.llm_flags_dir)
    llm_train = pd.read_csv(llm_dir / "train_exception_signals.csv")
    llm_test = pd.read_csv(llm_dir / "test_exception_signals.csv")
    train_df = train_df.merge(llm_train[["case_durable_id", *FLAG_KEYS]], on=CASE_ID_COLUMN, how="left")
    test_df = test_df.merge(llm_test[["case_durable_id", *FLAG_KEYS]], on=CASE_ID_COLUMN, how="left")
    train_df[FLAG_KEYS] = train_df[FLAG_KEYS].fillna(0)
    test_df[FLAG_KEYS] = test_df[FLAG_KEYS].fillna(0)
    llm_x_train = train_df[FLAG_KEYS].to_numpy(dtype="float32")
    llm_x_test = test_df[FLAG_KEYS].to_numpy(dtype="float32")

    print("Loading note text for TF-IDF+concept comparison arm...")
    phrase_dir = Path(args.phrase_dir)
    train_text_df = pd.read_parquet(phrase_dir / "train_high_signal_text.parquet")
    test_text_df = pd.read_parquet(phrase_dir / "test_high_signal_text.parquet")
    train_df = train_df.merge(train_text_df, on=CASE_ID_COLUMN, how="left")
    test_df = test_df.merge(test_text_df, on=CASE_ID_COLUMN, how="left")
    train_df["high_signal_text"] = train_df["high_signal_text"].fillna("")
    test_df["high_signal_text"] = test_df["high_signal_text"].fillna("")
    tfidf_x_train, tfidf_x_test = make_text_matrix(
        train_df["high_signal_text"], test_df["high_signal_text"], args.tfidf_max_features, args.svd_dim, args.seed
    )

    combined_x_train = np.concatenate([llm_x_train, tfidf_x_train], axis=1)
    combined_x_test = np.concatenate([llm_x_test, tfidf_x_test], axis=1)

    arms = {
        "llm_flags_only": (llm_x_train, llm_x_test),
        "tfidf_concept_only": (tfidf_x_train, tfidf_x_test),
        "llm_plus_tfidf": (combined_x_train, combined_x_test),
    }

    results = [{**structured_metrics, "method": "structured_baseline"}]
    clip_bound = float(np.quantile(np.abs(residual_train), 0.99))
    for name, (x_tr, x_te) in arms.items():
        print(f"Fitting residual corrector: {name} (dim={x_tr.shape[1]})...")
        test_resid_pred, meta = fit_residual_corrector(name, x_tr, residual_train, groups, x_te, args.n_folds, args.seed)
        corrected_test = y0_test + np.clip(test_resid_pred, -clip_bound, clip_bound)
        m = metric_dict(y_test, corrected_test)
        results.append({**m, "method": name})
        print(f"  {name}: OOF residual MAE={meta['oof_mae']:.4f} (best candidate: {meta['best_candidate']})  test MAE={m['MAE']:.4f}")

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    ref_mae = structured_metrics["MAE"]
    results_df["delta_vs_structured"] = results_df["MAE"] - ref_mae
    results_df.to_csv(output_dir / "llm_flag_eval_results.csv", index=False)

    print("\n=== RESULTS ===")
    print(results_df[["method", "MAE", "delta_vs_structured"]].to_string(index=False))
    print(f"\nStructured baseline: {ref_mae:.4f}")
    print(f"Existing best decoupled residual result (for comparison): ~31.70-31.84 MAE (~0.79-0.93 min gain)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
