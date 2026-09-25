"""Stage D: does the clinical-encoder note channel beat v4b?

The honest version of the question. Everything the earlier attempt lacked:

  * all 14,066 eligible cases, not a 2,352-case sample
  * the full 2,813-case holdout, scored once per configuration
  * the current leak-safe base, not the old one
  * paired bootstrap CIs on per-case absolute error

Leak discipline: the SVD over document vectors is fitted on TRAINING rows
only. The concept anchors need no fitting at all -- they are sentences I wrote
from domain knowledge, never touched by an outcome.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error as MAE

import crcnl_data as D
from v3_corrected import clean, predict, paired
from v4_leaksafe import design, STAFFING_RISK, TRAIN_START

OUT = Path("/home/claude/crcnl2/out")
EVP = Path("/mnt/user-data/uploads/PycharmProjects/_crcnl_ev/evidence_all.parquet")
SVD_DIM = 48


def main():
    raw = D.add_schedule_derived(D.load_tabular(with_lags=True))
    raw["surgeon_durable_id"] = raw["surgeon_durable_id"].astype(str)
    raw["bmi_missing"] = raw["bmi"].isna().astype("float32")
    df, stats = clean(raw)
    df[D.CASE] = df[D.CASE].astype(str)

    tr, te, _ = D.temporal_split(df)
    y = df[D.TARGET].to_numpy(dtype="float64")
    tr = tr[df[D.TIME].iloc[tr].to_numpy() >= TRAIN_START]
    print(f"train {len(tr)}  holdout {len(te)}")

    Xb = design(df, tr, STAFFING_RISK)

    # ---- note features, aligned to the cohort -------------------------------
    nf = pd.read_parquet(OUT / "note_features.parquet")
    nf["case_durable_id"] = nf.case_durable_id.astype(str)
    ev = pd.read_parquet(EVP)[["case_durable_id"]]
    ev["case_durable_id"] = ev.case_durable_id.astype(str)
    idx = {c: i for i, c in enumerate(ev.case_durable_id)}
    pos = df[D.CASE].map(idx)
    have = pos.notna().to_numpy()
    pos_f = pos.fillna(0).astype(int).to_numpy()
    print(f"cases with notes: {have.sum()} / {len(df)} ({have.mean():.3f})")

    A = nf.set_index("case_durable_id").reindex(df[D.CASE])
    sim_cols = [c for c in A.columns if c.startswith("sim_")]
    meta_cols = ["note_n_chunks", "note_n_notes", "note_ev_chars"]
    S = A[sim_cols + meta_cols].to_numpy(dtype="float64")
    S = np.nan_to_num(S, nan=0.0)
    Sdf = pd.DataFrame(S, columns=sim_cols + meta_cols, index=Xb.index)
    Sdf["note_missing"] = (~have).astype("float32")

    # ---- document vectors -> SVD fitted on TRAIN rows only ------------------
    dm = np.load(OUT / "doc_mean.npy")[pos_f]
    dx = np.load(OUT / "doc_max.npy")[pos_f]
    dm[~have] = 0.0; dx[~have] = 0.0
    Vs = []
    for name, Mat in (("dmean", dm), ("dmax", dx)):
        sv = TruncatedSVD(n_components=SVD_DIM, random_state=42)
        sv.fit(Mat[tr])                                   # TRAIN ONLY
        Z = sv.transform(Mat)
        Vs.append(pd.DataFrame(Z, columns=[f"{name}_{i}" for i in range(SVD_DIM)],
                               index=Xb.index))
        print(f"  {name} SVD{SVD_DIM} explains {sv.explained_variance_ratio_.sum():.3f}")
    V = pd.concat(Vs, axis=1)

    configs = {
        "A_v4b_no_notes": Xb,
        "B_+anchor_sims": pd.concat([Xb, Sdf], axis=1),
        "C_+doc_svd": pd.concat([Xb, V, Sdf[["note_missing"]]], axis=1),
        "D_+both": pd.concat([Xb, Sdf, V], axis=1),
    }

    res, preds = {}, {}
    print("\nholdout, scored once per configuration:")
    for k, X in configs.items():
        p = predict(X, y, tr, te)
        m = float(MAE(y[te], p))
        preds[k] = p
        res[k] = {"n_features": int(X.shape[1]), "holdout_mae": m}
        print(f"  {k:20s} {X.shape[1]:4d} feats   MAE {m:.4f}")

    base = np.abs(y[te] - preds["A_v4b_no_notes"])
    print("\npaired vs v4b (positive = notes help):")
    for k, p in preds.items():
        if k.startswith("A_"):
            continue
        d, lo, hi = paired(base, np.abs(y[te] - p))
        sig = "  *" if not (lo <= 0 <= hi) else ""
        res[k]["gain_vs_v4b"] = d
        res[k]["ci95"] = [lo, hi]
        print(f"  {k:20s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    # restrict to cases that actually have notes -- the fairer test
    hn = have[te]
    print(f"\nrestricted to the {int(hn.sum())} holdout cases WITH notes:")
    for k, p in preds.items():
        m = float(MAE(y[te][hn], p[hn]))
        res[k]["holdout_mae_notes_only"] = m
        print(f"  {k:20s} MAE {m:.4f}")

    res["n_train"] = int(len(tr)); res["n_holdout"] = int(len(te))
    res["n_holdout_with_notes"] = int(hn.sum())
    json.dump(res, open(OUT / "STAGE_D.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
