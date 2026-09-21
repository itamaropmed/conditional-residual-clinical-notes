"""Score the extractor against the hand-read gold labels, and against the
old binary regex concept flags as the incumbent baseline."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

from gold_labels import GOLD, FIELDS
from stage_b_extract import extract_frame

OUT = Path("/home/claude/crcnl2/out")

# The incumbent: the five binary flags from the shipped pipeline.
OLD_REGEX = {
    "redo_prior_ablation": [r"\bredo\b", r"repeat\s+ablation", r"prior\s+ablation"],
    "vt_substrate": [r"\bvt\b", r"\bvf\b", r"ventricular\s+tachycardia", r"icd\s+shock"],
    "device_complexity": [r"lead\s+(revision|extraction|failure|fracture)", r"generator\s+change"],
    "congenital": [r"\bfontan\b", r"transposition", r"congenital"],
    "access_difficulty": [r"\becmo\b", r"epicardial", r"pericardial\s+access"],
}


def old_flags(ev: str) -> dict[str, int]:
    return {k: int(bool(re.search("|".join(v), ev or "", re.I)))
            for k, v in OLD_REGEX.items()}


def main():
    ev = pd.read_parquet(OUT / "evidence_ordered.parquet")
    ev["case_durable_id"] = ev.case_durable_id.astype(str)
    gold_ids = set(GOLD)
    sub = ev[ev.case_durable_id.isin(gold_ids)].copy()
    print(f"scoring on {len(sub)} hand-read cases\n")

    pred = extract_frame(sub).set_index("case_durable_id")
    G = pd.DataFrame(GOLD, index=FIELDS).T

    print(f"{'field':24s} {'exact':>7s} {'±1':>7s} {'MAE':>7s}   {'old-regex AUC-ish':>18s}")
    print("-" * 74)
    rows = []
    for f in FIELDS:
        g = G[f].reindex(pred.index)
        p = pred[f].reindex(g.index)
        m = g.notna()
        g2, p2 = g[m].astype(int), p[m].astype(int)
        exact = float((g2 == p2).mean())
        within = float((np.abs(g2 - p2) <= 1).mean())
        mae = float(np.abs(g2 - p2).mean())

        old = ""
        if f in OLD_REGEX:
            of = sub.set_index("case_durable_id").evidence.map(
                lambda e: old_flags(e)[f]).reindex(g2.index)
            # how well does a binary flag track a graded truth?
            gb = (g2 > 0).astype(int)
            agree = float((of == gb).mean())
            old = f"{agree:.3f} agree"
        rows.append({"field": f, "exact": exact, "within1": within, "mae": mae})
        print(f"{f:24s} {exact:7.3f} {within:7.3f} {mae:7.3f}   {old:>18s}")

    r = pd.DataFrame(rows)
    print("-" * 74)
    print(f"{'MEAN':24s} {r.exact.mean():7.3f} {r.within1.mean():7.3f} {r.mae.mean():7.3f}")

    # what the graded scale buys: how much information does binarising destroy?
    print("\nInformation lost by binarising (gold distribution per field):")
    for f in FIELDS[:-1]:
        g = G[f].dropna().astype(int)
        vc = g.value_counts().sort_index().to_dict()
        nz = (g > 0).mean()
        print(f"  {f:24s} {vc}   nonzero {nz:.2f}")

    r.to_csv(OUT / "extractor_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
