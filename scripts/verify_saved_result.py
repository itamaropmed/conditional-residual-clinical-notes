#!/usr/bin/env python3
"""Recompute the selected 31.5564 held-out MAE from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "final_result_31_5564.json"
PREDICTIONS = (
    ROOT
    / "results"
    / "strict_meta_gate_top_refine"
    / "strict_meta_gate_top_refine_selected_predictions.csv"
)


def main() -> None:
    metadata = json.loads(RESULT.read_text(encoding="utf-8"))
    selected = metadata["selected_variant"]
    prediction_column = f"pred_{selected}"
    frame = pd.read_csv(PREDICTIONS)
    if prediction_column not in frame.columns:
        raise KeyError(f"Missing selected prediction column: {prediction_column}")
    target = frame["case_actual_duration_time"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    mae = float(np.mean(np.abs(target - prediction)))
    expected = float(metadata["final_metrics"]["heldout_mae"])
    if len(frame) != int(metadata["heldout_cases"]):
        raise AssertionError(f"Row-count mismatch: {len(frame)} != {metadata['heldout_cases']}")
    if not np.isclose(mae, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"MAE mismatch: {mae:.15f} != {expected:.15f}")
    print(f"heldout_rows={len(frame)}")
    print(f"selected_variant={selected}")
    print(f"heldout_mae={mae:.15f}")
    print("status=PASS")


if __name__ == "__main__":
    main()
