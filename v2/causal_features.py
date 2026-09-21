"""Causal feature engine.

Every feature here is computed from STRICTLY PAST rows only, by sorting on
scheduled_in_room and shifting within group. Nothing uses the current row's
outcome, and nothing uses a future row.

This replaces the shipped lag_* block, whose exact recipe DSC-127 Rule 1 says
is not recoverable -- these are re-derivable and therefore production-scorable.

Three families:
  1. procedure multi-hot   -- the `procedures` list carries technique and
     equipment markers (CRYO, 3D MAPPING - CARTO, CARTOSOUND) that the single
     primary_procedure_name throws away
  2. expanding history     -- surgeon, procedure, and surgeon x procedure
     means over several windows
  3. booking bias          -- the booked slot correlates 0.711 with the actual
     duration but has a systematic per-surgeon, per-procedure offset. Learning
     that offset from past cases is the largest single lever available.
"""
from __future__ import annotations

import ast
import numpy as np
import pandas as pd

TIME = "scheduled_in_room"
TARGET = "case_actual_duration_time"


# ------------------------------------------------------------- procedures
def parse_procs(v) -> list[str]:
    if isinstance(v, list):
        return sorted(set(v))
    if isinstance(v, str) and v.startswith("["):
        try:
            return sorted(set(ast.literal_eval(v)))
        except Exception:
            return []
    return []


def procedure_multihot(df: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    lists = df["procedures"].map(parse_procs)
    counts: dict[str, int] = {}
    for L in lists:
        for p in L:
            counts[p] = counts.get(p, 0) + 1
    keep = sorted(p for p, c in counts.items() if c >= min_count)
    M = np.zeros((len(df), len(keep)), dtype="float32")
    pos = {p: i for i, p in enumerate(keep)}
    for r, L in enumerate(lists):
        for p in L:
            j = pos.get(p)
            if j is not None:
                M[r, j] = 1.0
    out = pd.DataFrame(M, columns=[f"proc__{p}" for p in keep], index=df.index)
    out["proc__n_unique"] = lists.map(len).to_numpy(dtype="float32")
    return out


# --------------------------------------------------------- expanding stats
def _expanding(df: pd.DataFrame, keys: list[str], value: str, prefix: str,
               windows=(10, 30, None)) -> pd.DataFrame:
    """Strictly-past mean/std/count of `value` within `keys`.

    shift(1) inside the group is what makes it strictly past: row i sees rows
    0..i-1 of its own group and never itself.
    """
    g = df.groupby(keys, observed=True, sort=False)[value]
    out = pd.DataFrame(index=df.index)
    shifted = g.shift(1)
    sg = shifted.groupby([df[k] for k in keys], observed=True, sort=False)
    for w in windows:
        tag = "all" if w is None else str(w)
        if w is None:
            out[f"{prefix}_mean_{tag}"] = sg.transform(lambda s: s.expanding().mean())
            out[f"{prefix}_std_{tag}"] = sg.transform(lambda s: s.expanding().std())
            out[f"{prefix}_cnt"] = sg.transform(lambda s: s.expanding().count())
        else:
            out[f"{prefix}_mean_{tag}"] = sg.transform(
                lambda s: s.rolling(w, min_periods=1).mean())
            out[f"{prefix}_med_{tag}"] = sg.transform(
                lambda s: s.rolling(w, min_periods=1).median())
    return out


def build_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must already be sorted ascending by TIME."""
    d = df.copy()
    d["_sched"] = (pd.to_datetime(d["scheduled_out_of_room"], errors="coerce", utc=True)
                   - pd.to_datetime(d[TIME], errors="coerce", utc=True)
                   ).dt.total_seconds() / 60.0
    d["_overrun"] = d[TARGET] - d["_sched"]
    d["_ratio"] = d[TARGET] / d["_sched"].replace(0, np.nan)
    d["_surg"] = d["surgeon_durable_id"].astype(str)
    d["_proc"] = d["primary_procedure_name"].astype(str).fillna("__na__")
    d["_sp"] = d["_surg"] + "||" + d["_proc"]

    feats = [procedure_multihot(d)]

    # duration history
    feats.append(_expanding(d, ["_sp"], TARGET, "hx_sp_dur"))
    feats.append(_expanding(d, ["_proc"], TARGET, "hx_proc_dur"))
    feats.append(_expanding(d, ["_surg"], TARGET, "hx_surg_dur"))

    # booking bias: how wrong is the booked slot for this surgeon/procedure?
    feats.append(_expanding(d, ["_sp"], "_overrun", "hx_sp_over"))
    feats.append(_expanding(d, ["_proc"], "_overrun", "hx_proc_over"))
    feats.append(_expanding(d, ["_surg"], "_overrun", "hx_surg_over"))
    feats.append(_expanding(d, ["_sp"], "_ratio", "hx_sp_ratio", windows=(30, None)))
    feats.append(_expanding(d, ["_surg"], "_ratio", "hx_surg_ratio", windows=(30, None)))

    F = pd.concat(feats, axis=1)

    # the booked slot itself, and the slot corrected by learned bias
    F["sched_minutes"] = d["_sched"].to_numpy()
    F["sched_plus_sp_bias"] = d["_sched"].to_numpy() + F["hx_sp_over_mean_all"].to_numpy()
    F["sched_x_sp_ratio"] = d["_sched"].to_numpy() * F["hx_sp_ratio_mean_all"].to_numpy()
    F["sched_plus_surg_bias"] = d["_sched"].to_numpy() + F["hx_surg_over_mean_all"].to_numpy()

    # how far is this booking from what this pair usually gets booked?
    F["sched_minus_sp_dur_mean"] = d["_sched"].to_numpy() - F["hx_sp_dur_mean_all"].to_numpy()

    # experience / familiarity
    F["surg_case_index"] = d.groupby("_surg", observed=True).cumcount().to_numpy()
    F["sp_case_index"] = d.groupby("_sp", observed=True).cumcount().to_numpy()
    F["proc_case_index"] = d.groupby("_proc", observed=True).cumcount().to_numpy()

    # calendar
    t = pd.to_datetime(d[TIME], errors="coerce", utc=True)
    F["sched_hour"] = t.dt.hour.to_numpy(dtype="float32")
    F["sched_dow"] = t.dt.dayofweek.to_numpy(dtype="float32")
    F["sched_month_idx"] = ((t.dt.year - t.dt.year.min()) * 12 + t.dt.month).to_numpy(dtype="float32")

    # same-day load: how many cases this surgeon has that day, and this case's
    # position in the list. Known from the schedule, not from outcomes.
    day = t.dt.date.astype(str)
    F["surg_day_load"] = d.groupby([d["_surg"], day], observed=True)[TARGET].transform("size").to_numpy()
    F["surg_day_pos"] = d.groupby([d["_surg"], day], observed=True).cumcount().to_numpy()

    return F.astype("float32")
