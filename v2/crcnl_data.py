"""Shared data layer for the CRCNL rebuild.

Loads the de-identified tabular features and the LLM-extracted clinical fields,
builds the temporal patient-disjoint split, and exposes the design matrices.
No clinical note text is ever read here -- the note channel enters only through
the structured fields an extractor already produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/mnt/user-data/uploads/Clinical_notes/data/mayo")
LLM_V1 = Path("/mnt/user-data/uploads/Clinical_notes/llm_features_export_v1.parquet")

CASE = "case_durable_id"
PATIENT = "patient_durable_id"
TARGET = "case_actual_duration_time"
TIME = "scheduled_in_room"

# Columns that are only knowable after the case has happened. Excluded always.
POST_HOC = [
    "case_actual_patient_in_room", "case_actual_duration_time",
    "case_actual_prep_time", "case_actual_procedure_time", "case_actual_wrapup_time",
]

# The 11 domain LLM fields (EP/cardiac specific).
LLM11_ORD = [
    "mapping_complexity", "procedural_complexity_5", "anatomical_complexity",
    "n_complexity_drivers", "arrhythmia_complexity", "comorbidity_burden",
    "tricuspid_regurgitation_severity", "vt_morphologies_count",
]
LLM11_BIN = ["epicardial_access_anticipated", "obstructive_sleep_apnea", "refractory_af_markers"]

# The 17-field general export. Three-valued strings need explicit
# not_mentioned handling -- that is the whole point versus a regex flag,
# which cannot distinguish "absent" from "not written down".
V1_TRI = [
    "severe_cardiac_dysfunction", "chronic_anticoagulation_or_coagulopathy",
    "concurrent_additional_procedures_planned", "chronic_opioid_use_or_pain_disorder",
    "polytrauma_or_multisystem_trauma", "hemodynamic_instability_or_sepsis", "malnutrition",
]
V1_NUM = [
    "general_complexity", "num_llm_drivers", "prior_relevant_surgery_burden",
    "asa_status", "obesity", "most_recent_note_days", "context_source_ord",
    "severe_cardiac_dysfunction_ord",
]

TRI_MAP = {"yes": 1.0, "no": 0.0, "none": 0.0, "moderate": 0.5, "severe": 1.0}


def load_tabular(with_lags: bool = False) -> pd.DataFrame:
    name = ("mayo_hrs_tabular_features_with_lags_and_durations.parquet"
            if with_lags else "mayo_hrs_tabular_features_and_durations.parquet")
    df = pd.read_parquet(DATA / name)
    df = df[df[TARGET].notna()].copy()
    df[TIME] = pd.to_datetime(df[TIME], errors="coerce", utc=True)
    df = df[df[TIME].notna()].sort_values(TIME).reset_index(drop=True)
    return df


def load_llm() -> pd.DataFrame:
    """Merge both extractor outputs on case id, keyed for join."""
    top = pd.read_parquet(DATA / "mayo_hrs_llm_features_top11.parquet")
    top[CASE] = top[CASE].astype(str)

    v1 = pd.read_parquet(LLM_V1).rename(columns={"OR_CASE_ID": CASE})
    v1[CASE] = v1[CASE].astype(str)
    for c in V1_TRI:
        # Two columns per tri-state field: the graded value, and an explicit
        # "was this ever mentioned" indicator. A regex flag collapses these.
        v1[c + "__val"] = v1[c].map(TRI_MAP).astype("float32")
        v1[c + "__seen"] = (v1[c] != "not_mentioned").astype("float32")
    v1 = v1.drop(columns=V1_TRI + ["in_model_dataset"], errors="ignore")

    m = top.merge(v1, on=CASE, how="outer")
    return m


def llm_feature_columns(llm: pd.DataFrame) -> list[str]:
    cols = [c for c in llm.columns if c != CASE]
    return [c for c in cols if pd.api.types.is_numeric_dtype(llm[c])]


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.20,
                   strict_disjoint: bool = True):
    """Forward split on scheduled_in_room.

    With strict_disjoint (the CRCNL convention), training cases whose patient
    also appears in the holdout are dropped, so the two sides share no patient
    at all -- this reproduces the published 10,882 / 2,839 split. Also returns
    the new-patient-only subset of the holdout, so both numbers from Rule 11
    can be reported.
    """
    n = len(df)
    cut = int(round(n * (1.0 - test_fraction)))
    tr = np.arange(cut)
    te = np.arange(cut, n)
    pid = df[PATIENT].astype(str).to_numpy()

    if strict_disjoint:
        test_patients = set(pid[te])
        tr = tr[~np.isin(pid[tr], list(test_patients))]

    train_patients = set(pid[tr])
    is_new = ~np.isin(pid[te], list(train_patients))
    return tr, te, te[is_new]


def build_design(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature frame, target, and the raw time key. Categoricals are left as
    pandas category dtype for XGBoost's native handling; rare levels bucketed."""
    drop = set(POST_HOC + [CASE, PATIENT, "patient_birth_date", TIME,
                           "scheduled_setup_start", "scheduled_out_of_room",
                           "scheduled_cleanup_complete"])
    feat = [c for c in df.columns if c not in drop]
    X = df[feat].copy()

    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            X[c] = X[c].astype("int64") // 10**9
        elif pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
            X[c] = X[c].astype("float64")
        elif pd.api.types.is_bool_dtype(dt):
            X[c] = X[c].astype("float64")
        else:
            s = X[c].astype(str).fillna("__na__")
            vc = s.value_counts()
            rare = set(vc[vc < 25].index)          # Rule 7
            s = s.where(~s.isin(rare), "__rare__")
            X[c] = s.astype("category")

    y = df[TARGET].to_numpy(dtype="float64")
    return X, y, df[TIME].to_numpy()


def add_schedule_derived(df: pd.DataFrame) -> pd.DataFrame:
    """The booked-slot block. Audited clean earlier: 0 exact matches with the
    actual duration, 1.5% within one minute, correlation 0.70 -- a genuine
    pre-decision feature, not a leak."""
    out = df.copy()
    s_in = pd.to_datetime(out["scheduled_in_room"], errors="coerce", utc=True)
    s_out = pd.to_datetime(out["scheduled_out_of_room"], errors="coerce", utc=True)
    s_setup = pd.to_datetime(out["scheduled_setup_start"], errors="coerce", utc=True)
    s_clean = pd.to_datetime(out["scheduled_cleanup_complete"], errors="coerce", utc=True)
    out["sched_room_minutes"] = (s_out - s_in).dt.total_seconds() / 60.0
    out["sched_setup_lead_minutes"] = (s_in - s_setup).dt.total_seconds() / 60.0
    out["sched_cleanup_minutes"] = (s_clean - s_out).dt.total_seconds() / 60.0
    out["sched_hour"] = s_in.dt.hour.astype("float32")
    out["sched_dow"] = s_in.dt.dayofweek.astype("float32")
    return out
