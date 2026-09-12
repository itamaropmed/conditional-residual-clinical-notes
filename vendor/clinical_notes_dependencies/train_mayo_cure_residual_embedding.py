#!/usr/bin/env python3
"""Train and evaluate target-conditioned residual note embeddings on Mayo HRS data.

The output artifact of the CURE step is an embedding vector Z_i for each case.
During training, Z_i is learned from pre-cutoff clinical notes with residual
supervision from the structured-only model. During test, the embedder sees only
tabular metadata and pre-cutoff notes. It never sees the test target or test
residual.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeCV, SGDClassifier, SGDRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    def tqdm(iterable=None, **_: Any) -> Any:
        return iterable if iterable is not None else []

from train_mayo_tabular_xgboost_baseline import (
    CASE_ID_COLUMN,
    DATE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    add_safe_features,
    make_feature_frame,
    make_model,
    make_preprocessor,
    markdown_table,
    metric_dict,
    procedure_slice_metrics,
    read_table,
    require_xgboost,
    run_cv_objective,
    suggest_params,
    temporal_group_holdout,
)

NOTE_TEXT_METADATA_COLUMNS = {
    "note_text",
    "note_doc_types_used",
    "note_dates_used",
}

CLINICAL_TOKEN_PATTERN = r"(?u)\b(?=[^\W\d_]*[^\W\d_])\w[\w-]{1,}\b"

POST_PROCEDURE_DOC_TYPE_PATTERNS = [
    r"post\s*[- ]?\s*procedure",
    r"postprocedure",
    r"post\s+discharge",
    r"discharge",
    r"hospital\s+course",
    r"anesthesia\s+post",
]

SAME_DAY_PREOP_DOC_TYPE_PATTERNS = [
    r"preprocedure",
    r"pre\s*[- ]?\s*procedure",
    r"\bh\s*&\s*p\b",
    r"history\s+and\s+physical",
    r"consult",
    r"telephone\s+encounter",
]

SOURCE_CHANNEL_PATTERNS: dict[str, list[str]] = {
    "anesthesia_preop": [r"anesthesia.*pre", r"preprocedure.*anesthesia"],
    "consult_hnp": [r"consult", r"\bh\s*&\s*p\b", r"history\s+and\s+physical", r"outpatient\s+summary"],
    "telephone": [r"telephone", r"phone", r"message"],
    "diagnostic_device": [
        r"diagnostic",
        r"imaging",
        r"ecg",
        r"echo",
        r"ct\b",
        r"mri",
        r"monitor",
        r"holter",
        r"device",
        r"laboratory",
    ],
    "progress_plan": [r"progress", r"plan\s+of\s+care", r"nursing", r"transfer", r"addendum", r"research"],
}

CLINICAL_CONCEPT_PATTERNS: dict[str, list[str]] = {
    "redo_prior_ablation": [
        r"\bredo\b",
        r"repeat\s+ablation",
        r"prior\s+ablation",
        r"s/p\s+[^.]{0,80}\bablation\b",
        r"previous\s+[^.]{0,80}\bablation\b",
    ],
    "congenital_complex_anatomy": [
        r"\bfontan\b",
        r"tricuspid\s+atresia",
        r"transposition",
        r"coarctation",
        r"single\s+ventricle",
        r"tetralogy",
        r"congenital",
        r"\bvsd\b",
        r"\basd\b",
    ],
    "vt_pvc_instability": [
        r"\bvt\b",
        r"\bvf\b",
        r"ventricular\s+tachycardia",
        r"premature\s+ventricular",
        r"\bpvc",
        r"icd\s+shock",
        r"appropriate\s+icd",
        r"vt\s+storm",
        r"rvot",
        r"epicardial",
        r"scar",
    ],
    "device_lead_complexity": [
        r"\bcrt\b",
        r"lv\s+lead",
        r"lead\s+(revision|extraction|failure|fracture)",
        r"generator\s+change",
        r"venous\s+occlusion",
        r"\bicd\b",
        r"pacemaker",
        r"biventricular",
    ],
    "accessory_svt_mapping": [
        r"accessory\s+pathway",
        r"\bwpw\b",
        r"\bavnrt\b",
        r"supraventricular\s+tachycardia",
        r"\bsvt\b",
        r"mapping",
        r"septal",
        r"transseptal",
    ],
    "anticoag_thrombus_tee": [
        r"\btee\b",
        r"thrombus",
        r"left\s+atrial\s+appendage",
        r"\blaa\b",
        r"anticoag",
        r"heparin",
        r"apixaban",
        r"warfarin",
        r"rivaroxaban",
    ],
    "inpatient_instability": [
        r"\binpatient\b",
        r"icu",
        r"\becmo\b",
        r"\blvad\b",
        r"oxygen",
        r"\bcpap\b",
        r"pulmonary\s+embol",
        r"hemodynamic",
        r"volume\s+overload",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-data",
        default="data/mayo/mayo_hrs_tabular_features_and_durations.parquet",
        help="Path to Mayo tabular features/durations parquet.",
    )
    parser.add_argument(
        "--notes-data",
        default="mayo_hrs_notes_anonymized_merged.parquet",
        help="Path to anonymized Mayo notes parquet.",
    )
    parser.add_argument("--output-dir", default="output/mayo_cure_residual_embedding")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--tabular-n-trials", type=int, default=20)
    parser.add_argument("--tabular-timeout", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--svd-dim", type=int, default=192)
    parser.add_argument("--max-tfidf-features", type=int, default=40000)
    parser.add_argument("--max-notes-per-case", type=int, default=8)
    parser.add_argument("--max-note-chars", type=int, default=900)
    parser.add_argument("--max-case-note-chars", type=int, default=6000)
    parser.add_argument("--max-note-age-days", type=float, default=730.0)
    parser.add_argument(
        "--note-cutoff-policy",
        choices=["strict_date", "timestamp"],
        default="strict_date",
        help=(
            "strict_date keeps only notes from calendar dates before the procedure date. "
            "timestamp uses doc_date < scheduled_in_room and is less safe when doc_date is date-only."
        ),
    )
    parser.add_argument(
        "--include-same-day-preop-note-types",
        action="store_true",
        help="With strict_date, also allow same-day note types that look explicitly pre-op, never post-op/discharge.",
    )
    parser.add_argument(
        "--keep-postop-doc-types",
        action="store_true",
        help="Diagnostic only. By default, post-procedure/discharge doc types are excluded to avoid leakage.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test cap after sorting by date.")
    parser.add_argument("--max-note-rows", type=int, default=None, help="Optional smoke-test cap for streamed notes.")
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument(
        "--transformer-model",
        default="",
        help="Optional local HuggingFace encoder, e.g. UFNLP/gatortron-base or emilyalsentzer/Bio_ClinicalBERT.",
    )
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    parser.add_argument("--transformer-max-length", type=int, default=512)
    parser.add_argument(
        "--transformer-svd-dim",
        type=int,
        default=128,
        help="Train-only SVD compression dimension for frozen transformer note embeddings.",
    )
    parser.add_argument("--mil-max-chunks", type=int, default=12)
    parser.add_argument("--mil-chunk-svd-dim", type=int, default=128)
    parser.add_argument("--mil-hidden-dim", type=int, default=128)
    parser.add_argument("--mil-epochs", type=int, default=80)
    parser.add_argument("--mil-batch-size", type=int, default=128)
    parser.add_argument("--mil-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mil-contrastive-weight", type=float, default=0.05)
    parser.add_argument(
        "--residual-neighbor-ks",
        default="5,15,50",
        help="Comma-separated train-neighborhood sizes used for target-conditioned residual geometry.",
    )
    parser.add_argument(
        "--residual-neighbor-context-weight",
        type=float,
        default=1.0,
        help="Weight applied to structured context inside the residual-neighborhood geometry.",
    )
    parser.add_argument(
        "--residual-neighbor-note-weight",
        type=float,
        default=1.0,
        help="Weight applied to note embeddings inside the residual-neighborhood geometry.",
    )
    parser.add_argument(
        "--mil-backend",
        choices=["fast", "torch", "off"],
        default="fast",
        help="fast = vectorized sklearn residual-attention encoder; torch = experimental neural MIL; off = skip MIL.",
    )
    parser.add_argument(
        "--fusion-svd-dim",
        type=int,
        default=96,
        help="Compression dimension for the fused CURE target-conditioned embedding.",
    )
    parser.add_argument(
        "--fusion-pls-dim",
        type=int,
        default=12,
        help="Residual-supervised PLS coordinates for the fused embedding.",
    )
    parser.add_argument(
        "--sparse-residual-max-features",
        type=int,
        default=12000,
        help="Max TF-IDF features for the sparse residual lexical embedding arm.",
    )
    parser.add_argument(
        "--sparse-residual-regressors",
        default="huber",
        help="Comma-separated sparse residual regressors: huber,squared,epsilon.",
    )
    parser.add_argument(
        "--sparse-residual-max-interactions",
        type=int,
        default=2,
        help="Number of scaled context columns used as text interactions in the sparse residual arm.",
    )
    parser.add_argument(
        "--sparse-residual-top-procedures",
        type=int,
        default=4,
        help="Number of procedure-specific text interaction blocks in the sparse residual arm.",
    )
    parser.add_argument(
        "--sparse-residual-classifiers",
        action="store_true",
        help="Also add high-positive/high-negative/high-absolute residual sparse classifiers. Slower; off by default.",
    )
    parser.add_argument(
        "--skip-oof-ensemble",
        action="store_true",
        help="Skip the expensive OOF residual-ensemble arm during fast embedding search.",
    )
    parser.add_argument("--include-scheduled-duration", action="store_true", default=True)
    parser.add_argument("--drop-scheduled-duration", action="store_false", dest="include_scheduled_duration")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    return sorted(set(out)) or [5, 15, 50]


def clean_note_text(text: Any, max_chars: int) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def procedure_family(value: Any) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ["pvi", "atrial fibrillation", "afib", "flutter", "cti"]):
        return "af_pvi_flutter"
    if any(token in text for token in ["vt", "pvc", "ventricular"]):
        return "vt_pvc"
    if any(token in text for token in ["device", "icd", "pacemaker", "generator", "crt", "lead"]):
        return "device_lead"
    if any(token in text for token in ["svt", "avnrt", "accessory", "pathway", "wpw"]):
        return "svt_pathway"
    if "tilt" in text:
        return "tilt"
    return "other"


def source_channel_for_doc_type(doc_type: Any) -> str:
    text = str(doc_type or "").lower()
    for channel, patterns in SOURCE_CHANNEL_PATTERNS.items():
        if doc_type_matches(text, patterns):
            return channel
    return "other"


def concept_markers_for_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    markers = []
    for concept, patterns in CLINICAL_CONCEPT_PATTERNS.items():
        hits = 0
        for pattern in patterns:
            hits += len(re.findall(pattern, lowered, flags=re.IGNORECASE))
        if hits:
            markers.extend([f"concept_{concept}"] * min(4, hits))
    return markers


def source_aware_note_text_for_case(note_text: Any, procedure: Any) -> str:
    """Inject legal source/procedure/concept markers into pre-cutoff note text."""
    family = procedure_family(procedure)
    lines = str(note_text or "").splitlines()
    enriched_parts = [f"proc_family_{family}"]
    if not any(line.strip() for line in lines):
        return " ".join(enriched_parts)

    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        doc_type = "unknown"
        body = line
        match = re.match(r"^\[NOTE_TYPE:\s*([^\]]+)\]\s*(.*)$", line)
        if match:
            doc_type = match.group(1).strip()
            body = match.group(2).strip()
        channel = source_channel_for_doc_type(doc_type)
        markers = concept_markers_for_text(body)
        source_markers = [f"src_{channel}", f"src_{channel}"]
        source_markers.extend(f"src_{channel}_{marker}" for marker in markers)
        source_markers.extend(f"proc_{family}_{marker}" for marker in markers)
        enriched_parts.append(" ".join([*source_markers, body]))
    return "\n".join(enriched_parts)


def make_source_aware_note_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    procedures = enriched.get("primary_procedure_name", pd.Series([""] * len(enriched), index=enriched.index))
    enriched["note_text"] = [
        source_aware_note_text_for_case(note_text, proc)
        for note_text, proc in zip(enriched["note_text"].fillna(""), procedures.fillna(""))
    ]
    return enriched


HIGH_SIGNAL_SENTENCE_PATTERNS = [
    r"\bablation\b",
    r"\bpvi\b",
    r"\bvt\b",
    r"\bvf\b",
    r"\bpvc",
    r"\bsvt\b",
    r"\bwpw\b",
    r"\bavnrt\b",
    r"atrial\s+flutter",
    r"atrial\s+fibrillation",
    r"\bicd\b",
    r"\bcrt\b",
    r"lv\s+lead",
    r"generator",
    r"lead\s+(revision|extraction|failure|fracture)",
    r"accessory\s+pathway",
    r"\bfontan\b",
    r"congenital",
    r"tricuspid\s+atresia",
    r"transposition",
    r"\btee\b",
    r"thrombus",
    r"anticoag",
    r"\bheparin\b",
    r"mapping",
    r"epicardial",
    r"scar",
    r"shock",
    r"burden",
    r"recurrence",
    r"\bredo\b",
    r"prior\s+ablation",
    r"repeat\s+ablation",
    r"\becmo\b",
    r"\blvad\b",
    r"hemodynamic",
]


def is_high_signal_sentence(sentence: str) -> bool:
    text = str(sentence or "").lower()
    if concept_markers_for_text(text):
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in HIGH_SIGNAL_SENTENCE_PATTERNS)


def high_signal_source_aware_note_text_for_case(note_text: Any, procedure: Any, max_sentences: int = 36) -> str:
    family = procedure_family(procedure)
    kept: list[str] = [f"proc_family_{family}"]
    for line in str(note_text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        doc_type = "unknown"
        body = line
        match = re.match(r"^\[NOTE_TYPE:\s*([^\]]+)\]\s*(.*)$", line)
        if match:
            doc_type = match.group(1).strip()
            body = match.group(2).strip()
        channel = source_channel_for_doc_type(doc_type)
        sentences = re.split(r"(?<=[.!?])\s+|;\s+|\|\s+", body)
        selected = []
        for sent in sentences:
            sent = re.sub(r"\s+", " ", sent).strip()
            if len(sent) < 12:
                continue
            if is_high_signal_sentence(sent):
                selected.append(sent)
            if len(selected) >= 8:
                break
        if not selected and channel in {"consult_hnp", "telephone", "anesthesia_preop"}:
            selected = [sent.strip() for sent in sentences[:2] if len(sent.strip()) >= 12]
        for sent in selected:
            markers = concept_markers_for_text(sent)
            source_markers = [f"src_{channel}", f"hs_src_{channel}"]
            source_markers.extend(f"src_{channel}_{marker}" for marker in markers)
            source_markers.extend(f"proc_{family}_{marker}" for marker in markers)
            kept.append(" ".join([*source_markers, sent]))
            if len(kept) >= max_sentences + 1:
                break
        if len(kept) >= max_sentences + 1:
            break
    return "\n".join(kept)


def make_high_signal_source_aware_note_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    procedures = enriched.get("primary_procedure_name", pd.Series([""] * len(enriched), index=enriched.index))
    enriched["note_text"] = [
        high_signal_source_aware_note_text_for_case(note_text, proc)
        for note_text, proc in zip(enriched["note_text"].fillna(""), procedures.fillna(""))
    ]
    return enriched


def source_aware_marker_summary(texts: pd.Series) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for text in texts.fillna(""):
        for token in re.findall(r"\b(?:src|concept|proc)_[A-Za-z0-9_]+\b", str(text)):
            counts[token] += 1
    return [
        {"marker": marker, "case_count": count}
        for marker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:80]
    ]


def case_records_by_patient(case_df: pd.DataFrame) -> tuple[dict[str, list[dict[str, Any]]], dict[int, int]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_id_to_pos: dict[int, int] = {}
    for pos, row in enumerate(case_df.itertuples(index=False)):
        case_id = int(getattr(row, CASE_ID_COLUMN))
        patient_id = str(getattr(row, PATIENT_ID_COLUMN))
        cutoff = pd.Timestamp(getattr(row, DATE_COLUMN))
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        by_patient[patient_id].append({"case_id": case_id, "cutoff": cutoff, "pos": pos})
        case_id_to_pos[case_id] = pos
    for records in by_patient.values():
        records.sort(key=lambda x: x["cutoff"])
    return dict(by_patient), case_id_to_pos


def doc_type_matches(doc_type: str, patterns: list[str]) -> bool:
    text = str(doc_type or "").strip().lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def is_post_procedure_doc_type(doc_type: str) -> bool:
    return doc_type_matches(doc_type, POST_PROCEDURE_DOC_TYPE_PATTERNS)


def is_same_day_preop_doc_type(doc_type: str) -> bool:
    return doc_type_matches(doc_type, SAME_DAY_PREOP_DOC_TYPE_PATTERNS) and not is_post_procedure_doc_type(doc_type)


def note_allowed_before_case(
    doc_ts: pd.Timestamp,
    doc_type: str,
    cutoff: pd.Timestamp,
    note_cutoff_policy: str,
    include_same_day_preop_note_types: bool,
) -> bool:
    if note_cutoff_policy == "timestamp":
        return bool(doc_ts < cutoff)

    doc_day = doc_ts.date()
    cutoff_day = cutoff.date()
    if doc_day < cutoff_day:
        return True
    if doc_day > cutoff_day:
        return False
    return bool(include_same_day_preop_note_types and is_same_day_preop_doc_type(doc_type))


def attach_note_to_case_heaps(
    heaps: dict[int, list[tuple[float, str, str, str]]],
    records: list[dict[str, Any]],
    doc_ts: pd.Timestamp,
    doc_type: str,
    text: str,
    max_notes_per_case: int,
    max_note_age_days: float,
    note_cutoff_policy: str,
    include_same_day_preop_note_types: bool,
) -> None:
    if not text:
        return
    cutoffs = [r["cutoff"] for r in records]
    # The patient usually has one case, but this keeps the logic correct.
    for record, cutoff in zip(records, cutoffs):
        if not note_allowed_before_case(
            doc_ts=doc_ts,
            doc_type=doc_type,
            cutoff=cutoff,
            note_cutoff_policy=note_cutoff_policy,
            include_same_day_preop_note_types=include_same_day_preop_note_types,
        ):
            continue
        age_days = (cutoff - doc_ts).total_seconds() / 86400.0
        if age_days < 0 or age_days > max_note_age_days:
            continue
        case_id = int(record["case_id"])
        heap = heaps[case_id]
        note_score = doc_ts.timestamp()
        item = (note_score, doc_ts.isoformat(), doc_type, text)
        if len(heap) < max_notes_per_case:
            heapq.heappush(heap, item)
        elif note_score > heap[0][0]:
            heapq.heapreplace(heap, item)


def build_case_note_bundles(
    notes_path: Path,
    case_df: pd.DataFrame,
    max_notes_per_case: int,
    max_note_chars: int,
    max_case_note_chars: int,
    max_note_age_days: float,
    max_note_rows: int | None,
    note_cutoff_policy: str,
    include_same_day_preop_note_types: bool,
    keep_postop_doc_types: bool,
) -> pd.DataFrame:
    by_patient, case_id_to_pos = case_records_by_patient(case_df)
    needed_patients = set(by_patient)
    heaps: dict[int, list[tuple[float, str, str, str]]] = defaultdict(list)

    pf = pq.ParquetFile(notes_path)
    rows_seen = 0
    rows_matched_patient = 0
    rows_rejected_postop_doc_type = 0
    rows_attached = 0
    columns = ["epic_id", "doc_date", "doc_type", "content_markdown"]

    total_note_rows = pf.metadata.num_rows if pf.metadata is not None else None
    if max_note_rows is not None:
        total_note_rows = min(total_note_rows or max_note_rows, max_note_rows)
    total_batches = math.ceil(total_note_rows / 50000) if total_note_rows else None

    note_batches = tqdm(
        pf.iter_batches(columns=columns, batch_size=50000),
        total=total_batches,
        desc="Streaming notes",
        unit="batch",
    )
    for batch in note_batches:
        chunk = batch.to_pandas()
        rows_seen += len(chunk)
        if max_note_rows is not None and rows_seen > max_note_rows:
            chunk = chunk.iloc[: max(0, len(chunk) - (rows_seen - max_note_rows))]

        chunk = chunk[chunk["epic_id"].astype(str).isin(needed_patients)]
        rows_matched_patient += len(chunk)
        if not chunk.empty:
            chunk["doc_date"] = pd.to_datetime(chunk["doc_date"], errors="coerce", utc=True)
            chunk = chunk[chunk["doc_date"].notna()]
            if not keep_postop_doc_types:
                postop_mask = chunk["doc_type"].astype(str).map(is_post_procedure_doc_type)
                rows_rejected_postop_doc_type += int(postop_mask.sum())
                chunk = chunk[~postop_mask]
        for row in chunk.itertuples(index=False):
            patient_id = str(row.epic_id)
            text = clean_note_text(row.content_markdown, max_note_chars)
            before = sum(len(heaps[int(r["case_id"])]) for r in by_patient[patient_id])
            attach_note_to_case_heaps(
                heaps=heaps,
                records=by_patient[patient_id],
                doc_ts=pd.Timestamp(row.doc_date),
                doc_type=str(row.doc_type),
                text=text,
                max_notes_per_case=max_notes_per_case,
                max_note_age_days=max_note_age_days,
                note_cutoff_policy=note_cutoff_policy,
                include_same_day_preop_note_types=include_same_day_preop_note_types,
            )
            after = sum(len(heaps[int(r["case_id"])]) for r in by_patient[patient_id])
            rows_attached += max(0, after - before)

        if max_note_rows is not None and rows_seen >= max_note_rows:
            break
        note_batches.set_postfix(
            rows_seen=f"{rows_seen:,}",
            matched=f"{rows_matched_patient:,}",
            rejected_postop=f"{rows_rejected_postop_doc_type:,}",
            attached=f"{rows_attached:,}",
        )

    rows: list[dict[str, Any]] = []
    for case_id, pos in case_id_to_pos.items():
        notes = sorted(heaps.get(case_id, []), key=lambda x: x[0], reverse=True)
        parts = []
        doc_types = []
        doc_dates = []
        for _, doc_date, doc_type, text in notes:
            doc_types.append(doc_type)
            doc_dates.append(doc_date)
            parts.append(f"[NOTE_TYPE: {doc_type}] {text}")
        bundle = "\n".join(parts)
        if len(bundle) > max_case_note_chars:
            bundle = bundle[:max_case_note_chars].rsplit(" ", 1)[0]
        rows.append(
            {
                CASE_ID_COLUMN: case_id,
                "note_text": bundle,
                "n_pre_cutoff_notes_used": len(notes),
                "note_doc_types_used": "; ".join(doc_types[:max_notes_per_case]),
                "note_dates_used": "; ".join(doc_dates[:max_notes_per_case]),
                "note_chars": len(bundle),
            }
        )

    result = pd.DataFrame(rows)
    result.attrs["note_rows_seen"] = rows_seen
    result.attrs["note_rows_matched_patient"] = rows_matched_patient
    result.attrs["note_rows_rejected_postop_doc_type"] = rows_rejected_postop_doc_type
    result.attrs["note_cutoff_policy"] = note_cutoff_policy
    result.attrs["include_same_day_preop_note_types"] = include_same_day_preop_note_types
    result.attrs["keep_postop_doc_types"] = keep_postop_doc_types
    result.attrs["note_rows_attached_events"] = rows_attached
    return result


def train_tabular_with_optuna(
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    target: str,
    n_trials: int,
    timeout: int | None,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[str], list[str]]:
    _, _, numeric_cols, categorical_cols = make_feature_frame(train_df, target)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: run_cv_objective(
            trial=trial,
            XGBRegressor=XGBRegressor,
            train_df=train_df,
            target=target,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=seed,
            n_folds=n_folds,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    return dict(study.best_params), study.trials_dataframe(), numeric_cols, categorical_cols


def fit_predict_tabular(
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, list[str], list[str]]:
    X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(train_df, target)
    X_test, _, _, _ = make_feature_frame(test_df, target)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", make_model(XGBRegressor, params, seed)),
        ]
    )
    pipe.fit(X_train, y_train)
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    return pipe, train_pred, test_pred, numeric_cols, categorical_cols


def out_of_fold_structured_residuals(
    XGBRegressor: Any,
    train_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    seed: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(train_df, target)
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    oof_pred = np.zeros(len(train_df), dtype=float)

    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", make_model(XGBRegressor, params, seed + 1000 + fold_idx)),
            ]
        )
        pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
        oof_pred[val_idx] = pipe.predict(X_all.iloc[val_idx])
    return oof_pred, y_all.to_numpy(dtype=float) - oof_pred


def fit_tfidf_svd_embedding(
    train_text: pd.Series,
    test_text: pd.Series,
    max_features: int,
    min_df: int,
    svd_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_text = train_text.fillna("")
    test_text = test_text.fillna("")
    if not train_text.str.strip().any():
        z_train = np.zeros((len(train_text), 1), dtype="float32")
        z_test = np.zeros((len(test_text), 1), dtype="float32")
        meta = {
            "vectorizer": None,
            "svd": None,
            "scaler": None,
            "explained_variance_ratio_sum": 0.0,
            "n_terms": 0,
            "embedding_dim": 1,
            "fallback_reason": "no_nonempty_train_notes",
        }
        return z_train, z_test, meta

    last_error = None
    for candidate_min_df in [min_df, 1]:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=CLINICAL_TOKEN_PATTERN,
                ngram_range=(1, 2),
                min_df=candidate_min_df,
                max_df=0.97,
                max_features=max_features,
                sublinear_tf=True,
            )
            x_train = vectorizer.fit_transform(train_text)
            break
        except ValueError as exc:
            last_error = exc
    else:
        z_train = np.zeros((len(train_text), 1), dtype="float32")
        z_test = np.zeros((len(test_text), 1), dtype="float32")
        meta = {
            "vectorizer": None,
            "svd": None,
            "scaler": None,
            "explained_variance_ratio_sum": 0.0,
            "n_terms": 0,
            "embedding_dim": 1,
            "fallback_reason": f"tfidf_failed: {last_error}",
        }
        return z_train, z_test, meta

    x_test = vectorizer.transform(test_text)
    n_components = max(2, min(svd_dim, x_train.shape[0] - 2, x_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    z_train = svd.fit_transform(x_train)
    z_test = svd.transform(x_test)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    meta = {
        "vectorizer": vectorizer,
        "svd": svd,
        "scaler": scaler,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "n_terms": int(len(vectorizer.vocabulary_)),
        "embedding_dim": int(n_components),
    }
    return z_train, z_test, meta


def mean_pool_transformer(last_hidden: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def fit_transformer_note_embedding(
    train_text: pd.Series,
    test_text: pd.Series,
    model_name: str,
    batch_size: int,
    max_length: int,
    svd_dim: int,
    seed: int,
    local_files_only: bool = True,
    device_override: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Frozen clinical encoder baseline with train-only scaling/compression."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("torch and transformers are required for --transformer-model.") from exc

    train_text = train_text.fillna("").astype(str)
    test_text = test_text.fillna("").astype(str)
    if device_override == "auto":
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    else:
        device = device_override
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "dim", 0) or 768)

    def encode(texts: pd.Series) -> np.ndarray:
        arr = np.zeros((len(texts), hidden_size), dtype="float32")
        nonempty = [i for i, text in enumerate(texts.tolist()) if text.strip()]
        iterator = tqdm(
            range(0, len(nonempty), batch_size),
            desc=f"Encoding notes with {model_name}",
            unit="batch",
        )
        with torch.no_grad():
            for start in iterator:
                batch_idx = nonempty[start : start + batch_size]
                batch_text = [texts.iloc[i] for i in batch_idx]
                encoded = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded)
                pooled = mean_pool_transformer(output.last_hidden_state, encoded["attention_mask"])
                arr[batch_idx] = pooled.detach().cpu().numpy().astype("float32")
        return arr

    raw_train = encode(train_text)
    raw_test = encode(test_text)
    n_components = max(2, min(svd_dim, raw_train.shape[0] - 2, raw_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    z_train = svd.fit_transform(raw_train)
    z_test = svd.transform(raw_test)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    meta = {
        "model_name": model_name,
        "device": device,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "local_files_only": bool(local_files_only),
        "device_override": device_override,
        "raw_hidden_size": int(hidden_size),
        "embedding_dim": int(n_components),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "train_nonempty_notes": int(train_text.str.strip().astype(bool).sum()),
        "test_nonempty_notes": int(test_text.str.strip().astype(bool).sum()),
        "tokenizer": tokenizer,
        "model": model.cpu(),
        "svd": svd,
        "scaler": scaler,
    }
    return z_train.astype("float32"), z_test.astype("float32"), meta


def hidden_layer_embedding(mlp: MLPRegressor, x: np.ndarray) -> np.ndarray:
    hidden = x @ mlp.coefs_[0] + mlp.intercepts_[0]
    activation = getattr(mlp, "activation", "relu")
    if activation == "relu":
        return np.maximum(hidden, 0.0)
    if activation == "tanh":
        return np.tanh(hidden)
    if activation == "logistic":
        return 1.0 / (1.0 + np.exp(-hidden))
    return hidden


def fit_cure_residual_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    final_structured_train_pred: np.ndarray,
    final_structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    svd_dim: int,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_text = train_df["note_text"].fillna("")
    test_text = test_df["note_text"].fillna("")
    if not train_text.str.strip().any():
        z_train = np.zeros((len(train_df), 1), dtype="float32")
        z_test = np.zeros((len(test_df), 1), dtype="float32")
        train_residual_probe = np.zeros(len(train_df), dtype="float32")
        test_residual_probe = np.zeros(len(test_df), dtype="float32")
        meta = {
            "vectorizer": None,
            "svd": None,
            "low_scaler": None,
            "context_scaler": None,
            "residual_scaler": None,
            "mlp": None,
            "z_scaler": None,
            "context_cols": [],
            "residual_world_centroid_labels": [],
            "residual_world_count": 0,
            "residual_probe_clip_low_high": [0.0, 0.0],
            "n_terms": 0,
            "svd_dim": 0,
            "embedding_dim": 1,
            "mlp_n_iter": 0,
            "svd_explained_variance_ratio_sum": 0.0,
            "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_residual_probe)),
            "fallback_reason": "no_nonempty_train_notes",
        }
        return z_train, z_test, train_residual_probe, test_residual_probe, meta

    last_error = None
    for candidate_min_df in [min_df, 1]:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=CLINICAL_TOKEN_PATTERN,
                ngram_range=(1, 2),
                min_df=candidate_min_df,
                max_df=0.97,
                max_features=max_features,
                sublinear_tf=True,
            )
            x_text_train = vectorizer.fit_transform(train_text)
            break
        except ValueError as exc:
            last_error = exc
    else:
        z_train = np.zeros((len(train_df), 1), dtype="float32")
        z_test = np.zeros((len(test_df), 1), dtype="float32")
        train_residual_probe = np.zeros(len(train_df), dtype="float32")
        test_residual_probe = np.zeros(len(test_df), dtype="float32")
        meta = {
            "vectorizer": None,
            "svd": None,
            "low_scaler": None,
            "context_scaler": None,
            "residual_scaler": None,
            "mlp": None,
            "z_scaler": None,
            "context_cols": [],
            "residual_world_centroid_labels": [],
            "residual_world_count": 0,
            "residual_probe_clip_low_high": [0.0, 0.0],
            "n_terms": 0,
            "svd_dim": 0,
            "embedding_dim": 1,
            "mlp_n_iter": 0,
            "svd_explained_variance_ratio_sum": 0.0,
            "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_residual_probe)),
            "fallback_reason": f"tfidf_failed: {last_error}",
        }
        return z_train, z_test, train_residual_probe, test_residual_probe, meta

    x_text_test = vectorizer.transform(test_text)

    # Context available at inference: structured model prediction + scheduled room estimate.
    context_cols = ["scheduled_room_minutes", "scheduled_hour", "case_num_procedures", "patient_age"]
    context_train = train_df[context_cols].copy()
    context_test = test_df[context_cols].copy()
    context_train["structured_prediction_for_training"] = oof_structured_pred
    context_test["structured_prediction_for_training"] = final_structured_test_pred
    context_scaler = StandardScaler()
    c_train = context_scaler.fit_transform(context_train.fillna(context_train.median(numeric_only=True)))
    c_test = context_scaler.transform(context_test.fillna(context_train.median(numeric_only=True)))

    # Conditional text features: the same phrase can mean different things in different tabular regimes.
    conditioned_parts_train = [x_text_train]
    conditioned_parts_test = [x_text_test]
    for k in range(c_train.shape[1]):
        conditioned_parts_train.append(x_text_train.multiply(c_train[:, [k]]))
        conditioned_parts_test.append(x_text_test.multiply(c_test[:, [k]]))
    x_cond_train = sparse.hstack(conditioned_parts_train, format="csr")
    x_cond_test = sparse.hstack(conditioned_parts_test, format="csr")

    n_components = max(2, min(svd_dim, x_cond_train.shape[0] - 2, x_cond_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    low_train = svd.fit_transform(x_cond_train)
    low_test = svd.transform(x_cond_test)
    low_scaler = StandardScaler()
    low_train = low_scaler.fit_transform(low_train)
    low_test = low_scaler.transform(low_test)

    y_scaler = StandardScaler()
    y_res = y_scaler.fit_transform(oof_residual.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=(embedding_dim,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=min(128, max(16, len(train_df) // 20)),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    mlp.fit(low_train, y_res)
    z_train = hidden_layer_embedding(mlp, low_train)
    z_test = hidden_layer_embedding(mlp, low_test)

    train_residual_probe = y_scaler.inverse_transform(mlp.predict(low_train).reshape(-1, 1)).ravel()
    test_residual_probe = y_scaler.inverse_transform(mlp.predict(low_test).reshape(-1, 1)).ravel()
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    train_residual_probe = np.clip(train_residual_probe, lo, hi)
    test_residual_probe = np.clip(test_residual_probe, lo, hi)

    # Residual-world coordinates: prototypes are learned from train residual quantiles.
    # At test time, cases are embedded by similarity to those train-only worlds.
    quantile_edges = np.unique(np.quantile(oof_residual, np.linspace(0.0, 1.0, 7)))
    if len(quantile_edges) <= 2:
        residual_bins = np.zeros(len(oof_residual), dtype=int)
    else:
        residual_bins = np.digitize(oof_residual, quantile_edges[1:-1], right=True)
    centroids = []
    centroid_labels = []
    for bin_id in sorted(np.unique(residual_bins)):
        mask = residual_bins == bin_id
        if mask.sum() < 3:
            continue
        centroids.append(low_train[mask].mean(axis=0))
        centroid_labels.append(int(bin_id))
    if centroids:
        centroid_matrix = np.vstack(centroids)
        train_world = cosine_similarity(low_train, centroid_matrix)
        test_world = cosine_similarity(low_test, centroid_matrix)
    else:
        train_world = np.zeros((len(train_df), 0), dtype=float)
        test_world = np.zeros((len(test_df), 0), dtype=float)

    z_train = np.hstack([z_train, train_world, train_residual_probe.reshape(-1, 1)])
    z_test = np.hstack([z_test, test_world, test_residual_probe.reshape(-1, 1)])
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_test = z_scaler.transform(z_test)
    meta = {
        "vectorizer": vectorizer,
        "svd": svd,
        "low_scaler": low_scaler,
        "context_scaler": context_scaler,
        "residual_scaler": y_scaler,
        "mlp": mlp,
        "z_scaler": z_scaler,
        "context_cols": context_cols,
        "residual_world_centroid_labels": centroid_labels,
        "residual_world_count": int(len(centroid_labels)),
        "residual_probe_clip_low_high": [float(lo), float(hi)],
        "n_terms": int(len(vectorizer.vocabulary_)),
        "svd_dim": int(n_components),
        "embedding_dim": int(z_train.shape[1]),
        "mlp_n_iter": int(mlp.n_iter_),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_residual_probe)),
    }
    return z_train, z_test, train_residual_probe, test_residual_probe, meta


def make_sparse_residual_regressors(seed: int, names: list[str] | None = None) -> dict[str, SGDRegressor]:
    all_models = {
        "huber_elasticnet": SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.15,
            epsilon=0.10,
            max_iter=700,
            tol=1e-4,
            n_iter_no_change=5,
            random_state=seed,
            average=True,
        ),
        "squared_l2": SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=5e-5,
            max_iter=700,
            tol=1e-4,
            n_iter_no_change=5,
            random_state=seed + 1,
            average=True,
        ),
        "epsilon_insensitive": SGDRegressor(
            loss="epsilon_insensitive",
            penalty="elasticnet",
            alpha=5e-5,
            l1_ratio=0.10,
            epsilon=0.20,
            max_iter=700,
            tol=1e-4,
            n_iter_no_change=5,
            random_state=seed + 2,
            average=True,
        ),
    }
    aliases = {
        "huber": "huber_elasticnet",
        "squared": "squared_l2",
        "epsilon": "epsilon_insensitive",
    }
    if not names:
        return {"huber_elasticnet": all_models["huber_elasticnet"]}
    selected = []
    for name in names:
        key = aliases.get(name.strip(), name.strip())
        if key in all_models and key not in selected:
            selected.append(key)
    if not selected:
        selected = ["huber_elasticnet"]
    return {key: all_models[key] for key in selected}


def sparse_contribution_stats(x: sparse.csr_matrix, coef: np.ndarray) -> np.ndarray:
    weighted = x.multiply(coef).tocsr()
    pos = weighted.maximum(0).sum(axis=1).A.ravel()
    neg = weighted.minimum(0).sum(axis=1).A.ravel()
    abs_sum = np.abs(weighted).sum(axis=1).A.ravel()
    nnz = np.diff(weighted.indptr).astype("float32")
    return np.column_stack([pos, neg, abs_sum, nnz]).astype("float32")


def fit_sparse_residual_lexical_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    n_folds: int,
    seed: int,
    top_procedures: int = 4,
    max_interactions: int = 2,
    regressor_names: list[str] | None = None,
    enable_classifiers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train-safe residual-supervised sparse note embedding.

    This arm treats the embedding itself as an optimization object: coordinates
    are residual predictions, high-residual probabilities, and signed lexical
    contribution summaries produced out-of-fold for train cases. Test cases are
    transformed by models fit only on the train set.
    """
    train_text = train_df["note_text"].fillna("")
    test_text = test_df["note_text"].fillna("")
    if not train_text.str.strip().any():
        z_train = np.zeros((len(train_df), 1), dtype="float32")
        z_test = np.zeros((len(test_df), 1), dtype="float32")
        train_resid = np.zeros(len(train_df), dtype="float32")
        test_resid = np.zeros(len(test_df), dtype="float32")
        return z_train, z_test, train_resid, test_resid, {"backend": "sparse_residual_lexical", "fallback": "no_notes"}

    last_error = None
    for candidate_min_df in [min_df, 1]:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=CLINICAL_TOKEN_PATTERN,
                ngram_range=(1, 3),
                min_df=candidate_min_df,
                max_df=0.985,
                max_features=max_features,
                sublinear_tf=True,
                norm="l2",
            )
            x_text_train = vectorizer.fit_transform(train_text)
            break
        except ValueError as exc:
            last_error = exc
    else:
        z_train = np.zeros((len(train_df), 1), dtype="float32")
        z_test = np.zeros((len(test_df), 1), dtype="float32")
        train_resid = np.zeros(len(train_df), dtype="float32")
        test_resid = np.zeros(len(test_df), dtype="float32")
        return (
            z_train,
            z_test,
            train_resid,
            test_resid,
            {"backend": "sparse_residual_lexical", "fallback": f"tfidf_failed: {last_error}"},
        )

    x_text_test = vectorizer.transform(test_text)

    context = pd.DataFrame(index=train_df.index)
    context_test = pd.DataFrame(index=test_df.index)
    context["structured_pred"] = oof_structured_pred
    context_test["structured_pred"] = structured_test_pred
    for col in [
        "scheduled_room_minutes",
        "case_num_procedures",
        "case_num_providers",
        "patient_age",
        "scheduled_hour",
        "scheduled_dayofweek",
        "n_pre_cutoff_notes_used",
        "note_chars",
    ]:
        if col in train_df.columns:
            context[col] = pd.to_numeric(train_df[col], errors="coerce")
            context_test[col] = pd.to_numeric(test_df[col], errors="coerce")
    med = context.median(numeric_only=True)
    context_scaler = StandardScaler()
    c_train = context_scaler.fit_transform(context.fillna(med)).astype("float32")
    c_test = context_scaler.transform(context_test.fillna(med)).astype("float32")
    n_interactions = min(max_interactions, c_train.shape[1])
    interaction_cols = list(context.columns[:n_interactions])

    parts_train = [x_text_train]
    parts_test = [x_text_test]
    for idx in range(n_interactions):
        parts_train.append(x_text_train.multiply(c_train[:, [idx]]))
        parts_test.append(x_text_test.multiply(c_test[:, [idx]]))

    proc_col = "primary_procedure_name"
    top_proc_values: list[str] = []
    if proc_col in train_df.columns and top_procedures > 0:
        counts = train_df[proc_col].fillna("__missing__").astype(str).value_counts()
        top_proc_values = counts.head(top_procedures).index.tolist()
        train_proc = train_df[proc_col].fillna("__missing__").astype(str)
        test_proc = test_df[proc_col].fillna("__missing__").astype(str)
        for proc in top_proc_values:
            train_mask = (train_proc == proc).astype("float32").to_numpy()
            test_mask = (test_proc == proc).astype("float32").to_numpy()
            parts_train.append(x_text_train.multiply(train_mask[:, None]))
            parts_test.append(x_text_test.multiply(test_mask[:, None]))

    x_train = sparse.hstack(parts_train, format="csr")
    x_test = sparse.hstack(parts_test, format="csr")
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(oof_residual.reshape(-1, 1)).ravel()
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)

    regressors = make_sparse_residual_regressors(seed, regressor_names)
    train_blocks = []
    test_blocks = []
    model_rows = []
    full_models: dict[str, SGDRegressor] = {}
    best_regressor_name = ""
    best_oof_mae = float("inf")
    best_oof_pred = np.zeros(len(train_df), dtype="float32")
    best_test_pred = np.zeros(len(test_df), dtype="float32")
    best_train_contrib = np.zeros((len(train_df), 4), dtype="float32")
    best_test_contrib = np.zeros((len(test_df), 4), dtype="float32")

    for model_idx, (model_name, base_model) in enumerate(regressors.items()):
        oof_scaled = np.zeros(len(train_df), dtype="float32")
        contrib = np.zeros((len(train_df), 4), dtype="float32")
        fold_iter = splitter.split(x_train, y_scaled, groups=groups)
        fold_iter = tqdm(
            list(fold_iter),
            desc=f"Sparse residual {model_name}",
            unit="fold",
            leave=False,
        )
        for fold_idx, (fit_idx, val_idx) in enumerate(fold_iter, start=1):
            model = clone(base_model)
            model.random_state = seed + 100 * model_idx + fold_idx
            model.fit(x_train[fit_idx], y_scaled[fit_idx])
            oof_scaled[val_idx] = model.predict(x_train[val_idx]).astype("float32")
            contrib[val_idx] = sparse_contribution_stats(x_train[val_idx], model.coef_)
        oof_pred = y_scaler.inverse_transform(oof_scaled.reshape(-1, 1)).ravel()
        lo, hi = np.quantile(oof_residual, [0.005, 0.995])
        oof_pred = np.clip(oof_pred, lo, hi)
        oof_mae = float(mean_absolute_error(oof_residual, oof_pred))
        full_model = clone(base_model)
        full_model.random_state = seed + 1000 + model_idx
        full_model.fit(x_train, y_scaled)
        test_scaled = full_model.predict(x_test).astype("float32")
        test_pred = y_scaler.inverse_transform(test_scaled.reshape(-1, 1)).ravel()
        test_pred = np.clip(test_pred, lo, hi)
        test_contrib = sparse_contribution_stats(x_test, full_model.coef_)

        train_blocks.extend([oof_pred.reshape(-1, 1), contrib])
        test_blocks.extend([test_pred.reshape(-1, 1), test_contrib])
        full_models[model_name] = full_model
        model_rows.append({"model": model_name, "oof_residual_mae": oof_mae})
        if oof_mae < best_oof_mae:
            best_oof_mae = oof_mae
            best_regressor_name = model_name
            best_oof_pred = oof_pred.astype("float32")
            best_test_pred = test_pred.astype("float32")
            best_train_contrib = contrib
            best_test_contrib = test_contrib

    class_blocks_train = []
    class_blocks_test = []
    classifier_rows = []
    classifier_targets = {}
    if enable_classifiers:
        classifier_targets = {
            "high_positive": oof_residual >= np.quantile(oof_residual, 0.80),
            "high_negative": oof_residual <= np.quantile(oof_residual, 0.20),
            "high_absolute": np.abs(oof_residual) >= np.quantile(np.abs(oof_residual), 0.80),
            "extreme_positive": oof_residual >= np.quantile(oof_residual, 0.90),
            "extreme_negative": oof_residual <= np.quantile(oof_residual, 0.10),
        }
    for class_name, labels in classifier_targets.items():
        if len(np.unique(labels)) < 2:
            continue
        oof_prob = np.zeros(len(train_df), dtype="float32")
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_train, labels, groups=groups), start=1):
            if len(np.unique(labels[fit_idx])) < 2:
                oof_prob[val_idx] = float(labels[fit_idx].mean())
                continue
            clf = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=2e-5,
                l1_ratio=0.15,
                max_iter=700,
                tol=1e-4,
                n_iter_no_change=5,
                random_state=seed + 3000 + fold_idx,
                average=True,
            )
            clf.fit(x_train[fit_idx], labels[fit_idx].astype(int))
            oof_prob[val_idx] = clf.predict_proba(x_train[val_idx])[:, 1].astype("float32")
        clf_full = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.15,
            max_iter=700,
            tol=1e-4,
            n_iter_no_change=5,
            random_state=seed + 4000,
            average=True,
        )
        clf_full.fit(x_train, labels.astype(int))
        test_prob = clf_full.predict_proba(x_test)[:, 1].astype("float32")
        class_blocks_train.append(oof_prob.reshape(-1, 1))
        class_blocks_test.append(test_prob.reshape(-1, 1))
        classifier_rows.append(
            {
                "classifier": class_name,
                "positive_rate": float(labels.mean()),
                "mean_oof_probability": float(oof_prob.mean()),
            }
        )

    residual_projection_rows = []
    residual_projection_k = min(80, x_text_train.shape[1])
    if residual_projection_k >= 4:
        # Target-aware lexical coordinates: terms are selected by train residual
        # correlation only, then exposed to the downstream model as embedding
        # coordinates for train/test. This is not generic semantics; it is a
        # residual direction in sparse clinical language space.
        term_scores = np.asarray(x_text_train.T @ y_scaled).ravel()
        term_df = np.diff(x_text_train.tocsc().indptr)
        eligible = term_df >= max(2, min_df)
        term_scores = np.where(eligible, term_scores, 0.0)
        half_k = max(2, residual_projection_k // 2)
        pos_idx = np.argsort(term_scores)[-half_k:][::-1]
        neg_idx = np.argsort(term_scores)[:half_k]
        selected_idx = np.unique(np.concatenate([pos_idx, neg_idx]))
        selected_idx = selected_idx[np.abs(term_scores[selected_idx]) > 0]
        if len(selected_idx):
            selected_train = x_text_train[:, selected_idx].toarray().astype("float32")
            selected_test = x_text_test[:, selected_idx].toarray().astype("float32")
            selected_scores = term_scores[selected_idx].astype("float32")
            pos_mask = selected_scores > 0
            neg_mask = selected_scores < 0
            projection_train = selected_train @ selected_scores
            projection_test = selected_test @ selected_scores
            extra_train = [
                selected_train,
                projection_train.reshape(-1, 1),
                np.abs(selected_train).sum(axis=1, keepdims=True),
            ]
            extra_test = [
                selected_test,
                projection_test.reshape(-1, 1),
                np.abs(selected_test).sum(axis=1, keepdims=True),
            ]
            if pos_mask.any():
                extra_train.append(selected_train[:, pos_mask].sum(axis=1, keepdims=True))
                extra_test.append(selected_test[:, pos_mask].sum(axis=1, keepdims=True))
            if neg_mask.any():
                extra_train.append(selected_train[:, neg_mask].sum(axis=1, keepdims=True))
                extra_test.append(selected_test[:, neg_mask].sum(axis=1, keepdims=True))
            train_blocks.append(np.hstack(extra_train))
            test_blocks.append(np.hstack(extra_test))
            residual_projection_rows = [
                {
                    "term": str(vectorizer.get_feature_names_out()[idx]),
                    "score": float(term_scores[idx]),
                    "df": int(term_df[idx]),
                }
                for idx in selected_idx[np.argsort(np.abs(term_scores[selected_idx]))[::-1]][:40]
            ]

    procedure_head_rows = []
    procedure_head_train_blocks = []
    procedure_head_test_blocks = []
    if top_proc_values:
        proc_base_model = make_sparse_residual_regressors(seed, ["huber"])["huber_elasticnet"]
        train_proc_array = train_proc.to_numpy()
        test_proc_array = test_proc.to_numpy()
        for proc_idx, proc in enumerate(top_proc_values):
            proc_train_mask = train_proc_array == proc
            proc_test_mask = test_proc_array == proc
            proc_n = int(proc_train_mask.sum())
            proc_groups = groups[proc_train_mask]
            proc_unique_groups = np.unique(proc_groups)
            if proc_n < 120 or len(proc_unique_groups) < 3:
                continue
            proc_oof_scaled = np.zeros(len(train_df), dtype="float32")
            proc_test_scaled = np.zeros(len(test_df), dtype="float32")
            proc_split_count = min(n_folds, len(proc_unique_groups))
            proc_splitter = GroupKFold(n_splits=proc_split_count)
            proc_indices = np.where(proc_train_mask)[0]
            for fold_idx, (fit_local, val_local) in enumerate(
                proc_splitter.split(x_text_train[proc_train_mask], y_scaled[proc_train_mask], groups=proc_groups),
                start=1,
            ):
                fit_idx = proc_indices[fit_local]
                val_idx = proc_indices[val_local]
                model = clone(proc_base_model)
                model.random_state = seed + 5000 + 100 * proc_idx + fold_idx
                model.fit(x_text_train[fit_idx], y_scaled[fit_idx])
                proc_oof_scaled[val_idx] = model.predict(x_text_train[val_idx]).astype("float32")
            full_proc_model = clone(proc_base_model)
            full_proc_model.random_state = seed + 7000 + proc_idx
            full_proc_model.fit(x_text_train[proc_train_mask], y_scaled[proc_train_mask])
            if proc_test_mask.any():
                proc_test_scaled[proc_test_mask] = full_proc_model.predict(x_text_test[proc_test_mask]).astype("float32")
            proc_oof_pred = y_scaler.inverse_transform(proc_oof_scaled.reshape(-1, 1)).ravel().astype("float32")
            proc_test_pred = y_scaler.inverse_transform(proc_test_scaled.reshape(-1, 1)).ravel().astype("float32")
            proc_oof_pred[~proc_train_mask] = 0.0
            proc_test_pred[~proc_test_mask] = 0.0
            lo, hi = np.quantile(oof_residual, [0.005, 0.995])
            proc_oof_pred = np.clip(proc_oof_pred, lo, hi)
            proc_test_pred = np.clip(proc_test_pred, lo, hi)
            procedure_head_train_blocks.append(proc_oof_pred.reshape(-1, 1))
            procedure_head_test_blocks.append(proc_test_pred.reshape(-1, 1))
            proc_mae = mean_absolute_error(oof_residual[proc_train_mask], proc_oof_pred[proc_train_mask])
            zero_mae = mean_absolute_error(oof_residual[proc_train_mask], np.zeros(proc_n, dtype="float32"))
            procedure_head_rows.append(
                {
                    "procedure": str(proc),
                    "train_cases": proc_n,
                    "test_cases": int(proc_test_mask.sum()),
                    "oof_residual_mae": float(proc_mae),
                    "zero_residual_mae": float(zero_mae),
                    "delta_vs_zero": float(proc_mae - zero_mae),
                }
            )
        if procedure_head_train_blocks:
            train_blocks.append(np.hstack(procedure_head_train_blocks).astype("float32"))
            test_blocks.append(np.hstack(procedure_head_test_blocks).astype("float32"))

    z_train_raw = np.hstack([*train_blocks, *class_blocks_train, c_train]).astype("float32")
    z_test_raw = np.hstack([*test_blocks, *class_blocks_test, c_test]).astype("float32")
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train_raw).astype("float32")
    z_test = z_scaler.transform(z_test_raw).astype("float32")

    feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    best_model = full_models[best_regressor_name]
    base_vocab = len(feature_names)
    base_coef = np.asarray(best_model.coef_[:base_vocab])
    top_pos_idx = np.argsort(base_coef)[-30:][::-1]
    top_neg_idx = np.argsort(base_coef)[:30]
    meta = {
        "backend": "oof_sparse_residual_lexical_embedding",
        "vectorizer": vectorizer,
        "context_scaler": context_scaler,
        "residual_scaler": y_scaler,
        "models": full_models,
        "z_scaler": z_scaler,
        "n_terms": int(len(vectorizer.vocabulary_)),
        "conditional_feature_dim": int(x_train.shape[1]),
        "embedding_dim": int(z_train.shape[1]),
        "interaction_cols": interaction_cols,
        "top_procedure_interactions": top_proc_values,
        "regressor_selection": sorted(model_rows, key=lambda row: row["oof_residual_mae"]),
        "classifier_summary": classifier_rows,
        "residual_projection_terms": residual_projection_rows,
        "procedure_residual_heads": sorted(procedure_head_rows, key=lambda row: row["delta_vs_zero"]),
        "best_regressor": best_regressor_name,
        "train_residual_probe_mae_on_oof_residual": float(best_oof_mae),
        "best_positive_terms": [
            {"term": str(feature_names[i]), "coef": float(base_coef[i])} for i in top_pos_idx if base_coef[i] > 0
        ],
        "best_negative_terms": [
            {"term": str(feature_names[i]), "coef": float(base_coef[i])} for i in top_neg_idx if base_coef[i] < 0
        ],
        "best_train_contribution_stats_mean": best_train_contrib.mean(axis=0).tolist(),
        "best_test_contribution_stats_mean": best_test_contrib.mean(axis=0).tolist(),
    }
    return z_train, z_test, best_oof_pred, best_test_pred, meta


def fit_source_aware_sparse_residual_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    n_folds: int,
    seed: int,
    top_procedures: int = 4,
    max_interactions: int = 2,
    regressor_names: list[str] | None = None,
    enable_classifiers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Residual embedding with source/channel/concept markers added to legal notes.

    The only additional information is deterministic metadata already available
    before prediction time: note type, procedure family, and phrases present in
    the pre-cutoff text. Residual labels are used only inside the train folds.
    """
    source_train_df = make_source_aware_note_frame(train_df)
    source_test_df = make_source_aware_note_frame(test_df)
    z_train, z_test, train_probe, test_probe, meta = fit_sparse_residual_lexical_embedding(
        train_df=source_train_df,
        test_df=source_test_df,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        max_features=max_features,
        min_df=min_df,
        n_folds=n_folds,
        seed=seed,
        top_procedures=top_procedures,
        max_interactions=max_interactions,
        regressor_names=regressor_names,
        enable_classifiers=enable_classifiers,
    )
    meta = dict(meta)
    meta["backend"] = "source_aware_sparse_residual_lexical_embedding"
    meta["source_marker_summary_train"] = source_aware_marker_summary(source_train_df["note_text"])
    meta["procedure_family_counts_train"] = (
        train_df.get("primary_procedure_name", pd.Series([""] * len(train_df), index=train_df.index))
        .map(procedure_family)
        .value_counts()
        .to_dict()
    )
    return z_train, z_test, train_probe, test_probe, meta, source_train_df, source_test_df


def fit_high_signal_source_aware_sparse_residual_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    n_folds: int,
    seed: int,
    top_procedures: int = 4,
    max_interactions: int = 2,
    regressor_names: list[str] | None = None,
    enable_classifiers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Residual embedding over high-signal source-aware snippets only."""
    source_train_df = make_high_signal_source_aware_note_frame(train_df)
    source_test_df = make_high_signal_source_aware_note_frame(test_df)
    z_train, z_test, train_probe, test_probe, meta = fit_sparse_residual_lexical_embedding(
        train_df=source_train_df,
        test_df=source_test_df,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        max_features=max_features,
        min_df=min_df,
        n_folds=n_folds,
        seed=seed,
        top_procedures=top_procedures,
        max_interactions=max_interactions,
        regressor_names=regressor_names,
        enable_classifiers=enable_classifiers,
    )
    meta = dict(meta)
    meta["backend"] = "high_signal_source_aware_sparse_residual_lexical_embedding"
    meta["source_marker_summary_train"] = source_aware_marker_summary(source_train_df["note_text"])
    meta["mean_high_signal_chars_train"] = float(source_train_df["note_text"].str.len().mean())
    meta["mean_original_note_chars_train"] = float(train_df["note_text"].fillna("").str.len().mean())
    return z_train, z_test, train_probe, test_probe, meta, source_train_df, source_test_df


def require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "torch is required for the MIL residual encoder. Install it in the project venv and rerun."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def split_case_chunks(note_text: str, max_chunks: int) -> list[str]:
    text = str(note_text or "")
    if not text.strip():
        return []
    chunks: list[str] = []
    # Most bundles are one note per line, ordered from most recent to older.
    for part in re.split(r"\n+", text):
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) < 20:
            continue
        if len(part) <= 900:
            chunks.append(part)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", part)
            buf = []
            n_chars = 0
            for sent in sentences:
                if n_chars + len(sent) > 900 and buf:
                    chunks.append(" ".join(buf))
                    buf = [sent]
                    n_chars = len(sent)
                else:
                    buf.append(sent)
                    n_chars += len(sent)
            if buf:
                chunks.append(" ".join(buf))
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks]


def build_chunk_tensor(
    train_text: pd.Series,
    test_text: pd.Series,
    max_chunks: int,
    max_features: int,
    min_df: int,
    svd_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_chunks = [split_case_chunks(text, max_chunks) for text in train_text.fillna("")]
    test_chunks = [split_case_chunks(text, max_chunks) for text in test_text.fillna("")]
    flat_train_chunks = [chunk for chunks in train_chunks for chunk in chunks]
    if not flat_train_chunks:
        x_train = np.zeros((len(train_chunks), max_chunks, 1), dtype="float32")
        mask_train = np.zeros((len(train_chunks), max_chunks), dtype="float32")
        x_test = np.zeros((len(test_chunks), max_chunks, 1), dtype="float32")
        mask_test = np.zeros((len(test_chunks), max_chunks), dtype="float32")
        meta = {
            "chunk_vectorizer": None,
            "chunk_svd": None,
            "chunk_scaler": None,
            "n_terms": 0,
            "chunk_svd_dim": 1,
            "max_chunks": int(max_chunks),
            "train_chunks_total": 0,
            "test_chunks_total": int(sum(len(x) for x in test_chunks)),
            "train_cases_with_chunks": 0,
            "test_cases_with_chunks": int(sum(bool(x) for x in test_chunks)),
            "svd_explained_variance_ratio_sum": 0.0,
            "fallback_reason": "no_train_note_chunks",
        }
        return x_train, mask_train, x_test, mask_test, meta

    last_error = None
    for candidate_min_df in [min_df, 1]:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=CLINICAL_TOKEN_PATTERN,
                ngram_range=(1, 2),
                min_df=candidate_min_df,
                max_df=0.97,
                max_features=max_features,
                sublinear_tf=True,
            )
            x_flat_train = vectorizer.fit_transform(flat_train_chunks)
            break
        except ValueError as exc:
            last_error = exc
    else:
        x_train = np.zeros((len(train_chunks), max_chunks, 1), dtype="float32")
        mask_train = np.zeros((len(train_chunks), max_chunks), dtype="float32")
        x_test = np.zeros((len(test_chunks), max_chunks, 1), dtype="float32")
        mask_test = np.zeros((len(test_chunks), max_chunks), dtype="float32")
        meta = {
            "chunk_vectorizer": None,
            "chunk_svd": None,
            "chunk_scaler": None,
            "n_terms": 0,
            "chunk_svd_dim": 1,
            "max_chunks": int(max_chunks),
            "train_chunks_total": int(sum(len(x) for x in train_chunks)),
            "test_chunks_total": int(sum(len(x) for x in test_chunks)),
            "train_cases_with_chunks": int(sum(bool(x) for x in train_chunks)),
            "test_cases_with_chunks": int(sum(bool(x) for x in test_chunks)),
            "svd_explained_variance_ratio_sum": 0.0,
            "fallback_reason": f"chunk_tfidf_failed: {last_error}",
        }
        return x_train, mask_train, x_test, mask_test, meta

    n_components = max(2, min(svd_dim, x_flat_train.shape[0] - 2, x_flat_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    flat_train_low = svd.fit_transform(x_flat_train).astype("float32")

    low_scaler = StandardScaler()
    flat_train_low = low_scaler.fit_transform(flat_train_low).astype("float32")

    def pack(chunks_by_case: list[list[str]], fit_flat_low: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        tensor = np.zeros((len(chunks_by_case), max_chunks, n_components), dtype="float32")
        mask = np.zeros((len(chunks_by_case), max_chunks), dtype="float32")
        if fit_flat_low is not None:
            cursor = 0
            for i, chunks in enumerate(chunks_by_case):
                n = min(len(chunks), max_chunks)
                if n:
                    tensor[i, :n, :] = fit_flat_low[cursor : cursor + n]
                    mask[i, :n] = 1.0
                    cursor += n
            return tensor, mask

        flat = [chunk for chunks in chunks_by_case for chunk in chunks]
        if flat:
            low = low_scaler.transform(svd.transform(vectorizer.transform(flat))).astype("float32")
        else:
            low = np.zeros((0, n_components), dtype="float32")
        cursor = 0
        for i, chunks in enumerate(chunks_by_case):
            n = min(len(chunks), max_chunks)
            if n:
                tensor[i, :n, :] = low[cursor : cursor + n]
                mask[i, :n] = 1.0
                cursor += n
        return tensor, mask

    x_train, mask_train = pack(train_chunks, flat_train_low)
    x_test, mask_test = pack(test_chunks)
    meta = {
        "chunk_vectorizer": vectorizer,
        "chunk_svd": svd,
        "chunk_scaler": low_scaler,
        "n_terms": int(len(vectorizer.vocabulary_)),
        "chunk_svd_dim": int(n_components),
        "max_chunks": int(max_chunks),
        "train_chunks_total": int(sum(len(x) for x in train_chunks)),
        "test_chunks_total": int(sum(len(x) for x in test_chunks)),
        "train_cases_with_chunks": int(sum(bool(x) for x in train_chunks)),
        "test_cases_with_chunks": int(sum(bool(x) for x in test_chunks)),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    return x_train, mask_train, x_test, mask_test, meta


def build_mil_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    context_cols = [
        "scheduled_room_minutes",
        "scheduled_setup_lead_minutes",
        "scheduled_cleanup_minutes",
        "scheduled_hour",
        "scheduled_dayofweek",
        "case_num_procedures",
        "case_num_providers",
        "patient_age",
        "is_redo",
    ]
    train_context = train_df[context_cols].copy()
    test_context = test_df[context_cols].copy()
    train_context["structured_pred"] = oof_structured_pred
    test_context["structured_pred"] = structured_test_pred
    med = train_context.median(numeric_only=True)
    scaler = StandardScaler()
    c_train = scaler.fit_transform(train_context.fillna(med)).astype("float32")
    c_test = scaler.transform(test_context.fillna(med)).astype("float32")
    return c_train, c_test, scaler, [*context_cols, "structured_pred"]


def residual_bins(y: np.ndarray, n_bins: int = 6) -> np.ndarray:
    edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(y), dtype="int64")
    return np.digitize(y, edges[1:-1], right=True).astype("int64")


def fit_mil_residual_encoder(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    chunk_svd_dim: int,
    max_chunks: int,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    contrastive_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    torch, nn, F, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, mask_train, x_test, mask_test, chunk_meta = build_chunk_tensor(
        train_df["note_text"],
        test_df["note_text"],
        max_chunks=max_chunks,
        max_features=max_features,
        min_df=min_df,
        svd_dim=chunk_svd_dim,
        seed=seed,
    )
    c_train, c_test, context_scaler, context_cols = build_mil_context(
        train_df, test_df, oof_structured_pred, structured_test_pred
    )
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(oof_residual.reshape(-1, 1)).astype("float32").ravel()
    bins_train = residual_bins(oof_residual)

    device = "cpu"

    class ResidualMIL(nn.Module):
        def __init__(self, chunk_dim: int, context_dim: int, hidden_dim: int, z_dim: int):
            super().__init__()
            self.chunk_encoder = nn.Sequential(
                nn.Linear(chunk_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, z_dim),
                nn.ReLU(),
            )
            self.context_query = nn.Sequential(nn.Linear(context_dim, z_dim), nn.Tanh())
            self.residual_head = nn.Sequential(
                nn.Linear(z_dim + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x_chunks: Any, mask: Any, context: Any) -> tuple[Any, Any, Any]:
            z_chunks = self.chunk_encoder(x_chunks)
            query = self.context_query(context).unsqueeze(1)
            scores = (z_chunks * query).sum(dim=-1) / math.sqrt(z_chunks.shape[-1])
            scores = scores.masked_fill(mask <= 0, -1e4)
            attn = torch.softmax(scores, dim=1)
            empty = (mask.sum(dim=1, keepdim=True) <= 0).float()
            attn = attn * (1.0 - empty)
            z = (z_chunks * attn.unsqueeze(-1)).sum(dim=1)
            residual = self.residual_head(torch.cat([z, context], dim=1)).squeeze(1)
            return z, residual, attn

    def supcon_loss(z: Any, context: Any, labels: Any) -> Any:
        if contrastive_weight <= 0:
            return z.sum() * 0.0
        z_norm = F.normalize(z, p=2, dim=1)
        sim = z_norm @ z_norm.T / 0.20
        n = z.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        with torch.no_grad():
            dist = torch.cdist(context, context)
            threshold = torch.quantile(dist[~eye], 0.35) if n > 2 else dist.max()
            pos = (labels[:, None] == labels[None, :]) & (dist <= threshold) & (~eye)
        if not pos.any():
            return z.sum() * 0.0
        sim = sim.masked_fill(eye, -1e4)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        denom = pos.sum(dim=1).clamp_min(1)
        loss_per = -(log_prob * pos.float()).sum(dim=1) / denom
        active = pos.any(dim=1)
        return loss_per[active].mean() if active.any() else z.sum() * 0.0

    model = ResidualMIL(x_train.shape[-1], c_train.shape[-1], hidden_dim, embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()

    all_idx = np.arange(len(train_df))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)
    val_size = max(1, int(0.15 * len(all_idx)))
    val_idx = all_idx[:val_size]
    fit_idx = all_idx[val_size:]

    dataset = TensorDataset(
        torch.tensor(x_train[fit_idx], dtype=torch.float32),
        torch.tensor(mask_train[fit_idx], dtype=torch.float32),
        torch.tensor(c_train[fit_idx], dtype=torch.float32),
        torch.tensor(y_train[fit_idx], dtype=torch.float32),
        torch.tensor(bins_train[fit_idx], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_val = torch.tensor(x_train[val_idx], dtype=torch.float32, device=device)
    m_val = torch.tensor(mask_train[val_idx], dtype=torch.float32, device=device)
    c_val = torch.tensor(c_train[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 12
    wait = 0
    epoch_iter = tqdm(range(1, epochs + 1), desc="MIL residual encoder", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        for xb, mb, cb, yb, lb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            cb = cb.to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, pred, _ = model(xb, mb, cb)
            loss = loss_fn(pred, yb) + contrastive_weight * supcon_loss(z, cb, lb)
            train_losses.append(float(loss.detach().cpu()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, val_pred, _ = model(x_val, m_val, c_val)
            val_loss = float(loss_fn(val_pred, y_val).detach().cpu())
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                epoch_iter.set_postfix(
                    train_loss=f"{np.mean(train_losses):.4f}" if train_losses else "nan",
                    val_loss=f"{val_loss:.4f}",
                    best=f"{best_val:.4f}",
                    best_epoch=best_epoch,
                    stop="patience",
                )
                break
        epoch_iter.set_postfix(
            train_loss=f"{np.mean(train_losses):.4f}" if train_losses else "nan",
            val_loss=f"{val_loss:.4f}",
            best=f"{best_val:.4f}",
            best_epoch=best_epoch,
            wait=wait,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def encode(x: np.ndarray, m: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zs = []
        preds = []
        attns = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
                mb = torch.tensor(m[start : start + batch_size], dtype=torch.float32, device=device)
                cb = torch.tensor(c[start : start + batch_size], dtype=torch.float32, device=device)
                z, pred, attn = model(xb, mb, cb)
                zs.append(z.cpu().numpy())
                preds.append(pred.cpu().numpy())
                attns.append(attn.cpu().numpy())
        z_arr = np.vstack(zs)
        pred_scaled = np.concatenate(preds)
        attn_arr = np.vstack(attns)
        pred_resid = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        lo, hi = np.quantile(oof_residual, [0.005, 0.995])
        pred_resid = np.clip(pred_resid, lo, hi)
        return z_arr, pred_resid, attn_arr

    z_train, train_resid_pred, train_attn = encode(x_train, mask_train, c_train)
    z_test, test_resid_pred, test_attn = encode(x_test, mask_test, c_test)
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_test = z_scaler.transform(z_test)

    meta = {
        **chunk_meta,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "context_scaler": context_scaler,
        "context_cols": context_cols,
        "residual_scaler": y_scaler,
        "z_scaler": z_scaler,
        "device_used": device,
        "best_epoch": int(best_epoch),
        "best_validation_loss_scaled": float(best_val),
        "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_resid_pred)),
        "embedding_dim": int(z_train.shape[1]),
        "hidden_dim": int(hidden_dim),
        "contrastive_weight": float(contrastive_weight),
    }
    return z_train, z_test, train_resid_pred, test_resid_pred, meta


def masked_softmax_abs(scores: np.ndarray, mask: np.ndarray, temperature: float = 25.0) -> np.ndarray:
    scaled = np.abs(scores) / max(temperature, 1e-6)
    scaled = np.where(mask > 0, scaled, -1e9)
    row_max = np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled - row_max) * (mask > 0)
    denom = exp.sum(axis=1, keepdims=True)
    return np.divide(exp, denom, out=np.zeros_like(exp), where=denom > 0)


def fit_fast_mil_residual_encoder(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    max_features: int,
    min_df: int,
    chunk_svd_dim: int,
    max_chunks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fast target-aware residual chunk encoder.

    This is a pragmatic encoder for local iteration: it learns a residual scorer
    over note chunks using train residuals, turns the scorer into attention
    weights, and pools chunks into a case embedding. It is much faster than the
    experimental PyTorch MIL loop while preserving the key conditional idea.
    """
    x_train, mask_train, x_test, mask_test, chunk_meta = build_chunk_tensor(
        train_df["note_text"],
        test_df["note_text"],
        max_chunks=max_chunks,
        max_features=max_features,
        min_df=min_df,
        svd_dim=chunk_svd_dim,
        seed=seed,
    )
    c_train, c_test, context_scaler, context_cols = build_mil_context(
        train_df, test_df, oof_structured_pred, structured_test_pred
    )

    n_train, max_chunks_actual, chunk_dim = x_train.shape
    n_test = x_test.shape[0]

    train_rows = []
    train_targets = []
    for i in range(n_train):
        valid = np.where(mask_train[i] > 0)[0]
        if len(valid) == 0:
            continue
        context_block = np.repeat(c_train[i : i + 1], len(valid), axis=0)
        train_rows.append(np.hstack([x_train[i, valid], context_block]))
        train_targets.extend([oof_residual[i]] * len(valid))

    if train_rows:
        chunk_X = np.vstack(train_rows)
        chunk_y = np.asarray(train_targets, dtype="float32")
        chunk_scaler = StandardScaler()
        chunk_X = chunk_scaler.fit_transform(chunk_X)
        chunk_model = RidgeCV(alphas=np.logspace(-3, 3, 9))
        chunk_model.fit(chunk_X, chunk_y)
    else:
        chunk_scaler = None
        chunk_model = None

    def score_chunks(x: np.ndarray, mask: np.ndarray, context: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(x), max_chunks_actual), dtype="float32")
        if chunk_model is None or chunk_scaler is None:
            return scores
        for start in range(0, len(x), 2048):
            end = min(start + 2048, len(x))
            flat_chunks = x[start:end].reshape(-1, chunk_dim)
            flat_context = np.repeat(context[start:end], max_chunks_actual, axis=0)
            features = np.hstack([flat_chunks, flat_context])
            pred = chunk_model.predict(chunk_scaler.transform(features))
            scores[start:end] = pred.reshape(end - start, max_chunks_actual)
        return scores.astype("float32")

    train_scores = score_chunks(x_train, mask_train, c_train)
    test_scores = score_chunks(x_test, mask_test, c_test)
    train_weights = masked_softmax_abs(train_scores, mask_train)
    test_weights = masked_softmax_abs(test_scores, mask_test)

    z_pool_train = np.einsum("nk,nkd->nd", train_weights, x_train)
    z_pool_test = np.einsum("nk,nkd->nd", test_weights, x_test)

    # Keep simple fallback information for cases without notes.
    no_train_notes = (mask_train.sum(axis=1) <= 0).reshape(-1, 1).astype("float32")
    no_test_notes = (mask_test.sum(axis=1) <= 0).reshape(-1, 1).astype("float32")

    residual_features_train = np.hstack([z_pool_train, c_train, no_train_notes])
    residual_features_test = np.hstack([z_pool_test, c_test, no_test_notes])
    residual_feature_scaler = StandardScaler()
    residual_features_train_scaled = residual_feature_scaler.fit_transform(residual_features_train)
    residual_features_test_scaled = residual_feature_scaler.transform(residual_features_test)
    residual_head = RidgeCV(alphas=np.logspace(-3, 3, 9))
    residual_head.fit(residual_features_train_scaled, oof_residual)
    train_resid_pred = residual_head.predict(residual_features_train_scaled)
    test_resid_pred = residual_head.predict(residual_features_test_scaled)
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    train_resid_pred = np.clip(train_resid_pred, lo, hi)
    test_resid_pred = np.clip(test_resid_pred, lo, hi)

    z_train = np.hstack(
        [
            z_pool_train,
            train_scores.max(axis=1, keepdims=True),
            train_scores.mean(axis=1, keepdims=True),
            train_resid_pred.reshape(-1, 1),
            no_train_notes,
        ]
    )
    z_test = np.hstack(
        [
            z_pool_test,
            test_scores.max(axis=1, keepdims=True),
            test_scores.mean(axis=1, keepdims=True),
            test_resid_pred.reshape(-1, 1),
            no_test_notes,
        ]
    )
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_test = z_scaler.transform(z_test)

    meta = {
        **chunk_meta,
        "backend": "fast_residual_attention",
        "chunk_score_model": chunk_model,
        "chunk_score_scaler": chunk_scaler,
        "context_scaler": context_scaler,
        "context_cols": context_cols,
        "residual_feature_scaler": residual_feature_scaler,
        "residual_head": residual_head,
        "z_scaler": z_scaler,
        "embedding_dim": int(z_train.shape[1]),
        "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_resid_pred)),
        "train_cases_with_positive_mask": int((mask_train.sum(axis=1) > 0).sum()),
        "test_cases_with_positive_mask": int((mask_test.sum(axis=1) > 0).sum()),
    }
    return z_train, z_test, train_resid_pred, test_resid_pred, meta


def row_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, denom, out=np.zeros_like(x), where=denom > 0)


def weighted_residual_features(
    distances: np.ndarray,
    indices: np.ndarray,
    residuals: np.ndarray,
    ks: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build residual-neighborhood coordinates from train-only residuals."""
    max_k = indices.shape[1]
    sim = 1.0 - distances
    feature_blocks = []
    pred_by_k = []
    for k in ks:
        kk = max(1, min(k, max_k))
        idx = indices[:, :kk]
        sim_k = sim[:, :kk]
        residual_k = residuals[idx]
        temp = max(0.05, float(np.nanstd(sim_k)) + 0.05)
        logits = sim_k / temp
        logits = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
        pred = (weights * residual_k).sum(axis=1)
        centered = residual_k - pred[:, None]
        spread = np.sqrt((weights * centered * centered).sum(axis=1))
        abs_mean = (weights * np.abs(residual_k)).sum(axis=1)
        q10 = np.quantile(residual_k, 0.10, axis=1)
        q50 = np.quantile(residual_k, 0.50, axis=1)
        q90 = np.quantile(residual_k, 0.90, axis=1)
        feature_blocks.append(
            np.column_stack(
                [
                    pred,
                    spread,
                    abs_mean,
                    q10,
                    q50,
                    q90,
                    sim_k.max(axis=1),
                    sim_k.mean(axis=1),
                ]
            )
        )
        pred_by_k.append(pred)
    return np.hstack(feature_blocks), np.column_stack(pred_by_k)


def residual_world_similarity(
    train_geometry: np.ndarray,
    test_geometry: np.ndarray,
    residuals: np.ndarray,
    n_bins: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    bins = residual_bins(residuals, n_bins=n_bins)
    centroids = []
    labels = []
    for bin_id in sorted(np.unique(bins)):
        mask = bins == bin_id
        if mask.sum() < 3:
            continue
        centroids.append(train_geometry[mask].mean(axis=0))
        labels.append(int(bin_id))
    if not centroids:
        return (
            np.zeros((len(train_geometry), 0), dtype="float32"),
            np.zeros((len(test_geometry), 0), dtype="float32"),
            [],
        )
    centroid_matrix = row_normalize(np.vstack(centroids))
    return train_geometry @ centroid_matrix.T, test_geometry @ centroid_matrix.T, labels


def fit_residual_neighborhood_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    note_z_train: np.ndarray,
    note_z_test: np.ndarray,
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    ks: list[int],
    context_weight: float,
    note_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Target-conditioned residual-neighborhood embedding.

    The geometry is learned from train cases only. A test case is embedded by
    where it falls among train cases in the product space of structured context
    and note representation, then described by the residual distribution of
    those train neighbors.
    """
    c_train, c_test, context_scaler, context_cols = build_mil_context(
        train_df, test_df, oof_structured_pred, structured_test_pred
    )
    note_scaler = StandardScaler()
    note_train_scaled = note_scaler.fit_transform(note_z_train)
    note_test_scaled = note_scaler.transform(note_z_test)
    geometry_train = np.hstack([note_weight * note_train_scaled, context_weight * c_train])
    geometry_test = np.hstack([note_weight * note_test_scaled, context_weight * c_test])
    geometry_train = row_normalize(geometry_train.astype("float32"))
    geometry_test = row_normalize(geometry_test.astype("float32"))

    max_k = min(max(ks) + 1, max(2, len(train_df)))
    nn = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute")
    nn.fit(geometry_train)

    train_dist, train_idx = nn.kneighbors(geometry_train)
    # Drop self-neighbor for train rows when present, preventing direct residual echo.
    filtered_train_idx = np.zeros((len(train_idx), max_k - 1), dtype=int)
    filtered_train_dist = np.zeros((len(train_dist), max_k - 1), dtype="float32")
    for i in range(len(train_idx)):
        keep = train_idx[i] != i
        idx = train_idx[i][keep][: max_k - 1]
        dist = train_dist[i][keep][: max_k - 1]
        if len(idx) < max_k - 1:
            pad = np.resize(idx if len(idx) else np.array([i]), max_k - 1)
            pad_dist = np.resize(dist if len(dist) else np.array([1.0]), max_k - 1)
            idx = pad
            dist = pad_dist
        filtered_train_idx[i] = idx
        filtered_train_dist[i] = dist

    test_dist, test_idx = nn.kneighbors(geometry_test, n_neighbors=max_k - 1)
    train_neigh_features, train_pred_by_k = weighted_residual_features(
        filtered_train_dist, filtered_train_idx, oof_residual, ks
    )
    test_neigh_features, test_pred_by_k = weighted_residual_features(test_dist, test_idx, oof_residual, ks)
    train_world, test_world, world_labels = residual_world_similarity(geometry_train, geometry_test, oof_residual)

    train_no_notes = (train_df["n_pre_cutoff_notes_used"].to_numpy() <= 0).reshape(-1, 1).astype("float32")
    test_no_notes = (test_df["n_pre_cutoff_notes_used"].to_numpy() <= 0).reshape(-1, 1).astype("float32")
    z_train = np.hstack([train_neigh_features, train_world, train_no_notes])
    z_test = np.hstack([test_neigh_features, test_world, test_no_notes])
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_test = z_scaler.transform(z_test)

    train_resid_pred = train_pred_by_k[:, min(1, train_pred_by_k.shape[1] - 1)]
    test_resid_pred = test_pred_by_k[:, min(1, test_pred_by_k.shape[1] - 1)]
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    train_resid_pred = np.clip(train_resid_pred, lo, hi)
    test_resid_pred = np.clip(test_resid_pred, lo, hi)
    meta = {
        "backend": "residual_neighborhood_geometry",
        "ks": ks,
        "context_weight": float(context_weight),
        "note_weight": float(note_weight),
        "context_cols": context_cols,
        "context_scaler": context_scaler,
        "note_scaler": note_scaler,
        "nearest_neighbors": nn,
        "z_scaler": z_scaler,
        "residual_world_centroid_labels": world_labels,
        "embedding_dim": int(z_train.shape[1]),
        "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_resid_pred)),
        "train_cases_with_notes": int((train_df["n_pre_cutoff_notes_used"] > 0).sum()),
        "test_cases_with_notes": int((test_df["n_pre_cutoff_notes_used"] > 0).sum()),
    }
    return z_train, z_test, train_resid_pred, test_resid_pred, meta


def make_fusion_residual_model(model_name: str, seed: int) -> Any:
    if model_name == "RidgeCV":
        return RidgeCV(alphas=np.logspace(-3, 3, 13))
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=10,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown fusion residual model: {model_name}")


def out_of_fold_pls_coordinates(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, PLSRegression | None]:
    n_components = int(max(0, min(n_components, x_train.shape[1], len(x_train) - 2)))
    if n_components < 1:
        return (
            np.zeros((len(x_train), 0), dtype="float32"),
            np.zeros((len(x_test), 0), dtype="float32"),
            None,
        )
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)
    train_coords = np.zeros((len(x_train), n_components), dtype="float32")
    for fit_idx, val_idx in splitter.split(x_train, y, groups=groups):
        fold_components = max(1, min(n_components, len(fit_idx) - 2, x_train.shape[1]))
        pls = PLSRegression(n_components=fold_components, scale=False)
        pls.fit(x_train[fit_idx], y[fit_idx])
        coords = pls.transform(x_train[val_idx]).astype("float32")
        train_coords[val_idx, :fold_components] = coords
    full_pls = PLSRegression(n_components=n_components, scale=False)
    full_pls.fit(x_train, y)
    test_coords = full_pls.transform(x_test).astype("float32")
    return train_coords, test_coords, full_pls


def out_of_fold_residual_probe_coordinates(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    model_names = ["RidgeCV", "ExtraTrees", "RandomForest"]
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)
    oof_blocks = []
    test_blocks = []
    full_models: dict[str, Any] = {}
    rows = []
    for model_offset, model_name in enumerate(model_names):
        oof_pred = np.zeros(len(x_train), dtype="float32")
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y, groups=groups), start=1):
            model = make_fusion_residual_model(model_name, seed + 100 * model_offset + fold_idx)
            model.fit(x_train[fit_idx], y[fit_idx])
            oof_pred[val_idx] = model.predict(x_train[val_idx]).astype("float32")
        full_model = make_fusion_residual_model(model_name, seed + 1000 + model_offset)
        full_model.fit(x_train, y)
        test_pred = full_model.predict(x_test).astype("float32")
        oof_blocks.append(oof_pred.reshape(-1, 1))
        test_blocks.append(test_pred.reshape(-1, 1))
        full_models[model_name] = full_model
        rows.append(
            {
                "fusion_probe_model": model_name,
                "oof_residual_mae": float(mean_absolute_error(y, oof_pred)),
            }
        )
    train_probe = np.hstack(oof_blocks)
    test_probe = np.hstack(test_blocks)
    lo, hi = np.quantile(y, [0.005, 0.995])
    train_probe = np.clip(train_probe, lo, hi)
    test_probe = np.clip(test_probe, lo, hi)
    return train_probe, test_probe, full_models, pd.DataFrame(rows).sort_values("oof_residual_mae")


def fit_cure_target_fusion_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedding_blocks_train: list[np.ndarray],
    embedding_blocks_test: list[np.ndarray],
    oof_structured_pred: np.ndarray,
    structured_test_pred: np.ndarray,
    oof_residual: np.ndarray,
    svd_dim: int,
    pls_dim: int,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse note embeddings into a residual-target-conditioned vector.

    The train residual-probe coordinates are out-of-fold. That keeps the fused
    embedding target-aware without letting train rows carry their own residuals
    directly into the downstream residual corrector.
    """
    c_train, c_test, context_scaler, context_cols = build_mil_context(
        train_df, test_df, oof_structured_pred, structured_test_pred
    )
    has_train_notes = (train_df["n_pre_cutoff_notes_used"].to_numpy() > 0).reshape(-1, 1).astype("float32")
    has_test_notes = (test_df["n_pre_cutoff_notes_used"].to_numpy() > 0).reshape(-1, 1).astype("float32")
    note_count_train = np.log1p(train_df["n_pre_cutoff_notes_used"].to_numpy()).reshape(-1, 1).astype("float32")
    note_count_test = np.log1p(test_df["n_pre_cutoff_notes_used"].to_numpy()).reshape(-1, 1).astype("float32")

    raw_train = np.hstack([*embedding_blocks_train, c_train, has_train_notes, note_count_train]).astype("float32")
    raw_test = np.hstack([*embedding_blocks_test, c_test, has_test_notes, note_count_test]).astype("float32")
    raw_scaler = StandardScaler()
    raw_train_scaled = raw_scaler.fit_transform(raw_train)
    raw_test_scaled = raw_scaler.transform(raw_test)

    n_components = max(2, min(svd_dim, raw_train_scaled.shape[0] - 2, raw_train_scaled.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    low_train = svd.fit_transform(raw_train_scaled)
    low_test = svd.transform(raw_test_scaled)
    low_scaler = StandardScaler()
    low_train = low_scaler.fit_transform(low_train)
    low_test = low_scaler.transform(low_test)

    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    pls_train, pls_test, pls = out_of_fold_pls_coordinates(
        low_train, low_test, oof_residual.astype("float32"), groups, pls_dim, n_folds
    )
    probe_train, probe_test, probe_models, probe_selection = out_of_fold_residual_probe_coordinates(
        low_train,
        low_test,
        oof_residual.astype("float32"),
        groups,
        n_folds,
        seed,
    )

    probe_mean_train = probe_train.mean(axis=1, keepdims=True)
    probe_mean_test = probe_test.mean(axis=1, keepdims=True)
    probe_spread_train = probe_train.std(axis=1, keepdims=True)
    probe_spread_test = probe_test.std(axis=1, keepdims=True)
    train_resid_pred = probe_mean_train.ravel()
    test_resid_pred = probe_mean_test.ravel()

    z_train = np.hstack(
        [
            low_train,
            pls_train,
            probe_train,
            probe_mean_train,
            probe_spread_train,
            has_train_notes,
            note_count_train,
        ]
    )
    z_test = np.hstack(
        [
            low_test,
            pls_test,
            probe_test,
            probe_mean_test,
            probe_spread_test,
            has_test_notes,
            note_count_test,
        ]
    )
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train).astype("float32")
    z_test = z_scaler.transform(z_test).astype("float32")

    meta = {
        "backend": "target_conditioned_fusion",
        "raw_dim": int(raw_train.shape[1]),
        "embedding_dim": int(z_train.shape[1]),
        "svd_dim": int(n_components),
        "pls_dim": int(pls_train.shape[1]),
        "context_cols": context_cols,
        "raw_scaler": raw_scaler,
        "svd": svd,
        "low_scaler": low_scaler,
        "pls": pls,
        "probe_models": probe_models,
        "probe_selection": probe_selection.to_dict(orient="records"),
        "z_scaler": z_scaler,
        "context_scaler": context_scaler,
        "train_residual_probe_mae_on_oof_residual": float(mean_absolute_error(oof_residual, train_resid_pred)),
    }
    return z_train, z_test, train_resid_pred, test_resid_pred, meta


def fit_empty_mil_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    z_train = np.zeros((len(train_df), 1), dtype="float32")
    z_test = np.zeros((len(test_df), 1), dtype="float32")
    train_resid = np.zeros(len(train_df), dtype="float32")
    test_resid = np.zeros(len(test_df), dtype="float32")
    meta = {
        "backend": "off",
        "embedding_dim": 1,
        "train_residual_probe_mae_on_oof_residual": None,
        "n_terms": 0,
        "chunk_svd_dim": 0,
        "max_chunks": 0,
        "train_chunks_total": 0,
        "test_chunks_total": 0,
    }
    return z_train, z_test, train_resid, test_resid, meta


def add_embedding_columns(df: pd.DataFrame, embedding: np.ndarray, prefix: str) -> pd.DataFrame:
    emb = pd.DataFrame(
        embedding,
        index=df.index,
        columns=[f"{prefix}_{k:03d}" for k in range(embedding.shape[1])],
    )
    return pd.concat([df.copy(), emb], axis=1)


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in NOTE_TEXT_METADATA_COLUMNS if c in df.columns])


def evaluate_augmented_model(
    XGBRegressor: Any,
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, Pipeline]:
    pipe, _, pred, _, _ = fit_predict_tabular(
        XGBRegressor, modeling_frame(train_df), modeling_frame(test_df), target, params, seed
    )
    metrics = metric_dict(test_df[target].to_numpy(dtype=float), pred)
    metrics["method"] = name
    return metrics, pred, pipe


def evaluate_residual_corrector(
    XGBRegressor: Any,
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    oof_residual: np.ndarray,
    structured_test_pred: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, Pipeline]:
    residual_target = "__residual_target__"
    fit_train = train_df.copy()
    fit_test = test_df.copy()
    fit_train[residual_target] = oof_residual
    fit_test[residual_target] = 0.0

    # The residual corrector should not see the true duration column as a feature.
    fit_train = modeling_frame(fit_train).drop(columns=[target], errors="ignore")
    fit_test = modeling_frame(fit_test).drop(columns=[target], errors="ignore")

    X_train, y_train, numeric_cols, categorical_cols = make_feature_frame(fit_train, residual_target)
    X_test, _, _, _ = make_feature_frame(fit_test, residual_target)
    residual_params = dict(params)
    residual_params["n_estimators"] = min(int(residual_params.get("n_estimators", 600)), 800)
    residual_params["learning_rate"] = min(float(residual_params.get("learning_rate", 0.05)), 0.05)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", make_model(XGBRegressor, residual_params, seed)),
        ]
    )
    pipe.fit(X_train, y_train)
    residual_pred = pipe.predict(X_test)
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    residual_pred = np.clip(residual_pred, lo, hi)
    pred = structured_test_pred + residual_pred
    metrics = metric_dict(test_df[target].to_numpy(dtype=float), pred)
    metrics["method"] = name
    return metrics, pred, pipe


def residual_model_candidates(XGBRegressor: Any, base_params: dict[str, Any], seed: int) -> dict[str, Any]:
    xgb_params = dict(base_params)
    xgb_params.update(
        {
            "n_estimators": min(int(xgb_params.get("n_estimators", 500)), 500),
            "max_depth": min(int(xgb_params.get("max_depth", 4)), 4),
            "learning_rate": min(float(xgb_params.get("learning_rate", 0.04)), 0.035),
            "subsample": min(float(xgb_params.get("subsample", 0.8)), 0.85),
            "colsample_bytree": min(float(xgb_params.get("colsample_bytree", 0.8)), 0.85),
            "reg_alpha": max(float(xgb_params.get("reg_alpha", 0.0)), 0.1),
            "reg_lambda": max(float(xgb_params.get("reg_lambda", 1.0)), 3.0),
        }
    )
    return {
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            max_features=0.7,
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=8,
            max_features=0.7,
            random_state=seed + 1,
            n_jobs=-1,
        ),
        "ConservativeXGB": make_model(XGBRegressor, xgb_params, seed + 2),
    }


def fit_group_shrinkage(
    y_true_residual: np.ndarray,
    oof_pred_residual: np.ndarray,
    has_notes: np.ndarray,
) -> dict[str, float]:
    """Choose conservative residual-correction scales on train OOF predictions.

    This prevents a note arm from hurting every case when the residual learner is
    useful only for cases with enough pre-cutoff notes. The scale is selected on
    train-only out-of-fold residual predictions, then reused on test.
    """
    grid = np.linspace(0.0, 1.25, 51)
    scales: dict[str, float] = {}
    for name, mask in {
        "with_notes": has_notes.astype(bool),
        "without_notes": ~has_notes.astype(bool),
        "all": np.ones(len(y_true_residual), dtype=bool),
    }.items():
        if mask.sum() < 20:
            scales[name] = scales.get("all", 0.0)
            continue
        losses = [mean_absolute_error(y_true_residual[mask], alpha * oof_pred_residual[mask]) for alpha in grid]
        scales[name] = float(grid[int(np.argmin(losses))])
    return scales


def apply_group_shrinkage(
    pred_residual: np.ndarray,
    has_notes: np.ndarray,
    scales: dict[str, float],
) -> np.ndarray:
    out = pred_residual.copy()
    has_notes = has_notes.astype(bool)
    out[has_notes] *= scales.get("with_notes", scales.get("all", 0.0))
    out[~has_notes] *= scales.get("without_notes", scales.get("all", 0.0))
    return out


def fit_procedure_shrinkage(
    train_df: pd.DataFrame,
    y_true_residual: np.ndarray,
    oof_pred_residual: np.ndarray,
    min_support: int = 25,
) -> dict[str, Any]:
    """Choose residual-correction scales by procedure with global fallback."""
    grid = np.linspace(0.0, 1.25, 51)
    y = np.asarray(y_true_residual, dtype=float)
    pred = np.asarray(oof_pred_residual, dtype=float)

    global_losses = [mean_absolute_error(y, alpha * pred) for alpha in grid]
    global_alpha = float(grid[int(np.argmin(global_losses))])
    global_mae = float(min(global_losses))
    zero_mae = float(mean_absolute_error(y, np.zeros_like(y)))
    if global_mae >= zero_mae:
        global_alpha = 0.0
        global_mae = zero_mae

    procedures = (
        train_df["primary_procedure_name"].fillna("__missing__").astype(str).to_numpy()
        if "primary_procedure_name" in train_df.columns
        else np.asarray(["__all__"] * len(train_df), dtype=object)
    )
    scales: dict[str, float] = {}
    rows = []
    calibrated = np.zeros_like(pred)
    for proc in sorted(set(procedures)):
        mask = procedures == proc
        n = int(mask.sum())
        if n < min_support:
            alpha = global_alpha
            source = "global_fallback_low_support"
            proc_mae = float(mean_absolute_error(y[mask], alpha * pred[mask])) if n else 0.0
            proc_zero = float(mean_absolute_error(y[mask], np.zeros(mask.sum()))) if n else 0.0
        else:
            proc_losses = [mean_absolute_error(y[mask], alpha * pred[mask]) for alpha in grid]
            best_idx = int(np.argmin(proc_losses))
            alpha = float(grid[best_idx])
            proc_mae = float(proc_losses[best_idx])
            proc_zero = float(mean_absolute_error(y[mask], np.zeros(mask.sum())))
            if proc_mae >= proc_zero:
                alpha = 0.0
                proc_mae = proc_zero
                source = "zero_no_oof_gain"
            else:
                source = "procedure"
        scales[proc] = alpha
        calibrated[mask] = alpha * pred[mask]
        rows.append(
            {
                "procedure": proc,
                "n": n,
                "alpha": alpha,
                "source": source,
                "procedure_oof_mae": proc_mae,
                "procedure_zero_mae": proc_zero,
            }
        )

    return {
        "global_alpha": global_alpha,
        "global_oof_mae": global_mae,
        "global_zero_mae": zero_mae,
        "scales": scales,
        "oof_pred": calibrated,
        "oof_mae": float(mean_absolute_error(y, calibrated)),
        "table": rows,
        "min_support": int(min_support),
    }


def apply_procedure_shrinkage(
    test_df: pd.DataFrame,
    pred_residual: np.ndarray,
    proc_calibration: dict[str, Any],
) -> np.ndarray:
    pred = np.asarray(pred_residual, dtype=float)
    procedures = (
        test_df["primary_procedure_name"].fillna("__missing__").astype(str).to_numpy()
        if "primary_procedure_name" in test_df.columns
        else np.asarray(["__all__"] * len(test_df), dtype=object)
    )
    scales = proc_calibration.get("scales", {})
    fallback = float(proc_calibration.get("global_alpha", 0.0))
    out = np.zeros_like(pred)
    for i, proc in enumerate(procedures):
        out[i] = float(scales.get(str(proc), fallback)) * pred[i]
    return out


def residual_calibration_frame(
    source_df: pd.DataFrame,
    pred_residual: np.ndarray,
    target_residual: np.ndarray | None = None,
) -> pd.DataFrame:
    """Small context frame for calibrating a residual correction.

    This intentionally avoids raw note text and learned embedding columns. It
    learns when a residual correction is trustworthy from case context, note
    availability, and the correction magnitude/sign.
    """
    frame = pd.DataFrame(index=source_df.index)
    pred = np.asarray(pred_residual, dtype=float)
    frame["pred_residual"] = pred
    frame["abs_pred_residual"] = np.abs(pred)
    frame["positive_pred_residual"] = (pred > 0).astype("int64")
    frame["log_abs_pred_residual"] = np.log1p(np.abs(pred))
    frame["n_pre_cutoff_notes_used"] = source_df.get("n_pre_cutoff_notes_used", 0)
    frame["note_chars"] = source_df.get("note_chars", 0)
    frame["log_note_chars"] = np.log1p(pd.to_numeric(frame["note_chars"], errors="coerce").fillna(0))
    frame["has_notes"] = (pd.to_numeric(frame["n_pre_cutoff_notes_used"], errors="coerce").fillna(0) > 0).astype(
        "int64"
    )
    frame["pred_x_has_notes"] = frame["pred_residual"] * frame["has_notes"]
    frame["pred_x_log_note_chars"] = frame["pred_residual"] * frame["log_note_chars"]

    for col in [
        "primary_procedure_name",
        "primary_procedure_code",
        "procedure_category",
        "case_service",
        "site",
        "location_name",
        "surgeon_name",
        "note_doc_types_used",
    ]:
        if col in source_df.columns:
            frame[col] = source_df[col].astype(str)

    for col in [
        "scheduled_room_minutes",
        "scheduled_setup_lead_minutes",
        "scheduled_cleanup_minutes",
        "scheduled_hour",
        "scheduled_dayofweek",
        "case_num_procedures",
        "case_num_providers",
        "patient_age",
        "is_redo",
    ]:
        if col in source_df.columns:
            value = pd.to_numeric(source_df[col], errors="coerce")
            frame[col] = value
            frame[f"pred_x_{col}"] = frame["pred_residual"] * value.fillna(value.median())

    if target_residual is not None:
        frame["__calibration_target__"] = np.asarray(target_residual, dtype=float)
    else:
        frame["__calibration_target__"] = 0.0
    return frame


def adaptive_calibration_candidates(seed: int) -> dict[str, Any]:
    return {
        "CalibratedRidge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "CalibratedExtraTrees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=15,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def fit_adaptive_calibration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true_residual: np.ndarray,
    oof_pred_residual: np.ndarray,
    test_pred_residual: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = "__calibration_target__"
    train_cal = residual_calibration_frame(train_df, oof_pred_residual, y_true_residual)
    test_cal = residual_calibration_frame(test_df, test_pred_residual, None)
    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(train_cal, target)
    X_test, _, _, _ = make_feature_frame(test_cal, target)
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)

    rows = []
    fitted: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    for name, model in adaptive_calibration_candidates(seed).items():
        fold_preds = np.zeros(len(X_all), dtype=float)
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
            fold_model = clone(model)
            if hasattr(fold_model, "random_state"):
                setattr(fold_model, "random_state", seed + fold_idx)
            pipe = Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                    ("model", fold_model),
                ]
            )
            pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
            fold_preds[val_idx] = pipe.predict(X_all.iloc[val_idx])
        lo, hi = np.quantile(y_true_residual, [0.005, 0.995])
        fold_preds = np.clip(fold_preds, lo, hi)
        oof_mae = mean_absolute_error(y_all, fold_preds)
        rows.append({"calibrator": name, "oof_residual_mae": float(oof_mae)})
        oof_predictions[name] = fold_preds

        full_pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", clone(model)),
            ]
        )
        full_pipe.fit(X_all, y_all)
        test_predictions[name] = np.clip(full_pipe.predict(X_test), lo, hi)
        fitted[name] = full_pipe

    selection = pd.DataFrame(rows).sort_values("oof_residual_mae").reset_index(drop=True)
    best_name = str(selection.iloc[0]["calibrator"])
    return (
        {
            "best_calibrator": best_name,
            "pipeline": fitted[best_name],
            "oof_pred": oof_predictions[best_name],
            "test_pred": test_predictions[best_name],
            "selection": selection.to_dict(orient="records"),
        },
        selection,
    )


def evaluate_selected_residual_corrector(
    XGBRegressor: Any,
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    oof_residual: np.ndarray,
    structured_test_pred: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_folds: int,
) -> tuple[dict[str, Any], np.ndarray, Pipeline, dict[str, Any], np.ndarray]:
    residual_target = "__residual_target__"
    fit_train = modeling_frame(train_df.copy()).drop(columns=[target], errors="ignore")
    fit_test = modeling_frame(test_df.copy()).drop(columns=[target], errors="ignore")
    fit_train[residual_target] = oof_residual
    fit_test[residual_target] = 0.0

    X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(fit_train, residual_target)
    X_test, _, _, _ = make_feature_frame(fit_test, residual_target)
    groups = train_df[PATIENT_ID_COLUMN].astype(str).to_numpy()
    split_count = min(n_folds, len(np.unique(groups)))
    split_count = max(2, split_count)
    splitter = GroupKFold(n_splits=split_count)

    rows = []
    fitted_full: dict[str, Pipeline] = {}
    shrinkages: dict[str, dict[str, float]] = {}
    procedure_shrinkages: dict[str, dict[str, Any]] = {}
    calibration_modes: dict[str, str] = {}
    adaptive_calibrators: dict[str, dict[str, Any]] = {}
    selected_oof_predictions: dict[str, np.ndarray] = {}
    has_notes_train = train_df["n_pre_cutoff_notes_used"].to_numpy() > 0
    has_notes_test = test_df["n_pre_cutoff_notes_used"].to_numpy() > 0
    candidates = residual_model_candidates(XGBRegressor, params, seed)
    for model_name, model in candidates.items():
        fold_preds = np.zeros(len(X_all), dtype=float)
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
            pipe = Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                    ("model", model if model_name in {"RidgeCV"} else residual_model_candidates(XGBRegressor, params, seed + fold_idx)[model_name]),
                ]
            )
            pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
            fold_preds[val_idx] = pipe.predict(X_all.iloc[val_idx])
        scales = fit_group_shrinkage(y_all.to_numpy(dtype=float), fold_preds, has_notes_train)
        calibrated_fold_preds = apply_group_shrinkage(fold_preds, has_notes_train, scales)
        group_oof_mae = mean_absolute_error(y_all, calibrated_fold_preds)
        raw_oof_mae = mean_absolute_error(y_all, fold_preds)
        proc_calibration = fit_procedure_shrinkage(
            train_df=train_df,
            y_true_residual=y_all.to_numpy(dtype=float),
            oof_pred_residual=fold_preds,
            min_support=max(20, int(0.01 * len(train_df))),
        )
        proc_oof_mae = float(proc_calibration["oof_mae"])

        full_pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", residual_model_candidates(XGBRegressor, params, seed + 100)[model_name]),
            ]
        )
        full_pipe.fit(X_all, y_all)
        fitted_full[model_name] = full_pipe
        raw_test_pred_for_calibration = full_pipe.predict(X_test)
        lo, hi = np.quantile(oof_residual, [0.005, 0.995])
        raw_test_pred_for_calibration = np.clip(raw_test_pred_for_calibration, lo, hi)
        adaptive, adaptive_selection = fit_adaptive_calibration(
            train_df=train_df,
            test_df=test_df,
            y_true_residual=y_all.to_numpy(dtype=float),
            oof_pred_residual=fold_preds,
            test_pred_residual=raw_test_pred_for_calibration,
            groups=groups,
            n_folds=n_folds,
            seed=seed + 5000,
        )
        adaptive_oof_mae = float(adaptive_selection.iloc[0]["oof_residual_mae"])
        calibration_scores = {
            "group_shrinkage": group_oof_mae,
            "adaptive": adaptive_oof_mae,
            "procedure_shrinkage": proc_oof_mae,
        }
        best_calibration_mode = min(calibration_scores, key=calibration_scores.get)
        if best_calibration_mode == "adaptive":
            selected_oof_mae = adaptive_oof_mae
            calibration_modes[model_name] = "adaptive"
            selected_oof_predictions[model_name] = adaptive["oof_pred"]
        elif best_calibration_mode == "procedure_shrinkage":
            selected_oof_mae = proc_oof_mae
            calibration_modes[model_name] = "procedure_shrinkage"
            selected_oof_predictions[model_name] = proc_calibration["oof_pred"]
        else:
            selected_oof_mae = group_oof_mae
            calibration_modes[model_name] = "group_shrinkage"
            selected_oof_predictions[model_name] = calibrated_fold_preds
        adaptive_calibrators[model_name] = adaptive
        procedure_shrinkages[model_name] = proc_calibration
        rows.append(
            {
                "residual_model": model_name,
                "oof_residual_mae": float(selected_oof_mae),
                "raw_oof_residual_mae": float(raw_oof_mae),
                "group_shrinkage_oof_residual_mae": float(group_oof_mae),
                "adaptive_oof_residual_mae": float(adaptive_oof_mae),
                "procedure_shrinkage_oof_residual_mae": float(proc_oof_mae),
                "selected_calibration": calibration_modes[model_name],
                "selected_adaptive_calibrator": adaptive["best_calibrator"],
                "procedure_global_alpha": float(proc_calibration["global_alpha"]),
                "shrinkage_with_notes": float(scales.get("with_notes", 0.0)),
                "shrinkage_without_notes": float(scales.get("without_notes", 0.0)),
            }
        )
        shrinkages[model_name] = scales

    selection = pd.DataFrame(rows).sort_values("oof_residual_mae").reset_index(drop=True)
    best_name = str(selection.iloc[0]["residual_model"])
    best_pipe = fitted_full[best_name]
    residual_pred = best_pipe.predict(X_test)
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    residual_pred = np.clip(residual_pred, lo, hi)
    if calibration_modes[best_name] == "adaptive":
        residual_pred = adaptive_calibrators[best_name]["test_pred"]
    elif calibration_modes[best_name] == "procedure_shrinkage":
        residual_pred = apply_procedure_shrinkage(test_df, residual_pred, procedure_shrinkages[best_name])
    else:
        residual_pred = apply_group_shrinkage(residual_pred, has_notes_test, shrinkages[best_name])
    pred = structured_test_pred + residual_pred
    metrics = metric_dict(test_df[target].to_numpy(dtype=float), pred)
    metrics["method"] = f"{name}_{best_name}"
    details = {
        "selected_residual_model": best_name,
        "selected_calibration": calibration_modes[best_name],
        "selected_shrinkage": shrinkages[best_name],
        "selected_procedure_shrinkage": {
            key: value
            for key, value in procedure_shrinkages[best_name].items()
            if key not in {"oof_pred"}
        },
        "selected_adaptive_calibrator": adaptive_calibrators[best_name]["best_calibrator"],
        "adaptive_calibrator_selection": adaptive_calibrators[best_name]["selection"],
        "selection_table": selection.to_dict(orient="records"),
    }
    return metrics, pred, best_pipe, details, selected_oof_predictions[best_name]


def fit_procedure_residual_arm_chooser(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    oof_residual: np.ndarray,
    structured_test_pred: np.ndarray,
    train_arm_residuals: dict[str, np.ndarray],
    test_arm_residuals: dict[str, np.ndarray],
    min_support: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Choose among note-residual arms by procedure using train-only OOF errors.

    The chooser is deliberately conservative: every procedure can also choose a
    zero residual correction, which means it falls back to the structured model
    when the note arm has no train-side evidence of helping that procedure.
    """
    y = np.asarray(oof_residual, dtype=float)
    procedures_train = (
        train_df["primary_procedure_name"].fillna("__missing__").astype(str).to_numpy()
        if "primary_procedure_name" in train_df.columns
        else np.asarray(["__all__"] * len(train_df), dtype=object)
    )
    procedures_test = (
        test_df["primary_procedure_name"].fillna("__missing__").astype(str).to_numpy()
        if "primary_procedure_name" in test_df.columns
        else np.asarray(["__all__"] * len(test_df), dtype=object)
    )
    arm_names = ["zero", *train_arm_residuals.keys()]
    train_matrix = {"zero": np.zeros_like(y)}
    train_matrix.update({name: np.asarray(values, dtype=float) for name, values in train_arm_residuals.items()})
    test_matrix = {"zero": np.zeros(len(test_df), dtype=float)}
    test_matrix.update({name: np.asarray(values, dtype=float) for name, values in test_arm_residuals.items()})
    grid = np.linspace(0.0, 1.25, 51)

    def best_arm_for_mask(mask: np.ndarray) -> dict[str, Any]:
        zero_mae = float(mean_absolute_error(y[mask], np.zeros(mask.sum())))
        best = {
            "arm": "zero",
            "alpha": 0.0,
            "oof_mae": zero_mae,
            "zero_mae": zero_mae,
        }
        for arm in arm_names:
            values = train_matrix[arm][mask]
            for alpha in ([0.0] if arm == "zero" else grid):
                pred = float(alpha) * values
                mae = float(mean_absolute_error(y[mask], pred))
                if mae < best["oof_mae"]:
                    best = {
                        "arm": arm,
                        "alpha": float(alpha),
                        "oof_mae": mae,
                        "zero_mae": zero_mae,
                    }
        return best

    global_selection = best_arm_for_mask(np.ones(len(y), dtype=bool))
    selections: dict[str, dict[str, Any]] = {}
    rows = []
    train_selected = np.zeros_like(y)
    for proc in sorted(set(procedures_train)):
        mask = procedures_train == proc
        n = int(mask.sum())
        if n >= min_support:
            choice = best_arm_for_mask(mask)
            source = "procedure"
        else:
            choice = dict(global_selection)
            source = "global_fallback_low_support"
        selections[proc] = choice
        train_selected[mask] = choice["alpha"] * train_matrix[choice["arm"]][mask]
        rows.append(
            {
                "procedure": proc,
                "n_train": n,
                "selected_arm": choice["arm"],
                "alpha": float(choice["alpha"]),
                "source": source,
                "procedure_oof_mae": float(choice["oof_mae"]),
                "procedure_zero_mae": float(choice["zero_mae"]),
                "oof_delta_vs_zero": float(choice["oof_mae"] - choice["zero_mae"]),
            }
        )

    test_selected = np.zeros(len(test_df), dtype=float)
    for idx, proc in enumerate(procedures_test):
        choice = selections.get(str(proc), global_selection)
        test_selected[idx] = float(choice["alpha"]) * test_matrix[choice["arm"]][idx]

    pred = structured_test_pred + test_selected
    metrics = metric_dict(test_df[target].to_numpy(dtype=float), pred)
    metrics["method"] = "Structured_plus_CURE_procedure_residual_arm_chooser"
    details = {
        "global_selection": global_selection,
        "min_support": int(min_support),
        "train_oof_residual_mae": float(mean_absolute_error(y, train_selected)),
        "train_zero_residual_mae": float(mean_absolute_error(y, np.zeros_like(y))),
        "arm_names": arm_names,
        "selection_table": rows,
    }
    return metrics, pred, details


def fit_residual_ensemble_embedding(
    XGBRegressor: Any,
    embedding_frames: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    test_df: pd.DataFrame,
    target: str,
    oof_residual: np.ndarray,
    structured_test_pred: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, dict[str, Any]]:
    """Build an embedding from out-of-fold residual predictions.

    Each coordinate is a train-safe residual prediction from one note-feature
    world and one residual learner. For train rows, coordinates are produced
    out-of-fold by patient. For test rows, coordinates come from the same model
    family fit on all train rows. This turns the note representation search into
    an explicit residual-space optimization without using test targets.
    """
    if not embedding_frames:
        raise ValueError("At least one embedding frame is required.")

    groups = embedding_frames[0][1][PATIENT_ID_COLUMN].astype(str).to_numpy()
    split_count = max(2, min(n_folds, len(np.unique(groups))))
    splitter = GroupKFold(n_splits=split_count)
    candidate_names = ["RidgeCV", "ExtraTrees", "RandomForest"]
    train_blocks = []
    test_blocks = []
    artifacts: dict[str, Pipeline] = {}
    rows = []

    for frame_idx, (frame_name, train_frame, test_frame) in enumerate(embedding_frames):
        residual_target = "__residual_target__"
        fit_train = modeling_frame(train_frame.copy()).drop(columns=[target], errors="ignore")
        fit_test = modeling_frame(test_frame.copy()).drop(columns=[target], errors="ignore")
        fit_train[residual_target] = oof_residual
        fit_test[residual_target] = 0.0
        X_all, y_all, numeric_cols, categorical_cols = make_feature_frame(fit_train, residual_target)
        X_test, _, _, _ = make_feature_frame(fit_test, residual_target)
        candidates = residual_model_candidates(XGBRegressor, params, seed + frame_idx * 1000)

        for model_idx, model_name in enumerate(candidate_names):
            oof_pred = np.zeros(len(X_all), dtype="float32")
            for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
                fold_candidates = residual_model_candidates(
                    XGBRegressor,
                    params,
                    seed + frame_idx * 1000 + model_idx * 100 + fold_idx,
                )
                pipe = Pipeline(
                    [
                        ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                        ("model", fold_candidates[model_name]),
                    ]
                )
                pipe.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
                oof_pred[val_idx] = pipe.predict(X_all.iloc[val_idx]).astype("float32")

            full_pipe = Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                    ("model", candidates[model_name]),
                ]
            )
            full_pipe.fit(X_all, y_all)
            test_pred = full_pipe.predict(X_test).astype("float32")
            lo, hi = np.quantile(oof_residual, [0.005, 0.995])
            oof_pred = np.clip(oof_pred, lo, hi)
            test_pred = np.clip(test_pred, lo, hi)
            train_blocks.append(oof_pred.reshape(-1, 1))
            test_blocks.append(test_pred.reshape(-1, 1))
            key = f"{frame_name}__{model_name}"
            artifacts[key] = full_pipe
            rows.append(
                {
                    "residual_world": frame_name,
                    "model": model_name,
                    "oof_residual_mae": float(mean_absolute_error(oof_residual, oof_pred)),
                }
            )

    train_matrix = np.hstack(train_blocks)
    test_matrix = np.hstack(test_blocks)
    train_mean = train_matrix.mean(axis=1, keepdims=True)
    test_mean = test_matrix.mean(axis=1, keepdims=True)
    train_std = train_matrix.std(axis=1, keepdims=True)
    test_std = test_matrix.std(axis=1, keepdims=True)
    train_min = train_matrix.min(axis=1, keepdims=True)
    test_min = test_matrix.min(axis=1, keepdims=True)
    train_max = train_matrix.max(axis=1, keepdims=True)
    test_max = test_matrix.max(axis=1, keepdims=True)

    z_train_raw = np.hstack([train_matrix, train_mean, train_std, train_min, train_max])
    z_test_raw = np.hstack([test_matrix, test_mean, test_std, test_min, test_max])
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train_raw).astype("float32")
    z_test = z_scaler.transform(z_test_raw).astype("float32")

    meta_train = pd.DataFrame(
        {
            "patient_id": embedding_frames[0][1][PATIENT_ID_COLUMN].astype(str).to_numpy(),
            "has_notes": (embedding_frames[0][1]["n_pre_cutoff_notes_used"].to_numpy() > 0).astype(int),
        }
    )
    meta_test = pd.DataFrame(
        {
            "patient_id": test_df[PATIENT_ID_COLUMN].astype(str).to_numpy(),
            "has_notes": (test_df["n_pre_cutoff_notes_used"].to_numpy() > 0).astype(int),
        }
    )
    ensemble_train_frame = pd.concat(
        [
            meta_train,
            pd.DataFrame(z_train, columns=[f"ensemble_resid_z_{k:03d}" for k in range(z_train.shape[1])]),
        ],
        axis=1,
    )
    ensemble_test_frame = pd.concat(
        [
            meta_test,
            pd.DataFrame(z_test, columns=[f"ensemble_resid_z_{k:03d}" for k in range(z_test.shape[1])]),
        ],
        axis=1,
    )
    residual_target = "__residual_target__"
    ensemble_train_frame[residual_target] = oof_residual
    ensemble_test_frame[residual_target] = 0.0
    X_meta, y_meta, numeric_cols, categorical_cols = make_feature_frame(ensemble_train_frame, residual_target)
    X_meta_test, _, _, _ = make_feature_frame(ensemble_test_frame, residual_target)
    meta_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    meta_pipe.fit(X_meta, y_meta)
    residual_pred = meta_pipe.predict(X_meta_test)
    lo, hi = np.quantile(oof_residual, [0.005, 0.995])
    residual_pred = np.clip(residual_pred, lo, hi)
    pred = structured_test_pred + residual_pred
    details = {
        "backend": "oof_residual_ensemble_embedding",
        "embedding_dim": int(z_train.shape[1]),
        "base_coordinate_count": int(train_matrix.shape[1]),
        "selection_table": pd.DataFrame(rows).sort_values("oof_residual_mae").to_dict(orient="records"),
        "z_scaler": z_scaler,
        "base_residual_pipelines": artifacts,
        "meta_residual_pipeline": meta_pipe,
        "train_meta_residual_mae": float(mean_absolute_error(oof_residual, meta_pipe.predict(X_meta))),
    }
    return z_train, z_test, pred, meta_pipe, details


def safe_procedure_slice_metrics(test_df: pd.DataFrame, predictions: np.ndarray, target: str) -> pd.DataFrame:
    try:
        return procedure_slice_metrics(test_df, predictions, target)
    except KeyError:
        return pd.DataFrame(
            columns=[
                "primary_procedure_name",
                "N",
                "MAE",
                "MedianAbsError",
                "P90AbsError",
                "W30",
                "ActualMean",
                "PredMean",
            ]
        )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    tabular_path = Path(args.tabular_data)
    notes_path = Path(args.notes_data)
    if not tabular_path.is_absolute():
        tabular_path = root / tabular_path
    if not notes_path.is_absolute():
        notes_path = root / notes_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    XGBRegressor = require_xgboost()
    print("Loading and splitting tabular data...")
    df = read_table(tabular_path, args.target, args.max_cases)
    df = add_safe_features(df, args.include_scheduled_duration)
    split = temporal_group_holdout(df, args.test_fraction)
    train_df = split.train_df.copy()
    test_df = split.test_df.copy()

    overlap = set(train_df[PATIENT_ID_COLUMN].astype(str)) & set(test_df[PATIENT_ID_COLUMN].astype(str))
    if overlap:
        raise RuntimeError(f"Patient leakage in split: {len(overlap)} overlapping patients")

    print("Tuning structured-only XGBoost baseline...")
    best_params, trials_df, numeric_cols, categorical_cols = train_tabular_with_optuna(
        XGBRegressor,
        train_df,
        args.target,
        args.tabular_n_trials,
        args.tabular_timeout,
        args.n_folds,
        args.seed,
    )
    trials_df.to_csv(output_dir / "optuna_structured_trials.csv", index=False)

    print("Training final structured baseline and OOF residuals...")
    structured_pipe, structured_train_pred, structured_test_pred, _, _ = fit_predict_tabular(
        XGBRegressor, train_df, test_df, args.target, best_params, args.seed
    )
    oof_structured_pred, oof_residual = out_of_fold_structured_residuals(
        XGBRegressor, train_df, args.target, best_params, args.seed, args.n_folds
    )

    print("Building leakage-safe pre-cutoff note bundles...")
    all_cases = pd.concat([train_df, test_df], ignore_index=True)
    bundles = build_case_note_bundles(
        notes_path=notes_path,
        case_df=all_cases,
        max_notes_per_case=args.max_notes_per_case,
        max_note_chars=args.max_note_chars,
        max_case_note_chars=args.max_case_note_chars,
        max_note_age_days=args.max_note_age_days,
        max_note_rows=args.max_note_rows,
        note_cutoff_policy=args.note_cutoff_policy,
        include_same_day_preop_note_types=args.include_same_day_preop_note_types,
        keep_postop_doc_types=args.keep_postop_doc_types,
    )
    bundle_meta = dict(bundles.attrs)
    train_df = train_df.merge(bundles, on=CASE_ID_COLUMN, how="left")
    test_df = test_df.merge(bundles, on=CASE_ID_COLUMN, how="left")
    for frame in [train_df, test_df]:
        frame["note_text"] = frame["note_text"].fillna("")
        frame["n_pre_cutoff_notes_used"] = frame["n_pre_cutoff_notes_used"].fillna(0).astype(int)
        frame["note_chars"] = frame["note_chars"].fillna(0).astype(int)
        frame["note_doc_types_used"] = frame["note_doc_types_used"].fillna("")
        frame["note_dates_used"] = frame["note_dates_used"].fillna("")

    print("Training generic frozen TF-IDF/SVD note embedding...")
    z_generic_train, z_generic_test, generic_meta = fit_tfidf_svd_embedding(
        train_df["note_text"],
        test_df["note_text"],
        max_features=args.max_tfidf_features,
        min_df=args.min_df,
        svd_dim=args.svd_dim,
        seed=args.seed,
    )

    z_transformer_train = None
    z_transformer_test = None
    transformer_meta = None
    if args.transformer_model.strip():
        print(f"Training frozen transformer note embedding ({args.transformer_model})...")
        z_transformer_train, z_transformer_test, transformer_meta = fit_transformer_note_embedding(
            train_text=train_df["note_text"],
            test_text=test_df["note_text"],
            model_name=args.transformer_model.strip(),
            batch_size=args.transformer_batch_size,
            max_length=args.transformer_max_length,
            svd_dim=args.transformer_svd_dim,
            seed=args.seed,
        )

    print("Training CURE target-conditioned residual embedding...")
    z_cure_train, z_cure_test, train_resid_probe, test_resid_probe, cure_meta = fit_cure_residual_embedding(
        train_df=train_df,
        test_df=test_df,
        oof_structured_pred=oof_structured_pred,
        final_structured_train_pred=structured_train_pred,
        final_structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        max_features=args.max_tfidf_features,
        min_df=args.min_df,
        svd_dim=args.svd_dim,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    print("Training CURE sparse residual lexical embedding...")
    (
        z_sparse_train,
        z_sparse_test,
        sparse_train_resid_probe,
        sparse_test_resid_probe,
        sparse_meta,
    ) = fit_sparse_residual_lexical_embedding(
        train_df=train_df,
        test_df=test_df,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        max_features=min(args.max_tfidf_features, args.sparse_residual_max_features),
        min_df=args.min_df,
        n_folds=args.n_folds,
        seed=args.seed,
        top_procedures=args.sparse_residual_top_procedures,
        max_interactions=args.sparse_residual_max_interactions,
        regressor_names=[part.strip() for part in args.sparse_residual_regressors.split(",") if part.strip()],
        enable_classifiers=args.sparse_residual_classifiers,
    )

    print("Training CURE source-aware residual lexical embedding...")
    (
        z_source_sparse_train,
        z_source_sparse_test,
        source_sparse_train_resid_probe,
        source_sparse_test_resid_probe,
        source_sparse_meta,
        source_sparse_train_df,
        source_sparse_test_df,
    ) = fit_source_aware_sparse_residual_embedding(
        train_df=train_df,
        test_df=test_df,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        max_features=min(args.max_tfidf_features, args.sparse_residual_max_features),
        min_df=args.min_df,
        n_folds=args.n_folds,
        seed=args.seed + 17,
        top_procedures=args.sparse_residual_top_procedures,
        max_interactions=args.sparse_residual_max_interactions,
        regressor_names=[part.strip() for part in args.sparse_residual_regressors.split(",") if part.strip()],
        enable_classifiers=args.sparse_residual_classifiers,
    )

    print(f"Training CURE MIL residual encoder ({args.mil_backend})...")
    if args.mil_backend == "torch":
        z_mil_train, z_mil_test, mil_train_resid_probe, mil_test_resid_probe, mil_meta = fit_mil_residual_encoder(
            train_df=train_df,
            test_df=test_df,
            oof_structured_pred=oof_structured_pred,
            structured_test_pred=structured_test_pred,
            oof_residual=oof_residual,
            max_features=args.max_tfidf_features,
            min_df=args.min_df,
            chunk_svd_dim=args.mil_chunk_svd_dim,
            max_chunks=args.mil_max_chunks,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.mil_hidden_dim,
            epochs=args.mil_epochs,
            batch_size=args.mil_batch_size,
            learning_rate=args.mil_learning_rate,
            contrastive_weight=args.mil_contrastive_weight,
            seed=args.seed,
        )
    elif args.mil_backend == "fast":
        z_mil_train, z_mil_test, mil_train_resid_probe, mil_test_resid_probe, mil_meta = fit_fast_mil_residual_encoder(
            train_df=train_df,
            test_df=test_df,
            oof_structured_pred=oof_structured_pred,
            structured_test_pred=structured_test_pred,
            oof_residual=oof_residual,
            max_features=args.max_tfidf_features,
            min_df=args.min_df,
            chunk_svd_dim=args.mil_chunk_svd_dim,
            max_chunks=args.mil_max_chunks,
            seed=args.seed,
        )
    else:
        z_mil_train, z_mil_test, mil_train_resid_probe, mil_test_resid_probe, mil_meta = fit_empty_mil_embedding(
            train_df, test_df
        )

    print("Training CURE residual-neighborhood embedding...")
    residual_neighbor_ks = parse_int_list(args.residual_neighbor_ks)
    note_parts_train = [z_generic_train, z_cure_train, z_sparse_train, z_source_sparse_train, z_mil_train]
    note_parts_test = [z_generic_test, z_cure_test, z_sparse_test, z_source_sparse_test, z_mil_test]
    if z_transformer_train is not None and z_transformer_test is not None:
        note_parts_train.append(z_transformer_train)
        note_parts_test.append(z_transformer_test)
    z_note_ensemble_train = np.hstack(note_parts_train)
    z_note_ensemble_test = np.hstack(note_parts_test)
    (
        z_neighbor_train,
        z_neighbor_test,
        neighbor_train_resid_probe,
        neighbor_test_resid_probe,
        neighbor_meta,
    ) = fit_residual_neighborhood_embedding(
        train_df=train_df,
        test_df=test_df,
        note_z_train=z_note_ensemble_train,
        note_z_test=z_note_ensemble_test,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        ks=residual_neighbor_ks,
        context_weight=args.residual_neighbor_context_weight,
        note_weight=args.residual_neighbor_note_weight,
        seed=args.seed,
    )

    print("Training CURE target-conditioned fusion embedding...")
    fusion_parts_train = [
        z_generic_train,
        z_cure_train,
        z_sparse_train,
        z_source_sparse_train,
        z_mil_train,
        z_neighbor_train,
    ]
    fusion_parts_test = [
        z_generic_test,
        z_cure_test,
        z_sparse_test,
        z_source_sparse_test,
        z_mil_test,
        z_neighbor_test,
    ]
    if z_transformer_train is not None and z_transformer_test is not None:
        fusion_parts_train.append(z_transformer_train)
        fusion_parts_test.append(z_transformer_test)
    (
        z_fusion_train,
        z_fusion_test,
        fusion_train_resid_probe,
        fusion_test_resid_probe,
        fusion_meta,
    ) = fit_cure_target_fusion_embedding(
        train_df=train_df,
        test_df=test_df,
        embedding_blocks_train=fusion_parts_train,
        embedding_blocks_test=fusion_parts_test,
        oof_structured_pred=oof_structured_pred,
        structured_test_pred=structured_test_pred,
        oof_residual=oof_residual,
        svd_dim=args.fusion_svd_dim,
        pls_dim=args.fusion_pls_dim,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    print("Evaluating downstream models...")
    results: list[dict[str, Any]] = []
    structured_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), structured_test_pred)
    structured_metrics["method"] = "Structured_only_XGBoost"
    results.append(structured_metrics)

    generic_train = add_embedding_columns(train_df, z_generic_train, "generic_note_svd")
    generic_test = add_embedding_columns(test_df, z_generic_test, "generic_note_svd")
    metrics, generic_pred, generic_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_generic_TFIDF_SVD_notes",
        generic_train,
        generic_test,
        args.target,
        best_params,
        args.seed + 11,
    )
    results.append(metrics)

    transformer_pred = None
    transformer_resid_pred = None
    transformer_selected_resid_pred = None
    transformer_pipe = None
    transformer_resid_pipe = None
    transformer_selected_resid_pipe = None
    transformer_selected_details: dict[str, Any] | None = None
    transformer_selected_oof_resid = None
    if z_transformer_train is not None and z_transformer_test is not None:
        transformer_train = add_embedding_columns(train_df, z_transformer_train, "transformer_note_z")
        transformer_test = add_embedding_columns(test_df, z_transformer_test, "transformer_note_z")
        metrics, transformer_pred, transformer_pipe = evaluate_augmented_model(
            XGBRegressor,
            f"Structured_plus_{args.transformer_model}_frozen_notes",
            transformer_train,
            transformer_test,
            args.target,
            best_params,
            args.seed + 99,
        )
        results.append(metrics)
        metrics, transformer_resid_pred, transformer_resid_pipe = evaluate_residual_corrector(
            XGBRegressor,
            f"Structured_plus_{args.transformer_model}_residual_corrector",
            transformer_train,
            transformer_test,
            args.target,
            oof_residual,
            structured_test_pred,
            best_params,
            args.seed + 199,
        )
        results.append(metrics)
        (
            metrics,
            transformer_selected_resid_pred,
            transformer_selected_resid_pipe,
            transformer_selected_details,
            transformer_selected_oof_resid,
        ) = evaluate_selected_residual_corrector(
            XGBRegressor,
            f"Structured_plus_{args.transformer_model}_selected_residual_corrector",
            transformer_train,
            transformer_test,
            args.target,
            oof_residual,
            structured_test_pred,
            best_params,
            args.seed + 299,
            args.n_folds,
        )
        results.append(metrics)

    cure_train = add_embedding_columns(train_df, z_cure_train, "cure_resid_z")
    cure_test = add_embedding_columns(test_df, z_cure_test, "cure_resid_z")
    metrics, cure_pred, cure_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_residual_conditioned_embedding",
        cure_train,
        cure_test,
        args.target,
        best_params,
        args.seed + 22,
    )
    results.append(metrics)

    sparse_train = add_embedding_columns(train_df, z_sparse_train, "cure_sparse_resid_z")
    sparse_test = add_embedding_columns(test_df, z_sparse_test, "cure_sparse_resid_z")
    metrics, sparse_pred, sparse_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_sparse_residual_lexical_embedding",
        sparse_train,
        sparse_test,
        args.target,
        best_params,
        args.seed + 25,
    )
    results.append(metrics)

    sparse_direct_pred = structured_test_pred + sparse_test_resid_probe
    sparse_direct_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), sparse_direct_pred)
    sparse_direct_metrics["method"] = "Structured_plus_CURE_sparse_residual_lexical_direct_probe"
    results.append(sparse_direct_metrics)
    sparse_group_scales = fit_group_shrinkage(
        y_true_residual=oof_residual,
        oof_pred_residual=sparse_train_resid_probe,
        has_notes=train_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
    )
    sparse_group_oof = apply_group_shrinkage(
        sparse_train_resid_probe,
        train_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
        sparse_group_scales,
    )
    sparse_proc_cal = fit_procedure_shrinkage(
        train_df=train_df,
        y_true_residual=oof_residual,
        oof_pred_residual=sparse_train_resid_probe,
        min_support=max(20, int(0.01 * len(train_df))),
    )
    if sparse_proc_cal["oof_mae"] <= mean_absolute_error(oof_residual, sparse_group_oof):
        sparse_calibrated_resid = apply_procedure_shrinkage(test_df, sparse_test_resid_probe, sparse_proc_cal)
        sparse_calibration_mode = "procedure_shrinkage"
        sparse_calibration_oof_mae = float(sparse_proc_cal["oof_mae"])
    else:
        sparse_calibrated_resid = apply_group_shrinkage(
            sparse_test_resid_probe,
            test_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
            sparse_group_scales,
        )
        sparse_calibration_mode = "group_shrinkage"
        sparse_calibration_oof_mae = float(mean_absolute_error(oof_residual, sparse_group_oof))
    sparse_calibrated_direct_pred = structured_test_pred + sparse_calibrated_resid
    sparse_calibrated_metrics = metric_dict(
        test_df[args.target].to_numpy(dtype=float), sparse_calibrated_direct_pred
    )
    sparse_calibrated_metrics[
        "method"
    ] = f"Structured_plus_CURE_sparse_residual_lexical_calibrated_direct_probe_{sparse_calibration_mode}"
    results.append(sparse_calibrated_metrics)

    source_sparse_train = add_embedding_columns(train_df, z_source_sparse_train, "cure_source_sparse_resid_z")
    source_sparse_test = add_embedding_columns(test_df, z_source_sparse_test, "cure_source_sparse_resid_z")
    metrics, source_sparse_pred, source_sparse_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_source_aware_sparse_residual_lexical_embedding",
        source_sparse_train,
        source_sparse_test,
        args.target,
        best_params,
        args.seed + 26,
    )
    results.append(metrics)

    source_sparse_direct_pred = structured_test_pred + source_sparse_test_resid_probe
    source_sparse_direct_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), source_sparse_direct_pred)
    source_sparse_direct_metrics["method"] = "Structured_plus_CURE_source_aware_sparse_direct_probe"
    results.append(source_sparse_direct_metrics)
    source_sparse_group_scales = fit_group_shrinkage(
        y_true_residual=oof_residual,
        oof_pred_residual=source_sparse_train_resid_probe,
        has_notes=train_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
    )
    source_sparse_group_oof = apply_group_shrinkage(
        source_sparse_train_resid_probe,
        train_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
        source_sparse_group_scales,
    )
    source_sparse_proc_cal = fit_procedure_shrinkage(
        train_df=train_df,
        y_true_residual=oof_residual,
        oof_pred_residual=source_sparse_train_resid_probe,
        min_support=max(20, int(0.01 * len(train_df))),
    )
    if source_sparse_proc_cal["oof_mae"] <= mean_absolute_error(oof_residual, source_sparse_group_oof):
        source_sparse_calibrated_resid = apply_procedure_shrinkage(
            test_df, source_sparse_test_resid_probe, source_sparse_proc_cal
        )
        source_sparse_calibration_mode = "procedure_shrinkage"
        source_sparse_calibration_oof_mae = float(source_sparse_proc_cal["oof_mae"])
    else:
        source_sparse_calibrated_resid = apply_group_shrinkage(
            source_sparse_test_resid_probe,
            test_df["n_pre_cutoff_notes_used"].to_numpy() > 0,
            source_sparse_group_scales,
        )
        source_sparse_calibration_mode = "group_shrinkage"
        source_sparse_calibration_oof_mae = float(mean_absolute_error(oof_residual, source_sparse_group_oof))
    source_sparse_calibrated_direct_pred = structured_test_pred + source_sparse_calibrated_resid
    source_sparse_calibrated_metrics = metric_dict(
        test_df[args.target].to_numpy(dtype=float), source_sparse_calibrated_direct_pred
    )
    source_sparse_calibrated_metrics[
        "method"
    ] = f"Structured_plus_CURE_source_aware_sparse_calibrated_direct_probe_{source_sparse_calibration_mode}"
    results.append(source_sparse_calibrated_metrics)

    mil_train = add_embedding_columns(train_df, z_mil_train, "cure_mil_z")
    mil_test = add_embedding_columns(test_df, z_mil_test, "cure_mil_z")
    metrics, mil_pred, mil_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_MIL_residual_encoder_embedding",
        mil_train,
        mil_test,
        args.target,
        best_params,
        args.seed + 55,
    )
    results.append(metrics)

    metrics, generic_resid_pred, generic_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_generic_embedding_residual_corrector",
        generic_train,
        generic_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 33,
    )
    results.append(metrics)

    (
        metrics,
        generic_selected_resid_pred,
        generic_selected_resid_pipe,
        generic_selected_details,
        generic_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_generic_embedding_selected_residual_corrector",
        generic_train,
        generic_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 133,
        args.n_folds,
    )
    results.append(metrics)

    metrics, sparse_resid_pred, sparse_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_sparse_residual_lexical_corrector",
        sparse_train,
        sparse_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 35,
    )
    results.append(metrics)

    (
        metrics,
        sparse_selected_resid_pred,
        sparse_selected_resid_pipe,
        sparse_selected_details,
        sparse_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_sparse_residual_lexical_selected_corrector",
        sparse_train,
        sparse_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 135,
        args.n_folds,
    )
    results.append(metrics)

    metrics, source_sparse_resid_pred, source_sparse_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_source_aware_sparse_residual_corrector",
        source_sparse_train,
        source_sparse_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 36,
    )
    results.append(metrics)

    (
        metrics,
        source_sparse_selected_resid_pred,
        source_sparse_selected_resid_pipe,
        source_sparse_selected_details,
        source_sparse_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_source_aware_sparse_selected_corrector",
        source_sparse_train,
        source_sparse_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 136,
        args.n_folds,
    )
    results.append(metrics)

    metrics, mil_resid_pred, mil_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_MIL_embedding_residual_corrector",
        mil_train,
        mil_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 66,
    )
    results.append(metrics)

    (
        metrics,
        mil_selected_resid_pred,
        mil_selected_resid_pipe,
        mil_selected_details,
        mil_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_MIL_embedding_selected_residual_corrector",
        mil_train,
        mil_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 166,
        args.n_folds,
    )
    results.append(metrics)

    mil_direct_pred = structured_test_pred + mil_test_resid_probe
    mil_direct_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), mil_direct_pred)
    mil_direct_metrics["method"] = "Structured_plus_CURE_MIL_direct_residual_head"
    results.append(mil_direct_metrics)

    neighbor_train = add_embedding_columns(train_df, z_neighbor_train, "cure_neighbor_z")
    neighbor_test = add_embedding_columns(test_df, z_neighbor_test, "cure_neighbor_z")
    metrics, neighbor_pred, neighbor_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_residual_neighborhood_embedding",
        neighbor_train,
        neighbor_test,
        args.target,
        best_params,
        args.seed + 77,
    )
    results.append(metrics)

    metrics, neighbor_resid_pred, neighbor_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_residual_neighborhood_corrector",
        neighbor_train,
        neighbor_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 88,
    )
    results.append(metrics)

    (
        metrics,
        neighbor_selected_resid_pred,
        neighbor_selected_resid_pipe,
        neighbor_selected_details,
        neighbor_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_residual_neighborhood_selected_corrector",
        neighbor_train,
        neighbor_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 188,
        args.n_folds,
    )
    results.append(metrics)

    neighbor_direct_pred = structured_test_pred + neighbor_test_resid_probe
    neighbor_direct_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), neighbor_direct_pred)
    neighbor_direct_metrics["method"] = "Structured_plus_CURE_residual_neighborhood_direct_probe"
    results.append(neighbor_direct_metrics)

    fusion_train = add_embedding_columns(train_df, z_fusion_train, "cure_fusion_z")
    fusion_test = add_embedding_columns(test_df, z_fusion_test, "cure_fusion_z")
    metrics, fusion_pred, fusion_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_target_conditioned_fusion_embedding",
        fusion_train,
        fusion_test,
        args.target,
        best_params,
        args.seed + 101,
    )
    results.append(metrics)

    metrics, fusion_resid_pred, fusion_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_target_conditioned_fusion_corrector",
        fusion_train,
        fusion_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 102,
    )
    results.append(metrics)

    metrics, fusion_selected_resid_pred, fusion_selected_resid_pipe, fusion_selected_details, fusion_selected_oof_resid = (
        evaluate_selected_residual_corrector(
            XGBRegressor,
            "Structured_plus_CURE_target_conditioned_fusion_selected_corrector",
            fusion_train,
            fusion_test,
            args.target,
            oof_residual,
            structured_test_pred,
            best_params,
            args.seed + 202,
            args.n_folds,
        )
    )
    results.append(metrics)

    fusion_direct_pred = structured_test_pred + fusion_test_resid_probe
    fusion_direct_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), fusion_direct_pred)
    fusion_direct_metrics["method"] = "Structured_plus_CURE_target_conditioned_fusion_direct_probe"
    results.append(fusion_direct_metrics)

    stack_parts_train = [
        z_generic_train,
        z_cure_train,
        z_sparse_train,
        z_source_sparse_train,
        z_mil_train,
        z_neighbor_train,
        z_fusion_train,
    ]
    stack_parts_test = [
        z_generic_test,
        z_cure_test,
        z_sparse_test,
        z_source_sparse_test,
        z_mil_test,
        z_neighbor_test,
        z_fusion_test,
    ]
    if z_transformer_train is not None and z_transformer_test is not None:
        stack_parts_train.append(z_transformer_train)
        stack_parts_test.append(z_transformer_test)
    z_stack_train = np.hstack(stack_parts_train).astype("float32")
    z_stack_test = np.hstack(stack_parts_test).astype("float32")
    stack_train = add_embedding_columns(train_df, z_stack_train, "cure_stack_z")
    stack_test = add_embedding_columns(test_df, z_stack_test, "cure_stack_z")
    metrics, stack_pred, stack_pipe = evaluate_augmented_model(
        XGBRegressor,
        "Structured_plus_CURE_residual_stack_embedding",
        stack_train,
        stack_test,
        args.target,
        best_params,
        args.seed + 303,
    )
    results.append(metrics)

    metrics, stack_resid_pred, stack_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_residual_stack_corrector",
        stack_train,
        stack_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 304,
    )
    results.append(metrics)

    metrics, stack_selected_resid_pred, stack_selected_resid_pipe, stack_selected_details, stack_selected_oof_resid = (
        evaluate_selected_residual_corrector(
            XGBRegressor,
            "Structured_plus_CURE_residual_stack_selected_corrector",
            stack_train,
            stack_test,
            args.target,
            oof_residual,
            structured_test_pred,
            best_params,
            args.seed + 404,
            args.n_folds,
        )
    )
    results.append(metrics)

    ensemble_frames = [
        ("generic", generic_train, generic_test),
        ("cure", cure_train, cure_test),
        ("cure_sparse", sparse_train, sparse_test),
        ("cure_source_sparse", source_sparse_train, source_sparse_test),
        ("cure_mil", mil_train, mil_test),
        ("cure_neighbor", neighbor_train, neighbor_test),
        ("cure_fusion", fusion_train, fusion_test),
        ("cure_stack", stack_train, stack_test),
    ]
    if z_transformer_train is not None and z_transformer_test is not None:
        ensemble_frames.append(("transformer", transformer_train, transformer_test))
    if args.skip_oof_ensemble:
        print("Skipping CURE OOF residual-ensemble embedding.")
        z_ensemble_train = np.zeros((len(train_df), 1), dtype="float32")
        z_ensemble_test = np.zeros((len(test_df), 1), dtype="float32")
        ensemble_pred = structured_test_pred.copy()
        ensemble_aug_pred = structured_test_pred.copy()
        ensemble_resid_pred = structured_test_pred.copy()
        ensemble_selected_resid_pred = structured_test_pred.copy()
        ensemble_selected_oof_resid = np.zeros(len(train_df), dtype="float32")
        ensemble_meta_pipe = None
        ensemble_aug_pipe = None
        ensemble_resid_pipe = None
        ensemble_selected_resid_pipe = None
        ensemble_selected_details = {"selected_residual_model": "skipped"}
        ensemble_meta = {
            "backend": "skipped_oof_residual_ensemble",
            "embedding_dim": 1,
            "base_coordinate_count": 0,
            "train_meta_residual_mae": None,
            "selection_table": [],
        }
    else:
        print("Training CURE OOF residual-ensemble embedding...")
        (
            z_ensemble_train,
            z_ensemble_test,
            ensemble_pred,
            ensemble_meta_pipe,
            ensemble_meta,
        ) = fit_residual_ensemble_embedding(
            XGBRegressor=XGBRegressor,
            embedding_frames=ensemble_frames,
            test_df=test_df,
            target=args.target,
            oof_residual=oof_residual,
            structured_test_pred=structured_test_pred,
            params=best_params,
            seed=args.seed + 505,
            n_folds=args.n_folds,
        )
        ensemble_metrics = metric_dict(test_df[args.target].to_numpy(dtype=float), ensemble_pred)
        ensemble_metrics["method"] = "Structured_plus_CURE_OOF_residual_ensemble_direct"
        results.append(ensemble_metrics)

        ensemble_train = add_embedding_columns(train_df, z_ensemble_train, "cure_oof_ensemble_z")
        ensemble_test = add_embedding_columns(test_df, z_ensemble_test, "cure_oof_ensemble_z")
        metrics, ensemble_aug_pred, ensemble_aug_pipe = evaluate_augmented_model(
            XGBRegressor,
            "Structured_plus_CURE_OOF_residual_ensemble_embedding",
            ensemble_train,
            ensemble_test,
            args.target,
            best_params,
            args.seed + 506,
        )
        results.append(metrics)

        metrics, ensemble_resid_pred, ensemble_resid_pipe = evaluate_residual_corrector(
            XGBRegressor,
            "Structured_plus_CURE_OOF_residual_ensemble_corrector",
            ensemble_train,
            ensemble_test,
            args.target,
            oof_residual,
            structured_test_pred,
            best_params,
            args.seed + 507,
        )
        results.append(metrics)

        (
            metrics,
            ensemble_selected_resid_pred,
            ensemble_selected_resid_pipe,
            ensemble_selected_details,
            ensemble_selected_oof_resid,
        ) = (
            evaluate_selected_residual_corrector(
                XGBRegressor,
                "Structured_plus_CURE_OOF_residual_ensemble_selected_corrector",
                ensemble_train,
                ensemble_test,
                args.target,
                oof_residual,
                structured_test_pred,
                best_params,
                args.seed + 508,
                args.n_folds,
            )
        )
        results.append(metrics)

    metrics, cure_resid_pred, cure_resid_pipe = evaluate_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_embedding_residual_corrector",
        cure_train,
        cure_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 44,
    )
    results.append(metrics)

    (
        metrics,
        cure_selected_resid_pred,
        cure_selected_resid_pipe,
        cure_selected_details,
        cure_selected_oof_resid,
    ) = evaluate_selected_residual_corrector(
        XGBRegressor,
        "Structured_plus_CURE_embedding_selected_residual_corrector",
        cure_train,
        cure_test,
        args.target,
        oof_residual,
        structured_test_pred,
        best_params,
        args.seed + 144,
        args.n_folds,
    )
    results.append(metrics)

    arm_train_residuals = {
        "generic_selected": generic_selected_oof_resid,
        "cure_selected": cure_selected_oof_resid,
        "cure_sparse_selected": sparse_selected_oof_resid,
        "cure_source_sparse_selected": source_sparse_selected_oof_resid,
        "cure_mil_selected": mil_selected_oof_resid,
        "cure_neighbor_selected": neighbor_selected_oof_resid,
        "cure_fusion_selected": fusion_selected_oof_resid,
        "cure_stack_selected": stack_selected_oof_resid,
        "cure_oof_ensemble_selected": ensemble_selected_oof_resid,
    }
    arm_test_residuals = {
        "generic_selected": generic_selected_resid_pred - structured_test_pred,
        "cure_selected": cure_selected_resid_pred - structured_test_pred,
        "cure_sparse_selected": sparse_selected_resid_pred - structured_test_pred,
        "cure_source_sparse_selected": source_sparse_selected_resid_pred - structured_test_pred,
        "cure_mil_selected": mil_selected_resid_pred - structured_test_pred,
        "cure_neighbor_selected": neighbor_selected_resid_pred - structured_test_pred,
        "cure_fusion_selected": fusion_selected_resid_pred - structured_test_pred,
        "cure_stack_selected": stack_selected_resid_pred - structured_test_pred,
        "cure_oof_ensemble_selected": ensemble_selected_resid_pred - structured_test_pred,
    }
    if transformer_selected_oof_resid is not None and transformer_selected_resid_pred is not None:
        arm_train_residuals["transformer_selected"] = transformer_selected_oof_resid
        arm_test_residuals["transformer_selected"] = transformer_selected_resid_pred - structured_test_pred
    (
        procedure_arm_metrics,
        procedure_arm_pred,
        procedure_arm_details,
    ) = fit_procedure_residual_arm_chooser(
        train_df=train_df,
        test_df=test_df,
        target=args.target,
        oof_residual=oof_residual,
        structured_test_pred=structured_test_pred,
        train_arm_residuals=arm_train_residuals,
        test_arm_residuals=arm_test_residuals,
        min_support=max(30, int(0.01 * len(train_df))),
    )
    results.append(procedure_arm_metrics)
    pd.DataFrame(procedure_arm_details["selection_table"]).sort_values(
        ["n_train", "procedure"], ascending=[False, True]
    ).to_csv(output_dir / "procedure_residual_arm_chooser_selection.csv", index=False)

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df["delta_MAE_vs_structured"] = results_df["MAE"] - structured_metrics["MAE"]
    results_df.to_csv(output_dir / "benchmark_results.csv", index=False)

    predictions = test_df[
        [
            CASE_ID_COLUMN,
            PATIENT_ID_COLUMN,
            DATE_COLUMN,
            "primary_procedure_name",
            args.target,
            "n_pre_cutoff_notes_used",
            "note_chars",
            "note_doc_types_used",
            "note_dates_used",
        ]
    ].copy()
    predictions["structured_pred"] = structured_test_pred
    predictions["generic_embedding_pred"] = generic_pred
    predictions["cure_embedding_pred"] = cure_pred
    predictions["generic_residual_corrector_pred"] = generic_resid_pred
    predictions["generic_selected_residual_corrector_pred"] = generic_selected_resid_pred
    if transformer_pred is not None:
        predictions["transformer_embedding_pred"] = transformer_pred
    if transformer_resid_pred is not None:
        predictions["transformer_residual_corrector_pred"] = transformer_resid_pred
    if transformer_selected_resid_pred is not None:
        predictions["transformer_selected_residual_corrector_pred"] = transformer_selected_resid_pred
    predictions["cure_residual_corrector_pred"] = cure_resid_pred
    predictions["cure_selected_residual_corrector_pred"] = cure_selected_resid_pred
    predictions["cure_sparse_residual_lexical_embedding_pred"] = sparse_pred
    predictions["cure_sparse_residual_lexical_direct_probe_pred"] = sparse_direct_pred
    predictions["cure_sparse_residual_lexical_calibrated_direct_probe_pred"] = sparse_calibrated_direct_pred
    predictions["cure_sparse_residual_lexical_corrector_pred"] = sparse_resid_pred
    predictions["cure_sparse_residual_lexical_selected_corrector_pred"] = sparse_selected_resid_pred
    predictions["cure_source_aware_sparse_residual_lexical_embedding_pred"] = source_sparse_pred
    predictions["cure_source_aware_sparse_direct_probe_pred"] = source_sparse_direct_pred
    predictions["cure_source_aware_sparse_calibrated_direct_probe_pred"] = source_sparse_calibrated_direct_pred
    predictions["cure_source_aware_sparse_residual_corrector_pred"] = source_sparse_resid_pred
    predictions["cure_source_aware_sparse_selected_corrector_pred"] = source_sparse_selected_resid_pred
    predictions["cure_mil_embedding_pred"] = mil_pred
    predictions["cure_mil_residual_corrector_pred"] = mil_resid_pred
    predictions["cure_mil_selected_residual_corrector_pred"] = mil_selected_resid_pred
    predictions["cure_mil_direct_residual_head_pred"] = mil_direct_pred
    predictions["cure_neighbor_embedding_pred"] = neighbor_pred
    predictions["cure_neighbor_residual_corrector_pred"] = neighbor_resid_pred
    predictions["cure_neighbor_selected_residual_corrector_pred"] = neighbor_selected_resid_pred
    predictions["cure_neighbor_direct_residual_probe_pred"] = neighbor_direct_pred
    predictions["cure_fusion_embedding_pred"] = fusion_pred
    predictions["cure_fusion_residual_corrector_pred"] = fusion_resid_pred
    predictions["cure_fusion_selected_residual_corrector_pred"] = fusion_selected_resid_pred
    predictions["cure_fusion_direct_residual_probe_pred"] = fusion_direct_pred
    predictions["cure_stack_embedding_pred"] = stack_pred
    predictions["cure_stack_residual_corrector_pred"] = stack_resid_pred
    predictions["cure_stack_selected_residual_corrector_pred"] = stack_selected_resid_pred
    predictions["cure_oof_ensemble_direct_pred"] = ensemble_pred
    predictions["cure_oof_ensemble_embedding_pred"] = ensemble_aug_pred
    predictions["cure_oof_ensemble_residual_corrector_pred"] = ensemble_resid_pred
    predictions["cure_oof_ensemble_selected_residual_corrector_pred"] = ensemble_selected_resid_pred
    predictions["cure_procedure_residual_arm_chooser_pred"] = procedure_arm_pred
    predictions["structured_abs_error"] = (predictions["structured_pred"] - predictions[args.target]).abs()
    predictions["cure_abs_error"] = (predictions["cure_embedding_pred"] - predictions[args.target]).abs()
    predictions["cure_error_delta_vs_structured"] = predictions["cure_abs_error"] - predictions["structured_abs_error"]
    predictions.to_csv(output_dir / "test_predictions_with_note_embeddings.csv", index=False)

    safe_procedure_slice_metrics(test_df, structured_test_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_structured.csv", index=False
    )
    safe_procedure_slice_metrics(test_df, cure_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_cure_embedding.csv", index=False
    )
    safe_procedure_slice_metrics(test_df, sparse_selected_resid_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_cure_sparse_selected.csv", index=False
    )
    safe_procedure_slice_metrics(test_df, source_sparse_selected_resid_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_cure_source_aware_sparse_selected.csv", index=False
    )
    safe_procedure_slice_metrics(test_df, mil_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_cure_mil_embedding.csv", index=False
    )
    safe_procedure_slice_metrics(test_df, procedure_arm_pred, args.target).to_csv(
        output_dir / "procedure_slice_metrics_cure_procedure_arm_chooser.csv", index=False
    )

    embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_cure_test,
                columns=[f"cure_resid_z_{k:03d}" for k in range(z_cure_test.shape[1])],
            ),
        ],
        axis=1,
    )
    embeddings_out.to_csv(output_dir / "test_cure_embeddings.csv", index=False)
    sparse_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_sparse_test,
                columns=[f"cure_sparse_resid_z_{k:03d}" for k in range(z_sparse_test.shape[1])],
            ),
        ],
        axis=1,
    )
    sparse_embeddings_out.to_csv(output_dir / "test_cure_sparse_residual_lexical_embeddings.csv", index=False)
    source_sparse_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_source_sparse_test,
                columns=[f"cure_source_sparse_resid_z_{k:03d}" for k in range(z_source_sparse_test.shape[1])],
            ),
        ],
        axis=1,
    )
    source_sparse_embeddings_out.to_csv(
        output_dir / "test_cure_source_aware_sparse_residual_lexical_embeddings.csv", index=False
    )
    mil_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_mil_test,
                columns=[f"cure_mil_z_{k:03d}" for k in range(z_mil_test.shape[1])],
            ),
        ],
        axis=1,
    )
    mil_embeddings_out.to_csv(output_dir / "test_cure_mil_embeddings.csv", index=False)
    if z_transformer_test is not None:
        transformer_embeddings_out = pd.concat(
            [
                pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
                pd.DataFrame(
                    z_transformer_test,
                    columns=[f"transformer_note_z_{k:03d}" for k in range(z_transformer_test.shape[1])],
                ),
            ],
            axis=1,
        )
        transformer_embeddings_out.to_csv(output_dir / "test_transformer_note_embeddings.csv", index=False)
    neighbor_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_neighbor_test,
                columns=[f"cure_neighbor_z_{k:03d}" for k in range(z_neighbor_test.shape[1])],
            ),
        ],
        axis=1,
    )
    neighbor_embeddings_out.to_csv(output_dir / "test_cure_residual_neighborhood_embeddings.csv", index=False)
    fusion_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_fusion_test,
                columns=[f"cure_fusion_z_{k:03d}" for k in range(z_fusion_test.shape[1])],
            ),
        ],
        axis=1,
    )
    fusion_embeddings_out.to_csv(output_dir / "test_cure_target_conditioned_fusion_embeddings.csv", index=False)
    stack_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_stack_test,
                columns=[f"cure_stack_z_{k:03d}" for k in range(z_stack_test.shape[1])],
            ),
        ],
        axis=1,
    )
    stack_embeddings_out.to_csv(output_dir / "test_cure_residual_stack_embeddings.csv", index=False)
    ensemble_embeddings_out = pd.concat(
        [
            pd.DataFrame({CASE_ID_COLUMN: test_df[CASE_ID_COLUMN].to_numpy()}),
            pd.DataFrame(
                z_ensemble_test,
                columns=[f"cure_oof_ensemble_z_{k:03d}" for k in range(z_ensemble_test.shape[1])],
            ),
        ],
        axis=1,
    )
    ensemble_embeddings_out.to_csv(output_dir / "test_cure_oof_residual_ensemble_embeddings.csv", index=False)

    summary = {
        "tabular_path": str(tabular_path),
        "notes_path": str(notes_path),
        "target": args.target,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_patients": int(train_df[PATIENT_ID_COLUMN].nunique()),
        "test_patients": int(test_df[PATIENT_ID_COLUMN].nunique()),
        "patient_overlap_train_test": int(len(overlap)),
        "temporal_cutoff": str(split.cutoff),
        "best_structured_params": best_params,
        "note_bundle_meta": bundle_meta,
        "train_cases_with_notes": int((train_df["n_pre_cutoff_notes_used"] > 0).sum()),
        "test_cases_with_notes": int((test_df["n_pre_cutoff_notes_used"] > 0).sum()),
        "mean_train_notes_per_case": float(train_df["n_pre_cutoff_notes_used"].mean()),
        "mean_test_notes_per_case": float(test_df["n_pre_cutoff_notes_used"].mean()),
        "generic_embedding": {
            "n_terms": generic_meta["n_terms"],
            "embedding_dim": generic_meta["embedding_dim"],
            "svd_explained_variance_ratio_sum": generic_meta["explained_variance_ratio_sum"],
        },
        "cure_embedding": {
            key: value
            for key, value in cure_meta.items()
            if key
            not in {
                "vectorizer",
                "svd",
                "low_scaler",
                "context_scaler",
                "residual_scaler",
                "mlp",
                "z_scaler",
            }
        },
        "cure_sparse_residual_lexical_embedding": {
            key: value
            for key, value in sparse_meta.items()
            if key
            not in {
                "vectorizer",
                "context_scaler",
                "residual_scaler",
                "models",
                "z_scaler",
            }
        },
        "cure_sparse_residual_lexical_direct_calibration": {
            "selected_mode": sparse_calibration_mode,
            "selected_oof_residual_mae": sparse_calibration_oof_mae,
            "group_scales": sparse_group_scales,
            "procedure_shrinkage": {
                key: value for key, value in sparse_proc_cal.items() if key not in {"oof_pred"}
            },
        },
        "cure_source_aware_sparse_residual_lexical_embedding": {
            key: value
            for key, value in source_sparse_meta.items()
            if key
            not in {
                "vectorizer",
                "context_scaler",
                "residual_scaler",
                "models",
                "z_scaler",
            }
        },
        "cure_source_aware_sparse_direct_calibration": {
            "selected_mode": source_sparse_calibration_mode,
            "selected_oof_residual_mae": source_sparse_calibration_oof_mae,
            "group_scales": source_sparse_group_scales,
            "procedure_shrinkage": {
                key: value for key, value in source_sparse_proc_cal.items() if key not in {"oof_pred"}
            },
        },
        "cure_mil_embedding": {
            key: value
            for key, value in mil_meta.items()
            if key
            not in {
                "chunk_vectorizer",
                "chunk_svd",
                "chunk_scaler",
                "context_scaler",
                "residual_scaler",
                "z_scaler",
                "model_state_dict",
                "chunk_score_model",
                "chunk_score_scaler",
                "residual_feature_scaler",
                "residual_head",
            }
        },
        "cure_residual_neighborhood_embedding": {
            key: value
            for key, value in neighbor_meta.items()
            if key
            not in {
                "context_scaler",
                "note_scaler",
                "nearest_neighbors",
                "z_scaler",
            }
        },
        "cure_target_conditioned_fusion_embedding": {
            key: value
            for key, value in fusion_meta.items()
            if key
            not in {
                "raw_scaler",
                "svd",
                "low_scaler",
                "pls",
                "probe_models",
                "z_scaler",
                "context_scaler",
            }
        },
        "cure_residual_stack_embedding": {
            "backend": "no_bottleneck_residual_stack",
            "embedding_dim": int(z_stack_train.shape[1]),
            "includes_transformer": bool(z_transformer_train is not None),
        },
        "cure_oof_residual_ensemble_embedding": {
            key: value
            for key, value in ensemble_meta.items()
            if key not in {"z_scaler", "base_residual_pipelines", "meta_residual_pipeline"}
        },
        "selected_residual_correctors": {
            "generic": generic_selected_details,
            "cure": cure_selected_details,
            "cure_sparse_residual_lexical": sparse_selected_details,
            "cure_source_aware_sparse_residual_lexical": source_sparse_selected_details,
            "cure_mil": mil_selected_details,
            "cure_residual_neighborhood": neighbor_selected_details,
            "cure_target_conditioned_fusion": fusion_selected_details,
            "cure_residual_stack": stack_selected_details,
            "cure_oof_residual_ensemble": ensemble_selected_details,
        },
        "procedure_residual_arm_chooser": procedure_arm_details,
        "results": results_df.to_dict(orient="records"),
    }
    if transformer_meta is not None:
        summary["transformer_embedding"] = {
            key: value
            for key, value in transformer_meta.items()
            if key not in {"tokenizer", "model", "svd", "scaler"}
        }
        summary["selected_residual_correctors"]["transformer"] = transformer_selected_details
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "structured_xgb_pipeline.pkl").open("wb") as handle:
        pickle.dump(structured_pipe, handle)
    with (output_dir / "generic_embedding_xgb_pipeline.pkl").open("wb") as handle:
        pickle.dump(generic_pipe, handle)
    if transformer_pipe is not None:
        with (output_dir / "transformer_embedding_xgb_pipeline.pkl").open("wb") as handle:
            pickle.dump(transformer_pipe, handle)
    if transformer_resid_pipe is not None:
        with (output_dir / "transformer_residual_corrector_pipeline.pkl").open("wb") as handle:
            pickle.dump(transformer_resid_pipe, handle)
    if transformer_selected_resid_pipe is not None:
        with (output_dir / "transformer_selected_residual_corrector_pipeline.pkl").open("wb") as handle:
            pickle.dump(transformer_selected_resid_pipe, handle)
    with (output_dir / "cure_embedding_xgb_pipeline.pkl").open("wb") as handle:
        pickle.dump(cure_pipe, handle)
    with (output_dir / "cure_sparse_residual_lexical_pipeline.pkl").open("wb") as handle:
        pickle.dump(sparse_pipe, handle)
    with (output_dir / "cure_source_aware_sparse_residual_lexical_pipeline.pkl").open("wb") as handle:
        pickle.dump(source_sparse_pipe, handle)
    with (output_dir / "generic_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(generic_resid_pipe, handle)
    with (output_dir / "cure_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(cure_resid_pipe, handle)
    with (output_dir / "cure_sparse_residual_lexical_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(sparse_resid_pipe, handle)
    with (output_dir / "cure_source_aware_sparse_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(source_sparse_resid_pipe, handle)
    with (output_dir / "cure_mil_embedding_xgb_pipeline.pkl").open("wb") as handle:
        pickle.dump(mil_pipe, handle)
    with (output_dir / "cure_mil_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(mil_resid_pipe, handle)
    with (output_dir / "generic_selected_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(generic_selected_resid_pipe, handle)
    with (output_dir / "cure_selected_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(cure_selected_resid_pipe, handle)
    with (output_dir / "cure_sparse_residual_lexical_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(sparse_selected_resid_pipe, handle)
    with (output_dir / "cure_source_aware_sparse_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(source_sparse_selected_resid_pipe, handle)
    with (output_dir / "cure_mil_selected_residual_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(mil_selected_resid_pipe, handle)
    with (output_dir / "cure_residual_neighborhood_pipeline.pkl").open("wb") as handle:
        pickle.dump(neighbor_pipe, handle)
    with (output_dir / "cure_residual_neighborhood_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(neighbor_resid_pipe, handle)
    with (output_dir / "cure_residual_neighborhood_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(neighbor_selected_resid_pipe, handle)
    with (output_dir / "cure_target_conditioned_fusion_pipeline.pkl").open("wb") as handle:
        pickle.dump(fusion_pipe, handle)
    with (output_dir / "cure_target_conditioned_fusion_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(fusion_resid_pipe, handle)
    with (output_dir / "cure_target_conditioned_fusion_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(fusion_selected_resid_pipe, handle)
    with (output_dir / "cure_residual_stack_pipeline.pkl").open("wb") as handle:
        pickle.dump(stack_pipe, handle)
    with (output_dir / "cure_residual_stack_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(stack_resid_pipe, handle)
    with (output_dir / "cure_residual_stack_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(stack_selected_resid_pipe, handle)
    with (output_dir / "cure_oof_residual_ensemble_meta_pipeline.pkl").open("wb") as handle:
        pickle.dump(ensemble_meta_pipe, handle)
    with (output_dir / "cure_oof_residual_ensemble_embedding_pipeline.pkl").open("wb") as handle:
        pickle.dump(ensemble_aug_pipe, handle)
    with (output_dir / "cure_oof_residual_ensemble_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(ensemble_resid_pipe, handle)
    with (output_dir / "cure_oof_residual_ensemble_selected_corrector_pipeline.pkl").open("wb") as handle:
        pickle.dump(ensemble_selected_resid_pipe, handle)
    with (output_dir / "cure_embedding_artifacts.pkl").open("wb") as handle:
        pickle.dump(cure_meta, handle)
    with (output_dir / "cure_source_aware_sparse_residual_lexical_artifacts.pkl").open("wb") as handle:
        pickle.dump(source_sparse_meta, handle)
    with (output_dir / "cure_mil_embedding_artifacts.pkl").open("wb") as handle:
        pickle.dump(mil_meta, handle)
    with (output_dir / "cure_residual_neighborhood_artifacts.pkl").open("wb") as handle:
        pickle.dump(neighbor_meta, handle)
    with (output_dir / "cure_target_conditioned_fusion_artifacts.pkl").open("wb") as handle:
        pickle.dump(fusion_meta, handle)
    with (output_dir / "cure_oof_residual_ensemble_artifacts.pkl").open("wb") as handle:
        pickle.dump(ensemble_meta, handle)
    if transformer_meta is not None:
        transformer_meta_to_save = {
            key: value for key, value in transformer_meta.items() if key not in {"model", "tokenizer"}
        }
        with (output_dir / "transformer_note_embedding_artifacts.pkl").open("wb") as handle:
            pickle.dump(transformer_meta_to_save, handle)

    report = [
        "# Mayo CURE Residual Embedding Benchmark",
        "",
        "This experiment trains a note embedder on train cases only. The embedder uses residual supervision from the structured-only model during training, but test cases provide only tabular metadata and pre-cutoff notes.",
        "",
        "## Leakage Controls",
        "",
        f"- Note cutoff policy: `{args.note_cutoff_policy}`.",
        f"- Same-day pre-op-looking note types allowed: `{bool(args.include_same_day_preop_note_types)}`.",
        f"- Post-procedure/discharge note types kept: `{bool(args.keep_postop_doc_types)}`.",
        "- By default, post-procedure, discharge, hospital-course, and anesthesia-postprocedure note types are excluded.",
        "- Train and test patients are disjoint.",
        "- Test target and test residual are never used to build test embeddings.",
        "- The residual target for the embedder is based on out-of-fold structured predictions on train cases.",
        "- All `case_actual_*` columns are excluded from tabular features except the target.",
        "",
        "## Data",
        "",
        f"- Train cases: `{len(train_df)}`",
        f"- Test cases: `{len(test_df)}`",
        f"- Train patients: `{train_df[PATIENT_ID_COLUMN].nunique()}`",
        f"- Test patients: `{test_df[PATIENT_ID_COLUMN].nunique()}`",
        f"- Train/test patient overlap: `{len(overlap)}`",
        f"- Temporal cutoff: `{split.cutoff}`",
        f"- Train cases with pre-cutoff notes: `{int((train_df['n_pre_cutoff_notes_used'] > 0).sum())}`",
        f"- Test cases with pre-cutoff notes: `{int((test_df['n_pre_cutoff_notes_used'] > 0).sum())}`",
        f"- Note rows matched by patient: `{bundle_meta.get('note_rows_matched_patient', 0)}`",
        f"- Note rows rejected as post-op/discharge: `{bundle_meta.get('note_rows_rejected_postop_doc_type', 0)}`",
        f"- Note rows attached to cases: `{bundle_meta.get('note_rows_attached_events', 0)}`",
        "",
        "## Results",
        "",
        markdown_table(results_df),
        "",
        "## CURE Embedder",
        "",
        f"- Embedding dimension: `{z_cure_train.shape[1]}`",
        f"- TF-IDF terms: `{cure_meta['n_terms']}`",
        f"- Conditional SVD dimension: `{cure_meta['svd_dim']}`",
        f"- MLP iterations: `{cure_meta['mlp_n_iter']}`",
        f"- Train residual probe MAE on OOF residual: `{cure_meta['train_residual_probe_mae_on_oof_residual']:.3f}`",
        "",
        "## CURE MIL Residual Encoder",
        "",
        f"- Embedding dimension: `{z_mil_train.shape[1]}`",
        f"- Max chunks per case: `{mil_meta['max_chunks']}`",
        f"- Chunk SVD dimension: `{mil_meta['chunk_svd_dim']}`",
        f"- Chunk TF-IDF terms: `{mil_meta['n_terms']}`",
        f"- Backend: `{mil_meta.get('backend', 'torch')}`",
        f"- Best epoch: `{mil_meta.get('best_epoch', 'n/a')}`",
        (
            f"- Train residual-head MAE on OOF residual: `{mil_meta['train_residual_probe_mae_on_oof_residual']:.3f}`"
            if mil_meta.get("train_residual_probe_mae_on_oof_residual") is not None
            else "- Train residual-head MAE on OOF residual: `n/a`"
        ),
        "",
        "## CURE Residual-Neighborhood Embedding",
        "",
        f"- Embedding dimension: `{z_neighbor_train.shape[1]}`",
        f"- Neighbor sizes: `{neighbor_meta['ks']}`",
        f"- Context weight: `{neighbor_meta['context_weight']}`",
        f"- Note weight: `{neighbor_meta['note_weight']}`",
        f"- Residual-world bins: `{neighbor_meta['residual_world_centroid_labels']}`",
        f"- Train residual-neighborhood probe MAE on OOF residual: `{neighbor_meta['train_residual_probe_mae_on_oof_residual']:.3f}`",
        "",
        "## CURE Target-Conditioned Fusion Embedding",
        "",
        f"- Embedding dimension: `{z_fusion_train.shape[1]}`",
        f"- Raw fused dimension: `{fusion_meta['raw_dim']}`",
        f"- Fusion SVD dimension: `{fusion_meta['svd_dim']}`",
        f"- Residual-supervised PLS dimension: `{fusion_meta['pls_dim']}`",
        f"- Train residual-probe MAE on OOF residual: `{fusion_meta['train_residual_probe_mae_on_oof_residual']:.3f}`",
        f"- Probe selection: `{fusion_meta['probe_selection']}`",
        "",
        "## CURE Source-Aware Sparse Residual Embedding",
        "",
        f"- Embedding dimension: `{z_source_sparse_train.shape[1]}`",
        f"- TF-IDF terms: `{source_sparse_meta['n_terms']}`",
        f"- Train residual probe MAE on OOF residual: `{source_sparse_meta['train_residual_probe_mae_on_oof_residual']:.3f}`",
        f"- Calibration mode: `{source_sparse_calibration_mode}`",
        f"- Top source/concept markers: `{source_sparse_meta.get('source_marker_summary_train', [])[:8]}`",
        "",
        "## CURE Residual Stack Embedding",
        "",
        f"- Embedding dimension: `{z_stack_train.shape[1]}`",
        f"- Includes generic note SVD, CURE residual embedding, sparse residual embedding, source-aware sparse embedding, CURE MIL, residual-neighborhood, target-conditioned fusion, and optional transformer vectors.",
        f"- Transformer included: `{bool(z_transformer_train is not None)}`",
        "",
        "## CURE OOF Residual-Ensemble Embedding",
        "",
        f"- Embedding dimension: `{z_ensemble_train.shape[1]}`",
        f"- Base residual-coordinate count: `{ensemble_meta['base_coordinate_count']}`",
        (
            f"- Train meta residual MAE: `{ensemble_meta['train_meta_residual_mae']:.3f}`"
            if ensemble_meta.get("train_meta_residual_mae") is not None
            else "- Train meta residual MAE: `skipped`"
        ),
        f"- Best base residual worlds: `{ensemble_meta['selection_table'][:5]}`",
        "",
        *(
            [
                "## Frozen Transformer Note Embedding",
                "",
                f"- Model: `{transformer_meta['model_name']}`",
                f"- Device: `{transformer_meta['device']}`",
                f"- Max length: `{transformer_meta['max_length']}`",
                f"- Raw hidden size: `{transformer_meta['raw_hidden_size']}`",
                f"- Compressed embedding dimension: `{transformer_meta['embedding_dim']}`",
                f"- Train cases with nonempty notes: `{transformer_meta['train_nonempty_notes']}`",
                f"- Test cases with nonempty notes: `{transformer_meta['test_nonempty_notes']}`",
                "",
            ]
            if transformer_meta is not None
            else []
        ),
        "## Saved Files",
        "",
        "- `benchmark_results.csv`",
        "- `metrics_summary.json`",
        "- `test_predictions_with_note_embeddings.csv`",
        "- `test_cure_embeddings.csv`",
        "- `test_cure_source_aware_sparse_residual_lexical_embeddings.csv`",
        "- `test_cure_mil_embeddings.csv`",
        *(
            ["- `test_transformer_note_embeddings.csv`"]
            if transformer_meta is not None
            else []
        ),
        "- `test_cure_residual_neighborhood_embeddings.csv`",
        "- `test_cure_target_conditioned_fusion_embeddings.csv`",
        "- `test_cure_residual_stack_embeddings.csv`",
        "- `test_cure_oof_residual_ensemble_embeddings.csv`",
        "- `procedure_slice_metrics_structured.csv`",
        "- `procedure_slice_metrics_cure_embedding.csv`",
        "- `procedure_slice_metrics_cure_mil_embedding.csv`",
        "- `cure_embedding_artifacts.pkl`",
        "- `cure_source_aware_sparse_residual_lexical_artifacts.pkl`",
        "- `cure_mil_embedding_artifacts.pkl`",
        "- `cure_residual_neighborhood_artifacts.pkl`",
        "- `cure_target_conditioned_fusion_artifacts.pkl`",
        "- `cure_oof_residual_ensemble_artifacts.pkl`",
    ]
    (output_dir / "cure_residual_embedding_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\nBenchmark results sorted by MAE:")
    print(results_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")
    print(f"Open report: {output_dir / 'cure_residual_embedding_report.md'}")


if __name__ == "__main__":
    main()
