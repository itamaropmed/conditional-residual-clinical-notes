"""Stage A of the note channel: retrieve and rerank the passages that actually
bear on procedural complexity, so the extractor sees evidence instead of the
last twelve notes in date order.

The raw bundles are dominated by imaging reports and unrelated encounters. A
regex over the whole bundle fires on incidental mentions; that is the
documented reason every concept flag showed negative gain lift. Here each
bundle is split into passages, scored against concept queries with BM25 plus a
negation//history-aware adjustment, and compressed to a short evidence brief.

No model downloads: BM25 and the scoring rules are self-contained.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

STAGE = Path("/mnt/user-data/uploads/PycharmProjects/_crcnl_stage")
OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- concepts
# Each concept is a retrieval query, not a match rule. The query pulls
# candidate passages; the graded value is decided later by reading them.
CONCEPTS: dict[str, str] = {
    "redo_prior_ablation":
        "redo repeat ablation prior ablation previous pulmonary vein isolation "
        "recurrence after ablation reconnection prior procedure",
    "vt_substrate":
        "ventricular tachycardia VT storm ICD shocks premature ventricular complex "
        "PVC burden scar substrate epicardial mapping RVOT inducible",
    "af_persistence":
        "persistent atrial fibrillation longstanding permanent refractory antiarrhythmic "
        "failed amiodarone cardioversion left atrial enlargement",
    "device_lead_complexity":
        "lead extraction lead revision generator change CRT biventricular ICD pacemaker "
        "lead fracture venous occlusion upgrade",
    "congenital_anatomy":
        "congenital heart disease Fontan transposition tetralogy single ventricle "
        "atrial septal defect ventricular septal defect corrected anatomy",
    "structural_access":
        "transseptal puncture difficult access femoral vein occlusion IVC filter "
        "epicardial access pericardial adhesions prior sternotomy",
    "anticoagulation_thrombus":
        "anticoagulation warfarin apixaban rivaroxaban heparin bridging thrombus "
        "left atrial appendage TEE clot",
    "cardiac_dysfunction":
        "reduced ejection fraction cardiomyopathy heart failure NYHA class "
        "severe LV dysfunction inotropes decompensated",
    "comorbid_burden":
        "obstructive sleep apnea obesity BMI diabetes chronic kidney disease COPD "
        "morbid obesity dialysis",
    "concurrent_procedures":
        "additional procedure planned combined staged same setting also undergo "
        "plus concomitant",
}

NEG = re.compile(r"\b(no|not|without|denies|negative for|ruled out|free of|absent|"
                 r"unremarkable for|never)\b[^.;]{0,40}$", re.I)
HIST = re.compile(r"\b(history of|h/o|s/p|status post|prior|previous|remote|in 20\d\d)\b", re.I)
HEDGE = re.compile(r"\b(possible|suspected|cannot exclude|may represent|question of|"
                   r"consider|likely|probable)\b", re.I)
NOISE_HDR = re.compile(r"Diagnostic imaging study|Technique:|Contrast:|Post-processing|"
                       r"Date/Time of Exam|Reason For Exam", re.I)

TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")


def passages(bundle: str, target_chars: int = 420) -> list[str]:
    """Split a bundle into overlapping passages at sentence boundaries."""
    chunks = []
    for block in bundle.split("\n\n"):
        block = block.strip()
        if len(block) < 60:
            continue
        sents = re.split(r"(?<=[.;])\s+", block)
        cur = ""
        for s in sents:
            if len(cur) + len(s) > target_chars and cur:
                chunks.append(cur.strip()); cur = s
            else:
                cur += " " + s
        if cur.strip():
            chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 60]


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = max(self.len.mean(), 1.0)
        self.tf = [Counter(d) for d in docs]
        dfc = Counter()
        for d in docs:
            dfc.update(set(d))
        self.idf = {t: np.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in dfc.items()}

    def score(self, q: list[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype="float32")
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rerank_adjust(passage: str, base: float) -> float:
    """Cheap cross-encoder stand-in: penalise the three failure modes a regex
    cannot see -- negation, remote history, and hedged imaging boilerplate."""
    p = base
    if NEG.search(passage):
        p *= 0.25
    if HIST.search(passage):
        p *= 0.70          # real but weaker evidence than an active finding
    if HEDGE.search(passage):
        p *= 0.80
    if NOISE_HDR.search(passage):
        p *= 0.45
    return p


def build_evidence(bundle: str, top_k: int = 6, per_concept: int = 2) -> dict:
    ps = passages(bundle)
    if not ps:
        return {"evidence": "", "hits": {}}
    toks = [TOKEN.findall(p.lower()) for p in ps]
    bm = BM25(toks)
    chosen: dict[int, float] = {}
    hits: dict[str, float] = {}
    for name, q in CONCEPTS.items():
        qs = TOKEN.findall(q.lower())
        sc = bm.score(qs)
        adj = np.array([rerank_adjust(ps[i], sc[i]) for i in range(len(ps))])
        order = np.argsort(-adj)[:per_concept]
        best = float(adj[order[0]]) if len(order) else 0.0
        hits[name] = best
        for i in order:
            if adj[i] > 0:
                chosen[int(i)] = max(chosen.get(int(i), 0.0), float(adj[i]))
    keep = sorted(chosen.items(), key=lambda kv: -kv[1])[:top_k]
    keep = sorted(i for i, _ in keep)
    return {"evidence": "\n---\n".join(ps[i] for i in keep), "hits": hits}


def main():
    b = pd.read_parquet(STAGE / "preop_bundles_sample.parquet")
    recs, hitrows = [], []
    for r in b.itertuples():
        e = build_evidence(r.bundle)
        recs.append({"case_durable_id": r.case_durable_id,
                     "evidence": e["evidence"],
                     "ev_chars": len(e["evidence"]),
                     "n_notes": r.n_notes})
        hitrows.append({"case_durable_id": r.case_durable_id, **e["hits"]})
    ev = pd.DataFrame(recs)
    hs = pd.DataFrame(hitrows)
    ev.to_parquet(OUT / "evidence_sample.parquet", index=False, compression="zstd")
    hs.to_parquet(OUT / "retrieval_scores.parquet", index=False)
    print(f"{len(ev)} cases | evidence median {int(ev.ev_chars.median())} chars "
          f"(bundles were 14,000) | compression {14000/max(ev.ev_chars.median(),1):.1f}x")
    print("\nretrieval score coverage (fraction of cases with any hit):")
    for c in CONCEPTS:
        print(f"  {c:26s} {(hs[c] > 0).mean():.3f}   median {hs[c].median():.2f}")


if __name__ == "__main__":
    main()
