"""Stage B: the extractor.

Graded, negation-aware, history-aware rules over the retrieved evidence. This
replaces the binary regex concept flags, which the audit found had negative
gain lift on every single flag.

Three things the regex layer could not do, and this does:
  1. Grade. "PVC burden 21%" and "PVC burden <1%" both matched the old
     \\bpvc pattern and became the same 1.0.
  2. Negate. "No evidence of thrombus" fired the thrombus flag.
  3. Separate absent from unmentioned. -1 is a distinct state, and the
     residual model gets an explicit seen-mask so it can learn the difference.

Rules were written against 44 hand-labelled cases and are scored against them
in eval_extractor.py. Nothing here is fitted to the outcome.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from gold_labels import FIELDS

NEG_NEAR = r"(?:no|not|without|denies|negative for|ruled out|free of|absent)\s+(?:\w+\s+){0,4}"


def _has(text: str, pat: str) -> bool:
    return re.search(pat, text, re.I) is not None


def _neg(text: str, pat: str) -> bool:
    """True when the concept appears only under negation."""
    if not _has(text, pat):
        return False
    pos = [m.start() for m in re.finditer(pat, text, re.I)]
    for p in pos:
        window = text[max(0, p - 60):p]
        if not re.search(NEG_NEAR + r"$", window, re.I):
            return False        # at least one unnegated mention
    return True


def _count(text: str, pat: str) -> int:
    return len(re.findall(pat, text, re.I))


def extract(ev: str) -> dict[str, int]:
    t = ev or ""
    if len(t.strip()) < 80:
        return {f: -1 for f in FIELDS} | {"expected_complexity": 2}
    out: dict[str, int] = {}

    # ---- redo / prior ablation ------------------------------------------
    abl_prior = _count(t, r"\b(?:redo|repeat)\s+\w*\s*ablation|prior ablation|"
                          r"previous ablation|ablation (?:in|on) \d|s/p[^.]{0,30}ablation|"
                          r"underwent[^.]{0,40}ablation|ablation \d{1,2}/\d")
    if _has(t, r"ablation history:\s*no|no prior ablation|cardioversion or ablation history:\s*no"):
        out["redo_prior_ablation"] = 0
    elif abl_prior >= 2:
        out["redo_prior_ablation"] = 2
    elif abl_prior == 1:
        out["redo_prior_ablation"] = 1
    elif _has(t, r"ablation"):
        out["redo_prior_ablation"] = 0
    else:
        out["redo_prior_ablation"] = -1

    # ---- VT substrate ----------------------------------------------------
    burst = _has(t, r"vt storm|icd shock|appropriate shock|multiple shocks|"
                    r"hemodynamically unstable ventricular|extensive substrate|"
                    r"complex vt circuit|epicardial")
    vt = _has(t, r"\bventricular tachycardia\b|\bVT\b|\bNSVT\b|VT-NS|scar substrate|"
                 r"substrate mapping|inducible ventricular")
    m = re.search(r"pvc burden[^0-9%]{0,12}(\d{1,2}(?:\.\d)?)\s*%", t, re.I)
    pvc_pct = float(m.group(1)) if m else None
    if burst:
        out["vt_substrate"] = 3
    elif vt:
        out["vt_substrate"] = 2
    elif pvc_pct is not None and pvc_pct >= 5:
        out["vt_substrate"] = 1
    elif _has(t, r"\bPVC\b|premature ventricular|ventricular ectopy"):
        out["vt_substrate"] = 1 if (pvc_pct is None or pvc_pct >= 5) else 0
    else:
        out["vt_substrate"] = -1 if not _has(t, r"ectopy|ventricular") else 0

    # ---- AF type ---------------------------------------------------------
    if _has(t, r"longstanding persistent|long-standing persistent|permanent atrial fib|"
               r"chronic atrial fib|permanent\b[^.]{0,20}fibrillation"):
        out["af_type"] = 3
    elif _has(t, r"persistent atrial (?:fibrillation|fib)|persistent symptomatic atrial"):
        out["af_type"] = 2
    elif _has(t, r"paroxysmal atrial (?:fibrillation|fib)|\bPAF\b"):
        out["af_type"] = 1
    elif _has(t, r"atrial fibrillation|\bafib\b|\bAF\b"):
        out["af_type"] = 1
    else:
        out["af_type"] = -1

    # ---- device complexity ----------------------------------------------
    if _has(t, r"lead extraction|lead revision|lead fracture|generator change|"
               r"\bCRT\b|biventricular|epicardial lead|battery.{0,20}(?:ERI|depletion)|"
               r"upgrade"):
        out["device_complexity"] = 3
    elif _has(t, r"ICD implant|implant planned|\+/- ICD|pacemaker implant|device implant"):
        out["device_complexity"] = 2
    elif _has(t, r"device lead|implantable pulse generator|\bICD\b|pacemaker|"
                 r"\bILR\b|loop recorder|interrogation"):
        out["device_complexity"] = 1
    else:
        out["device_complexity"] = -1 if not _has(t, r"device") else 0

    # ---- congenital ------------------------------------------------------
    if _has(t, r"fontan|transposition|tetralogy|single ventricle|truncus|ebstein|"
               r"congenital pulmonary valve stenosis|congenital heart|"
               r"anomaly heart congenital"):
        out["congenital"] = 2
    elif _has(t, r"fenestrated atrial septum|atrial septal defect|\bASD\b|\bVSD\b|\bPFO\b|"
                 r"aneurysmal atrial septum"):
        out["congenital"] = 1 if not _neg(t, r"atrial septal defect|\bASD\b|\bVSD\b") else 0
    else:
        out["congenital"] = 0

    # ---- access difficulty ----------------------------------------------
    if _has(t, r"epicardial access|pericardial adhesion|prior sternotomy|"
               r"venous occlusion|IVC filter|mechanical (?:aortic|mitral) valve|"
               r"mechanical mitral valve|transplant|esophagectomy|"
               r"valve replacement|bioprosthesis|prosthetic valve"):
        out["access_difficulty"] = 2
    elif _has(t, r"transseptal|difficult access|prior cardiac surgery|"
                 r"cardiac surgery|valve repair|\bPCI\b|\bCABG\b"):
        out["access_difficulty"] = 1
    else:
        out["access_difficulty"] = 0

    # ---- anticoagulation / thrombus --------------------------------------
    thr = _has(t, r"thrombus|clot in|LAA thrombus")
    thr_neg = _neg(t, r"thrombus")
    if thr and not thr_neg:
        out["anticoag_thrombus"] = 2
    elif _has(t, r"warfarin|coumadin|apixaban|eliquis|rivaroxaban|xarelto|"
                 r"anticoagulat|heparin|\bDOAC\b"):
        out["anticoag_thrombus"] = 1
    elif thr_neg:
        out["anticoag_thrombus"] = 0
    else:
        out["anticoag_thrombus"] = -1

    # ---- LV dysfunction --------------------------------------------------
    efs = [float(x) for x in re.findall(r"ejection fraction[^0-9]{0,18}(\d{2})\s*%", t, re.I)]
    efs += [float(x) for x in re.findall(r"\bEF[^0-9a-z]{0,8}(\d{2})\s*[%-]", t, re.I)]
    if efs:
        e = min(efs)
        out["lv_dysfunction"] = 2 if e < 40 else (1 if e < 55 else 0)
    elif _has(t, r"severely reduced (?:left )?ventricular|EF 1\d|EF 2\d|"
                 r"severe (?:LV|left ventricular) (?:systolic )?dysfunction"):
        out["lv_dysfunction"] = 2
    elif _has(t, r"mildly reduced|moderately reduced|cardiomyopathy|heart failure"):
        out["lv_dysfunction"] = 1
    elif _has(t, r"normal left ventricular systolic function|normal (?:LV|left ventricular) function"):
        out["lv_dysfunction"] = 0
    else:
        out["lv_dysfunction"] = -1

    # ---- comorbid burden -------------------------------------------------
    hits = sum(bool(_has(t, p)) for p in [
        r"obstructive sleep apnea|\bOSA\b|\bCPAP\b|\bBiPAP\b",
        r"morbid obesity|obesity|BMI 4\d",
        r"diabetes|\bDM\b|prediabetes",
        r"chronic kidney|\bCKD\b|dialysis|renal failure",
        r"\bCOPD\b|pulmonary disease",
        r"hypertension|\bHTN\b",
        r"stroke|\bTIA\b",
        r"peripheral vascular|coronary artery disease|\bCAD\b",
        r"transplant|rejection|cancer|malignan",
    ])
    out["comorbid_burden"] = 2 if hits >= 4 else (1 if hits >= 2 else 0)

    # ---- concurrent procedures ------------------------------------------
    combo = sum(bool(_has(t, p)) for p in [
        r"\bPVI\b|pulmonary vein isolation", r"\bCTI\b|cavotricuspid|flutter ablation",
        r"mitral (?:line|isthmus)", r"LAA (?:closure|occlusion)",
        r"\bAV node ablation\b", r"\+/-|and possible|combined|staged|concomitant",
    ])
    out["concurrent_procedures"] = 2 if combo >= 3 else (1 if combo >= 1 else 0)

    # ---- overall -----------------------------------------------------
    graded = [out[f] for f in FIELDS[:-1] if out[f] > 0]
    sev = sum(graded)
    out["expected_complexity"] = int(np.clip(1 + round(sev / 3.0), 1, 5))
    return out


def extract_frame(ev: pd.DataFrame) -> pd.DataFrame:
    rows = [{"case_durable_id": r.case_durable_id, **extract(r.evidence)}
            for r in ev.itertuples()]
    return pd.DataFrame(rows)
