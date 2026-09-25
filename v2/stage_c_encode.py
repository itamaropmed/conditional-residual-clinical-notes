"""Stage C: clinical encoding with Bio_ClinicalBERT.

No regex in the feature path. The note channel becomes two things:

  1. A pooled document vector. Each case's retrieved evidence is split into
     chunks, encoded, and mean/max pooled -- the encoder's own reading of the
     text, reduced by SVD fitted on training rows only.

  2. Similarity to clinical concept anchors. For each of twelve duration-
     relevant concepts I wrote canonical sentences in the register these notes
     actually use -- having read the corpus first, so "PVC burden of 21% on
     Holter" rather than a keyword list. Those anchors are encoded once, and
     each case gets max and mean cosine similarity of its chunks to each
     anchor set. That is semantic matching: a note saying "frequent ventricular
     ectopy in bigeminy" scores on the PVC anchor without sharing a token
     with it.

The anchors are authored from domain knowledge, never fitted to the outcome,
so they carry no leakage. Everything downstream of them (SVD, scaler) is fit
on training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MODEL = "/home/claude/bcb"
EV = Path("/mnt/user-data/uploads/PycharmProjects/_crcnl_ev/evidence_all.parquet")
OUT = Path("/home/claude/crcnl2/out"); OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)

# Twelve concepts that plausibly move operative time in an EP lab, each with
# sentences phrased the way the notes phrase them.
ANCHORS: dict[str, list[str]] = {
    "redo_ablation": [
        "He had an ablation in 2018 after which he had recurrent arrhythmias leading to a second ablation.",
        "Prior pulmonary vein isolation with rate-dependent reconnection of the right superior pulmonary vein.",
        "This is a redo procedure for recurrent atrial fibrillation after previous catheter ablation.",
    ],
    "vt_substrate": [
        "Extensive substrate and complex ventricular tachycardia circuits were found on prior mapping.",
        "He received appropriate ICD shocks for sustained ventricular tachycardia.",
        "Two different hemodynamically unstable ventricular tachycardias were induced.",
    ],
    "pvc_burden": [
        "Holter showed a PVC burden of 21 percent with frequent ventricular ectopy in bigeminy and trigeminy.",
        "Frequent ventricular ectopy consisting of 35 percent of the recorded period.",
    ],
    "af_persistent": [
        "She is a 67-year-old female with symptomatic persistent atrial fibrillation.",
        "He has been in atrial fibrillation for over two years and it is considered permanent.",
        "Longstanding persistent atrial fibrillation despite amiodarone and multiple cardioversions.",
    ],
    "device_complex": [
        "History of complete heart block status post CRT-D with epicardial leads and early battery depletion.",
        "Planned lead extraction for lead fracture with venous occlusion.",
        "Generator change with upgrade to a biventricular device.",
    ],
    "congenital": [
        "Echocardiogram shows anatomy that is classic for Ebstein anomaly with a diminutive septal leaflet.",
        "Congenital pulmonary valve stenosis status post replacement and repeat bioprosthesis.",
        "Adult congenital heart disease with single ventricle physiology after Fontan palliation.",
    ],
    "access_difficulty": [
        "Prior sternotomy with pericardial adhesions is expected to complicate epicardial access.",
        "Complex ventricular tachycardia circuits in the proximity of his mechanical aortic valve.",
        "Femoral venous occlusion noted on imaging; alternative access will be required.",
    ],
    "thrombus_anticoag": [
        "The left atrial appendage thrombus TEE is pending before the procedure.",
        "She continues on warfarin with an INR goal of 2.0 to 3.0 through the procedure.",
    ],
    "lv_dysfunction": [
        "Severely enlarged left ventricular chamber size with calculated ejection fraction of 15 percent.",
        "Ischemic cardiomyopathy with an ejection fraction of 25 to 30 percent and moderate right ventricular dysfunction.",
    ],
    "comorbid_burden": [
        "Morbid obesity with body mass index of 42, obstructive sleep apnea on CPAP, and type 2 diabetes.",
        "Chronic kidney disease, peripheral vascular disease, prior stroke and hyperlipidemia.",
        "Status post heart transplant with multiple episodes of acute cellular rejection.",
    ],
    "concurrent_procedures": [
        "The plan is for pulmonary vein isolation with an empiric cavotricuspid isthmus line in the same setting.",
        "Electrophysiology study with possible ablation and ICD implant at the same sitting.",
        "Combined left atrial appendage closure and atrial fibrillation ablation.",
    ],
    "routine_simple": [
        "She is a pleasant 55-year-old woman presenting for a straightforward supraventricular tachycardia ablation.",
        "Routine device battery check; all device function appears normal.",
        "Normal left ventricular systolic function with no regional wall motion abnormalities.",
    ],
}


def chunks(text: str, tok, max_tokens: int = 256, max_chunks: int = 2) -> list[str]:
    parts = [p.strip() for p in text.split("\n---\n") if len(p.strip()) > 40]
    return parts[:max_chunks] if parts else []


@torch.no_grad()
def encode(texts: list[str], tok, model, bs: int = 16, max_len: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", truncation=True,
                max_length=max_len, padding=True)
        h = model(**b).last_hidden_state              # (B, T, 768)
        mask = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean over tokens
        out.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
    return np.vstack(out).astype("float32")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    names = list(ANCHORS)
    flat, owner = [], []
    for i, n in enumerate(names):
        for s in ANCHORS[n]:
            flat.append(s); owner.append(i)
    A = encode(flat, tok, model)
    owner = np.array(owner)
    np.save(OUT / "anchor_vecs.npy", A)
    print(f"anchors: {len(flat)} sentences over {len(names)} concepts")

    ev = pd.read_parquet(EV)
    ev["case_durable_id"] = ev.case_durable_id.astype(str)
    print(f"cases: {len(ev)}")

    doc_mean = np.zeros((len(ev), 768), dtype="float32")
    doc_max = np.zeros((len(ev), 768), dtype="float32")
    sim_max = np.zeros((len(ev), len(names)), dtype="float32")
    sim_mean = np.zeros((len(ev), len(names)), dtype="float32")
    nchunk = np.zeros(len(ev), dtype="float32")

    import time
    t0 = time.time()

    # Flatten every (case, chunk) pair and encode in length-sorted batches.
    # Encoding one case at a time spends most of each batch on padding; sorting
    # by length and batching across cases cuts the work several-fold.
    evidence = ev.evidence.tolist()
    owners_row, texts = [], []
    for r in range(len(ev)):
        for c in chunks(evidence[r], tok):
            owners_row.append(r); texts.append(c)
    owners_row = np.array(owners_row)
    print(f"chunks: {len(texts)} across {len(ev)} cases", flush=True)

    order = np.argsort([len(x) for x in texts])
    BS = 32
    nb = (len(order) + BS - 1) // BS

    # Checkpoint every 50 batches. This container suspends compute between
    # turns, so a long job has to be able to resume where it stopped.
    CK, CKI = OUT / "E_all.npy", OUT / "E_done.npy"
    if CK.exists() and CKI.exists():
        E_all = np.load(CK)
        done_to = int(np.load(CKI)[0])
        print(f"resuming at batch {done_to}/{nb}", flush=True)
    else:
        E_all = np.zeros((len(texts), 768), dtype="float32")
        done_to = 0

    for k, bi in enumerate(range(0, len(order), BS)):
        if k < done_to:
            continue
        sel = order[bi:bi + BS]
        E_all[sel] = encode([texts[i][:900] for i in sel], tok, model, bs=BS)
        if k and k % 50 == 0:
            np.save(CK, E_all); np.save(CKI, np.array([k]))
            el = time.time() - t0
            rate = el / max(k - done_to, 1)
            print(f"  batch {k}/{nb}  {el:.0f}s  eta {rate*(nb-k)/60:.0f}m", flush=True)
    np.save(CK, E_all); np.save(CKI, np.array([nb]))

    S_all = E_all @ A.T
    for r in range(len(ev)):
        m = owners_row == r
        if not m.any():
            continue
        E = E_all[m]
        doc_mean[r] = E.mean(0); doc_max[r] = E.max(0); nchunk[r] = int(m.sum())
        S = S_all[m]
        for i in range(len(names)):
            col = S[:, owner == i]
            sim_max[r, i] = col.max(); sim_mean[r, i] = col.mean()

    np.save(OUT / "doc_mean.npy", doc_mean)
    np.save(OUT / "doc_max.npy", doc_max)
    out = pd.DataFrame({"case_durable_id": ev.case_durable_id.to_numpy(),
                        "note_n_chunks": nchunk,
                        "note_n_notes": ev.n_notes.to_numpy(dtype="float32"),
                        "note_ev_chars": ev.ev_chars.to_numpy(dtype="float32")})
    for i, n in enumerate(names):
        out[f"sim_max__{n}"] = sim_max[:, i]
        out[f"sim_mean__{n}"] = sim_mean[:, i]
    out.to_parquet(OUT / "note_features.parquet", index=False)
    print(f"wrote note_features.parquet  {out.shape}  in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
