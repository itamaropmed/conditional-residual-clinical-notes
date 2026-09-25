#!/usr/bin/env python3
"""Extract structured clinical fields from pre-op note evidence using Claude
Code running locally, in headless mode. No API key needed -- `claude -p` uses
the subscription already signed in on this machine.

    python claude_code_extract.py --evidence evidence_all.parquet \
        --out llm_features_v2.parquet --batch 8

Design notes
------------
* Batched. One `claude -p` call carries several cases, because per-call
  overhead dominates at this volume. 8 is a reasonable default; raise it until
  the JSON starts coming back malformed.
* Resumable. Every completed case is appended to a JSONL sidecar and skipped on
  restart, so rate limits or a closed laptop cost you nothing.
* Strict schema. The model is told to return JSON only, one object per case,
  and anything unparseable is retried once at batch size 1 before being
  recorded as a failure.
* Graded, with an explicit not-mentioned state. `-1` means the evidence does
  not settle the field. That distinction is the whole reason to use a language
  model rather than pattern matching, and it is the thing the old regex layer
  could not represent.

Input is the retrieved evidence brief (~2,600 chars/case), not the raw bundle
(~14,000). That compression is 5.4x fewer tokens for the same clinical content.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

FIELDS = {
    "redo_prior_ablation": "0 none stated | 1 one prior | 2 multiple prior",
    "vt_substrate": "0 none | 1 PVC burden only | 2 VT or scar substrate | 3 VT storm, ICD shocks or epicardial expected",
    "af_type": "0 none | 1 paroxysmal | 2 persistent | 3 longstanding or permanent",
    "device_complexity": "0 none | 1 device in situ | 2 implant planned | 3 extraction, revision or CRT",
    "congenital": "0 none | 1 simple (ASD/VSD/PFO) | 2 complex",
    "access_difficulty": "0 none | 1 possible | 2 likely",
    "anticoag_thrombus": "0 none | 1 on anticoagulation | 2 thrombus or LAA concern",
    "lv_dysfunction": "0 EF >= 55 | 1 EF 40-54 | 2 EF < 40",
    "comorbid_burden": "0 low | 1 moderate | 2 high",
    "concurrent_procedures": "0 single | 1 two planned | 2 three or more / combined",
    "expected_complexity": "1 routine .. 5 major",
}

SYSTEM = """You are extracting structured features from pre-operative clinical \
notes for an electrophysiology lab, to predict how long each case will take.

Rules:
- Return ONLY a JSON array. No prose, no markdown fences.
- One object per case, in the order given, each with "case_id" and every field.
- Use -1 when the evidence does not settle a field. Do not guess. "-1" and "0" \
are different: 0 means the notes indicate absence, -1 means the notes are silent.
- Respect negation. "No evidence of thrombus" is 0, not 2.
- Respect recency. A remote history is weaker evidence than an active finding.
- Grade on severity, not mere mention. A PVC burden of 21% and one of <1% are \
not the same.

Fields and their scales:
""" + "\n".join(f"- {k}: {v}" for k, v in FIELDS.items())


def build_prompt(batch: list[tuple[str, str]]) -> str:
    parts = [SYSTEM, "\nCases:\n"]
    for cid, ev in batch:
        parts.append(f'\n=== case_id: {cid} ===\n{ev[:6000]}\n')
    parts.append(f"\nReturn a JSON array of exactly {len(batch)} objects.")
    return "".join(parts)


def call_claude(prompt: str, model: str, timeout: int = 300) -> str:
    """Headless Claude Code. --print returns the response and exits."""
    cmd = ["claude", "--print", "--model", model]
    r = subprocess.run(cmd, input=prompt, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[:400]}")
    return r.stdout


def parse(out: str) -> list[dict]:
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        raise ValueError("no JSON array in response")
    rows = json.loads(m.group(0))
    clean = []
    for r in rows:
        if "case_id" not in r:
            continue
        rec = {"case_durable_id": str(r["case_id"])}
        for f in FIELDS:
            v = r.get(f, -1)
            try:
                rec[f] = int(v)
            except (TypeError, ValueError):
                rec[f] = -1
        clean.append(rec)
    return clean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True,
                    help="parquet with case_durable_id and evidence columns")
    ap.add_argument("--out", default="llm_features_v2.parquet")
    ap.add_argument("--sidecar", default="llm_features_v2.jsonl")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="a smaller model is the right call here: the schema is "
                         "rigid and the volume is high")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cases (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls, to stay under rate limits")
    a = ap.parse_args()

    ev = pd.read_parquet(a.evidence)
    ev["case_durable_id"] = ev.case_durable_id.astype(str)
    ev = ev[ev.evidence.str.len() > 80]

    side = Path(a.sidecar)
    done: set[str] = set()
    if side.exists():
        for line in side.open():
            try:
                done.add(json.loads(line)["case_durable_id"])
            except Exception:
                pass
        print(f"resuming: {len(done)} cases already extracted")

    todo = [(r.case_durable_id, r.evidence) for r in ev.itertuples()
            if r.case_durable_id not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} cases to extract, batch size {a.batch}, model {a.model}")

    fh = side.open("a")
    ok = fail = 0
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        try:
            rows = parse(call_claude(build_prompt(batch), a.model))
        except Exception as e:
            print(f"  batch at {i} failed ({e}); retrying one at a time", flush=True)
            rows = []
            for one in batch:
                try:
                    rows += parse(call_claude(build_prompt([one]), a.model))
                except Exception as e2:
                    print(f"    case {one[0]} failed: {e2}", flush=True)
                    fail += 1
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        ok += len(rows)
        if (i // a.batch) % 10 == 0:
            el = time.time() - t0
            rate = ok / max(el, 1e-9)
            print(f"  {ok} done, {fail} failed, {el/60:.1f}m elapsed, "
                  f"eta {(len(todo)-ok)/max(rate,1e-9)/60:.0f}m", flush=True)
        if a.sleep:
            time.sleep(a.sleep)
    fh.close()

    rows = [json.loads(l) for l in side.open()]
    out = pd.DataFrame(rows).drop_duplicates("case_durable_id")
    for f in FIELDS:
        out[f"{f}__seen"] = (out[f] >= 0).astype("int8")
    out.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}  {out.shape}   ({ok} extracted, {fail} failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
