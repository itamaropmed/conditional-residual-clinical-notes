"""Gold labels produced by reading the retrieved evidence briefs.

Case identifiers are salted SHA-256 prefixes, not the real case_durable_id
values, so no clinical description in this file is linkable to a record.
Regenerate the mapping locally with:
    "c"+sha256(("crcnl-v2-gold"+case_durable_id).encode()).hexdigest()[:12]

Schema. Every field is graded, and -1 means the evidence does not settle it.
That -1 is the whole point: a regex flag cannot distinguish "the note says
there is no prior ablation" from "the note never mentions ablation", and the
audit showed those two states pull case duration in opposite directions.

  redo_prior_ablation   0 none stated | 1 one prior | 2 multiple prior
  vt_substrate          0 none | 1 PVC burden only | 2 VT or scar substrate
                        3 VT storm / ICD shocks / epicardial expected
  af_type               0 none | 1 paroxysmal | 2 persistent | 3 longstanding
  device_complexity     0 none | 1 device in situ | 2 implant planned
                        3 extraction, revision or CRT
  congenital            0 none | 1 simple (ASD/VSD/PFO) | 2 complex
  access_difficulty     0 none | 1 possible | 2 likely
  anticoag_thrombus     0 none | 1 on anticoagulation | 2 thrombus/LAA concern
  lv_dysfunction        0 EF >= 55 | 1 EF 40-54 | 2 EF < 40
  comorbid_burden       0 low | 1 moderate | 2 high
  concurrent_procedures 0 single | 1 two planned | 2 three or more / combined
  expected_complexity   1 (routine) .. 5 (major)
"""

FIELDS = [
    "redo_prior_ablation", "vt_substrate", "af_type", "device_complexity",
    "congenital", "access_difficulty", "anticoag_thrombus", "lv_dysfunction",
    "comorbid_burden", "concurrent_procedures", "expected_complexity",
]

# case_durable_id -> labels, in FIELDS order
GOLD: dict[str, list[int]] = {
    # 0  normal EF 63, LA enlargement, possible infiltrative process, no EP history
    "c1a48e324d5f4": [0, 0, -1, 0, 0, 0, -1, 0, 1, -1, 2],
    # 1  post cardiac surgery, AF, severely dilated LA, device leads present, EF 60-65
    "cc74736b12b03": [0, 0, 2, 1, 0, 1, 1, 0, 1, -1, 3],
    # 2  PVC ablation, 35.7% ventricular ectopy burden, OSA on CPAP, EF 60
    "c71dd18aa11a8": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2],
    # 3  SVT / atrial tachycardia, ILR in situ, EF 52, prior esophagectomy
    "caf708860a387": [0, 0, 0, 1, 0, 1, -1, 1, 1, -1, 2],
    # 4  SVT ablation, EF 64, RVSP 54 raised, moderate LA enlargement
    "c65e8f0b7f5ba": [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 2],
    # 5  AF + flutter ablation, tachy-induced CM recovered 59, amiodarone, Xarelto, DM
    "c312525fab301": [0, 0, 2, 0, 0, 0, 1, 0, 2, 1, 3],
    # 6  paediatric/young SVT ablation, otherwise unremarkable
    "c3cbbbb4ebc0e": [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1],
    # 7  paroxysmal AF, CT PV normal anatomy, no LAA thrombus
    "c7e35b4865996": [0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 2],
    # 8  accessory pathway / WPW, prior ablation attempt described, near His
    "c8f194f8936bf": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    # 9  truncus arteriosus, congenital anomaly, morbid obesity, EPS +/- ICD implant
    "c4d2d8a5223c4": [0, 2, -1, 2, 2, 1, -1, -1, 2, 1, 4],
    # 10 AF, PVI/CTI ablation, CHA2DS2VASc 4, HF + CAD + DM, dysautonomia
    "c15eb415896c2": [0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 3],
    # 11 pacemaker in situ since 2013, routine battery check, low PVC burden
    "c6cd97fef631a": [0, 1, 0, 1, 0, 0, -1, -1, 1, -1, 2],
    # 12 echo only: mild RV enlargement, borderline LA; no EP history in evidence
    "cc9d8b81150f5": [-1, -1, -1, -1, 0, -1, -1, 0, -1, -1, 2],
    # 13 persistent symptomatic AF + flutter, cardioversion, flecainide, no LAA thrombus
    "c797127895f83": [0, 0, 2, 0, 0, 0, -1, 0, 0, 1, 3],
    # 14 SVT 10 years, morbid obesity BMI 40-44.9, sleep apnea, prediabetes, HTN
    "cddcffa6f028d": [0, 0, 0, 0, 0, 0, -1, -1, 2, 0, 2],
    # 15 AF, ablations 1/2018 AND 7/2019, RSPV reconnection + mitral line, EF 51-55, Xarelto
    "cecf438b90f2f": [2, 0, 2, 0, 0, 0, 1, 1, 0, 1, 4],
    # 16 PVC ablation 4/2022, PVC burden 20%, mildly reduced EF on Entresto
    "c9c801e2e8e1a": [1, 1, 0, 0, 0, 0, -1, 1, 0, 0, 3],
    # 17 PVI + repeat mitral isthmus 2020, stroke, LAA occlusion device, Coumadin, MV repair, OSA
    "cbecd8dc94ed5": [2, 1, 2, 1, 0, 2, 1, -1, 2, 1, 4],
    # 18 persistent AF, 3 cardioversions, obesity, OSA CPAP, tachy-induced CM EF 44%
    "cff560f47261a": [0, 0, 2, 0, 0, 0, -1, 1, 2, 0, 3],
    # 19 TEE pre-ablation, EF 61, normal LA, sigmoid septum
    "ce1f81239146c": [0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 2],
    # 20 warfarin, congenital pulmonary stenosis s/p replacement 1976 + bioprosthesis 2022
    "c6b30c12a803a": [-1, 0, -1, 0, 2, 2, 1, -1, 2, 1, 4],
    # 21 AF ablation, enlarged LA, device lead in RV, raised LVOT velocities
    "c86c08da3be27": [0, 0, -1, 1, 0, 0, -1, 0, 1, 0, 3],
    # 22 logistics boilerplate only -- the bundle carries no clinical signal
    "cdab1e3952dc2": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
    # 23 PVI ablation, mildly enlarged LA, PVC burden <1%
    "c693e42879a18": [0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 2],
    # 24 redo VT ablation, prior VT ablations, ICD ATP, amiodarone toxicity,
    #    mechanical aortic valve, extensive substrate, complex VT circuits
    "cf652c1d73d39": [2, 3, -1, 1, 0, 2, -1, -1, 2, 0, 5],
    # 25 prior RF PVI with extra carina lesions + empiric CTI, recurrences beyond 3 months
    "c7162e0a6f004": [1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 3],
    # 26 pacemaker interrogation only, device since 2014
    "ce1f6b15c6f76": [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 2],
    # 27 21 y.o. syncope with classic vasovagal prodrome, sinus tachycardia
    "c5ccd7ddcd620": [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1],
    # 28 ICD interrogation, increased septal thickness, LVOT Valsalva gradient 58
    "c1f6d391388dc": [-1, -1, -1, 1, 0, -1, -1, 0, 1, -1, 3],
    # 29 Ebstein anomaly, severely enlarged RV with mod-severe dysfunction, EPS before surgery
    "c3601d9814da3": [0, 0, 0, 0, 2, 1, -1, 0, 1, 1, 4],
    # 30 ischemic CMP, prior VT ablation, CHB s/p CRT-D epicardial leads, battery depletion,
    #    flutter, LAA closure planned, CHADS-VASc 6, stroke/TIA
    "cd5c3f124d86e": [1, 2, 2, 3, 0, 2, 1, 2, 2, 1, 5],
    # 31 HCM, abnormal strain -11%, LVOT gradients
    "ca12da6d03ee6": [-1, -1, -1, -1, 0, -1, -1, 0, 1, -1, 3],
    # 32 VT redo ablation, prior VT ablation 2017, EF 25-30, mod-severe RV dysfunction
    "cb4988ed2ef76": [1, 2, -1, 1, 0, 1, -1, 2, 1, 0, 4],
    # 33 atrial flutter ablation, otherwise boilerplate
    "c1cf5ff2bb32d": [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 2],
    # 34 PVI ablation, CAD s/p PCI with residual disease, ILR, AF, asymmetric LVH
    "cf1de710f8203": [0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3],
    # 35 EF 71, fenestrated aneurysmal atrial septum with three small shunts
    "c59b724492b6b": [0, 0, -1, 0, 1, 0, -1, 0, 0, -1, 2],
    # 36 HCM reverse-curve septum, mid-cavitary obstruction, strain -6%
    "cc7b24262327b": [-1, -1, -1, -1, 0, -1, -1, 0, 1, -1, 3],
    # 37 PVC burden 21%, sleep apnea on CPAP
    "c9469dbf9acd0": [0, 1, -1, 0, 0, 0, -1, -1, 1, 0, 2],
    # 38 AF pre-ablation, mechanical mitral valve 2015, moderate AS, HFpEF, OSA,
    #    atypical flutter, LAA thrombus TEE pending
    "c9d77c191aed7": [0, 0, 2, 0, 0, 2, 2, 1, 2, 1, 4],
    # 39 AF on amiodarone, spontaneous cardioversion, Xarelto continued
    "cbbe99929d6ab": [0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 2],
    # 40 heart transplant 2021, multiple rejection episodes, stroke, CKD, severe TR
    "c2e5b5646e599": [-1, -1, -1, -1, 0, 2, -1, -1, 2, -1, 5],
    # 41 sinus rhythm, PVC <1%, PAC <1%, boilerplate
    "c4ca77c3306e1": [0, 0, -1, 0, 0, 0, -1, -1, 0, 0, 2],
    # 42 ICM EF 15%, anterior MI, OHCA, dual-chamber ICD, permanent AF, suboptimal BiV%
    "c903ce111fe1e": [-1, 1, 3, 3, 0, 1, -1, 2, 2, 0, 5],
    # 43 chronic AF s/p AV node ablation, device at ERI, NSVT, anticoagulated
    "cf75702b80e40": [1, 2, 3, 2, 0, -1, 1, -1, 1, 0, 4],
}
