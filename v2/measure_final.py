"""The procedure multi-hot gain, measured on the fully leak-safe v4b config."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error as MAE
import crcnl_data as D
from v3_corrected import clean, predict, paired
from v4_leaksafe import design, STAFFING_RISK, TRAIN_START

raw = D.add_schedule_derived(D.load_tabular(with_lags=True))
raw["surgeon_durable_id"] = raw["surgeon_durable_id"].astype(str)
raw["bmi_missing"] = raw["bmi"].isna().astype("float32")
df, stats = clean(raw)
tr, te, _ = D.temporal_split(df)
y = df[D.TARGET].to_numpy("float64")
tr = tr[df[D.TIME].iloc[tr].to_numpy() >= TRAIN_START]

X = design(df, tr, STAFFING_RISK)
base = [c for c in X.columns if not c.startswith("proc__")]
p_no = predict(X[base], y, tr, te)
p_yes = predict(X, y, tr, te)
m_no, m_yes = float(MAE(y[te], p_no)), float(MAE(y[te], p_yes))
d, lo, hi = paired(np.abs(y[te]-p_no), np.abs(y[te]-p_yes))
resid = np.abs(y[te]-p_yes); q = np.quantile(y[te], [0.01, 0.99])
mid = (y[te] >= q[0]) & (y[te] <= q[1])
out = {"config": "v4b leak-safe", "n_train": int(len(tr)), "n_holdout": int(len(te)),
       "mae_without_multihot": m_no, "mae_final": m_yes,
       "multihot_gain": d, "ci95": [lo, hi],
       "mae_middle_98pct": float(resid[mid].mean()), "mae_tails": float(resid[~mid].mean())}
print(json.dumps(out, indent=2))
json.dump(out, open("out/FINAL_LEAKSAFE.json","w"), indent=2)
