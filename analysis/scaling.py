"""Sec. 3.3 and App. B.2: where and why PRS exceeds the per-recording z-score.
  - margin of PRS over the z-score per quarter of tail weight (Tab. 1 runs, five seeds), clipping per quarter
  - PRS with one operation changed at a time, seed 42 (train.py --scheme fedbn --scaling mean_iqr|med_sd|noclip
    --seed 42, both folds), next to PRS, the z-score and no scaling
  - artifact injection (analysis/artifact.py), if its results exist
Needs results/summary/recordings.npz (analysis/recordings.py). Writes results/summary/scaling.npz.
    python analysis/scaling.py > results/summary/scaling.log"""
import os

import numpy as np
from scipy.stats import wilcoxon

from common import DS, RESULTS, SEEDS, SUMMARY, preds, table

R = np.load(f"{SUMMARY}/recordings.npz")
row = {(d, str(n)): i for i, (d, n) in enumerate(zip(R["ds"], R["name"]))}


def per_recording(run, key):
    """{(ds, recording): accuracy in %} over both folds of a run."""
    out = {}
    for k in (0, 1):
        f = preds(run, k)
        for ds in DS:
            ok = np.split(f[f"{ds}|{key}"] == f[f"{ds}|labels"], np.cumsum(f[f"{ds}|lengths"])[:-1])
            out.update({(ds, str(n)): 100 * o.mean() for n, o in zip(f[f"{ds}|names"], ok)})
    return out


def quarters(recs, keep):
    client, tail = np.array([r[0] for r in recs]), R["tail"][[row[r] for r in recs]]
    qs = np.full(len(recs), -1)
    for ds in DS:
        m = (client == ds) & keep
        qs[m] = np.searchsorted(np.percentile(tail[m], [25, 50, 75]), tail[m])
    return qs


out = {}
print("=== PRS - z-score per quarter of tail weight (lightest to heaviest within a client), mean over seeds")
recs = sorted(per_recording("fedbn_robust_s42", "Q"))
low = np.isin([r[1] for r in recs], R["isruc_low"]) & np.array([r[0] == "ISRUC" for r in recs])
qs, qs_wo = quarters(recs, np.ones(len(recs), bool)), quarters(recs, ~low)
st = R["stats"][[row[r] for r in recs]]
n_ep = R["epochs"][[row[r] for r in recs]]
print("samples clipped by PRS per quarter " + " ".join(
    f"{100 * np.average(st[qs == j, :, 4].mean(1), weights=n_ep[qs == j]):.3f}%" for j in range(4)))
for key in ("S", "Q"):
    acc = {sc: np.mean([[per_recording(f"fedbn_{sc}_s{s}", key)[r] for r in recs] for s in SEEDS], 0)
           for sc in ("robust", "zscore")}
    d = acc["robust"] - acc["zscore"]
    print(f"{key}: " + " ".join(f"{d[qs == j].mean():+.2f}" for j in range(4)) + " | without the ISRUC low range "
          + " ".join(f"{d[qs_wo == j].mean():+.2f}" for j in range(4))
          + f" | heaviest quarter positive {np.mean(d[qs == 3] > 0):.0%}, Wilcoxon p {wilcoxon(d[qs == 3]).pvalue:.2g}")

print("\n=== One operation of PRS changed at a time, seed 42, both folds: ACC MF1 kappa with stored / PAR statistics")
SC = ("none", "robust", "mean_iqr", "med_sd", "noclip", "zscore")
avg = {}
for sc in SC:
    for key in ("S", "Q"):
        avg[sc, key] = table(f"fedbn_{sc}_s42", key)[:, :3].mean(0)
        out[f"avg|{sc}|{key}"] = avg[sc, key]
    print(f"{sc:<9} " + " | ".join(" ".join(f"{x:.2f}" for x in avg[sc, key]) for key in ("S", "Q")))
for key in ("S", "Q"):
    a = {sc: avg[sc, key][0] for sc in SC}
    print(f"{key}: PRS - z-score {a['robust'] - a['zscore']:+.2f} | IQR in place of SD "
          f"{((a['robust'] - a['med_sd']) + (a['mean_iqr'] - a['zscore'])) / 2:+.2f} | median in place of mean "
          f"{((a['robust'] - a['mean_iqr']) + (a['med_sd'] - a['zscore'])) / 2:+.2f} | clipping {a['robust'] - a['noclip']:+.2f}")
    s42 = {sc: per_recording(f"fedbn_{sc}_s42", key) for sc in SC[1:]}
    for sc in SC[1:-1]:
        d = np.array([s42[sc][r] - s42["zscore"][r] for r in recs])
        out[f"tail|{sc}|{key}"] = np.array([d[qs == j].mean() for j in range(4)])
        print(f"   {sc:<9} - z-score per quarter " + " ".join(f"{x:+.2f}" for x in out[f"tail|{sc}|{key}"]))

if all(os.path.exists(f"{SUMMARY}/artifact_{sc}.npz") for sc in SC[1:]):
    print("\n=== Artifacts in 0, 2 and 5% of the epochs of the fold-0 test recordings: accuracy on the clean epochs (PAR)")
    for sc in SC[1:]:
        a = np.load(f"{SUMMARY}/artifact_{sc}.npz")
        acc = 100 * a["correct_clean"] / a["n_clean"]                         # (rho, clients)
        out[f"art|{sc}"] = acc.mean(1)
        print(f"{sc:<9} " + " | ".join(" ".join(f"{v:5.1f}" for v in c) + f" -> {c.mean():5.2f}" for c in acc)
              + " | change " + " ".join(f"{acc.mean(1)[i] - acc.mean(1)[0]:+.2f}" for i in (1, 2)))
    a = np.load(f"{SUMMARY}/artifact_robust.npz")
    print("growth of a channel (median over recordings and channels): SD x "
          + " ".join(f"{x:.2f}" for x in a["sd_growth"]) + ", IQR x " + " ".join(f"{x:.3f}" for x in a["iqr_growth"]))
np.savez(f"{SUMMARY}/scaling.npz", **out)
