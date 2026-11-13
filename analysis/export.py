"""Release tables (App. B.1): the recording manifest of the subject splits, the metrics of every run, seed, fold, client
and evaluation, and the selected rounds of every run.
    python analysis/export.py      -> results/release/{manifest.csv, metrics.csv, selected_rounds.csv}"""
import csv
import os

import numpy as np

from common import DS, RESULTS, STAGES, confusion, scores, subject_of
from fedsleepalign.data import split_halves

OUT = f"{RESULTS}/release"
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "database", "recording", "subject", "half"])
    for split in (42, 1):
        for ds, halves in split_halves(split).items():
            for h, names in enumerate(halves):
                w.writerows([split, ds, n[:-4], subject_of(ds, n), h] for n in names)
runs = sorted(f[:-10] for f in os.listdir(f"{RESULTS}/preds") if f.endswith("_fold0.npz"))
with open(f"{OUT}/metrics.csv", "w", newline="") as f, open(f"{OUT}/selected_rounds.csv", "w", newline="") as g:
    w, v = csv.writer(f), csv.writer(g)
    w.writerow(["run", "fold", "client", "evaluation", "accuracy", "macro_f1", "kappa"] + [f"recall_{s}" for s in STAGES])
    v.writerow(["run", "fold", "evaluation", "rounds"])
    for run in runs:
        for k in (0, 1):
            if not os.path.exists(f"{RESULTS}/preds/{run}_fold{k}.npz"):
                continue
            d = np.load(f"{RESULTS}/preds/{run}_fold{k}.npz")
            keys = sorted({key.split("|")[1] for key in d.files} - {"labels", "names", "lengths"})
            for ds in [ds for ds in DS if f"{ds}|labels" in d.files]:
                for key in keys:
                    w.writerow([run, k, ds, key] + [f"{x:.6f}" for x in scores(confusion(d[f"{ds}|labels"], d[f"{ds}|{key}"]))])
            r = np.load(f"{RESULTS}/rounds/{run}_fold{k}.npz")
            v.writerows([run, k, key, " ".join(map(str, b))] for key, b in zip(r["keys"], r["best"]))
print("wrote", sorted(os.listdir(OUT)))
