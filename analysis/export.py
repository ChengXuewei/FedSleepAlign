"""Release files (App. B.1), written from the outputs of train.py:
    manifest.csv         database, subject and test half of every recording for the subject splits 42 and 1
    metrics.csv          per run, fold, client and evaluation: accuracy, macro F1, kappa, stage recalls and the
                         confusion matrix (true stage by row, W N1 N2 N3 REM, 25 counts in row-major order)
    selected_rounds.csv  per run, fold and evaluation: the M selected rounds
    recordings.csv       per run, fold and test recording: kept epochs and correctly staged epochs per evaluation
analysis/release_tables.py rebuilds Tab. 1, 2 and 4, the paired tests and the bootstrap from these files alone.
    python analysis/export.py      -> results/release/"""
import csv
import os

import numpy as np

from common import DS, RESULTS, STAGES, confusion, scores, subject_of
from fedsleepalign.data import split_halves

EVAL = ("S", "R", "Q", "S_R", "S_Q", "R_S", "Q_S", "S-subj", "R-subj", "Q-subj")
OUT = f"{RESULTS}/release"
os.makedirs(OUT, exist_ok=True)
stem = lambda n: n[:-4] if n.endswith(".npz") else n

with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "database", "recording", "subject", "half"])
    for split in (42, 1):
        for ds, halves in split_halves(split).items():
            for h, names in enumerate(halves):
                w.writerows([split, ds, stem(n), subject_of(ds, n), h] for n in names)

runs = sorted(f[:-10] for f in os.listdir(f"{RESULTS}/preds") if f.endswith("_fold0.npz"))
files = [open(f"{OUT}/{name}.csv", "w", newline="") for name in ("metrics", "selected_rounds", "recordings")]
met, sel, rec = (csv.writer(f) for f in files)
met.writerow(["run", "fold", "client", "evaluation", "accuracy", "macro_f1", "kappa"]
             + [f"recall_{s}" for s in STAGES] + ["confusion"])
sel.writerow(["run", "fold", "evaluation", "rounds"])
rec.writerow(["run", "fold", "client", "recording", "epochs"] + [f"correct_{e}" for e in EVAL])
for run in runs:
    for k in (0, 1):
        if not os.path.exists(f"{RESULTS}/preds/{run}_fold{k}.npz"):
            continue
        d = np.load(f"{RESULTS}/preds/{run}_fold{k}.npz")
        keys = sorted({name.split("|")[1] for name in d.files} - {"labels", "names", "lengths"}, key=EVAL.index)
        for ds in [ds for ds in DS if f"{ds}|labels" in d.files]:
            y, n = d[f"{ds}|labels"], d[f"{ds}|lengths"]
            idx = np.repeat(np.arange(len(n)), n)                      # test recording of every epoch
            correct = {}
            for e in keys:
                cm = confusion(y, d[f"{ds}|{e}"])
                met.writerow([run, k, ds, e] + [f"{x:.6f}" for x in scores(cm)] + [" ".join(map(str, cm.ravel()))])
                correct[e] = np.bincount(idx, weights=d[f"{ds}|{e}"] == y, minlength=len(n)).astype(int)
            for i, name in enumerate(d[f"{ds}|names"]):
                rec.writerow([run, k, ds, stem(str(name)), n[i]] + [correct[e][i] if e in correct else "" for e in EVAL])
        r = np.load(f"{RESULTS}/rounds/{run}_fold{k}.npz")
        sel.writerows([run, k, key, " ".join(map(str, b))] for key, b in zip(r["keys"], r["best"]))
for f in files:
    f.close()
print("wrote", sorted(os.listdir(OUT)))
