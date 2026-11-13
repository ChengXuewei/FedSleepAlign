"""Tab. 1, 2 and 4 with the paired tests and the subject bootstrap, rebuilt from the release files alone (manifest.csv,
metrics.csv, recordings.csv), without data, GPU or training. The numbers equal those of analysis/ablation.py.
    python analysis/release_tables.py [RELEASE]        (default: release/ of this repository)"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel

from common import DS, SEEDS, holm, pm, scores

REL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release")
read = lambda name: list(csv.DictReader(open(f"{REL}/{name}.csv")))
SC = ("none", "zscore", "robust")
runs = {f"fedbn_{sc}_s{s}" for sc in SC for s in SEEDS}

cm = defaultdict(int)                                   # (run, client, evaluation) -> confusion matrix over both folds
for r in read("metrics"):
    if r["run"] in runs:
        cm[r["run"], r["client"], r["evaluation"]] += np.array(r["confusion"].split(), int).reshape(5, 5)
subject = {(r["database"], r["recording"]): r["subject"] for r in read("manifest") if r["split"] == "42"}
correct, epochs = defaultdict(int), defaultdict(int)    # per subject over both folds
for r in read("recordings"):
    if r["run"] in runs:
        s = subject[r["client"], r["recording"]]
        epochs[r["run"], r["client"], s] += int(r["epochs"])
        for e in ("S", "Q"):
            correct[r["run"], r["client"], s, e] += int(r[f"correct_{e}"])


def res(sc, key):
    """(seeds, clients + average, [ACC, MF1, kappa]) in %."""
    a = np.array([[100 * scores(cm[f"fedbn_{sc}_s{s}", ds, key])[:3] for ds in DS] for s in SEEDS])
    return np.concatenate([a, a.mean(1, keepdims=True)], 1)


acc = lambda sc, key: res(sc, key)[:, 5, 0]
print("=== Tab. 1: accuracy per client and averages (ACC, MF1, kappa), mean +- SD over seeds (%)")
for key in ("S", "Q", "R"):
    for sc in SC:
        r = res(sc, key)
        print(f"{sc:<7}{key:<2}" + " | ".join(pm(r[:, d, 0]) for d in range(5)) + " || "
              + " | ".join(pm(r[:, 5, m]) for m in range(3)))

print("\n=== Tab. 2: models selected with S or Q (rows) tested with stored or PAR statistics (columns)")
for sc in ("none", "robust"):
    for sel, (a, b) in (("S", ("S", "S_Q")), ("Q", ("Q_S", "Q"))):
        print(f"{sc:<7} selected with {sel}: stored {pm(acc(sc, a))}  PAR {pm(acc(sc, b))}")

print("\n=== Tab. 4: paired t-tests of the average over the clients, Holm over 10 comparisons x 3 metrics, d_z, "
      "subject bootstrap 95% (pointwise)")
C = [(("robust", "S"), ("none", "S")), (("zscore", "S"), ("none", "S")), (("robust", "S"), ("zscore", "S")),
     (("robust", "Q"), ("none", "Q")), (("zscore", "Q"), ("none", "Q")), (("robust", "Q"), ("zscore", "Q")),
     (("none", "Q"), ("none", "S")), (("zscore", "Q"), ("zscore", "S")), (("robust", "Q"), ("robust", "S")),
     (("robust", "Q"), ("none", "S"))]
p = np.array([[ttest_rel(res(*a)[:, 5, m], res(*b)[:, 5, m]).pvalue for m in range(3)] for a, b in C])
adj = holm(p.ravel()).reshape(p.shape)
subj = {ds: sorted({s for (run, c, s) in epochs if c == ds and run == "fedbn_none_s42"}) for ds in DS}
n = {ds: np.array([epochs["fedbn_none_s42", ds, s] for s in subj[ds]]) for ds in DS}
rng = np.random.default_rng(0)                          # the resamples of analysis/ablation.py
boot = {ds: [rng.integers(0, len(n[ds]), len(n[ds])) for _ in range(2000)] for ds in DS}
B = {}
for sc in SC:
    for key in ("S", "Q"):
        c = {ds: np.array([[correct[f"fedbn_{sc}_s{s}", ds, j, key] for s in SEEDS] for j in subj[ds]]) for ds in DS}
        B[sc, key] = 100 * np.mean([[c[ds][i].sum(0).mean() / n[ds][i].sum() for i in boot[ds]] for ds in DS], axis=0)
for (a, b), q in zip(C, adj):
    d, e = res(*a)[:, 5] - res(*b)[:, 5], B[a] - B[b]
    print(f"{'-'.join(a)} vs {'-'.join(b):<10} " + " | ".join(pm(d[:, m]) for m in range(3))
          + f" | Holm p {q[0]:.2g} (MF1 {q[1]:.2g}, kappa {q[2]:.2g}) | d_z {d[:, 0].mean() / d[:, 0].std(ddof=1):.1f}"
          + f" | bootstrap [{np.percentile(e, 2.5):.2f}, {np.percentile(e, 97.5):.2f}]")
