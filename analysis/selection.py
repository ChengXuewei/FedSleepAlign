"""Sec. 3.3 and App. B.7: model selection on held-out subjects and the variability over test subjects.
  - new subject split (train.py --scheme fedbn --scaling none|robust --seed 42 --split 1 --val dual, both folds): the
    top-M rounds selected on held-out windows or on held-out subjects
  - the five seeds of Tab. 1: SD over seeds, subject bootstrap, folds, and the gain of FedSleepAlign over FedBN per
    test subject (Fig. 6)
Writes results/summary/selection.npz.     python analysis/selection.py > results/summary/selection.log"""
import numpy as np

from common import DS, SEEDS, SUMMARY, subjects, table

VARIANTS = {"FedBN": ("none", "S"), "FedBN + PRS": ("robust", "S"), "FedBN + PAR": ("none", "Q"),
            "FedSleepAlign": ("robust", "Q")}
out = {}
print("=== New subject split, seed 42: average accuracy with the rounds selected on windows / on subjects (%)")
for v, (sc, w) in VARIANTS.items():
    win, subj = table(f"fedbn_{sc}_s42_p1_dual", w)[:, 0], table(f"fedbn_{sc}_s42_p1_dual", f"{w}-subj")[:, 0]
    orig = table(f"fedbn_{sc}_s42", w)[:, 0]
    out[f"orig|{v}"], out[f"win|{v}"], out[f"subj|{v}"] = orig, win, subj
    print(f"{v:<14} original split {orig.mean():.2f} | new split windows {win.mean():.2f}, subjects {subj.mean():.2f} "
          f"(difference {subj.mean() - win.mean():+.2f}; largest per client {np.abs(subj - win).max():.2f})")

print("\n=== Five seeds, original split: SD over seeds, subject bootstrap (2,000 resamples), folds (average accuracy, %)")
C = {v: subjects(f"fedbn_{sc}_s{{seed}}", w) for v, (sc, w) in VARIANTS.items()}
rng = np.random.default_rng(0)
boot = {ds: [rng.integers(0, len(n), len(n)) for _ in range(2000)] for ds, (_, n, _) in C["FedBN"].items()}
avg = lambda c, sel=None: 100 * np.mean([c[ds][0][sel[ds] if sel else slice(None)].sum(0)
                                         / c[ds][1][sel[ds] if sel else slice(None)].sum() for ds in DS], axis=0)
for v, c in C.items():
    seeds = avg(c)
    bs = np.array([100 * np.mean([c[ds][0][b[i]].mean(1).sum() / c[ds][1][b[i]].sum() for ds, b in boot.items()])
                   for i in range(2000)])
    folds = [avg(c, {ds: c[ds][2] == k for ds in DS}).mean() for k in (0, 1)]
    print(f"{v:<14} {seeds.mean():.2f} | SD seeds {seeds.std(ddof=1):.2f} | SD bootstrap {bs.std(ddof=1):.2f} "
          f"[{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}] | folds {folds[0]:.2f} {folds[1]:.2f}")
print("\n=== Per test subject (accuracy averaged over the seeds): FedSleepAlign - FedBN")
acc = {v: {ds: 100 * C[v][ds][0].mean(1) / C[v][ds][1] for ds in DS} for v in C}
for ds in DS:
    d = acc["FedSleepAlign"][ds] - acc["FedBN"][ds]
    out[f"gain|{ds}"] = d
    print(f"{ds:<13} {len(d)} subjects | median gain {np.median(d):+.1f}, improved {100 * np.mean(d > 0):.0f}% | "
          f"FedSleepAlign 5-95% [{np.percentile(acc['FedSleepAlign'][ds], 5):.1f}, "
          f"{np.percentile(acc['FedSleepAlign'][ds], 95):.1f}]")
d = np.concatenate([out[f"gain|{ds}"] for ds in DS])
print(f"all {len(d)} subjects: improved {100 * np.mean(d > 0):.0f}%, by more than one point {100 * np.mean(d > 1):.0f}%")
np.savez(f"{SUMMARY}/selection.npz", **out)
