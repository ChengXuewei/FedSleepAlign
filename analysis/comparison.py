"""Sec. 3.5 and App. B.5-B.6: FedSleepAlign against local training, centralized training, FedAvg and Hybrid BN under the
matched protocol (PRS, seed 42), from the runs
    train.py --scheme local --client I | central | fedavg | hbn | fedbn   (--scaling robust --seed 42, both folds)
and FedBN without scaling (fedbn_none_s42). Writes results/summary/comparison.npz for figures/make_figures.py.
    python analysis/comparison.py > results/summary/comparison.log"""
import numpy as np

from common import DS, RESULTS, SUMMARY, preds, scores, confusion, table

ROWS = [("Local (stored)", "local_robust_s42", "S"), ("Local (PAR)", "local_robust_s42", "Q"),
        ("Central (stored)", "central_robust_s42", "S"), ("Central (PAR)", "central_robust_s42", "Q"),
        ("FedAvg (stored)", "fedavg_robust_s42", "S"), ("FedAvg (PAR)", "fedavg_robust_s42", "Q"),
        ("HybridBN", "hbn_robust_s42", "S"), ("FedBN+PRS (stored)", "fedbn_robust_s42", "S"),
        ("FedSleepAlign", "fedbn_robust_s42", "Q"), ("FedBN (stored)", "fedbn_none_s42", "S")]
res = {label: table(run, key)[:, :3] for label, run, key in ROWS}
print("=== Fig. 2: accuracy per client over both folds | average ACC, MF1, kappa (%)")
for label in res:
    print(f"{label:<20}" + " ".join(f"{x:6.2f}" for x in res[label][:, 0]) + " | "
          + " ".join(f"{x:6.2f}" for x in res[label].mean(0)))
loc, fsa, fb = res["Local (PAR)"][:, 0], res["FedSleepAlign"][:, 0], res["FedBN (stored)"][:, 0]
print(f"\nFedSleepAlign - local per client " + " ".join(f"{x:+.2f}" for x in fsa - loc) + f"; FedBN - local "
      + " ".join(f"{x:+.2f}" for x in fb - loc) + f"; gap closed {(fsa - fb).mean() / (loc - fb).mean():.0%}")
for k in ("Central (PAR)", "HybridBN"):
    print(f"{k} - local per client " + " ".join(f"{x:+.2f}" for x in res[k][:, 0] - loc))
for a, b in (("Local (PAR)", "Local (stored)"), ("Central (PAR)", "Central (stored)"), ("FedAvg (PAR)", "FedAvg (stored)"),
             ("FedSleepAlign", "FedBN+PRS (stored)")):
    d = res[a][:, 0] - res[b][:, 0]
    print(f"PAR - stored, {b.split(' (')[0]:<10} average {d.mean():+.2f}, per client " + " ".join(f"{x:+.2f}" for x in d))

print("\n=== Per fold: FedAvg against FedBN with PRS (average accuracy over the clients, %)")
for label, run in (("FedAvg", "fedavg_robust_s42"), ("FedBN+PRS", "fedbn_robust_s42")):
    for key in ("S", "Q"):
        print(f"{label:<10} {key}: " + " ".join(
            f"fold {k} {np.mean([100 * scores(confusion(preds(run, k)[f'{ds}|labels'], preds(run, k)[f'{ds}|{key}']))[0] for ds in DS]):.2f}"
            for k in (0, 1)))
for run in ("fedavg_robust_s42", "fedbn_robust_s42"):
    dev = np.array([np.load(f"{RESULTS}/rounds/{run}_fold{k}.npz")["dev"][10:, :, 2, :, 0] for k in (0, 1)])
    print(f"{run}: largest mean deviation of the stored from the PAR statistics over the layers "
          f"{dev.mean(axis=(0, 1, 2)).max():.2f} SD")
a = np.concatenate([np.load(f"{RESULTS}/rounds/hbn_robust_s42_fold{k}.npz")["hbn_a"].ravel() for k in (0, 1)])
print(f"Hybrid BN mixing weights between {a.min():.3f} and {a.max():.3f}")
np.savez(f"{SUMMARY}/comparison.npz", **res)
