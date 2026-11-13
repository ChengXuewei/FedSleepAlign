"""App. B.2: artifacts injected at test time. In every fold-0 test recording, a random fraction rho of the epochs
receives on every channel a raised-cosine excursion over the 30-s epoch with a peak of ten IQRs of the channel and a
random sign, before the scaling; the epochs contaminated at the larger rho include those of the smaller one. The models
of the fold-0 run selected under PAR (train.py --scheme fedbn --scaling X --seed 42 --fold 0 --save-weights selected)
predict the contaminated recordings, and the accuracy is measured on the epochs that no rho contaminates, so it changes
only through the statistics of the recording and the context of the windows. One GPU.
    python analysis/artifact.py --scaling X      -> results/summary/artifact_<X>.npz"""
import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

from common import DS, RESULTS, SUMMARY
from fedsleepalign import bn
from fedsleepalign.config import BATCH, DATA, DATABASES
from fedsleepalign.data import SCALINGS, Windows, load_recording, split_halves, test_windows
from fedsleepalign.evaluate import majority, predict, reconstruct
from fedsleepalign.federated import device, load
from fedsleepalign.model import build_model

RHOS, AMP = (0.0, 0.02, 0.05), 10.0
BUMP = (1 - np.cos(2 * np.pi * np.arange(3000) / 3000)) / 2


def contaminate(ds, j, names, rho):
    """Raw channels [EEG 1, EEG 2, EOG] of the recordings with a fraction rho of their epochs contaminated, the mask of
    the epochs that the largest rho leaves clean, and the growth of the SD and the IQR of every channel."""
    chans, clean, growth = [[], [], []], [], []
    for i, n in enumerate(names):
        xs = [load_recording(f"{DATA}/{c}/{n}") for c in DATABASES[ds]]
        y = xs[0][1]
        rng = np.random.default_rng(1000 * j + i)
        order, signs = rng.permutation(len(y)), rng.choice([-1.0, 1.0], size=(3, len(y)))
        hit = order[:int(round(rho * len(y)))]
        clean.append(~np.isin(np.arange(len(y)), order[:int(round(max(RHOS) * len(y)))]))
        for ch, ((x, _), chan) in enumerate(zip(xs, chans)):
            x0 = x.astype(np.float64)
            q1, q3 = np.percentile(x0, [25, 75])
            x1 = x0.copy()
            x1[hit] += AMP * (q3 - q1) * signs[ch, hit, None] * BUMP
            chan.append((x1.astype(np.float32), y))
            growth.append([x1.std() / x0.std(), np.subtract(*np.percentile(x1, [75, 25])) / (q3 - q1)])
    return chans, np.concatenate(clean), np.array(growth)


def main(scaling):
    run, halves = f"fedbn_{scaling}_s42", split_halves(42)
    wdir = f"{RESULTS}/weights/{run}_fold0"
    log = np.load(f"{RESULTS}/rounds/{run}_fold0.npz")
    rounds = sorted(log["best"][list(log["keys"]).index("Q")])                        # the rounds selected under PAR
    members = [load(f"{wdir}/r{r:02d}.pt") for r in rounds]
    model = build_model("fedbn").to(device)
    correct, n_clean, growth = np.zeros((len(RHOS), len(DS))), np.zeros((len(RHOS), len(DS))), {}
    for j, ds in enumerate(DS):
        for ri, rho in enumerate(RHOS):
            raw, clean, g = contaminate(ds, j, halves[ds][0], rho)
            growth.setdefault(rho, []).append(g)
            channels = [[(SCALINGS[scaling](x), y) for x, y in chan] for chan in raw]
            wx, wy, meta, lengths = test_windows(channels)
            loader = DataLoader(Windows(wx, wy), batch_size=BATCH)
            votes = []
            for m in members:
                model.load_state_dict({**m["shared"], **m["spec"]["Q"][j]})
                v, labels_w = predict(model, loader)
                votes.append(v)
            preds, labels = reconstruct(majority(np.stack(votes)), labels_w, meta, lengths)
            correct[ri, j], n_clean[ri, j] = np.sum((preds == labels)[clean]), clean.sum()
            print(f"{ds:<13} rho {rho:.2f}: accuracy on the clean epochs {100 * correct[ri, j] / n_clean[ri, j]:.2f}",
                  flush=True)
    med = {rho: np.median(np.concatenate(growth[rho]), axis=0) for rho in RHOS[1:]}
    np.savez(f"{SUMMARY}/artifact_{scaling}.npz", rhos=np.array(RHOS), correct_clean=correct, n_clean=n_clean,
             sd_growth=np.array([med[r][0] for r in RHOS[1:]]), iqr_growth=np.array([med[r][1] for r in RHOS[1:]]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scaling", choices=("robust", "mean_iqr", "med_sd", "noclip", "zscore"), required=True)
    main(p.parse_args().scaling)
