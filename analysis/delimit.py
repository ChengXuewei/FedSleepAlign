"""App. B.1: PRS with the statistics of the whole stored recording. SleepEDF-78 and SHHS keep 30 min of wake around
sleep by the hypnogram, whereas a deployment without labels has only the stored recording. Here the median and IQR of
every channel come from all epochs of the stored recording (read by the preprocessing scripts from the raw files), and
the models of the fold-0 run selected under PAR (train.py --scheme fedbn --scaling robust --seed 42 --fold 0
--save-weights selected) predict the same kept epochs. One GPU; pyedflib.
    python analysis/delimit.py --sleepedf-raw RAW --shhs-raw RAW      -> results/summary/delimit.npz"""
import argparse
import glob
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

from common import DS, RESULTS, SUMMARY, confusion, scores
from fedsleepalign.config import BATCH, DATA, DATABASES
from fedsleepalign.data import Windows, load_channels, load_recording, split_halves, test_windows
from fedsleepalign.evaluate import majority, predict, reconstruct
from fedsleepalign.federated import device, load
from fedsleepalign.model import build_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
import shhs, sleepedf                                                            # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sleepedf-raw", required=True)
p.add_argument("--shhs-raw", required=True)
a = p.parse_args()


def stored_statistics(ds, name):
    """(q1, median, q3) of every channel over all epochs of the stored recording."""
    if ds == "SleepEDF-78":
        hyp = glob.glob(f"{a.sleepedf_raw}/{name[:6]}*-Hypnogram.edf")[0]
        x = sleepedf.read(f"{a.sleepedf_raw}/{name}-PSG.edf", hyp)[0]
    else:
        x = shhs.read(f"{a.shhs_raw}/edfs/shhs1/{name}.edf",
                      f"{a.shhs_raw}/annotations-events-profusion/shhs1/{name}-profusion.xml")[0]
    return [np.percentile(v, [25, 50, 75]) for v in x.values()], len(next(iter(x.values())))


halves, out = split_halves(42), {}
log = np.load(f"{RESULTS}/rounds/fedbn_robust_s42_fold0.npz")
members = [load(f"{RESULTS}/weights/fedbn_robust_s42_fold0/r{r:02d}.pt")
           for r in sorted(log["best"][list(log["keys"]).index("Q")])]
model = build_model("fedbn").to(device)
saved = np.load(f"{RESULTS}/preds/fedbn_robust_s42_fold0.npz")
for ds in ("SleepEDF-78", "SHHS"):
    j, names = DS.index(ds), halves[ds][0]
    channels, ratio, hours = [[], [], []], [], []
    for n in names:
        q, n_stored = stored_statistics(ds, n[:-4])
        for c, chan, (q1, q2, q3) in zip(DATABASES[ds], channels, q):
            x, y = load_recording(f"{DATA}/{c}/{n}")
            chan.append((np.clip((x - q2) / max(q3 - q1, 1e-6), -20.0, 20.0).astype(np.float32), y))
            k1, k3 = np.percentile(x, [25, 75])
            ratio.append((q3 - q1) / (k3 - k1))
        hours.append([n_stored / 120, len(y) / 120])
    wx, wy, meta, lengths = test_windows(channels)
    loader = DataLoader(Windows(wx, wy), batch_size=BATCH)
    votes = []
    for m in members:
        model.load_state_dict({**m["shared"], **m["spec"]["Q"][j]})
        v, labels_w = predict(model, loader)
        votes.append(v)
    preds, labels = reconstruct(majority(np.stack(votes)), labels_w, meta, lengths)
    trimmed, stored = 100 * scores(confusion(labels, saved[f"{ds}|Q"]))[:3], 100 * scores(confusion(labels, preds))[:3]
    hours = np.mean(hours, 0)
    print(f"{ds}: stored {hours[0]:.1f} h, kept {hours[1]:.1f} h on average; IQR stored / kept median "
          f"{np.median(ratio):.3f} | ACC MF1 kappa trimmed " + " ".join(f"{v:.2f}" for v in trimmed) + " | stored "
          + " ".join(f"{v:.2f}" for v in stored) + " | change " + " ".join(f"{s - t:+.2f}" for s, t in zip(stored, trimmed)))
    out[f"{ds}|trimmed"], out[f"{ds}|stored"], out[f"{ds}|hours"] = trimmed, stored, hours
np.savez(f"{SUMMARY}/delimit.npz", **out)
