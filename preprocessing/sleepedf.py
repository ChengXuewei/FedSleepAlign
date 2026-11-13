"""SleepEDF-78: the 153 Sleep Cassette recordings of Sleep-EDF Expanded (PhysioNet sleep-edfx), processed as in
TinySleepNet: R&K stages of the hypnogram with S3 and S4 merged into N3, 30 min of wake kept before the first and after
the last sleep epoch, movement and unscored epochs removed. Channels Fpz-Cz, Pz-Oz and the horizontal EOG, all 100 Hz.
    python preprocessing/sleepedf.py RAW OUT      RAW holds SC4*-PSG.edf and SC4*-Hypnogram.edf"""
import glob
import os

import numpy as np
import pyedflib

from common import args, run

CHANNELS = {"fpz_cz": "EEG Fpz-Cz", "pz_oz": "EEG Pz-Oz", "eog": "EOG horizontal"}
FOLDERS = {k: f"sleepedf/{k}" for k in CHANNELS}
ANN2LABEL = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 3,
             "Sleep stage R": 4, "Movement time": 5, "Sleep stage ?": 6}


def read(psg, hypnogram):
    """All epochs of the stored recording, their labels and the indices of the kept epochs."""
    f, a = pyedflib.EdfReader(psg), pyedflib.EdfReader(hypnogram)
    names = f.getSignalLabels()
    x = {k: f.readSignal(names.index(c)).reshape(-1, 3000).astype(np.float32) for k, c in CHANNELS.items()}
    _, durations, stages = a.readAnnotations()
    f.close(); a.close()
    y = np.concatenate([np.full(int(d) // 30, ANN2LABEL[s]) for d, s in zip(durations, stages)])[:len(x["fpz_cz"])]
    sleep = np.flatnonzero(y != 0)                    # any label other than wake, as TinySleepNet
    period = np.arange(max(sleep[0] - 60, 0), min(sleep[-1] + 60, len(y) - 1) + 1)
    return x, y, period[y[period] < 5], CHANNELS


def records(raw):
    hyp = {os.path.basename(h)[:6]: h for h in glob.glob(f"{raw}/SC4*-Hypnogram.edf")}
    return [(os.path.basename(p)[:-8], p, hyp[os.path.basename(p)[:6]])
            for p in sorted(glob.glob(f"{raw}/SC4*-PSG.edf")) if os.path.basename(p)[:6] in hyp]


if __name__ == "__main__":
    raw, out = args()
    run(records(raw), read, out, FOLDERS)
