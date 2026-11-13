"""PhysioNet-2018 (You Snooze You Win challenge), the 994 training recordings with public labels. The stage of a 30-s
epoch is the stage that covers most of its samples; epochs dominated by "undefined" are removed. Channels F3-M2, O1-M2
and E1-M2, resampled from 200 Hz to 100 Hz by the Fourier method per epoch. split_halves draws 200 subjects.
    python preprocessing/physionet2018.py RAW OUT      RAW holds tr*-*/<record>.{hea,mat} and <record>-arousal.mat"""
import glob
import os

import h5py
import numpy as np
import scipy.signal
import wfdb

from common import args, run

CHANNELS = {"f3_m2": "F3-M2", "o1_m2": "O1-M2", "eog": "E1-M2"}
FOLDERS = {k: f"physionet2018/{k}" for k in CHANNELS}
STAGES = ["wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined"]


def read(record_dir):
    name = os.path.basename(record_dir)
    rec = wfdb.rdrecord(f"{record_dir}/{name}")
    sig = {k: rec.p_signal[:, rec.sig_name.index(c)].astype(np.float32) for k, c in CHANNELS.items()}
    k = round(30 * rec.fs)
    with h5py.File(f"{record_dir}/{name}-arousal.mat", "r") as f:
        stages = [f["data"]["sleep_stages"][s][0] for s in STAGES]
    n = min(len(sig["f3_m2"]), len(stages[0])) // k
    y = np.stack([s[:n * k].astype(np.int64).reshape(n, k).sum(1) for s in stages]).argmax(0)
    kept = np.flatnonzero(y != 5)
    x = {key: np.zeros((n, 3000), np.float32) for key in CHANNELS}
    for key, s in sig.items():
        for i in kept:
            x[key][i] = scipy.signal.resample(s[i * k:(i + 1) * k], 3000).astype(np.float32)
    return x, y, kept, CHANNELS


if __name__ == "__main__":
    raw, out = args()
    run([(os.path.basename(d), d) for d in sorted(glob.glob(f"{raw}/tr*-*"))], read, out, FOLDERS)
