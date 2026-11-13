"""HMC (Haaglanden Medisch Centrum sleep staging database, PhysioNet), 151 recordings scored with the AASM rules.
Epochs follow the 30-s grid of the stage annotations; unscored epochs ("Sleep stage ?") are removed, which leaves every
night contiguous. Channels F4-M1, O2-M1 and E1-M2 at 256 Hz; every night is resampled to 100 Hz as a whole by the
polyphase filter of scipy.signal.resample_poly (25/64), whose low-pass removes content above 50 Hz.
    python preprocessing/hmc.py RAW OUT      RAW holds RECORDS and recordings/SN###.edf, SN###_sleepscoring.edf"""
import os

import numpy as np
import pyedflib
import scipy.signal

from common import args, run

CHANNELS = {"f4_m1": ["EEG F4-M1", "EEG F4-A1", "F4-M1", "F4-A1"], "o2_m1": ["EEG O2-M1", "EEG O2-A1", "O2-M1", "O2-A1"],
            "e1_m2": ["EOG E1-M2", "EOG E1-A2", "E1-M2", "E1-A2", "EOG LOC-A2"]}
FOLDERS = {k: f"hmc/{k}" for k in CHANNELS}
STAGES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def pick(names, aliases):
    for alias in aliases:
        if alias in names:
            return [names.index(alias)], alias
    for alias in aliases:
        act, _, ref = alias.partition("-")
        act = act.split(" ")[-1]
        if act in names and ref in names:
            return [names.index(act), names.index(ref)], alias
    raise KeyError(f"none of {aliases}")


def read(psg, scoring):
    f = pyedflib.EdfReader(psg)
    a = pyedflib.EdfReader(scoring)
    offset = (a.getStartdatetime() - f.getStartdatetime()).total_seconds()
    stages = {}
    for onset, dur, desc in zip(*a.readAnnotations()):
        code = str(desc).split("@@")[0].strip()
        if code.startswith("Sleep stage ") and code[12:].strip() in STAGES and abs(dur - 30) < 1e-3:
            stages[int(round((onset + offset) / 30))] = STAGES[code[12:].strip()]
    a.close()
    names, sig, used = f.getSignalLabels(), {}, {}
    for key, aliases in CHANNELS.items():
        idx, used[key] = pick(names, aliases)
        sig[key] = f.readSignal(idx[0]).astype(np.float32)
        if len(idx) == 2:
            sig[key] = sig[key] - f.readSignal(idx[1]).astype(np.float32)
        k = round(30 * f.getSampleFrequency(idx[0]))
    f.close()
    n = min(len(s) for s in sig.values()) // k
    kept = sorted(i for i in stages if 0 <= i < n)
    assert kept == list(range(kept[0], kept[-1] + 1)), "the scored epochs of a night are contiguous"
    n = kept[-1] + 1                                    # the stored recording ends with its last scored epoch
    y = np.full(n, -1)
    y[kept] = [stages[i] for i in kept]
    x = {}
    for key, s in sig.items():                          # the scored part of the night, resampled as a whole
        night = np.stack([s[i * k:(i + 1) * k] for i in kept]).reshape(-1).astype(np.float64)
        x[key] = np.zeros((n, 3000), np.float32)
        x[key][kept] = scipy.signal.resample_poly(night, 25, 64).reshape(len(kept), 3000).astype(np.float32)
    return x, y, kept, used


if __name__ == "__main__":
    raw, out = args()
    with open(f"{raw}/RECORDS") as f:
        ids = [os.path.basename(line.strip())[:-4] for line in f if line.strip()]
    run([(i, f"{raw}/recordings/{i}.edf", f"{raw}/recordings/{i}_sleepscoring.edf") for i in ids], read, out, FOLDERS)
