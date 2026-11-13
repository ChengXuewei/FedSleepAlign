"""ISRUC-Sleep, subgroups I-III (126 recordings of 118 subjects), with the stages of the first scorer (stage 4 merged
into N3). Channels F3-A2, O1-A2 and LOC-A2 (re-derived from the electrodes when only these are stored), resampled from
200 Hz to 100 Hz by the Fourier method per 30-s epoch. The recording without an F3 derivation (I-8) is skipped.
    python preprocessing/isruc.py RAW OUT      RAW holds subgroupI/<n>/<n>.rec, subgroupII/<n>/<s>/<s>.rec, ..."""
import glob
import os

import numpy as np
import pyedflib
import scipy.signal

from common import args, run

CHANNELS = {"f3_a2": ["F3-A2", "F3-M2"], "o1_a2": ["O1-A2", "O1-M2"], "eog": ["LOC-A2", "E1-M2"]}
FOLDERS = {k: f"isruc/{k}" for k in CHANNELS}
CODES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}


def pick(names, aliases):
    """Indices of a stored derivation, or of its active electrode and reference."""
    for alias in aliases:
        if alias in names:
            return [names.index(alias)], alias
    for alias in aliases:
        act, _, ref = alias.partition("-")
        if act in names and ref in names:
            return [names.index(act), names.index(ref)], alias
    raise KeyError(f"none of {aliases}")


def read(rec, scoring):
    y = np.array([CODES[c] for c in np.loadtxt(scoring, dtype=int)], dtype=np.int32)
    f = pyedflib.EdfReader(rec)
    names, sig, fs, used = f.getSignalLabels(), {}, {}, {}
    try:
        for key, aliases in CHANNELS.items():
            idx, used[key] = pick(names, aliases)
            fs[key] = f.getSampleFrequency(idx[0])
            sig[key] = f.readSignal(idx[0]).astype(np.float32)
            if len(idx) == 2:
                sig[key] = sig[key] - f.readSignal(idx[1]).astype(np.float32)
    finally:
        f.close()
    n = min(len(y), *(len(s) // round(30 * fs[k]) for k, s in sig.items()))
    x = {}
    for key, s in sig.items():
        k = round(30 * fs[key])
        x[key] = np.stack([scipy.signal.resample(s[i * k:(i + 1) * k], 3000).astype(np.float32) for i in range(n)])
    return x, y[:n], np.arange(n), used


def records(raw):
    num = lambda p: (0, int(os.path.basename(p))) if os.path.basename(p).isdigit() else (1, os.path.basename(p))
    out = []
    for group, prefix in (("subgroupI", "I"), ("subgroupIII", "III")):
        for d in sorted(glob.glob(f"{raw}/{group}/*/"), key=lambda p: num(p.rstrip("/"))):
            n = os.path.basename(d.rstrip("/"))
            out.append((f"{prefix}-{n}", f"{d}{n}.rec", f"{d}{n}_1.txt"))
    for d in sorted(glob.glob(f"{raw}/subgroupII/*/"), key=lambda p: num(p.rstrip("/"))):
        for sd in sorted(glob.glob(f"{d}*/"), key=lambda p: num(p.rstrip("/"))):
            n, s = os.path.basename(d.rstrip("/")), os.path.basename(sd.rstrip("/"))
            out.append((f"II-{n}-{s}", f"{sd}{s}.rec", f"{sd}{s}_1.txt"))
    return out


if __name__ == "__main__":
    raw, out = args()
    run(records(raw), read, out, FOLDERS)
