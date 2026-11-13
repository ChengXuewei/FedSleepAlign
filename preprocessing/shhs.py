"""SHHS: the baseline (SHHS-1) recordings of the 329 subjects with regular sleep (apnea-hypopnea index below 5)
selected by Eldele et al. (AttnSleep), listed in shhs_records.txt. R&K stages of the profusion XML with S3 and S4
merged into N3; 30 min of wake kept before the first and after the last sleep epoch; other codes removed. Channels
C4-A1 and C3-A2 at 125 Hz and the difference of the left and right EOG at 50 Hz, resampled to 100 Hz by the Fourier
method per epoch. The data are distributed by the NSRR (sleepdata.org) after a data access request.
    python preprocessing/shhs.py RAW OUT      RAW holds edfs/shhs1/<id>.edf and annotations-events-profusion/shhs1/"""
import os
import re

import numpy as np
import pyedflib
import scipy.signal

from common import args, run

CHANNELS = {"c4_a1": "EEG", "c3_a2": "EEG(sec)", "eog_lr": ("EOG(L)", "EOG(R)")}
LABELS = {"c4_a1": "C4-A1", "c3_a2": "C3-A2", "eog_lr": "EOG(L)-EOG(R)"}
FOLDERS = {k: f"shhs/{k}" for k in CHANNELS}
ALIASES = {"EEG(sec)": ("EEG(sec)", "EEG(SEC)", "EEG2", "EEG 2")}
STAGES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}


def signal(edf, label):
    names = edf.getSignalLabels()
    i = next(names.index(a) for a in ALIASES.get(label, (label,)) if a in names)
    return edf.readSignal(i), edf.getSampleFrequency(i)


def read(edf_path, xml_path):
    stages = np.array([int(s) for s in re.findall(r"<SleepStage>(\d+)</SleepStage>", open(xml_path).read())])
    edf = pyedflib.EdfReader(edf_path)
    sig = {}
    for key, lab in CHANNELS.items():
        if isinstance(lab, tuple):
            (l, fs), (r, _) = signal(edf, lab[0]), signal(edf, lab[1])
            sig[key] = (l - r, fs)
        else:
            sig[key] = signal(edf, lab)
    edf.close()
    n = min(len(stages), *(len(s) // int(round(30 * fs)) for s, fs in sig.values()))
    y = np.array([STAGES.get(s, -1) for s in stages[:n]])
    sleep = np.flatnonzero(y > 0)
    period = np.arange(max(0, sleep[0] - 60), min(n, sleep[-1] + 61))
    x = {key: scipy.signal.resample(s[:n * int(round(30 * fs))].reshape(n, -1), 3000, axis=1).astype(np.float32)
         for key, (s, fs) in sig.items()}
    return x, y, period[y[period] >= 0], LABELS


if __name__ == "__main__":
    raw, out = args()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shhs_records.txt")) as f:
        ids = [line.strip() for line in f if line.strip()]
    run([(i, f"{raw}/edfs/shhs1/{i}.edf", f"{raw}/annotations-events-profusion/shhs1/{i}-profusion.xml") for i in ids],
        read, out, FOLDERS)
