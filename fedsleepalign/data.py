"""Recordings, subject split, per-recording scaling and windows."""
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from .augment import augment
from .config import DATA, DATABASES, SEQLEN


def subject_of(ds, name):
    """SleepEDF-78 SC4ssN*: subject ss (two nights); ISRUC II-s-n: session n of subject s; otherwise one recording per
    subject."""
    if ds == "SleepEDF-78":
        return name[3:5]
    if ds == "ISRUC" and name.startswith("II-"):
        return name.rsplit("-", 1)[0]
    return name


def split_halves(split=42):
    """{database: [recordings of half 0, recordings of half 1]}, split by subject; fold k tests on half k. Only
    recordings with all three channels are used, and 200 subjects of PhysioNet-2018 are drawn. Split 42 is that of the
    paper; any other value divides the same subjects with its own generator."""
    rng = np.random.RandomState(42)
    halves = {}
    for i, (ds, chans) in enumerate(DATABASES.items()):
        names = sorted(set.intersection(*({f for f in os.listdir(f"{DATA}/{c}") if f.endswith(".npz")}
                                          for c in chans)))
        subjects = sorted({subject_of(ds, n) for n in names})
        if ds == "PhysioNet2018":
            subjects = sorted(rng.choice(subjects, 200, replace=False))
        perm = rng.permutation(subjects)
        if split != 42:
            perm = np.random.RandomState(1000 * split + i).permutation(subjects)
        halves[ds] = [[n for n in names if subject_of(ds, n) in h] for h in map(set, np.array_split(perm, 2))]
    return halves


# ── per-recording scaling of every channel ───────────────────────────────────────────────────────────────────
def robust_scale(x):
    """PRS: minus the median, divided by the IQR (floored at 1e-6), clipped at +-20."""
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return np.clip((x - q2) / max(q3 - q1, 1e-6), -20.0, 20.0).astype(np.float32)


def _scaling(center, scale, kappa):
    """Minus the median or mean, divided by the IQR or SD, clipped at +-kappa (None: no clipping)."""
    def f(x):
        x64 = x.astype(np.float64)
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        u = (x64 - (q2 if center == "median" else x64.mean())) / max(q3 - q1 if scale == "iqr" else x64.std(), 1e-6)
        return (u if kappa is None else np.clip(u, -kappa, kappa)).astype(np.float32)
    return f


SCALINGS = {
    "none": lambda x: x,
    "robust": robust_scale,                              # PRS
    "zscore": _scaling("mean", "sd", 27.0),              # 27 SDs = 20 IQRs for Gaussian activity
    "mean_iqr": _scaling("mean", "iqr", 20.0),           # PRS with the mean in place of the median
    "med_sd": _scaling("median", "sd", 27.0),            # PRS with the SD in place of the IQR
    "noclip": _scaling("median", "iqr", None),           # PRS without clipping
}


def load_recording(path):
    """One channel of one recording: x (epochs, 3000) in uV at 100 Hz, y (epochs,) in 0..4 = W, N1, N2, N3, REM."""
    with np.load(path) as f:
        return f["x"].reshape(len(f["y"]), -1).astype(np.float32), f["y"].astype(np.int32)


def load_channels(ds, names, scaling):
    """[EEG 1, EEG 2, EOG], each a list of (scaled x, y) over the recordings."""
    return [[(SCALINGS[scaling](x), y) for x, y in (load_recording(f"{DATA}/{c}/{n}") for n in names)]
            for c in DATABASES[ds]]


# ── windows of L epochs; the network receives EEG 1 + EEG 2, EEG 1 - EEG 2 and the EOG ─────────────────────────
def _window(e1, e2, eog, y, s, e):
    return np.stack([e1[s:e] + e2[s:e], e1[s:e] - e2[s:e], eog[s:e]]).astype(np.float32), y[s:e].astype(np.int64)


def training_windows(channels, seqlen=SEQLEN):
    """Non-overlapping windows from the start of every recording (trailing epochs left out)."""
    out = [_window(e1, e2, eog, y, w * seqlen, (w + 1) * seqlen)
           for (e1, y), (e2, _), (eog, _) in zip(*channels) for w in range(len(y) // seqlen)]
    return [w[0] for w in out], [w[1] for w in out]


def test_windows(channels, seqlen=SEQLEN):
    """Windows covering every epoch: consecutive windows plus a final one that overlaps its predecessor. Returns
    windows, labels, (recording, start, end) of every window and the recording lengths."""
    wx, wy, meta, lengths = [], [], [], []
    for r, ((e1, y), (e2, _), (eog, _)) in enumerate(zip(*channels)):
        n = len(y)
        lengths.append(n)
        starts = [0] if n <= seqlen else list(range(0, n - seqlen + 1, seqlen))
        if n > seqlen and starts[-1] + seqlen < n:
            starts.append(n - seqlen)
        for s in starts:
            x, lab = _window(e1, e2, eog, y, s, min(s + seqlen, n))
            wx.append(x); wy.append(lab); meta.append((r, s, min(s + seqlen, n)))
    return wx, wy, meta, lengths


class Windows(Dataset):
    def __init__(self, wx, wy):
        self.x, self.y = [torch.from_numpy(w) for w in wx], [torch.from_numpy(w) for w in wy]
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class Samples(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def make_client(ds, names, scaling, val="window"):
    """The training half of one database, kept by its client: 90% of its windows for training plus augmented copies of
    10% of them, and 10% for validation ("win"). With val == "dual", 10% of its subjects (at least one) are first
    held out as a subject-disjoint validation set ("subj")."""
    vsets = {}
    if val == "dual":
        subjects = sorted({subject_of(ds, n) for n in names})
        held = set(random.sample(subjects, max(1, round(0.1 * len(subjects)))))
        vsets["subj"] = Windows(*training_windows(load_channels(ds, [n for n in names if subject_of(ds, n) in held],
                                                                scaling)))
        names = [n for n in names if subject_of(ds, n) not in held]
    windows = Windows(*training_windows(load_channels(ds, names, scaling)))
    n = int(0.9 * len(windows))
    train, vsets["win"] = random_split(windows, [n, len(windows) - n])
    copies = [(augment(train[i][0].permute(1, 0, 2)).permute(1, 0, 2), train[i][1])
              for i in random.sample(range(n), int(0.1 * n))]
    return SimpleNamespace(train=Samples([train[i] for i in range(n)] + copies), val=vsets)
