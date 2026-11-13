"""Loading of the test predictions written by train.py, metrics and helpers shared by the analysis scripts."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fedsleepalign.config import DATABASES, RESULTS, STAGES      # noqa: E402
from fedsleepalign.data import subject_of                          # noqa: E402
from fedsleepalign.evaluate import confusion, scores               # noqa: E402

DS = list(DATABASES)
SEEDS = (42, 142, 242, 342, 442)
SUMMARY = f"{RESULTS}/summary"
os.makedirs(SUMMARY, exist_ok=True)


def preds(run, k):
    return np.load(f"{RESULTS}/preds/{run}_fold{k}.npz")


def pooled(run, key, ds):
    """Labels and predictions of evaluation `key` over the test epochs of both folds of one client. Local runs are
    stored per client (<run>_c<i>)."""
    name = f"{run}_c{DS.index(ds)}" if run.startswith("local_") else run
    f = [preds(name, k) for k in (0, 1)]
    return np.concatenate([x[f"{ds}|labels"] for x in f]), np.concatenate([x[f"{ds}|{key}"] for x in f])


def table(run, key):
    """(clients, 8) scores in % over both folds: accuracy, macro F1, kappa and the five stage recalls."""
    return np.array([100 * scores(confusion(*pooled(run, key, ds))) for ds in DS])


def per_seed(runs, key):
    """(seeds, clients + average, 8) scores in %, the average taken over the clients."""
    a = np.array([table(r, key) for r in runs])
    return np.concatenate([a, a.mean(1, keepdims=True)], axis=1)


def subjects(run, key, seeds=SEEDS):
    """{ds: (correct epochs (subjects, seeds), epochs (subjects,), fold (subjects,))} over both folds of the runs
    <run> with the given seeds (`run` contains '{seed}')."""
    out = {}
    for ds in DS:
        per_seed_ok, subj, fold = [], None, None
        for s in seeds:
            f = [preds(run.format(seed=s), k) for k in (0, 1)]
            rec = np.concatenate([np.repeat(x[f"{ds}|names"], x[f"{ds}|lengths"]) for x in f])
            ok = np.concatenate([x[f"{ds}|{key}"] == x[f"{ds}|labels"] for x in f])
            fk = np.concatenate([np.full(int(np.sum(x[f"{ds}|lengths"])), k) for k, x in enumerate(f)])
            subj = np.array([subject_of(ds, str(r)) for r in rec])
            us, idx = np.unique(subj, return_inverse=True)
            per_seed_ok.append(np.bincount(idx, weights=ok, minlength=len(us)))
            fold = np.array([fk[idx == i][0] for i in range(len(us))])
        out[ds] = (np.stack(per_seed_ok, 1), np.bincount(idx, minlength=len(us)), fold)
    return out


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(p) - i) * p[j])); adj[j] = run
    return adj


pm = lambda x: f"{np.mean(x):.2f} +- {np.std(x, ddof=1):.2f}"
