"""App. B.4 and B.6: calibration analyses at fixed weights, on the weights of every round of FedBN with PRS
(train.py --scheme fedbn --scaling robust --seed 42 --fold 0 --save-weights all). The clients are rebuilt exactly as
in training, and calibration windows are drawn uniformly without replacement from a client's training pool with a
generator seeded by (seed, fold, round, client). One GPU for every part except summary.
    --part timing     one-pass, layer-wise (one forward pass per BN layer, as in training) and module-by-module
                      layer-wise statistics (every module applied once to all cached batches, Prop. 2): time and agreement
    --part bsweep     the same 320 windows per client and round in batches of b windows: layer-wise statistics against
                      the one-pass statistics for b = 1, 2, 4, 8, 32 (BN output error and pooled validation accuracy)
    --part btest      test accuracy of every bsweep variant with its own top-10 rounds
    --part budget     layer-wise statistics from N windows against those of the whole training pool
    --part perclient  top-10 rounds selected by every client's own validation accuracy instead of the pooled one
    --part summary    the numbers of App. B.4 and B.6 from the saved parts
    python analysis/calibration.py --part P [--rounds A-B]      -> results/calibration/"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import linregress
from torch.utils.data import DataLoader

from common import DS, RESULTS, confusion, scores
from fedsleepalign import bn
from fedsleepalign.config import BATCH, ROUNDS, TOP_M
from fedsleepalign.data import Windows, load_channels, split_halves, test_windows
from fedsleepalign.evaluate import epoch_accuracy, majority, predict, reconstruct
from fedsleepalign.federated import device, load
from fedsleepalign.model import build_model
from train import build_clients

RUN, K, SEED = "fedbn_robust_s42", 0, 42
OUT = f"{RESULTS}/calibration"
B_SIZES, N_CAL, BUDGETS, REPS, BUDGET_ROUNDS = (1, 2, 4, 8, 32), 320, (32, 64, 128, 320, 640, 1280), 3, (10, 20, 30, 40, 50)


def clients():
    halves = split_halves(42)
    cl, _ = build_clients(SimpleNamespace(scheme="fedbn", seed=SEED, scaling="robust", val="window"), DS, halves, K)
    return halves, cl


def weights(r):
    return load(f"{RESULTS}/weights/{RUN}_fold{K}/r{r:02d}.pt")


def stored(w, i):
    return {**w["shared"], **w["spec"]["S"][i]}


def draw(client, n, key):
    """n windows drawn uniformly without replacement from the client's training pool (augmented copies included)."""
    idx = np.random.default_rng(key).choice(len(client.train), size=min(n, len(client.train)), replace=False)
    return torch.stack([client.train[int(i)][0] for i in idx])


def split(x, b):
    return [x[j:j + b] for j in range(0, len(x), b)]


@torch.no_grad()
def reestimate_by_module(model, state, batches):
    """The statistics of bn.reestimate, computed module by module: every module of the residual blocks is applied once
    to every cached batch, and a BN layer receives its pooled statistics before it is applied."""
    model.load_state_dict(state); model.eval()
    xs = [bn.cnn_input(x.to(device)) for x in batches]
    for bi in (0, 3):
        blk = model.cnn[bi]
        m1, ms, h = bn.Moments(), bn.Moments(), []
        for x in xs:
            a = blk.conv1(x); m1.add(a); h.append(a)
            ms.add(blk.downsample[0](x))
        m1.set(blk.bn1); ms.set(blk.downsample[1])
        m2 = bn.Moments()
        for j in range(len(h)):
            h[j] = blk.conv2(blk.relu(blk.bn1(h[j]))); m2.add(h[j])
        m2.set(blk.bn2)
        if bi == 0:
            xs = [model.cnn[2](model.cnn[1](blk.relu(blk.bn2(g) + blk.downsample(x)))) for x, g in zip(xs, h)]
    return bn.bn_entries(model, model.state_dict())


def bn_error(est, ref, names, eps=1e-5):
    """Per layer, RMS and maximum over channels of the normalized BN output error of Prop. 1(ii)."""
    out = []
    for n in names:
        mu, var, mu0, var0 = (t[f"{n}.{s}"].double() for t, s in ((est, "running_mean"), (est, "running_var"),
                                                                   (ref, "running_mean"), (ref, "running_var")))
        s, s0 = (var + eps).sqrt(), (var0 + eps).sqrt()
        e = (mu - mu0) ** 2 / s ** 2 + var0 * (1 / s - 1 / s0) ** 2
        out.append([e.mean().sqrt().item(), e.max().sqrt().item()])
    return np.array(out)


def val_accuracy(model, w, specs, cl):
    accs = []
    for i, c in enumerate(cl):
        model.load_state_dict({**w["shared"], **specs[i]})
        accs.append(epoch_accuracy(model, DataLoader(c.val["win"], batch_size=BATCH)))
    return np.average(accs, weights=[len(c.val["win"]) for c in cl]), accs


def variants(model, w, cl, r):
    """BN entries of every client: Q (layer-wise) and R<b> (one-pass, batches of b) from the same 320 windows."""
    out = {"Q": [], **{f"R{b}": [] for b in B_SIZES}}
    for i, c in enumerate(cl):
        x, st = draw(c, N_CAL, [SEED, K, r, i]), stored(w, i)
        out["Q"].append(reestimate_by_module(model, st, split(x, 32)))
        for b in B_SIZES:
            out[f"R{b}"].append(bn.reestimate_onepass(model, st, split(x, b)))
    return out


def test_members(model, halves, members, spec_of):
    """{ds: {member: window predictions}}, labels, meta and lengths of the fold-K test recordings."""
    out = {}
    for j, ds in enumerate(DS):
        wx, wy, meta, lengths = test_windows(load_channels(ds, halves[ds][K], "robust"))
        loader, pred = DataLoader(Windows(wx, wy), batch_size=BATCH), {}
        for m in members:
            model.load_state_dict(spec_of(m, j))
            pred[m], labels_w = predict(model, loader)
        out[ds] = (pred, labels_w, meta, lengths)
    return out


def main(part, rounds):
    model = build_model("fedbn").to(device)
    layers = [n for n, _ in bn.norm_layers(model)]
    if part == "summary":
        return summary()
    halves, cl = clients()
    if part == "timing":
        res = []
        for r in rounds:
            w = weights(r)
            for i, c in enumerate(cl):
                batches, st, t = split(draw(c, N_CAL, [SEED, K, r, i]), 32), stored(w, i), {}
                for name, fn in (("one-pass", bn.reestimate_onepass), ("layer-wise", bn.reestimate),
                                 ("by module", reestimate_by_module)):
                    times = []
                    for _ in range(3):
                        torch.cuda.synchronize(); t0 = time.time(); est = fn(model, st, batches)
                        torch.cuda.synchronize(); times.append(time.time() - t0)
                    t[name] = (np.median(times), est)
                a, b = t["layer-wise"][1], t["by module"][1]
                rel = max(max(((b[f"{n}.running_mean"] - a[f"{n}.running_mean"]).abs() / a[f"{n}.running_var"].sqrt()).max().item(),
                              ((b[f"{n}.running_var"] - a[f"{n}.running_var"]).abs() / a[f"{n}.running_var"]).max().item())
                          for n in layers)
                res.append([r, i, t["one-pass"][0], t["layer-wise"][0], t["by module"][0], rel])
                print(f"round {r} {DS[i]:<13} " + " ".join(f"{k} {v[0]:.2f}s" for k, v in t.items())
                      + f" | max relative difference {rel:.1e}", flush=True)
        np.savez(f"{OUT}/timing.npz", res=np.array(res), gpu=torch.cuda.get_device_name())
    elif part == "bsweep":
        names = ["S", "Q"] + [f"R{b}" for b in B_SIZES]
        val = np.zeros((len(rounds), len(names), 1 + len(cl)))
        err = np.zeros((len(rounds), len(B_SIZES), len(cl), len(layers), 2))
        for ri, r in enumerate(rounds):
            w = weights(r)
            var = {"S": w["spec"]["S"], **variants(model, w, cl, r)}
            for j, n in enumerate(names):
                p, accs = val_accuracy(model, w, var[n], cl)
                val[ri, j] = [p, *accs]
            for bj, b in enumerate(B_SIZES):
                for i in range(len(cl)):
                    err[ri, bj, i] = bn_error(var[f"R{b}"][i], var["Q"][i], layers)
            print(f"round {r:02d} validation " + " ".join(f"{n} {100 * val[ri, j, 0]:.2f}" for j, n in enumerate(names)),
                  flush=True)
        np.savez(f"{OUT}/bsweep_r{rounds[0]:02d}-{rounds[-1]:02d}.npz", rounds=np.array(rounds), names=np.array(names),
                 val=val, err=err)
    elif part == "btest":
        parts = [np.load(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.startswith("bsweep_r")]
        rr, val, names = np.concatenate([p["rounds"] for p in parts]), np.concatenate([p["val"] for p in parts]), list(parts[0]["names"])
        top = {n: sorted(rr[np.argsort(-val[:, j, 0], kind="stable")[:TOP_M]].tolist()) for j, n in enumerate(names)
               if n in ("Q", "R1", "R4", "R32")}
        spec = {}
        for r in sorted({r for t in top.values() for r in t}):
            w = weights(r)
            var = variants(model, w, cl, r)
            spec.update({(r, n): (w["shared"], var[n]) for n in top if r in top[n]})
        res = test_members(model, halves, list(spec), lambda m, j: {**spec[m][0], **spec[m][1][j]})
        out = {}
        for ds, (pred, labels_w, meta, lengths) in res.items():
            for n, t in top.items():
                p, y = reconstruct(majority(np.stack([pred[r, n] for r in t])), labels_w, meta, lengths)
                out[f"{ds}|{n}"], out[f"{ds}|labels"] = p, y
        np.savez(f"{OUT}/btest.npz", **out)
    elif part == "budget":
        err = np.full((len(rounds), len(BUDGETS), REPS, len(cl), len(layers), 2), np.nan)
        val = np.full((len(rounds), len(BUDGETS) + 1, REPS), np.nan)
        for ri, r in enumerate(rounds):
            w = weights(r)
            full = [bn.reestimate(model, stored(w, i), [x for x, _ in DataLoader(c.train, batch_size=BATCH)])
                    for i, c in enumerate(cl)]
            val[ri, -1] = val_accuracy(model, w, full, cl)[0]
            for ni, n in enumerate(BUDGETS):
                for rep in range(REPS):
                    est = [bn.reestimate(model, stored(w, i), split(draw(c, n, [SEED, K, r, i, n, rep]), 32))
                           for i, c in enumerate(cl)]
                    err[ri, ni, rep] = [bn_error(e, f, layers) for e, f in zip(est, full)]
                    val[ri, ni, rep] = val_accuracy(model, w, est, cl)[0]
            print(f"round {r:02d} error " + " ".join(f"N={n}: {x:.4f}" for n, x in
                                                      zip(BUDGETS, err[ri, ..., 0].max(-1).mean((1, 2)))), flush=True)
        np.savez(f"{OUT}/budget.npz", rounds=np.array(rounds), budgets=np.array(BUDGETS), err=err, val=val,
                 pool=np.array([len(c.train) for c in cl]))
    elif part == "perclient":
        log = np.load(f"{RESULTS}/rounds/{RUN}_fold{K}.npz")
        q = list(log["keys"]).index("Q")
        sel = {"pooled": sorted(log["best"][q])}
        sel.update({f"own{j}": sorted(np.argsort(-log["acc"][:, q, j], kind="stable")[:TOP_M] + 1) for j in range(len(DS))})
        ws = {r: weights(r) for r in sorted({r for s in sel.values() for r in s})}
        res = test_members(model, halves, list(ws), lambda r, j: {**ws[r]["shared"], **ws[r]["spec"]["Q"][j]})
        out = {}
        for j, (ds, (pred, labels_w, meta, lengths)) in enumerate(res.items()):
            for name, rounds_ in (("pooled", sel["pooled"]), ("own", sel[f"own{j}"])):
                p, y = reconstruct(majority(np.stack([pred[r] for r in rounds_])), labels_w, meta, lengths)
                out[f"{name}|{ds}"] = 100 * np.mean(p == y)
            print(f"{ds:<13} pooled {out[f'pooled|{ds}']:.2f} | own {out[f'own|{ds}']:.2f}", flush=True)
        np.savez(f"{OUT}/perclient.npz", **out)


def summary():
    t = np.load(f"{OUT}/timing.npz"); r = t["res"]; rounds = np.unique(r[:, 0])
    tot = lambda j: np.median([r[r[:, 0] == x, j].sum() for x in rounds])
    print(f"timing ({t['gpu']}), five clients: one-pass {tot(2):.1f}s, layer-wise {tot(3):.1f}s, by module {tot(4):.1f}s; "
          f"largest relative difference {r[:, 5].max():.1e}")
    parts = [np.load(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.startswith("bsweep_r")]
    rr, val, err = (np.concatenate([p[k] for p in parts]) for k in ("rounds", "val", "err"))
    names, late = list(parts[0]["names"]), rr > 10
    print("validation, rounds 11-50: " + " ".join(f"{n} {100 * val[late, j, 0].mean():.2f}" for j, n in enumerate(names)))
    print("BN output error of one-pass against layer-wise (largest layer): " + " ".join(
        f"b={b} {err[late, j, :, :, 0].max(-1).mean():.2f}" for j, b in enumerate(B_SIZES)))
    q, r1 = names.index("Q"), names.index("R1")
    print(f"layer-wise above one-pass with single windows in {np.sum(val[late, q, 0] > val[late, r1, 0])} of "
          f"{late.sum()} rounds, by {100 * (val[late, q, 0] - val[late, r1, 0]).mean():.1f} points")
    if os.path.exists(f"{OUT}/btest.npz"):
        d = np.load(f"{OUT}/btest.npz")
        for n in ("Q", "R32", "R4", "R1"):
            s = np.mean([scores(confusion(d[f"{ds}|labels"], d[f"{ds}|{n}"])) for ds in DS], 0)
            print(f"test fold {K} {n:<4} ACC {100 * s[0]:.2f} MF1 {100 * s[1]:.2f} kappa {100 * s[2]:.2f}")
    b = np.load(f"{OUT}/budget.npz")
    e = b["err"][..., 0].max(-1).mean((0, 2, 3))
    print(f"budget: pool sizes {b['pool'].tolist()}; error " + " ".join(f"N={n} {x:.3f}" for n, x in zip(b["budgets"], e))
          + f"; slope {linregress(np.log(b['budgets']), np.log(e)).slope:.2f}; validation "
          + " ".join(f"{100 * x:.2f}" for x in b["val"].mean((0, 2))))
    if os.path.exists(f"{OUT}/perclient.npz"):
        d = np.load(f"{OUT}/perclient.npz")
        print("per-client selection: " + " ".join(f"{ds} {d[f'own|{ds}'] - d[f'pooled|{ds}']:+.2f}" for ds in DS)
              + f" | average {np.mean([d[f'pooled|{ds}'] for ds in DS]):.2f} -> {np.mean([d[f'own|{ds}'] for ds in DS]):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("timing", "bsweep", "btest", "budget", "perclient", "summary"), required=True)
    p.add_argument("--rounds", help="A-B (inclusive)")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lo, hi = map(int, a.rounds.split("-")) if a.rounds else (1, ROUNDS)
    rounds = list(range(lo, hi + 1))
    main(a.part, {"timing": [r for r in (25, 50) if lo <= r <= hi], "budget": [r for r in BUDGET_ROUNDS if lo <= r <= hi]}
         .get(a.part, rounds))
