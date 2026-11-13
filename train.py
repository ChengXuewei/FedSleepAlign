"""Training and testing of every run in the paper.

Schemes (--scheme):
    fedbn    FedBN: the BN layers stay local, all other parameters are averaged (Eq. 1)
    fedavg   FedAvg: all parameters and BN statistics are averaged
    hbn      FedAvg with Hybrid BN (Chen et al., ICML 2025), following the official FedHBN code
    local    one client trains alone (--client 0..4)
    central  the training windows of all clients pooled into one training set
Scalings (--scaling): none, robust (PRS), zscore, and the PRS variants mean_iqr, med_sd, noclip.

Every round trains one local epoch per client, aggregates, and evaluates the model of every client with three kinds of
BN statistics on its validation windows: S, stored during local training; R, re-estimated in one pass; Q, re-estimated
layer by layer (PAR). Each kind selects its own top-10 rounds by the pooled validation accuracy, and the test applies
the majority vote of these rounds to every epoch of the test recordings of fold k. The crossed ensembles S_R, S_Q
(rounds of S with the statistics of R or Q) and R_S, Q_S give Tab. 2. Hybrid BN evaluates its global statistics (S).
With --val dual, every client also holds out 10% of its training subjects, and every kind is selected on them as well
("-subj"). FedBN and FedAvg log the deviations of Tab. 5 and the weight drift after every round.

    python train.py --scheme fedbn --scaling robust --seed 42 --fold 0     (FedSleepAlign is fedbn/robust with Q)
writes RESULTS/preds/<run>_fold<k>.npz (test predictions) and RESULTS/rounds/<run>_fold<k>.npz (per-round logs),
where <run> = <scheme>_<scaling>_s<seed>[_p<split>][_dual][_c<client>]. A run resumes from its last round; with
--budget MIN it exits with code 3 when the next round would exceed MIN minutes (to continue in a new job)."""
import argparse, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS; set before CUDA starts
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from fedsleepalign import bn
from fedsleepalign.config import BATCH, CAL_BATCHES, DATABASES, RESULTS, ROUNDS, SEQLEN, TOP_M
from fedsleepalign.data import SCALINGS, Windows, load_channels, make_client, split_halves, subject_of, test_windows
from fedsleepalign.evaluate import epoch_accuracy, majority, predict, reconstruct
from fedsleepalign.federated import device, fed_average, load, local_epoch, save, set_seed, sq_norm, to_cpu
from fedsleepalign.model import build_model

torch.set_num_threads(4)
HBN_MOMENTUM = 0.02                  # the official 0.002 for 500 rounds, scaled to 50 rounds
CROSSED = {"S_R": ("S", "R"), "S_Q": ("S", "Q"), "R_S": ("R", "S"), "Q_S": ("Q", "S")}    # (rounds of, statistics)
STAT_KEYS = ("running_mean", "running_var", "num_batches_tracked")


def run_name(a):
    return (f"{a.scheme}_{a.scaling}_s{a.seed}" + (f"_p{a.split}" if a.split != 42 else "")
            + ("_dual" if a.val == "dual" else "") + (f"_c{a.client}" if a.scheme == "local" else "")
            + ("_smoke" if a.smoke else ""))


def build_clients(a, dbs, halves, k):
    """The training halves of the clients and the initial model. FedBN and FedAvg draw the clients in turn from one
    generator; runs that train a client alone, pool the clients or hold out subjects draw every client from its own
    generator (seed, client), so its data do not depend on the other clients. Both orders reproduce the paper's runs."""
    per_client = a.scheme in ("hbn", "local", "central") or a.val == "dual"
    clients = []
    for ds in dbs:
        if per_client:
            set_seed(1000 * a.seed + list(DATABASES).index(ds))
        elif not clients:
            set_seed(a.seed)
        clients.append(make_client(ds, halves[ds][1 - k], a.scaling, a.val))
    if per_client:
        set_seed(a.seed)
    return clients, build_model(a.scheme, SEQLEN).state_dict()


def train(a, run, k, work, budget):
    """Trains fold k and returns the selected rounds; the selected models are kept in work/fold<k>_rNN.pt."""
    rounds, top_m = (3, 2) if a.smoke else (ROUNDS, TOP_M)
    halves = split_halves(a.split)
    if a.smoke:
        halves = {ds: [h[:4] for h in hs] for ds, hs in halves.items()}
    dbs = list(DATABASES) if a.scheme != "local" else [list(DATABASES)[a.client]]
    ways = ("S",) if a.scheme == "hbn" else ("S", "R", "Q")
    vsets = ("win", "subj") if a.val == "dual" else ("win",)
    keys = [w if v == "win" else f"{w}-subj" for v in vsets for w in ways]
    federated, t0 = a.scheme in ("fedbn", "fedavg"), time.time()
    ckpt = f"{work}/fold{k}.pt"
    ck = load(ckpt)
    if ck is not None and ck["round"] == rounds:
        return ck["best"], halves, dbs, keys
    model = build_model(a.scheme, SEQLEN).to(device)
    layers = [n for n, _ in bn.norm_layers(model)]
    bn_keys = [f"{n}.{key}" for n, m in bn.norm_layers(model) for key in m.state_dict()]
    own_keys = {"fedbn": bn_keys, "hbn": [f"{n}.factor" for n in layers]}.get(a.scheme, [])
    avg_keys = [key for key in model.state_dict() if key not in own_keys
                and not (a.scheme == "hbn" and key.rsplit(".", 1)[1].startswith(("global_", "running_")))]
    param_keys = [key for key in avg_keys if not key.endswith(STAT_KEYS)]
    first_keys = [key for key in param_keys if key.startswith(("cnn.0.conv1.", "cnn.0.downsample.0."))]

    clients, init = build_clients(a, dbs, halves, k)
    units = [c.train for c in clients] if a.scheme != "central" else [ConcatDataset([c.train for c in clients])]
    loaders = [DataLoader(u, batch_size=BATCH, shuffle=True) for u in units]
    vloaders = [{v: DataLoader(d, batch_size=BATCH) for v, d in c.val.items()} for c in clients]
    sizes = [len(u) for u in units]
    if ck is None:
        for ds, c in zip(dbs, clients):
            print(f"fold {k} | {ds:<13} | train windows {len(c.train)} | validation "
                  + ", ".join(f"{v} {len(d.dataset)}" for v, d in vloaders[dbs.index(ds)].items())
                  + f" | train subjects {len({subject_of(ds, n) for n in halves[ds][1 - k]})}")
        init = {key: v.to(device) for key, v in init.items()}
        ck = dict(round=0, shared=to_cpu(init), own=[to_cpu({key: init[key] for key in own_keys}) for _ in units],
                  best={key: {} for key in keys}, best_val={key: [0.0] * top_m for key in keys}, diag=[])
    shared = {key: v.to(device) for key, v in ck["shared"].items()}
    own = [{key: v.to(device) for key, v in o.items()} for o in ck["own"]]
    best, best_val, diag = ck["best"], {key: np.array(v) for key, v in ck["best_val"].items()}, ck["diag"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0]).to(device))
    round_time, wdir = 0.0, f"{RESULTS}/weights/{run}_fold{k}"

    for r in range(ck["round"], rounds):
        if budget and round_time and time.time() - t0 + 1.3 * round_time > budget:
            print(f"(stopped after round {r} to continue in a new job)")
            sys.exit(3)
        t_round = time.time()
        set_seed(a.seed + 1 + r)                     # a round's randomness depends only on its index
        prev = shared
        states = [local_epoch(model, {**shared, **o}, l, criterion) for o, l in zip(own, loaders)]
        t_local = time.time() - t_round
        shared = {**shared, **fed_average(states, sizes, avg_keys)}
        own = [{key: s[key] for key in own_keys} for s in states]

        t = time.time()                              # the evaluation statistics of every client
        batches = [bn.draw_batches(l, CAL_BATCHES * (len(clients) if a.scheme == "central" else 1)) for l in loaders]
        dev = []
        if a.scheme == "hbn":
            coll = [bn.hbn_collect(model, {**shared, **o}, b) for o, b in zip(own, batches)]
            spec = {"S": [bn.hbn_global(coll, layers, shared, 1.0)] * len(clients)}
            shared.update(bn.hbn_global(coll, layers, shared, HBN_MOMENTUM))    # for the next round's training
        else:
            spec = {"S": own if a.scheme == "fedbn" else [{}] * len(units), "R": [], "Q": []}
            for o, b, s in zip(own, batches, states):
                spec["R"].append(bn.reestimate_onepass(model, {**shared, **o}, b))
                spec["Q"].append(bn.reestimate(model, {**shared, **o}, b))
                if federated:                        # Tab. 5: residual, aggregation, total, one-pass
                    q_local = bn.reestimate(model, s, b)
                    stored = {**shared, **o}
                    dev.append([bn.deviation(s, q_local, layers), bn.deviation(q_local, spec["Q"][-1], layers),
                                bn.deviation(stored, spec["Q"][-1], layers), bn.deviation(spec["R"][-1], spec["Q"][-1], layers)])
            spec = {w: x * (len(clients) // len(x)) for w, x in spec.items()}  # central: one set for all clients
        t_stats = time.time() - t

        acc = {key: [] for key in keys}
        for w in ways:
            for j, vl in enumerate(vloaders):
                model.load_state_dict({**shared, **spec[w][j]})
                for v in vsets:
                    acc[w if v == "win" else f"{w}-subj"].append(epoch_accuracy(model, vl[v]))
        val = {key: np.average(acc[key], weights=[len(vl["subj" if key.endswith("-subj") else "win"].dataset)
                                                   for vl in vloaders]) for key in keys}
        for key in keys:                             # every evaluation keeps its own top-M rounds
            if val[key] > best_val[key].min():
                j = int(best_val[key].argmin())
                best_val[key][j], best[key][j] = val[key], r + 1
        needed = {rr for key in keys for rr in best[key].values()}
        if r + 1 in needed:
            save(f"{work}/fold{k}_r{r + 1:02d}.pt", shared=to_cpu(shared), spec={w: [to_cpu(x) for x in spec[w]]
                                                                                 for w in ways})
        if a.save_weights == "all":                  # every round, for analysis/calibration.py
            os.makedirs(wdir, exist_ok=True)
            save(f"{wdir}/r{r + 1:02d}.pt", shared=to_cpu(shared), spec={w: [to_cpu(x) for x in spec[w]] for w in ways})
        for f in os.listdir(work):
            if f.startswith(f"fold{k}_r") and int(f[len(f"fold{k}_r"):][:2]) not in needed:
                os.remove(f"{work}/{f}")
        d = dict(val=[val[key] for key in keys], acc=[acc[key] for key in keys], t_local=t_local, t_stats=t_stats)
        if federated:
            d.update(dev=dev, upd=[sq_norm(s, prev, param_keys) ** 0.5 for s in states],
                     drift=[sq_norm(shared, s, param_keys) ** 0.5 for s in states],
                     drift1=[sq_norm(shared, s, first_keys) ** 0.5 for s in states],
                     mean_upd=sq_norm(shared, prev, param_keys) ** 0.5)
        if a.scheme == "hbn":
            d["hbn_a"] = [[torch.sigmoid(o[f"{n}.factor"]).mean().item() for n in layers] for o in own]
        diag.append(d)
        print(f"round {r + 1:02d} | " + " ".join(f"{key} {100 * val[key]:.2f}" for key in keys)
              + f" | local {t_local:.0f}s, statistics {t_stats:.1f}s", flush=True)
        save(ckpt, round=r + 1, shared=to_cpu(shared), own=[to_cpu(o) for o in own], best=best,
             best_val={key: v.tolist() for key, v in best_val.items()}, diag=diag)
        round_time = max(round_time, time.time() - t_round)

    print("selected rounds | " + " | ".join(f"{key} {sorted(best[key].values())}" for key in keys))
    save(ckpt, round=rounds, best=best, diag=diag)
    os.makedirs(f"{RESULTS}/rounds", exist_ok=True)
    extra = {f: np.array([d[f] for d in diag]) for f in ("dev", "upd", "drift", "drift1", "mean_upd", "hbn_a")
             if f in diag[0]}
    np.savez(f"{RESULTS}/rounds/{run}_fold{k}.npz", keys=np.array(keys), clients=np.array(dbs),
             val=np.array([d["val"] for d in diag]), acc=np.array([d["acc"] for d in diag]),
             t_local=np.array([d["t_local"] for d in diag]), t_stats=np.array([d["t_stats"] for d in diag]),
             best=np.array([sorted(best[key].values()) for key in keys]), **extra)
    return best, halves, dbs, keys


def test(a, run, k, work, best, halves, dbs, keys):
    """Majority votes of the selected rounds on every epoch of the test recordings of fold k."""
    tests = {key: (key, key.split("-")[0]) for key in keys}
    if {"R", "Q"} <= set(keys):
        tests.update(CROSSED)
    members = sorted({(rr, w) for rk, w in tests.values() for rr in best[rk].values()})
    kept = {rr: load(f"{work}/fold{k}_r{rr:02d}.pt") for rr in {m[0] for m in members}}
    model, out = build_model(a.scheme, SEQLEN).to(device), {}
    for j, ds in enumerate(dbs):
        wx, wy, meta, lengths = test_windows(load_channels(ds, halves[ds][k], a.scaling))
        loader = DataLoader(Windows(wx, wy), batch_size=BATCH)
        pred = {}
        for rr, w in members:
            model.load_state_dict({**kept[rr]["shared"], **kept[rr]["spec"][w][j]})
            pred[rr, w], labels_w = predict(model, loader)
        for name, (rk, w) in tests.items():
            preds, labels = reconstruct(majority(np.stack([pred[rr, w] for rr in sorted(best[rk].values())])),
                                        labels_w, meta, lengths)
            out[f"{ds}|{name}"] = preds
        out[f"{ds}|labels"], out[f"{ds}|names"], out[f"{ds}|lengths"] = labels, np.array(halves[ds][k]), lengths
        print(f"test {ds:<13} | " + " ".join(f"{name} {100 * np.mean(out[f'{ds}|{name}'] == labels):.2f}"
                                             for name in tests), flush=True)
    os.makedirs(f"{RESULTS}/preds", exist_ok=True)
    np.savez(f"{RESULTS}/preds/{run}_fold{k}.npz", **out)
    if a.save_weights == "selected":                 # the selected models, for analysis/artifact.py and delimit.py
        os.makedirs(f"{RESULTS}/weights/{run}_fold{k}", exist_ok=True)
        for rr in kept:
            os.replace(f"{work}/fold{k}_r{rr:02d}.pt", f"{RESULTS}/weights/{run}_fold{k}/r{rr:02d}.pt")
    for f in os.listdir(work):
        if f.startswith((f"fold{k}_r", f"fold{k}.pt")):
            os.remove(f"{work}/{f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scheme", choices=("fedbn", "fedavg", "hbn", "local", "central"), default="fedbn")
    p.add_argument("--scaling", choices=tuple(SCALINGS), default="robust")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, choices=(0, 1), required=True)
    p.add_argument("--split", type=int, default=42, help="subject split; 42 is that of the paper")
    p.add_argument("--val", choices=("window", "dual"), default="window")
    p.add_argument("--client", type=int, choices=range(len(DATABASES)), help="the client of --scheme local")
    p.add_argument("--save-weights", choices=("none", "selected", "all"), default="none")
    p.add_argument("--budget", type=float, default=0, help="minutes this job may use (0: unlimited)")
    p.add_argument("--smoke", action="store_true", help="four recordings per half and client, 3 rounds, top-2")
    a = p.parse_args()
    assert (a.client is not None) == (a.scheme == "local"), "--client goes with --scheme local"
    run = run_name(a)
    work = f"{RESULTS}/tmp/{run}"
    os.makedirs(work, exist_ok=True)
    print(f"===== {run} fold {a.fold} on {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")
    best, halves, dbs, keys = train(a, run, a.fold, work, 60 * a.budget)
    test(a, run, a.fold, work, best, halves, dbs, keys)
