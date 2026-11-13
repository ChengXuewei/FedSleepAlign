"""Tab. 1, 2, 4 and 5 and the numbers of Sec. 3.3-3.4 and App. B.1-B.3, from the FedBN runs with the scalings none,
zscore and robust (PRS) for the five seeds (train.py --scheme fedbn --scaling X --seed S --fold K).
    python analysis/ablation.py > results/summary/ablation.log"""
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

from common import DS, RESULTS, SEEDS, STAGES, SUMMARY, confusion, holm, pm, pooled, per_seed, subjects

SC = ("none", "zscore", "robust")
KEYS = ("S", "R", "Q", "S_R", "S_Q", "R_S", "Q_S")
res = {(sc, key): per_seed([f"fedbn_{sc}_s{s}" for s in SEEDS], key) for sc in SC for key in KEYS}  # (seeds, 6, 8)
acc = lambda sc, key: res[sc, key][:, 5, 0]

print("=== Tab. 1: accuracy per client and averages (ACC, MF1, kappa) over the clients, mean +- SD over seeds (%)")
for key in ("S", "Q", "R"):
    for sc in SC:
        r = res[sc, key]
        print(f"{sc:<7}{key:<2}" + " | ".join(pm(r[:, d, 0]) for d in range(5)) + " || "
              + " | ".join(pm(r[:, 5, m]) for m in range(3)))

print("\n=== Tab. 2: models selected with S or Q (rows) tested with S or Q statistics (columns)")
for sc in ("none", "robust"):
    for sel, (a, b) in (("S", ("S", "S_Q")), ("Q", ("Q_S", "Q"))):
        print(f"{sc:<7} selected with {sel}: stored {pm(acc(sc, a))}  PAR {pm(acc(sc, b))}")
    st1, st2 = acc(sc, "S_Q") - acc(sc, "S"), acc(sc, "Q") - acc(sc, "Q_S")       # statistics switched, fixed models
    se1, se2 = acc(sc, "Q") - acc(sc, "S_Q"), acc(sc, "Q_S") - acc(sc, "S")       # selection switched
    total = acc(sc, "Q") - acc(sc, "S")
    print(f"        statistics switched {st1.mean():+.2f} (rounds of S) / {st2.mean():+.2f} (rounds of Q); selection "
          f"{se1.mean():+.2f} (PAR) / {se2.mean():+.2f} (stored); interaction {(st2 - st1).mean():+.2f}; share of the "
          f"gain {st1.mean() / total.mean():.0%} (statistics first) / {st2.mean() / total.mean():.0%} (selection first)")

print("\n=== Tab. 4: paired comparisons of the average accuracy; Holm over 10 comparisons x 3 metrics; d_z; subject "
      "bootstrap 95% (pointwise)")
C = [(("robust", "S"), ("none", "S")), (("zscore", "S"), ("none", "S")), (("robust", "S"), ("zscore", "S")),
     (("robust", "Q"), ("none", "Q")), (("zscore", "Q"), ("none", "Q")), (("robust", "Q"), ("zscore", "Q")),
     (("none", "Q"), ("none", "S")), (("zscore", "Q"), ("zscore", "S")), (("robust", "Q"), ("robust", "S")),
     (("robust", "Q"), ("none", "S"))]
p = np.array([[ttest_rel(res[a][:, 5, m], res[b][:, 5, m]).pvalue for m in range(3)] for a, b in C])
adj = holm(p.ravel()).reshape(p.shape)
SUBJ = {(sc, key): subjects(f"fedbn_{sc}_s{{seed}}", key) for sc in SC for key in ("S", "Q")}
rng = np.random.default_rng(0)
boot = {ds: [rng.integers(0, len(n), len(n)) for _ in range(2000)] for ds, (_, n, _) in SUBJ["none", "S"].items()}
B = {key: 100 * np.mean([[c[i].sum(0).mean() / n[i].sum() for i in boot[ds]] for ds, (c, n, _) in SUBJ[key].items()],
                        axis=0) for key in SUBJ}
for (a, b), q in zip(C, adj):
    d, e = res[a][:, 5, :3] - res[b][:, 5, :3], B[a] - B[b]
    print(f"{'-'.join(a)} vs {'-'.join(b):<10} per seed " + " ".join(f"{x:5.2f}" for x in d[:, 0]) + " | "
          + " | ".join(pm(d[:, m]) for m in range(3)) + f" | Holm p {q[0]:.2g} (MF1 {q[1]:.2g}, kappa {q[2]:.2g})"
          + f" | d_z {d[:, 0].mean() / d[:, 0].std(ddof=1):.1f} | bootstrap [{np.percentile(e, 2.5):.2f}, "
          f"{np.percentile(e, 97.5):.2f}]")
inter = (acc("robust", "Q") - acc("robust", "S")) - (acc("none", "Q") - acc("none", "S"))
print(f"interaction PRS x PAR {pm(inter)} (p = {ttest_1samp(inter, 0).pvalue:.2g}); combined gain "
      f"{(acc('robust', 'Q') - acc('none', 'S')).mean():.2f} vs separate gains "
      f"{(acc('robust', 'S') - acc('none', 'S')).mean() + (acc('none', 'Q') - acc('none', 'S')).mean():.2f}")
for name, (a, b) in (("PRS alone", (("robust", "S"), ("none", "S"))), ("PAR alone", (("none", "Q"), ("none", "S"))),
                     ("PRS - z-score with PAR", (("robust", "Q"), ("zscore", "Q")))):
    d = res[a][:, :5, 0] - res[b][:, :5, 0]
    other = d[:, [0, 2, 3, 4]].mean(1)
    print(f"{name:<23} per client " + " ".join(f"{x:+.2f}" for x in d.mean(0)) + f" | without ISRUC {pm(other)} "
          f"(p = {ttest_1samp(other, 0).pvalue:.2g}) | ISRUC share of the gain {d[:, 1].mean() / d.sum(1).mean():.0%}")
print("subjects weighted equally (average accuracy): " + " ".join(
    f"{sc} {key} {100 * np.mean([(c / n[:, None]).mean(1).mean() for c, n, _ in SUBJ[sc, key].values()]):.2f}"
    for sc in SC for key in ("S", "Q")))

print("\n=== ISRUC recordings with an F3 IQR below 5 uV (low range) and the others, mean +- SD over seeds")
low = np.load(f"{SUMMARY}/recordings.npz")["isruc_low"]
for sc in ("none", "robust"):
    for key in ("S", "Q"):
        lo, hi = [], []
        for s in SEEDS:
            f = [np.load(f"{RESULTS}/preds/fedbn_{sc}_s{s}_fold{k}.npz") for k in (0, 1)]
            rec = np.concatenate([np.repeat(x["ISRUC|names"], x["ISRUC|lengths"]) for x in f])
            ok = np.concatenate([x[f"ISRUC|{key}"] == x["ISRUC|labels"] for x in f])
            is_low = np.isin(rec, low)
            lo.append(100 * ok[is_low].mean()); hi.append(100 * ok[~is_low].mean())
        print(f"{sc:<7}{key}: low range {pm(lo)} ({len(low)} recordings), others {pm(hi)}")

print("\n=== Validation per round (Fig. 4), selection and Tab. 5 deviations, rounds 11-50, ten runs per scaling")
for sc in SC:
    R = [np.load(f"{RESULTS}/rounds/fedbn_{sc}_s{s}_fold{k}.npz") for s in SEEDS for k in (0, 1)]
    keys = list(R[0]["keys"]); v = 100 * np.array([r["val"] for r in R])                   # (runs, rounds, keys)
    S, Q = v[..., keys.index("S")], v[..., keys.index("Q")]
    best = np.array([r["best"] for r in R])
    print(f"{sc:<7} PAR higher in {np.sum(Q > S)}/{Q.size} rounds; gap Q - S {np.mean(Q[:, 10:] - S[:, 10:]):.2f}; "
          f"selected rounds in 41-50: S {np.mean(best[:, keys.index('S')] > 40):.0%}, Q "
          f"{np.mean(best[:, keys.index('Q')] > 40):.0%}; drift {np.mean([r['drift'][10:].mean() for r in R]):.4f}")
    dev = np.array([r["dev"][10:] for r in R])                                              # (runs, 40, 5, 4, 6, 2)
    for j, what in enumerate(("residual", "aggregation", "total", "one-pass")):
        print(f"        {what:<12} mean difference (SD) " + " ".join(f"{x:.3f}" for x in dev[..., j, :, 0].mean((0, 1, 2)))
              + " | variance ratio " + " ".join(f"{x:.2f}" for x in dev[..., j, :, 1].mean((0, 1, 2))))
for a, b in (("robust", "none"),):
    da = [np.load(f"{RESULTS}/rounds/fedbn_{a}_s{s}_fold{k}.npz")["drift"][10:].mean() for s in SEEDS for k in (0, 1)]
    db = [np.load(f"{RESULTS}/rounds/fedbn_{b}_s{s}_fold{k}.npz")["drift"][10:].mean() for s in SEEDS for k in (0, 1)]
    print(f"drift {a} / {b}: {np.mean(da) / np.mean(db):.3f}")

print("\n=== One-pass (R) against layer-wise (Q) statistics: accuracy, and Q - R per client (mean +- SD over seeds)")
for sc in ("none", "robust"):
    d = res[sc, "Q"][:, :, 0] - res[sc, "R"][:, :, 0]
    print(f"{sc:<7} R average {pm(acc(sc, 'R'))}, Q - R average {pm(d[:, 5])} | per client "
          + " ".join(f"{pm(d[:, i])}" for i in range(5)))

print("\n=== Stage recall (%) pooled over seeds and folds, and the share of confusions of FedSleepAlign")
for sc, key in (("none", "S"), ("robust", "S"), ("none", "Q"), ("robust", "Q")):
    print(f"{sc:<7}{key} " + " | ".join(f"{ds} " + " ".join(f"{c} {res[sc, key][:, i, 3 + j].mean():.1f}"
                                                            for j, c in enumerate(STAGES)) for i, ds in enumerate(DS)))
cm = sum(confusion(*pooled(f"fedbn_robust_s{s}", "Q", ds)) for s in SEEDS for ds in DS)
err = cm - np.diag(np.diag(cm))
print(f"FedSleepAlign errors: N2 <-> N3 {(err[2, 3] + err[3, 2]) / err.sum():.0%}, N1 <-> W/N2/REM "
      f"{(err[1, [0, 2, 4]].sum() + err[[0, 2, 4], 1].sum()) / err.sum():.0%}")
np.savez(f"{SUMMARY}/ablation.npz", **{f"{sc}|{key}": res[sc, key] for sc, key in res})
