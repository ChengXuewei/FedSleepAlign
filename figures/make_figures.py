"""All data figures of the paper, drawn with matplotlib as PDF from the outputs of train.py and analysis/*.py
(Fig. 1 is the TikZ diagram FL_structure.tex, Fig. 3 comes from data_augmentation.py). Text is set by LaTeX in Times,
as in cvpr.sty, and every figure has the width at which the paper includes it, so font sizes are final.
    comparison.pdf      fig:comparison   results/summary/comparison.npz       (analysis/comparison.py)
    round_accuracy.pdf  fig:rounds       results/rounds/fedbn_{none,robust}_s*_fold*.npz
    calibration.pdf     fig:calibration  results/calibration/                 (analysis/calibration.py)
    selection.pdf       fig:selection    results/summary/selection.npz        (analysis/selection.py)
    confusion.pdf       fig:confusion    results/preds/fedbn_{none,robust}_s*_fold*.npz
    scaling.pdf         fig:scaling      results/summary/scaling.npz          (analysis/scaling.py)
    python figures/make_figures.py [name ...]      (all figures without arguments; output in $FSA_FIGURES or figures/pdf)"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fedsleepalign.config import RESULTS                                          # noqa: E402

SUMMARY, CALIB = f"{RESULTS}/summary", f"{RESULTS}/calibration"
FIG = os.environ.get("FSA_FIGURES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf"))
os.makedirs(FIG, exist_ok=True)
DS = ["SleepEDF-78", "ISRUC", "PhysioNet2018", "HMC", "SHHS"]
DS_TEX = ["SleepEDF-78", "ISRUC", "PhysioNet-2018", "HMC", "SHHS"]
SEEDS = [42, 142, 242, 342, 442]
COL, TEXT_W = 3.28, 6.875                         # column and text width of cvpr.sty (in)
PAR, STORED, HBN, CENT, INK, INK2, GRID = "#27965a", "#898781", "#3468af", "#3a3936", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["Times"], "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.edgecolor": STORED, "axes.linewidth": 0.5, "xtick.color": STORED, "ytick.color": STORED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.titlepad": 3, "axes.labelpad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.4,
    "lines.solid_capstyle": "round", "legend.frameon": False, "savefig.pad_inches": 0.01,
})


def save(fig, name, width=COL):
    """Save with a tight bounding box whose width is `width`, the width at which main.tex includes the figure, so that
    LaTeX does not rescale it and the font sizes stay as set."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(4):
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        fig.set_figwidth(fig.get_figwidth() + width - w)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def rows(ax, labels, grid="x"):
    """Categorical y axis with the first label on top."""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis=grid)
    ax.set_axisbelow(True)


def mark(ax, x, y, kind, **kw):
    style = dict(stored=dict(marker="o", mfc="white", mec=STORED, mew=0.8, ms=4.2),
                 par=dict(marker="o", mfc=PAR, mec="white", mew=0.4, ms=4.6),
                 hbn=dict(marker="D", mfc=HBN, mec="white", mew=0.4, ms=4.0),
                 cent=dict(marker="s", mfc=CENT, mec="white", mew=0.4, ms=3.8),
                 local=dict(marker="x", color=INK, mew=1.0, ms=4.4))[kind]
    ax.plot(x, y, ls="none", zorder=3, **{**style, **kw})


# ── fig:comparison ───────────────────────────────────────────────────────────────────────────────────────────────────
def comparison():
    S = dict(np.load(f"{SUMMARY}/comparison.npz"))
    acc = lambda k: S[k][:, 0]
    schemes = [("Local", "Local (stored)", "Local (PAR)"), ("Centralized", "Central (stored)", "Central (PAR)"),
               ("FedAvg", "FedAvg (stored)", "FedAvg (PAR)"), ("Hybrid BN", "HybridBN", None),
               ("FedBN", "FedBN+PRS (stored)", "FedSleepAlign")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.08), gridspec_kw=dict(wspace=0.85))
    rows(a, [s[0] for s in schemes])
    for i, (name, s, p) in enumerate(schemes):
        if p is None:
            mark(a, acc(s).mean(), i, "hbn")
            continue
        a.plot([acc(s).mean(), acc(p).mean()], [i, i], color=STORED, lw=0.6, zorder=2)
        mark(a, acc(s).mean(), i, "stored"); mark(a, acc(p).mean(), i, "par")
    a.annotate("FedSleepAlign", (acc("FedSleepAlign").mean(), 4), xytext=(0, 4.5), textcoords="offset points",
               ha="center", va="bottom", fontsize=5.5, color=INK2)
    a.set_xlim(78, 83); a.set_xticks(range(78, 84)); a.set_xlabel(r"Accuracy (\%)"); a.set_title("(a) Average over clients")
    rows(b, DS_TEX)
    loc, cen, fsa = acc("Local (PAR)"), acc("Central (PAR)"), acc("FedSleepAlign")
    for i in range(5):
        b.plot([min(loc[i], cen[i], fsa[i]), max(loc[i], cen[i], fsa[i])], [i, i], color=GRID, lw=0.8, zorder=1)
    mark(b, loc, np.arange(5) - 0.17, "local"); mark(b, cen, np.arange(5) + 0.17, "cent"); mark(b, fsa, np.arange(5), "par")
    b.set_xlim(76, 88); b.set_xticks(range(76, 89, 4)); b.set_xlabel(r"Accuracy (\%)"); b.set_title("(b) Accuracy per client")
    handles = [Line2D([], [], ls="none", marker="o", mfc="white", mec=STORED, mew=0.8, ms=4),
               Line2D([], [], ls="none", marker="o", mfc=PAR, mec="white", ms=4.4),
               Line2D([], [], ls="none", marker="D", mfc=HBN, mec="white", ms=3.8),
               Line2D([], [], ls="none", marker="x", color=INK, mew=1.0, ms=4.2),
               Line2D([], [], ls="none", marker="s", mfc=CENT, mec="white", ms=3.6)]
    fig.legend(handles, ["stored", "PAR", "Hybrid BN", "local", "centralized"], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, -0.1), handletextpad=0.2, columnspacing=0.9)
    save(fig, "comparison")


# ── fig:rounds ───────────────────────────────────────────────────────────────────────────────────────────────────────
def round_accuracy():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.75), sharex=True, gridspec_kw=dict(hspace=0.42))
    t = np.arange(1, 51)
    for ax, sc, title in zip(axes, ("none", "robust"), ("(a) FedBN", "(b) FedBN with PRS")):
        logs = [np.load(f"{RESULTS}/rounds/fedbn_{sc}_s{s}_fold{k}.npz") for s in SEEDS for k in range(2)]
        keys = list(logs[0]["keys"])
        val = 100 * np.array([d["val"][:, [keys.index("S"), keys.index("R"), keys.index("Q")]] for d in logs])
        for col, color, ls, label in ((0, STORED, (0, (3, 1.5)), "stored"), (2, PAR, "-", "PAR")):
            m, sd = val[:, :, col].mean(0), val[:, :, col].std(0, ddof=1)
            ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.16, lw=0)
            ax.plot(t, m, color=color, ls=ls, lw=1.0)
            ax.text(51, m[-1], label, fontsize=7, color=INK2, va="center")
        gap = (val[:, 10:, 2] - val[:, 10:, 0]).mean()
        ax.text(49.5, 61.5, rf"mean gap in rounds 11--50: ${gap:+.1f}$ points", fontsize=7, ha="right", va="bottom")
        ax.set_ylim(60, 85); ax.set_xlim(1, 50); ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_ylabel(r"Val.\ accuracy (\%)"); ax.set_title(title); ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[1].set_xlabel("Round")
    save(fig, "round_accuracy")


# ── fig:calibration ──────────────────────────────────────────────────────────────────────────────────────────────────
def calibration():
    P = [np.load(f"{CALIB}/{f}") for f in sorted(os.listdir(CALIB)) if f.startswith("bsweep_r")]
    rounds, val = np.concatenate([p["rounds"] for p in P]), np.concatenate([p["val"] for p in P])
    names, late = list(P[0]["names"]), rounds > 10
    B = [int(n[1:]) for n in names if n.startswith("R")]
    one = np.array([100 * val[late, names.index(f"R{b}"), 0].mean() for b in B])
    lay = 100 * val[late, names.index("Q"), 0].mean()
    d = np.load(f"{CALIB}/budget.npz")
    N, err = d["budgets"], d["err"][..., 0].max(axis=-1).mean(axis=(0, 2, 3))
    slope = np.polyfit(np.log(N), np.log(err), 1)[0]
    ref = np.exp(np.mean(np.log(err) + 0.5 * np.log(N))) * N ** -0.5
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw=dict(wspace=0.55))
    a.axhline(lay, color=PAR, lw=1.0); a.text(B[0], lay + 0.15, "layer-wise", fontsize=7, color=INK2, va="bottom")
    a.plot(B, one, color=STORED, ls=(0, (3, 1.5)), lw=1.0, marker="o", ms=2.8, mfc=STORED, mec=STORED)
    a.text(B[0] * 1.25, one[0] - 0.25, "one-pass", fontsize=7, color=INK2, va="top")
    a.set_xscale("log", base=2); a.set_xticks(B, [str(b) for b in B]); a.minorticks_off()
    a.set_xlim(B[0] / 1.3, B[-1] * 1.3); a.set_ylim(75, 85)
    a.set_xlabel("Windows per batch"); a.set_ylabel(r"Val.\ accuracy (\%)"); a.set_title("(a) Batch size")
    a.grid(axis="y"); a.set_axisbelow(True)
    b.plot(N, ref, color=STORED, ls=":", lw=0.8); b.plot(N, err, color=PAR, lw=1.0, marker="o", ms=2.8)
    b.text(N[-2], ref[-2] * 0.82, r"slope $-1/2$", fontsize=7, color=INK2, ha="right", va="top")
    b.text(0.98, 0.98, rf"fitted slope ${slope:.2f}$", transform=b.transAxes, fontsize=7, color=INK2, ha="right", va="top")
    b.set_xscale("log"); b.set_yscale("log"); b.minorticks_off()
    b.set_xticks([32, 100, 320, 1000], ["32", "100", "320", "1000"]); b.set_yticks([0.02, 0.05, 0.1, 0.2], ["0.02", "0.05", "0.1", "0.2"])
    b.set_xlim(N[0] / 1.4, N[-1] * 1.4); b.set_ylim(0.017, 0.26)
    b.set_xlabel(r"Windows $N$"); b.set_ylabel("BN output error"); b.set_title("(b) Calibration budget")
    b.grid(axis="y"); b.set_axisbelow(True)
    save(fig, "calibration")


# ── fig:selection ────────────────────────────────────────────────────────────────────────────────────────────────────
def selection():
    S = dict(np.load(f"{SUMMARY}/selection.npz"))
    var = [(v, v) for v in ("FedBN", "FedBN + PRS", "FedBN + PAR", "FedSleepAlign")]
    m = lambda k: S[k][:, 0].mean()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_W, 1.55), gridspec_kw=dict(wspace=0.75))
    rows(a, [v[1] for v in var]); rows(b, [v[1] for v in var])
    for i, (k, _) in enumerate(var):
        a.plot(m(f"orig|{k}"), i - 0.15, ls="none", marker="x", color=INK2, mew=0.9, ms=4.4, zorder=3)
        mark(a, m(f"win|{k}"), i + 0.15, "stored")
        b.plot(m(f"subj|{k}") - m(f"win|{k}"), i, ls="none", marker="o", mfc=HBN, mec="white", mew=0.4, ms=4.6)
    a.set_xlim(73, 81); a.set_xticks(range(73, 82, 2)); a.set_xlabel(r"Accuracy (\%), held-out windows")
    a.set_title("(a) Subject splits")
    b.axvline(0, color=STORED, lw=0.5); b.set_xlim(-0.5, 0.5); b.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    b.set_xlabel(r"Held-out subjects $-$ windows (points)"); b.set_title("(b) Subject-disjoint selection")
    for i, ds in enumerate(DS):
        d = S[f"gain|{ds}"]
        c.boxplot(d, positions=[i], vert=False, widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor="#c9e5d4", edgecolor="#1f7a49", lw=0.6),
                  medianprops=dict(color="#1f7a49", lw=0.9), whiskerprops=dict(color="#1f7a49", lw=0.6),
                  capprops=dict(color="#1f7a49", lw=0.6))
        c.text(np.percentile(d, 95) + 0.8, i, rf"{100 * np.mean(d > 0):.0f}\%", fontsize=6, color=INK2, va="center")
    rows(c, DS_TEX)                                        # after boxplot, which resets the ticks
    c.axvline(0, color=STORED, lw=0.5); c.set_xlim(-10, 40); c.set_xticks(range(-10, 41, 10))
    c.set_xlabel(r"FedSleepAlign $-$ FedBN (points)"); c.set_title("(c) Gain per test subject")
    handles = [Line2D([], [], ls="none", marker="x", color=INK2, mew=0.9, ms=4.2),
               Line2D([], [], ls="none", marker="o", mfc="white", mec=STORED, mew=0.8, ms=4),
               Line2D([], [], ls="none", marker="o", mfc=HBN, mec="white", ms=4.4)]
    fig.legend(handles, ["original split", "new split, held-out windows", "new split, difference"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.36, -0.1), handletextpad=0.2, columnspacing=1.2)
    save(fig, "selection", TEXT_W)


# ── fig:confusion ────────────────────────────────────────────────────────────────────────────────────────────────────
def confusion():
    """(a) Recall-normalized confusion matrices of the four variants of Tab. 1, counts summed over the clients, folds and
    seeds; (b) the accuracy of layer-wise minus one-pass statistics per client, mean +- SD over the seeds; (c) the recall
    of every stage per client (the seeds share the test epochs, so pooling equals averaging over them)."""
    var = [("none", "S", "FedBN"), ("robust", "S", "FedBN + PRS"), ("none", "Q", "FedBN + PAR"),
           ("robust", "Q", "FedSleepAlign")]
    cm = {v[2]: np.zeros((5, 5)) for v in var}
    cmc = {(v[2], ds): np.zeros((5, 5)) for v in var for ds in DS}
    diff = {sc: np.zeros((len(SEEDS), 5)) for sc in ("none", "robust")}
    for si, s in enumerate(SEEDS):
        for sc in ("none", "robust"):
            F = [np.load(f"{RESULTS}/preds/fedbn_{sc}_s{s}_fold{k}.npz") for k in range(2)]
            for di, ds in enumerate(DS):
                y = np.concatenate([f[f"{ds}|labels"] for f in F])
                p = {t: np.concatenate([f[f"{ds}|{t}"] for f in F]) for t in ("S", "Q", "R")}
                diff[sc][si, di] = 100 * (np.mean(p["Q"] == y) - np.mean(p["R"] == y))
                for sc_v, t, name in var:
                    if sc_v == sc:
                        np.add.at(cm[name], (y, p[t]), 1); np.add.at(cmc[name, ds], (y, p[t]), 1)
    stages = ["W", "N1", "N2", "N3", "REM"]
    fig = plt.figure(figsize=(TEXT_W, 3.35))
    for i, (_, _, name) in enumerate(var):
        ax = fig.add_axes([0.035 + i * 0.158, 0.6, 0.125, 0.3])
        r = 100 * cm[name] / cm[name].sum(1, keepdims=True)
        ax.imshow(r, cmap="Greens", vmin=0, vmax=100)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{r[a, b]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if r[a, b] > 60 else INK)
        ax.set_xticks(range(5), stages, fontsize=6); ax.set_yticks(range(5), stages if i == 0 else [], fontsize=6)
        ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(name, fontsize=7.5); ax.set_xlabel("Predicted", fontsize=6.5, labelpad=1)
        if i == 0:
            ax.set_ylabel("True", fontsize=6.5, labelpad=1)
            ax.text(-0.42, 1.2, "(a)", transform=ax.transAxes, fontsize=8, va="bottom")
    ax = fig.add_axes([0.745, 0.6, 0.235, 0.3])
    rows(ax, DS_TEX)
    for sc, dy, color, label in (("none", -0.15, STORED, "FedBN"), ("robust", 0.15, PAR, "with PRS")):
        m, sd = diff[sc].mean(0), diff[sc].std(0, ddof=1)
        ax.errorbar(m, np.arange(5) + dy, xerr=sd, ls="none", marker="o", ms=3.2, color=color, mec="white", mew=0.3,
                    elinewidth=0.7, capsize=1.5, label=label)
    ax.axvline(0, color=STORED, lw=0.5); ax.set_xlim(-3, 1.5)
    ax.set_xlabel(r"Layer-wise $-$ one-pass (points)", fontsize=7); ax.set_title("(b) One-pass vs layer-wise", fontsize=7.5)
    ax.legend(loc="lower left", fontsize=6, handletextpad=0.1, borderaxespad=0.1)
    style = {"FedBN": dict(marker="o", mfc="white", mec=STORED, mew=0.8, ms=3.6),
             "FedBN + PRS": dict(marker="D", mfc="white", mec=STORED, mew=0.8, ms=3.2),
             "FedBN + PAR": dict(marker="o", mfc=PAR, mec="white", mew=0.3, ms=4.0),
             "FedSleepAlign": dict(marker="D", mfc=PAR, mec="white", mew=0.3, ms=3.6)}
    for d, (ds, label) in enumerate(zip(DS, DS_TEX)):
        ax = fig.add_axes([0.06 + d * 0.19, 0.12, 0.16, 0.3])
        for v, (_, _, name) in enumerate(var):
            m = cmc[name, ds]
            ax.plot(np.arange(5) + (v - 1.5) * 0.13, 100 * np.diag(m) / m.sum(1), ls="none", zorder=3, **style[name],
                    label=name)
        ax.set_xticks(range(5), stages, fontsize=6); ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25), [str(t) for t in range(0, 101, 25)] if d == 0 else [])
        ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_title(label, fontsize=7.5)
        if d == 0:
            ax.set_ylabel(r"Recall (\%)", fontsize=7)
            ax.text(-0.3, 1.12, "(c)", transform=ax.transAxes, fontsize=8, va="bottom")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.025),
               handletextpad=0.2, columnspacing=1.5)
    save(fig, "confusion", TEXT_W)
    for name, m in cm.items():
        print(name, "recall " + " ".join(f"{x:.1f}" for x in 100 * np.diag(m) / m.sum(1)))
    for ds in DS:
        print(ds, " | ".join(name + " " + " ".join(f"{x:.1f}" for x in 100 * np.diag(cmc[name, ds]) / cmc[name, ds].sum(1))
                             for _, _, name in var))
    for sc in diff:
        print(sc, "Q - R per client", " ".join(f"{a:+.2f}+-{b:.2f}" for a, b in zip(diff[sc].mean(0), diff[sc].std(0, ddof=1))))
    errs = cm["FedSleepAlign"] - np.diag(np.diag(cm["FedSleepAlign"]))
    print("FedSleepAlign share of errors N2<->N3 %.1f%%, N1<->(W,N2,REM) %.1f%%" % (
        100 * (errs[2, 3] + errs[3, 2]) / errs.sum(), 100 * (errs[1, [0, 2, 4]].sum() + errs[[0, 2, 4], 1].sum()) / errs.sum()))


# ── fig:scaling ──────────────────────────────────────────────────────────────────────────────────────────────────────
def scaling():
    """Scalings that change one operation of PRS at a time (analysis/scaling.py -> results/summary/scaling.npz): (a) accuracy
    with stored and PAR statistics, (b) margin over the z-score per quarter of tail weight with PAR statistics, (c)
    change of the accuracy on clean epochs when artifacts enter 2% or 5% of the epochs."""
    S = dict(np.load(f"{SUMMARY}/scaling.npz"))
    var = [("none", "none"), ("robust", "PRS"), ("mean_iqr", "mean, IQR"), ("med_sd", "median, SD"),
           ("noclip", "no clip"), ("zscore", "z-score")]
    color = dict(robust=PAR, mean_iqr=HBN, med_sd="#c0711c", noclip=CENT, zscore=STORED)
    shift = dict(robust=-0.06, mean_iqr=-0.02, med_sd=0.02, noclip=0.06, zscore=0.1)     # keeps coinciding lines visible
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_W, 1.65), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1]))
    rows(a, [v[1] for v in var])
    for i, (sc, _) in enumerate(var):
        if f"avg|{sc}|S" in S:
            st, pa = S[f"avg|{sc}|S"][0], S[f"avg|{sc}|Q"][0]
            a.plot([st, pa], [i, i], color=STORED, lw=0.6, zorder=2)
            mark(a, st, i, "stored"); mark(a, pa, i, "par")
    a.set_xlim(74, 81.5); a.set_xticks(range(74, 82, 2)); a.set_xlabel(r"Accuracy (\%)")
    a.set_title("(a) Accuracy, seed 42")
    a.legend([Line2D([], [], ls="none", marker="o", mfc="white", mec=STORED, mew=0.8, ms=4),
              Line2D([], [], ls="none", marker="o", mfc=PAR, mec="white", ms=4.4)], ["stored", "PAR"],
             loc="lower left", handletextpad=0.1, borderaxespad=0.2)
    for sc, label in var[1:-1]:
        if f"tail|{sc}|Q" in S:
            b.plot(np.arange(1, 5) + shift[sc], S[f"tail|{sc}|Q"], color=color[sc], lw=1.0, marker="o", ms=2.8,
                   ls=(0, (3, 1.5)) if sc == "noclip" else "-", label=label)
    b.axhline(0, color=STORED, lw=0.5); b.set_xticks(range(1, 5), ["1", "2", "3", "4"]); b.set_xlim(0.7, 4.3)
    b.set_xlabel("Quarter of tail weight"); b.set_ylabel(r"Margin over z-score (points)")
    b.set_title("(b) Margin by tail weight, PAR"); b.grid(axis="y"); b.set_axisbelow(True)
    rho = np.array([0, 2, 5])
    for sc, label in var[1:]:
        if f"art|{sc}" in S:
            d = S[f"art|{sc}"] - S[f"art|{sc}"][0]
            c.plot(rho + 2.5 * shift[sc], d, color=color[sc], lw=1.0, marker="o", ms=2.8,
                   ls=(0, (3, 1.5)) if sc == "noclip" else "-",
                   label=label)
    c.axhline(0, color=STORED, lw=0.5); c.set_xticks(rho); c.set_xlim(-0.4, 5.5); c.set_ylim(-2.05, 0.15)
    c.set_xlabel(r"Epochs with artifacts (\%)"); c.set_ylabel(r"Change on clean epochs (points)")
    c.set_title("(c) Artifacts, PAR"); c.grid(axis="y"); c.set_axisbelow(True)
    handles = [Line2D([], [], color=color[sc], lw=1.0, marker="o", ms=2.8, ls=(0, (3, 1.5)) if sc == "noclip" else "-")
               for sc, _ in var[1:]]
    fig.legend(handles, [v[1] for v in var[1:]], loc="upper center", ncol=5, bbox_to_anchor=(0.62, -0.08),
               handlelength=1.6, handletextpad=0.3, columnspacing=1.4)
    save(fig, "scaling", TEXT_W)

FIGURES = dict(comparison=comparison, round_accuracy=round_accuracy, calibration=calibration, selection=selection,
               confusion=confusion, scaling=scaling)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
