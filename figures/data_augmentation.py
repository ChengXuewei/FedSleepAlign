"""
The seven data augmentations of augment.py, each applied alone to one epoch (Fig. augment, Appendix B.1 of main.tex).

Recording SC4001E0 of SleepEDF-78 (Fpz-Cz), epoch 615 (N2), scaled by PRS (median and IQR of the recording, clipped
at +-20 IQR as in fedsleepalign/data.py). Every augmentation is applied to the whole 30-s epoch with the operations of
augment.py and the parameters given in the panel titles; the random draws (noise, filter kernel, Fourier phases, crop
start) come from a seeded generator. The panels show samples 1630-1930 (3 s with a sleep spindle).

Run:    python figures/data_augmentation.py      (output: data_augmentation.pdf in $FSA_FIGURES or figures/pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.environ.get("FSA_FIGURES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf"))
NPZ = os.path.join(os.environ.get("FSA_DATA", "data"), "sleepedf/fpz_cz/SC4001E0.npz")
EPOCH, START, STOP = 615, 1630, 1930        # N2 epoch and the 3-s excerpt (samples at 100 Hz)
KAPPA, SEED, FS = 20.0, 0, 100

C_ORIG = "#2a78d6"     # original (blue); validated against C_AUG: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1
C_AUG = "#eb6834"      # augmented (orange)
C_INK2 = "#52514e"     # secondary text and scale bar
LW_ORIG, LW_AUG = 0.95, 0.5   # the original is drawn wider underneath, so it stays visible where the two coincide

CM = 1 / 2.54
FIG_W = 6.875                               # \textwidth of cvpr.sty in inches (figure* at \linewidth)
PANEL_W, PANEL_H = 3.95 * CM, 2.30 * CM     # panel height twice that of the former TikZ version (1.15 cm)
HSEP, TITLE_H, ROW_GAP, LEGEND_H, BOTTOM = 0.48 * CM, 0.80 * CM, 0.30 * CM, 0.45 * CM, 0.05 * CM
YLIM = (-2.0, 1.25)                         # in IQR units after PRS, identical for all panels


def prs(x):
    """Per-recording robust scaling of one channel (n_epochs, 3000), as robust_scale in fedsleepalign/data.py."""
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return np.clip((x - q2) / max(q3 - q1, 1e-6), -KAPPA, KAPPA).astype(np.float32)


def augmentations(x, g):
    """Each operation of augment.py alone on x (1, 1, L) with the parameters of the panel titles."""
    L = x.shape[-1]
    out = {}
    out["noise"] = x + torch.randn(x.shape, generator=g) * 0.01 * x.std()
    s = 50                                                   # +0.5 s, padded with the boundary value
    y = torch.roll(x, shifts=s, dims=-1)
    y[..., :s] = y[..., s:s + 1].expand_as(y[..., :s])
    out["shift"] = y
    out["scale"] = x * 1.1
    w = torch.rand(5, generator=g) - 0.5
    w = w / w.abs().sum()
    out["filter"] = F.conv1d(x, w.view(1, 1, -1), padding="same")
    y = F.interpolate(x, size=max(2, int(L * 0.9)), mode="linear", align_corners=True)
    out["warp"] = F.interpolate(y, size=L, mode="linear", align_corners=True)
    X = torch.fft.fft(x)
    phase = torch.rand(X.real.shape, generator=g) * 0.2 - 0.1
    out["phase"] = torch.fft.ifft(X * torch.exp(1j * phase)).real
    size = max(2, int(L * 0.9))
    a = torch.randint(0, L - size + 1, (1,), generator=g).item()
    out["crop"] = F.interpolate(x[..., a:a + size], size=L, mode="linear", align_corners=True)
    return {k: v.squeeze().numpy() for k, v in out.items()}


PANELS = [   # key, title, subtitle
    (None, "Original (after PRS)", r"N2 epoch, Fpz-Cz"),
    ("noise", "Additive noise", r"Gaussian, $0.01\times$ channel SD"),
    ("shift", "Time shift", r"up to $\pm$0.5\,s, here $+$0.5\,s"),
    ("scale", "Amplitude scaling", r"factor in [0.9, 1.1], here 1.1"),
    ("filter", "Random filtering", r"five-tap kernel, unit $\ell_1$ norm"),
    ("warp", "Time warping", r"factor in [0.9, 1.1], here 0.9"),
    ("phase", "Phase perturbation", r"Fourier phases within $\pm$0.1\,rad"),
    ("crop", "Cropping", r"90--100\% segment, here 90\%"),
]


def main():
    with np.load(NPZ) as d:
        assert int(d["y"][EPOCH]) == 2 and float(d["fs"]) == FS        # N2 at 100 Hz
        x = prs(d["x"].squeeze())[EPOCH]
    g = torch.Generator().manual_seed(SEED)
    aug = augmentations(torch.tensor(x).view(1, 1, -1), g)
    t = np.arange(STOP - START) / FS
    orig = x[START:STOP]
    for k, v in aug.items():
        assert YLIM[0] < v[START:STOP].min() and v[START:STOP].max() < YLIM[1], k

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Times"],   # mathptmx, the Times of cvpr.sty
        "pdf.fonttype": 42, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    fig_h = LEGEND_H + 2 * (TITLE_H + PANEL_H) + ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h))
    left0 = (FIG_W - 4 * PANEL_W - 3 * HSEP) / 2
    for i, (key, title, sub) in enumerate(PANELS):
        r, c = divmod(i, 4)
        bottom = BOTTOM + (1 - r) * (PANEL_H + TITLE_H + ROW_GAP)
        ax = fig.add_axes([(left0 + c * (PANEL_W + HSEP)) / FIG_W, bottom / fig_h, PANEL_W / FIG_W, PANEL_H / fig_h])
        ax.set_xlim(0, t[-1]); ax.set_ylim(*YLIM); ax.set_axis_off()
        ax.plot(t, orig, color=C_ORIG, lw=LW_ORIG if key else 0.7, zorder=2)
        if key:
            ax.plot(t, aug[key][START:STOP], color=C_AUG, lw=LW_AUG, zorder=3)
        ax.text(0.5, 1 + 0.42 * CM / PANEL_H, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        ax.text(0.5, 1 + 0.10 * CM / PANEL_H, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
                color=C_INK2)
        if key is None:                                     # scale bars: 0.5 s and 1 IQR
            x0, y0 = 0.03, -1.85
            ax.plot([x0, x0 + 0.5], [y0, y0], color=C_INK2, lw=0.6, solid_capstyle="butt", clip_on=False)
            ax.plot([x0, x0], [y0, y0 + 1.0], color=C_INK2, lw=0.6, solid_capstyle="butt", clip_on=False)
            ax.text(x0 + 0.25, y0 - 0.06, r"0.5\,s", ha="center", va="top", fontsize=5.5, color=C_INK2)
            ax.text(x0 + 0.04, y0 + 0.5, "1 IQR", ha="left", va="center", fontsize=5.5, color=C_INK2)
    handles = [plt.Line2D([], [], color=C_ORIG, lw=1.4), plt.Line2D([], [], color=C_AUG, lw=1.4)]
    fig.legend(handles, ["original", "augmented"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, fontsize=7, handlelength=1.8, columnspacing=2.0, borderaxespad=0.1)
    os.makedirs(HERE, exist_ok=True)
    fig.savefig(os.path.join(HERE, "data_augmentation.pdf"))


if __name__ == "__main__":
    main()
