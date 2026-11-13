"""BN statistics of a client model: PAR (layer-wise re-estimation), the one-pass comparator, deviations between two
sets of statistics, and the statistics exchange of Hybrid BN."""
import torch
import torch.nn as nn

from .model import HybridBN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_layers(model):
    """(name, layer) of the (hybrid) BN layers in forward order B1.1, B1.2, B1.s, B2.1, B2.2, B2.s."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.BatchNorm1d, HybridBN))]


def bn_entries(model, state):
    """Copies of the entries of `state` that belong to BN layers (affine parameters and statistics)."""
    return {f"{n}.{k}": state[f"{n}.{k}"].clone() for n, m in norm_layers(model) for k in m.state_dict()}


def cnn_input(x):
    """(B, 3, L, 3000) windows -> (B * L, 3, 3000) epochs, the input of the convolutional blocks."""
    B, C, W, L = x.shape
    return x.permute(0, 2, 1, 3).reshape(B * W, C, L)


def draw_batches(loader, n):
    """The first n batches of a fresh shuffle of a client's training windows (augmented copies included)."""
    out = []
    for b, (inputs, _) in enumerate(loader):
        if b == n:
            break
        out.append(inputs)
    return out


class Moments:
    """Count, sum and sum of squares per channel over all epochs and positions, accumulated in double precision."""
    def __init__(self): self.n, self.s, self.q = 0, 0.0, 0.0
    def add(self, x):
        var, mean = torch.var_mean(x, dim=(0, 2), correction=0)
        nb = x.shape[0] * x.shape[2]
        self.n += nb; self.s = self.s + nb * mean.double(); self.q = self.q + nb * (var.double() + mean.double() ** 2)
    def set(self, bn):
        mean = self.s / self.n
        bn.running_mean.copy_(mean.float()); bn.running_var.copy_(((self.q - self.n * mean ** 2) / (self.n - 1)).float())


@torch.no_grad()
def reestimate(model, state, batches):
    """PAR (Algorithm 1): with all modules in evaluation mode, every BN layer in forward order gets the pooled mean and
    Bessel-corrected variance of its input over the batches, while the layers before it normalize with their
    re-estimated statistics. Returns the BN entries of `state` with these statistics."""
    model.load_state_dict(state)
    model.eval()
    layers = norm_layers(model)
    for _, bn in layers:
        bn.reset_running_stats()
    for _, bn in layers:
        mom = Moments()
        hook = bn.register_forward_pre_hook(lambda module, inp: mom.add(inp[0]))
        for x in batches:
            model.cnn(cnn_input(x.to(device)))
        hook.remove()
        mom.set(bn)
    return bn_entries(model, model.state_dict())


@torch.no_grad()
def reestimate_onepass(model, state, batches):
    """The one-pass comparator: all BN layers in training mode accumulate the cumulative average of the batch means and
    unbiased batch variances over the batches, dropout off. Returns the BN entries of `state` with these statistics."""
    model.load_state_dict(state)
    model.eval()
    bns = [m for _, m in norm_layers(model)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()      # momentum None: cumulative average
    for x in batches:
        model.cnn(cnn_input(x.to(device)))
    for m in bns:
        m.momentum = 0.1; m.eval()
    return bn_entries(model, model.state_dict())


def deviation(a, b, names):
    """Per BN layer, the median over its channels of |mean_a - mean_b| / sd_b and of var_a / var_b (Tab. 5)."""
    out = []
    for n in names:
        ma, va = a[f"{n}.running_mean"].double(), a[f"{n}.running_var"].double()
        mb, vb = b[f"{n}.running_mean"].double(), b[f"{n}.running_var"].double()
        out.append([((ma - mb).abs() / vb.sqrt()).median().item(), (va / vb).median().item()])
    return out


@torch.no_grad()
def hbn_collect(model, state, batches):
    """Hybrid BN, client side: per layer the count-merged mixed moments of the batches under `state` (hybrid layers in
    training mode, dropout off)."""
    model.load_state_dict(state)
    model.eval()
    layers = [m for _, m in norm_layers(model)]
    for m in layers:
        m.train(); m.collect = True
        m.running_num.zero_(); m.running_mean.zero_(); m.running_var.zero_()
    for x in batches:
        model.cnn(cnn_input(x.to(device)))
    for m in layers:
        m.eval(); m.collect = False
    return [(m.running_num.clone(), m.running_mean.clone(), m.running_var.clone()) for m in layers]


def hbn_global(collected, names, prev, momentum):
    """Hybrid BN, server side: the clients' moments pooled by their counts (variance including the spread of the
    means), moved from the previous global moments with `momentum`."""
    out = {}
    for j, name in enumerate(names):
        n = torch.cat([c[j][0] for c in collected])[:, None]
        mu, var = torch.stack([c[j][1] for c in collected]), torch.stack([c[j][2] for c in collected])
        g_mean = (n * mu).sum(0) / n.sum()
        g_var = (n * (var + (mu - g_mean) ** 2)).sum(0) / (n.sum() - 1)
        for key, g in ((f"{name}.global_mean", g_mean), (f"{name}.global_var", g_var)):
            out[key] = momentum * g + (1 - momentum) * prev[key]
    return out
