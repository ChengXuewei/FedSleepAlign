"""Seeding, checkpoints, the local epoch of a client and the weighted averaging of the server."""
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # deterministic cuBLAS; set before CUDA starts
import numpy as np
import torch
import torch.optim as optim

torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def to_cpu(state):
    return {k: v.to("cpu", copy=True) for k, v in state.items()}


def save(path, **obj):
    """Atomic save that keeps the previous version as <path>.bak."""
    with open(path + ".part", "wb") as f:
        torch.save(obj, f)
        f.flush(); os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(path + ".part", path)


def load(path, default=None):
    """The saved object, or the previous version if the file is unreadable."""
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                return torch.load(p, weights_only=False)
            except Exception as e:
                print(f"(unreadable {p}: {str(e)[:80]})")
    return default


def local_epoch(model, state, loader, criterion):
    """One epoch over a client's training windows from `state`, with Adam restarted every round."""
    model.load_state_dict(state)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.train()
    for inputs, labels in loader:
        optimizer.zero_grad()
        # (B, 5, L) logits -> (B * L, 5): the same loss as the 2-d form, which is not deterministic on CUDA
        loss = criterion(model(inputs.to(device)).transpose(1, 2).flatten(0, 1), labels.flatten().to(device))
        loss.backward()
        optimizer.step()
    return {k: v.clone() for k, v in model.state_dict().items()}


def fed_average(states, sizes, keys):
    """The `keys` entries of the client states averaged with weights n_k / n (Eq. 1)."""
    w = np.asarray(sizes, dtype=np.float64) / np.sum(sizes)
    return {k: sum(wi * s[k].double() for wi, s in zip(w, states)).to(states[0][k].dtype) for k in keys}


def sq_norm(a, b, keys):
    """||a - b||^2 over the entries `keys`."""
    return sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in keys)
