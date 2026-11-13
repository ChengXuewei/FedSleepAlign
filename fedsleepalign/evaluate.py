"""Validation accuracy, test predictions, the majority vote of the ensemble and the metrics."""
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def epoch_accuracy(model, loader):
    """Accuracy over all epochs of the windows in `loader`."""
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        correct += (model(inputs).argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def predict(model, loader):
    """Predicted stages (windows, L) and labels of all windows in `loader`."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(torch.softmax(model(x.to(device)), dim=1).argmax(1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def majority(votes):
    """Majority vote over the first axis of (members, windows, L); ties go to the stage listed first."""
    return (votes[..., None] == np.arange(5)).sum(axis=0).argmax(axis=-1)


def reconstruct(preds_w, labels_w, meta, lengths):
    """Window predictions -> one prediction per epoch of every test recording (concatenated)."""
    preds = [np.full(n, -1, dtype=np.int64) for n in lengths]
    labels = [np.full(n, -1, dtype=np.int64) for n in lengths]
    for w, (r, s, e) in enumerate(meta):
        preds[r][s:e], labels[r][s:e] = preds_w[w][:e - s], labels_w[w][:e - s]
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    assert (preds >= 0).all(), "every epoch is covered by a window"
    return preds, labels


def confusion(y, p):
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def scores(cm):
    """[accuracy, macro F1, Cohen's kappa, recall of W, N1, N2, N3, REM] of a confusion matrix (rows: true stage)."""
    n, tp = cm.sum(), np.diag(cm)
    recall, precision = tp / cm.sum(1).clip(min=1), tp / cm.sum(0).clip(min=1)
    f1 = 2 * precision * recall / (precision + recall).clip(min=1e-12)
    acc, pe = tp.sum() / n, (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return np.array([acc, f1.mean(), (acc - pe) / (1 - pe), *recall])
