"""Output format shared by the preprocessing scripts: one <OUT>/<folder>/<recording>.npz per channel and recording
with the kept 30-s epochs at 100 Hz (x, epochs x 3000, uV), their stages (y: 0..4 = W, N1, N2, N3, REM), their indices
in the stored recording (epoch_index) and the number of 30-s epochs of the stored recording (n_all_epochs)."""
import os
import sys

import numpy as np


def save(out, folders, record, x, y, kept, labels):
    """x: {channel: all epochs of the stored recording (n, 3000)}, y: all labels, kept: indices of the kept epochs."""
    kept = np.asarray(kept)
    for key, folder in folders.items():
        os.makedirs(f"{out}/{folder}", exist_ok=True)
        np.savez(f"{out}/{folder}/{record}.npz", x=x[key][kept].astype(np.float32), y=np.asarray(y)[kept].astype(np.int32),
                 fs=100, ch_label=labels[key], epoch_index=kept.astype(np.int32), n_all_epochs=len(x[key]))


def run(records, read, out, folders):
    """Preprocesses [(record, *paths)] with read(*paths) -> (x, y, kept, labels); records that fail are reported."""
    failed = []
    for record, *paths in records:
        try:
            x, y, kept, labels = read(*paths)
        except Exception as e:                        # e.g. a missing derivation
            print(f"{record}: skipped ({e})", flush=True)
            failed.append(record)
            continue
        save(out, folders, record, x, y, kept, labels)
        print(f"{record}: {len(kept)} of {len(y)} epochs", flush=True)
    print(f"done, {len(records) - len(failed)} recordings" + (f", skipped {failed}" if failed else ""))


def args():
    if len(sys.argv) != 3:
        sys.exit(f"usage: python {sys.argv[0]} RAW_DIR OUT_DIR")
    return sys.argv[1], sys.argv[2]
