"""Statistics of the recordings (Tab. 3, Sec. 3.1, App. B.1-B.2): per recording and channel the median, IQR, mean, SD
and the share of samples clipped by PRS (20 IQRs) and the z-score (27 SDs); the spread of IQR / SD within a client;
the ISRUC recordings with an F3 IQR below 5 uV; the tail weight of every recording; and the windows that span epochs
removed by the preprocessing. With --isruc-raw RAW, also the physical range and date in the EDF headers of ISRUC.
    python analysis/recordings.py [--isruc-raw RAW] > results/summary/recordings.log"""
import argparse
import sys

import numpy as np

from common import DS, SUMMARY, subject_of
from fedsleepalign.config import DATA, DATABASES, SEQLEN, STAGES
from fedsleepalign.data import split_halves

p = argparse.ArgumentParser()
p.add_argument("--isruc-raw", help="ISRUC raw folder (subgroupI, II, III) for the header check")
a = p.parse_args()

halves = split_halves(42)
rows, info = [], []
for ds in DS:
    for name in halves[ds][0] + halves[ds][1]:
        st, meta = [], None
        for c in DATABASES[ds]:
            with np.load(f"{DATA}/{c}/{name}") as f:
                x, y = f["x"].astype(np.float32), f["y"]
                if "epoch_index" in f:
                    meta = (f["epoch_index"], int(f["n_all_epochs"]) if "n_all_epochs" in f else None)
            q1, q2, q3 = np.percentile(x, [25, 50, 75])
            x64 = x.astype(np.float64); mu, sd = x64.mean(), x64.std()
            st.append([q2, q3 - q1, mu, sd, np.mean(np.abs(x - q2) > 20 * (q3 - q1)), np.mean(np.abs(x64 - mu) > 27 * sd)])
        rows.append(st); info.append((ds, name, y, meta))
stats = np.array(rows)                                         # (recordings, channels, [median IQR mean SD clip_r clip_z])
client = np.array([i[0] for i in info]); names = np.array([i[1] for i in info])
epochs = np.array([len(i[2]) for i in info])

print("=== Tab. 3: subjects / recordings, epochs (stored, kept), IQR per channel (median [5%, 95%], uV), stages (%)")
for ds in DS:
    m = client == ds
    y = np.concatenate([i[2] for i in info if i[0] == ds])
    stored = (sum(i[3][1] for i in info if i[0] == ds)
              if all(i[3] and i[3][1] is not None for i in info if i[0] == ds) else "-")
    print(f"{ds:<13} {len({subject_of(ds, n) for n in names[m]})} / {m.sum()} | epochs stored {stored}, kept "
          f"{epochs[m].sum()} | IQR " + " | ".join(
              f"{np.median(stats[m, j, 1]):.1f} [{np.percentile(stats[m, j, 1], 5):.1f}, {np.percentile(stats[m, j, 1], 95):.1f}]"
              for j in range(3)) + " | " + "/".join(f"{100 * np.mean(y == s):.0f}" for s in range(len(STAGES))))

print("\n=== IQR / SD per channel: median and ratio of the 95th to the 5th percentile over the recordings of a client")
ratio = stats[..., 1] / stats[..., 3]
for ds in DS:
    q = np.percentile(ratio[client == ds], [5, 50, 95], axis=0)
    print(f"{ds:<13} median " + " ".join(f"{v:.2f}" for v in q[1]) + " | 95% / 5% " + " ".join(f"{v:.2f}" for v in q[2] / q[0]))
w = epochs[:, None].repeat(3, 1)
print(f"samples clipped: PRS {100 * np.average(stats[..., 4], weights=w):.3f}%, z-score "
      f"{100 * np.average(stats[..., 5], weights=w):.4f}%; smallest IQR {stats[..., 1].min():.2f} uV")

isruc_low = names[(client == "ISRUC") & (stats[:, 0, 1] < 5)]
print(f"\nISRUC recordings with an F3 IQR below 5 uV: {len(isruc_low)} of {np.sum(client == 'ISRUC')}; median IQR "
      + " ".join(f"{np.median(stats[np.isin(names, isruc_low), j, 1]):.2f}" for j in range(3)) + " uV, others "
      + " ".join(f"{np.median(stats[(client == 'ISRUC') & ~np.isin(names, isruc_low), j, 1]):.2f}" for j in range(3)))
if a.isruc_raw:
    import pyedflib
    from collections import Counter
    for group, sel in (("low range", np.isin(names, isruc_low)), ("others", (client == "ISRUC") & ~np.isin(names, isruc_low))):
        ranges, years = Counter(), Counter()
        for n in names[sel]:
            p_ = n[:-4].split("-")
            rec = (f"{a.isruc_raw}/subgroupII/{p_[1]}/{p_[2]}/{p_[2]}.rec" if p_[0] == "II"
                   else f"{a.isruc_raw}/subgroup{p_[0]}/{p_[1]}/{p_[1]}.rec")
            f = pyedflib.EdfReader(rec)
            labels = f.getSignalLabels()
            ranges[tuple(sorted({f.getSignalHeader(i)["physical_max"] for i, l in enumerate(labels)
                                 if l.split("-")[0] in ("F3", "O1", "LOC", "A1", "A2", "M1", "M2", "E1")}))] += 1
            years[f.getStartdatetime().year] += 1
            f.close()
        print(f"  {group}: EDF physical range (uV) {dict(ranges)}, recording years {dict(sorted(years.items()))}")

tail = np.log(stats[..., 3] / stats[..., 1]).mean(1)           # tail weight: mean over the channels of log(SD / IQR)
print("\n=== Windows of L epochs that span epochs removed by the preprocessing (from epoch_index)")
for ds in DS:
    span = total = removed = 0
    for i in info:
        if i[0] == ds and i[3] is not None:
            idx = i[3][0]
            removed += int(np.sum(np.diff(idx) - 1))
            n = len(idx) // SEQLEN
            total += n
            span += sum(bool((np.diff(idx[w * SEQLEN:(w + 1) * SEQLEN]) > 1).any()) for w in range(n))
    print(f"{ds:<13} removed epochs inside the kept period {removed}, windows spanning a gap {span} of {total}"
          + (f" ({100 * span / total:.1f}%)" if total else ""))
np.savez(f"{SUMMARY}/recordings.npz", ds=client, name=names, stats=stats, tail=tail, epochs=epochs, isruc_low=isruc_low)
