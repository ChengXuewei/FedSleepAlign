"""Paths, databases and constants shared by all scripts."""
import os

DATA = os.environ.get("FSA_DATA", "data")            # preprocessed recordings, written by preprocessing/*.py
RESULTS = os.environ.get("FSA_RESULTS", "results")   # predictions, per-round logs, weights and summaries

# one client per database: folders of EEG 1 (frontal; central for SHHS), EEG 2 (occipital; parietal for SleepEDF-78,
# the other central derivation for SHHS) and the EOG, each holding one <recording>.npz per recording
DATABASES = {
    "SleepEDF-78": ("sleepedf/fpz_cz", "sleepedf/pz_oz", "sleepedf/eog"),
    "ISRUC": ("isruc/f3_a2", "isruc/o1_a2", "isruc/eog"),
    "PhysioNet2018": ("physionet2018/f3_m2", "physionet2018/o1_m2", "physionet2018/eog"),
    "HMC": ("hmc/f4_m1", "hmc/o2_m1", "hmc/e1_m2"),
    "SHHS": ("shhs/c4_a1", "shhs/c3_a2", "shhs/eog_lr"),
}
STAGES = ("W", "N1", "N2", "N3", "REM")
SEQLEN = 20          # epochs per window (L)
ROUNDS = 50          # communication rounds (T), one local epoch each
TOP_M = 10           # rounds in the majority-vote ensemble (M)
CAL_BATCHES = 10     # calibration batches of PAR per client (B)
BATCH = 32           # windows per batch
