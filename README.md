# FedSleepAlign

Code of the paper *FedSleepAlign: A Federated Alignment Framework for Multi-Source Sleep Staging*.

Five public polysomnography (PSG) databases act as the clients of a federation. Starting from FedBN, which keeps the
batch normalization (BN) layers local, FedSleepAlign adds two label-free and communication-free alignment strategies:

- **PRS** (per-recording robust scaling): every channel of every recording is standardized by its median and
  interquartile range (IQR) and clipped at 20 IQRs (`fedsleepalign/data.py`);
- **PAR** (post-aggregation re-estimation): after every aggregation, every client re-estimates the statistics of its BN
  layers under the aggregated weights, layer by layer in forward order, from B = 10 batches of its training windows
  (Algorithm 1 of the paper, `fedsleepalign/bn.py`).

This repository contains everything needed to reproduce the paper: preprocessing of the five databases, all training
runs, every analysis behind the tables, figures and quoted numbers, and the figure scripts.

## Contents

```
fedsleepalign/        the method
  config.py           paths (FSA_DATA, FSA_RESULTS), databases and channels, constants (L, T, M, B)
  data.py             subject split, PRS and the z-score controls, windows, clients and augmentation copies
  augment.py          the seven augmentations
  model.py            the network (residual CNN, BiGRU, Transformer, BiLSTM over the epochs of a window), Hybrid BN
  bn.py               PAR, the one-pass comparator, deviations between BN statistics, Hybrid BN statistics
  federated.py        seeding and determinism, checkpoints, the local epoch, the weighted average of the server
  evaluate.py         validation accuracy, test prediction, majority vote, metrics
train.py              every training run (FedBN, FedAvg, Hybrid BN, local, centralized; all scalings)
preprocessing/        raw databases -> 30-s epochs at 100 Hz, one file per channel and recording
analysis/             the numbers of the paper (one script per part, see below)
release/              subject splits and results of all runs of the paper (see Release files)
figures/              make_figures.py (all data figures), data_augmentation.py, FL_structure.tex (overview diagram)
scripts/run_all.sh    all training runs of the paper
```

## Installation

Python 3.10 with

```
pip install -r requirements.txt
```

(PyTorch 2 with CUDA, NumPy, SciPy, pyedflib, wfdb, h5py and Matplotlib). The figures are typeset with LaTeX
(`text.usetex`), so a TeX installation with the Times font is needed for `figures/`. The experiments of the paper used
PyTorch 2.13 with CUDA 13.0, NumPy 2.2 and SciPy 1.15.

## Data

| Client | Source | Recordings used | Channels |
|---|---|---|---|
| SleepEDF-78 | [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/), Sleep Cassette | 153 (78 subjects) | Fpz-Cz, Pz-Oz, horizontal EOG |
| ISRUC | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/), subgroups I-III | 125 (117 subjects) | F3-A2, O1-A2, LOC-A2 |
| PhysioNet-2018 | [PhysioNet/CinC Challenge 2018](https://physionet.org/content/challenge-2018/), training set | 200 of 994 | F3-M2, O1-M2, E1-M2 |
| HMC | [HMC sleep staging database](https://physionet.org/content/hmc-sleep-staging/) | 151 | F4-M1, O2-M1, E1-M2 |
| SHHS | [SHHS-1](https://sleepdata.org/datasets/shhs) (NSRR, data access request) | 329 (list in `preprocessing/shhs_records.txt`) | C4-A1, C3-A2, EOG(L)-EOG(R) |

Each script reads one raw database and writes `<OUT>/<database>/<channel>/<recording>.npz` with the kept 30-s epochs
at 100 Hz (`x`, epochs x 3000, in uV), their stages (`y`, 0-4 = W, N1, N2, N3, REM), their indices in the stored
recording (`epoch_index`) and the number of epochs of the stored recording (`n_all_epochs`):

```
cd preprocessing
python sleepedf.py      RAW/sleep-edfx/sleep-cassette  DATA
python isruc.py         RAW/isruc                      DATA     # subgroupI/, subgroupII/, subgroupIII/
python physionet2018.py RAW/challenge-2018/training    DATA
python hmc.py           RAW/hmc-sleep-staging          DATA     # RECORDS, recordings/
python shhs.py          RAW/shhs/polysomnography       DATA     # edfs/shhs1/, annotations-events-profusion/shhs1/
export FSA_DATA=DATA
```

R&K stages S3 and S4 are merged into N3. SleepEDF-78 and SHHS keep 30 min of wake before the first and after the last
sleep epoch, and movement, unscored and undefined epochs are removed. Signals recorded at other rates are resampled to
100 Hz by the Fourier method per epoch; HMC nights are resampled as a whole by a polyphase filter. The scripts reproduce
the data of the paper exactly. The subject split (`split_halves` in `fedsleepalign/data.py`) draws 200 PhysioNet-2018
subjects and divides the subjects of every database into two halves for two-fold cross-validation; `analysis/export.py`
writes the resulting manifest.

## Training

```
python train.py --scheme fedbn --scaling robust --seed 42 --fold 0      # FedSleepAlign (evaluation Q) and FedBN + PRS (S)
```

| Option | Values |
|---|---|
| `--scheme` | `fedbn` (BN local), `fedavg`, `hbn` (Hybrid BN, Chen et al., ICML 2025), `local --client I`, `central` |
| `--scaling` | `none`, `robust` (PRS), `zscore`, and the PRS variants `mean_iqr`, `med_sd`, `noclip` |
| `--seed`, `--fold` | training seed (42, 142, 242, 342, 442 in the paper), test half 0 or 1 |
| `--split`, `--val dual` | another subject split; additional subject-disjoint validation set |
| `--save-weights` | `selected` (the selected models) or `all` (every round), for `analysis/` |
| `--budget MIN` | exit with code 3 before MIN minutes, to continue in a new job (runs resume from their last round) |

A run trains for 50 rounds of one local epoch per client (Adam, learning rate 1e-4, weight decay 1e-3, batches of 32
windows of L = 20 epochs). After every round it evaluates every client model on its validation windows with three kinds
of BN statistics, **S** (stored during local training), **R** (one-pass re-estimation) and **Q** (PAR), and each kind
keeps its own top-10 rounds by the pooled validation accuracy. The test applies the majority vote of these rounds to
every epoch of the test recordings; FedBN is `fedbn/none` with S, and FedSleepAlign is `fedbn/robust` with Q. Outputs:

```
$FSA_RESULTS/preds/<run>_fold<k>.npz     test predictions per client and evaluation (S, R, Q, crossed S_Q, Q_S, ...)
$FSA_RESULTS/rounds/<run>_fold<k>.npz    validation accuracy per round, selected rounds, deviations of the BN statistics
$FSA_RESULTS/weights/<run>_fold<k>/      saved models (--save-weights)
```

`scripts/run_all.sh` lists all runs of the paper. A federated run takes about 4-5 hours per fold on one RTX 6000 Ada.
With deterministic algorithms, a run repeats exactly on the same GPU model.

## Analysis

Run from the repository root with `FSA_DATA` and `FSA_RESULTS` set (`python analysis/<script>.py`); the logs and
`results/summary/*.npz` hold the numbers of the paper.

| Script | Paper | Needs |
|---|---|---|
| `recordings.py [--isruc-raw RAW]` | Tab. 3, Sec. 3.1, App. B.1 (amplitudes, clipping, ISRUC low-range recordings, gaps) | data |
| `ablation.py` | Tab. 1, 2, 4, 5; Sec. 3.3-3.4; App. B.2-B.3 | FedBN runs of Tab. 1, `recordings.py` |
| `comparison.py` | Sec. 3.5, App. B.5-B.6, fig:comparison | local, central, FedAvg, Hybrid BN runs |
| `selection.py` | Sec. 3.3, App. B.7, fig:selection | runs on split 1 with `--val dual`, Tab. 1 runs |
| `artifact.py --scaling X` (GPU) | App. B.2 | fold-0 models of seed 42 (PRS, its variants, z-score) |
| `scaling.py` | Sec. 3.3, App. B.2, fig:scaling | PRS-variant runs, `artifact.py`, `recordings.py` |
| `calibration.py --part P` (GPU) | App. B.4, B.6, fig:calibration | every round of `fedbn_robust_s42` fold 0 |
| `delimit.py` (GPU) | App. B.1 | fold-0 models of `fedbn_robust_s42`, raw SleepEDF and SHHS |
| `export.py` | App. B.1 (the files in `release/`) | all runs |
| `release_tables.py` | Tab. 1, 2, 4, paired tests, bootstrap | `release/` only |

`calibration.py` has the parts `timing`, `bsweep --rounds A-B`, `btest`, `budget`, `perclient` and `summary`.

## Release files

`release/` holds the subject splits and the results of all runs of the paper, written by `analysis/export.py`:

| File | Content |
|---|---|
| `manifest.csv` | database, subject and test half of every recording for the subject splits 42 (paper) and 1 (App. B.7) |
| `metrics.csv` | accuracy, macro F1, kappa, stage recalls and confusion matrix per run, fold, client and evaluation |
| `selected_rounds.csv` | the M = 10 selected rounds per run, fold and evaluation |
| `recordings.csv` | kept and correctly staged epochs per run, fold, test recording and evaluation |

Runs are named `<scheme>_<scaling>_s<seed>`, with `_p1_dual` for the subject split 1 and `_c<i>` for the local
training of client i. The evaluations are `S`, `R` and `Q`, the crossed `S_Q` (rounds selected with S, tested with Q),
`Q_S`, `S_R` and `R_S`, and, for `--val dual`, the selection on subject-disjoint validation (`S-subj`, ...).

```
python analysis/release_tables.py      # Tab. 1, 2 and 4, the paired tests and the bootstrap from release/ alone
```

It runs in a few seconds on a CPU and needs neither the data nor the trained models.

## Figures

```
python figures/make_figures.py          # all data figures as PDF in $FSA_FIGURES (default figures/pdf)
python figures/data_augmentation.py     # the augmentation examples
pdflatex figures/FL_structure.tex       # the overview diagram
```
