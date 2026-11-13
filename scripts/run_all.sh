#!/bin/bash
# Every training run of the paper, one after another (a federated run takes about 4-5 h per fold on one RTX 6000 Ada).
# On a cluster, submit the lines as separate jobs; a run resumes from its last round, and --budget MIN makes it exit
# with code 3 before a job's time limit so that it can continue in a new job.
#   FSA_DATA=/path/to/data FSA_RESULTS=/path/to/results bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
T="python train.py"

for fold in 0 1; do
  # Tab. 1, 2, 4, 5, fig:rounds, fig:confusion: FedBN with three scalings and five seeds; FedSleepAlign = robust, Q.
  # Fold 0 of seed 42 keeps models for analysis/: every round of PRS (calibration, artifacts, delimitation) and the
  # selected rounds of the z-score (artifacts).
  for scaling in none zscore robust; do
    for seed in 42 142 242 342 442; do
      case "$seed-$fold-$scaling" in
        42-0-robust) extra="--save-weights all" ;;
        42-0-zscore) extra="--save-weights selected" ;;
        *) extra="" ;;
      esac
      $T --scheme fedbn --scaling $scaling --seed $seed --fold $fold $extra
    done
  done
  # Sec. 3.5, fig:comparison, App. B.5-B.6: matched comparison, seed 42
  $T --scheme fedavg --scaling robust --seed 42 --fold $fold
  $T --scheme hbn --scaling robust --seed 42 --fold $fold
  $T --scheme central --scaling robust --seed 42 --fold $fold
  for client in 0 1 2 3 4; do $T --scheme local --client $client --scaling robust --seed 42 --fold $fold; done
  # Sec. 3.3, App. B.7, fig:selection: subject-disjoint selection on a new subject split
  for scaling in none robust; do $T --scheme fedbn --scaling $scaling --seed 42 --split 1 --val dual --fold $fold; done
  # App. B.2, fig:scaling: PRS with one operation changed at a time
  for scaling in mean_iqr med_sd noclip; do
    if [ "$fold" = 0 ]; then extra="--save-weights selected"; else extra=""; fi
    $T --scheme fedbn --scaling $scaling --seed 42 --fold $fold $extra
  done
done
