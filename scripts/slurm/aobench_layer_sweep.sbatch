#!/bin/bash
# AObench layer-window sweep on the school cluster (account brain, partition Brain_GPU, A100s).
# Submit from the repo root, after `mkdir -p logs` (Slurm does not create the log folder, and the
# job fails silently without it). Any argument after the file name goes to
# scripts/aobench_layer_sweep.py:
#
#   sbatch scripts/slurm/aobench_layer_sweep.sbatch --starts 21 13 --limit 10   # pipeline check
#   sbatch scripts/slurm/aobench_layer_sweep.sbatch                             # full experiment
#
# The run resumes where it stopped: if the job is cut, submit the same command again.
# On a 40 GB A100, Qwen3-32B as a bf16 judge does not fit on one GPU: run the two phases as
# separate jobs, the judge with two GPUs (it spreads over every GPU the job gets):
#   sbatch scripts/slurm/aobench_layer_sweep.sbatch --phase generate
#   sbatch --gres=gpu:a100:2 scripts/slurm/aobench_layer_sweep.sbatch --phase judge
# (options given to sbatch on the command line override the #SBATCH lines below)

#SBATCH --job-name=aobench-sweep
#SBATCH --account=brain
#SBATCH --partition=Brain_GPU
#SBATCH --qos=high
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# uv, its Python and the caches are on /Brain: a job cannot read the home folder
source scripts/slurm/cluster_env.sh
# Anything else that writes to ~ (matplotlib, torch, ...) gets a home folder on /Brain too
export HOME="$PROCOM_DIR/jobhome/$USER"
mkdir -p "$HOME"
command -v uv >/dev/null || { echo "uv not found in $PROCOM_DIR/tools/bin"; exit 1; }

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# -u: unbuffered output, so the log file shows progress live
uv run python -u scripts/aobench_layer_sweep.py "$@"

echo "Finished $(date)"
