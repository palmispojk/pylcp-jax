#!/bin/bash
# Stage 1 - blue MOT: first stage of the pipeline; writes blue_mot_final_state.pkl for later stages.
#
# Usage (from this directory):  mkdir -p logs && sbatch blue_mot_sim.sh
# Edit the resources below for your cluster, or override on the command line,
# e.g.  sbatch --partition=<gpu-partition> --time=12:00:00 blue_mot_sim.sh
#
#SBATCH --job-name=blue_mot
##SBATCH --partition=<your-gpu-partition>   # remove one '#' and set; omit for cluster default
#SBATCH --gres=gpu:1                        # simulations run on JAX; one GPU is enough
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                           # lower this for fewer atoms
#SBATCH --time=24:00:00                       # walltime limit (HH:MM:SS)
#SBATCH --output=logs/blue_mot_%j.out         # logs/ must exist before submitting
#SBATCH --error=logs/blue_mot_%j.err

set -euo pipefail

# SLURM runs a copy of this script, so work from the directory you submitted in.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Python interpreter: $PYTHON if set, else the repo's .venv, else python3 on PATH.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ..)"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON="python3"
fi

export PYTHONUNBUFFERED=1   # stream prints to the log as they happen
"$PYTHON" -u blue_mot_sim.py
