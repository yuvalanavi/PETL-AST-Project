#!/bin/bash
# =============================================================================
# Smoke test using NGC PyTorch container (per SLURM guide)
# 5 folds × 3 epochs — validates the full pipeline
# Expected: ~15-30 min
# =============================================================================

#SBATCH --job-name=petl-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

REPO_DIR="$HOME/petl-ast/PETL-AST-Project"
CMD="${REPO_DIR}/slurm/run_smoke_test.sh"

srun easy_ngc \
    --modules="${REPO_DIR}/slurm/requirements_cluster.txt" \
    --cmd "${CMD}" \
    nvcr.io/nvidia/pytorch:22.12-py3
