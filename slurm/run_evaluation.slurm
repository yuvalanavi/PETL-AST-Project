#!/bin/bash
# =============================================================================
# Evaluate all 5 fold checkpoints on their respective test sets.
# Expected: ~5 min (inference only)
#
# Usage:  sbatch slurm/run_evaluation.slurm
# =============================================================================

#SBATCH --job-name=petl-eval
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gres=gpu:titan:1

STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/yuvalanavi"
VENV_DIR="${STORAGE}/petl-ast/venv"
REPO_DIR="${STORAGE}/petl-ast/PETL-AST-Project"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p slurm/logs

echo "=== PETL-AST Evaluation ==="
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

python evaluation.py \
    --data_path data \
    --checkpoint_dir outputs \
    --device cuda \
    --num_workers 4

echo ""
echo "=== Done: $(date) ==="
