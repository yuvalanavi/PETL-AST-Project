#!/bin/bash
# =============================================================================
# Single fold training (Conformer Adapter on GSC)
#
# Usage:  sbatch slurm/full_training_gsc.slurm
# =============================================================================

#SBATCH --job-name=petl-train-fold0
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gres=gpu:rtx3090:1

STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/yuvalanavi"
VENV_DIR="${STORAGE}/petl-ast/venv"
REPO_DIR="${STORAGE}/petl-ast/PETL-AST-Project"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p outputs slurm/logs

# --- CHANGED: Hardcode the fold number here ---
FOLD=0

echo "=== PETL-AST Single Training — Fold ${FOLD} ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $(python --version 2>&1)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

bash download_gsc_data.sh

export START_FOLD=${FOLD}
export END_FOLD=$((FOLD + 1))

python train.py \
    --data_path 'data' \
    --dataset_name 'GSC' \
    --method 'adapter' \
    --adapter_block 'conformer' \
    --adapter_type 'Pfeiffer' \
    --seq_or_par 'parallel' \
    --reduction_rate_adapter 96 \
    --kernel_size 31 \
    --device cuda \
    --num_workers 4 \
    --save_best_ckpt True \
    --output_path '/outputs'

echo ""
echo "=== Fold ${FOLD} Results ==="
python -c "
import os
log = 'outputs/training.log'
if os.path.exists(log):
    with open(log) as f:
        for l in f:
            l = l.strip()
            if any(k in l for k in ['Trainloss at epoch 74', 'Folds accuracy', 'Avg accuracy', 'Std accuracy', 'Training time']):
                print(l)
ckpt = f'outputs/bestmodel_fold${FOLD}'
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f'Checkpoint: {ckpt} ({size_mb:.1f} MB)')
else:
    print('!! No checkpoint saved')
"
echo "=== Fold ${FOLD} Done: $(date) ==="