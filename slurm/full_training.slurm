#!/bin/bash
# =============================================================================
# Full training: 1 fold × 50 epochs per job (Conformer Adapter on ESC-50)
# Each fold runs as a separate parallel job via SLURM job array.
# Expected: ~5 hours per fold on TITAN Xp
#
# Usage:  sbatch --array=0-4 slurm/full_training.slurm
# =============================================================================

#SBATCH --job-name=petl-train
#SBATCH --output=slurm/logs/%x_fold%a_%j.out
#SBATCH --error=slurm/logs/%x_fold%a_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=420
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
mkdir -p outputs slurm/logs

FOLD=${SLURM_ARRAY_TASK_ID}

echo "=== PETL-AST Full Training — Fold ${FOLD} ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $(python --version 2>&1)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

bash download_data.sh

export START_FOLD=${FOLD}
export END_FOLD=$((FOLD + 1))

python train.py \
    --data_path 'data' \
    --dataset_name 'ESC-50' \
    --method 'adapter' \
    --adapter_block 'conformer' \
    --adapter_type 'Pfeiffer' \
    --seq_or_par 'parallel' \
    --reduction_rate_adapter 96 \
    --kernel_size 8 \
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
            if any(k in l for k in ['Trainloss at epoch 49', 'Folds accuracy', 'Avg accuracy', 'Std accuracy', 'Training time']):
                print(l)
ckpt = f'outputs/bestmodel_fold${FOLD}'
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f'Checkpoint: {ckpt} ({size_mb:.1f} MB)')
else:
    print('!! No checkpoint saved')
"
echo "=== Fold ${FOLD} Done: $(date) ==="
