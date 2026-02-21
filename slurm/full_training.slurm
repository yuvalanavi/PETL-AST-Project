#!/bin/bash
# =============================================================================
# Full training: 5 folds × 50 epochs (Conformer Adapter on ESC-50)
# Expected: ~3-5 hours on a single GPU
#
# Usage:  sbatch slurm/full_training.slurm
# =============================================================================

#SBATCH --job-name=petl-train
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=480
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

rm -f outputs/bestmodel_fold* outputs/training.log

echo "=== PETL-AST Full Training ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version 2>&1)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Download data if needed
bash download_data.sh

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
echo "=== Results ==="
python -c "
import os
log = 'outputs/training.log'
if os.path.exists(log):
    with open(log) as f:
        c = f.read()
    for l in c.split('\n'):
        if any(k in l for k in ['Folds accuracy', 'Avg accuracy', 'Std accuracy', 'Training time']):
            print(l.strip())
ckpts = [f for f in os.listdir('outputs') if f.startswith('bestmodel')]
print(f'Checkpoints: {len(ckpts)}')
"
echo "=== Done: $(date) ==="
