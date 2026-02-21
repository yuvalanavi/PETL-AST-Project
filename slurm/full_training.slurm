#!/bin/bash
# =============================================================================
# Full training: 5 folds × 50 epochs
# Expected: ~3-5 hours on RTX 3090 / V100
# =============================================================================

#SBATCH --job-name=petl-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=480
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

PROJECT_DIR="$HOME/petl-ast"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"
VENV_DIR="${PROJECT_DIR}/venv"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

rm -f outputs/bestmodel_fold* outputs/training.log
mkdir -p outputs

echo "=== PETL-AST Full Training ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version 2>&1)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

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
    --save_best_ckpt True --output_path '/outputs'

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
