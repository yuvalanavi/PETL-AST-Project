#!/bin/bash
# =============================================================================
# SLURM full training: 5 folds × 50 epochs
# Expected runtime: ~3-5 hours on RTX 3090 / V100
# =============================================================================

#SBATCH --job-name=petl-train
#SBATCH --output=/home/yandex/APDL2526a/petl-ast/PETL-AST-Project/outputs/full_training_%j.out
#SBATCH --error=/home/yandex/APDL2526a/petl-ast/PETL-AST-Project/outputs/full_training_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=480
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

PROJECT_DIR="/home/yandex/APDL2526a/petl-ast"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"
VENV_DIR="${PROJECT_DIR}/venv"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

# Clean smoke test outputs before full run
rm -f outputs/bestmodel_fold* outputs/training.log
mkdir -p outputs

echo "=== PETL-AST Full Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run full training (5 folds × 50 epochs)
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
echo "=== Training complete: $(date) ==="
echo ""

# Summary
python -c "
import os
output_dir = 'outputs'
log = os.path.join(output_dir, 'training.log')
if os.path.exists(log):
    with open(log) as f:
        content = f.read()
    for line in content.split('\n'):
        if any(k in line for k in ['Folds accuracy', 'Avg accuracy', 'Std accuracy', 'Training time']):
            print(line.strip())
ckpts = [f for f in os.listdir(output_dir) if f.startswith('bestmodel_fold')]
print(f'Checkpoints saved: {len(ckpts)}')
"
