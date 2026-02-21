#!/bin/bash
# =============================================================================
# Smoke test: 5 folds × 3 epochs — validates the full pipeline
# Expected: ~15-30 min on RTX 3090 / V100
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

PROJECT_DIR="$HOME/petl-ast"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"
VENV_DIR="${PROJECT_DIR}/venv"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p outputs

echo "=== PETL-AST Smoke Test ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version 2>&1)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Override epochs to 3 for smoke test
cp hparams/train.yaml hparams/train.yaml.bak
python -c "
import yaml
with open('hparams/train.yaml') as f:
    hp = yaml.safe_load(f)
hp['epochs_ESC'] = 3
with open('hparams/train.yaml', 'w') as f:
    yaml.dump(hp, f, default_flow_style=False)
print('epochs_ESC set to 3')
"

# Run training
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

EXIT_CODE=$?

# Restore original yaml
cp hparams/train.yaml.bak hparams/train.yaml
rm hparams/train.yaml.bak

echo ""
echo "=== Verification ==="
python -c "
import os
d = 'outputs'
log = os.path.join(d, 'training.log')
if not os.path.exists(log):
    print('!! MISSING training.log'); exit(1)
with open(log) as f:
    c = f.read()
# Check NaN
nan_lines = [l for l in c.split('\n') if 'Trainloss' in l and 'nan' in l]
if nan_lines:
    print(f'!! NaN in {len(nan_lines)} lines - BROKEN')
else:
    print('OK: No NaN in losses')
# Show key results
for l in c.split('\n'):
    if any(k in l for k in ['Trainloss at epoch 0', 'Trainloss at epoch 2', 'Folds accuracy', 'Avg accuracy', 'Training time']):
        print(f'  {l.strip()}')
ckpts = [f for f in os.listdir(d) if f.startswith('bestmodel')]
print(f'  Checkpoints: {len(ckpts)}/5')
"

echo ""
echo "=== Done (exit code ${EXIT_CODE}): $(date) ==="
