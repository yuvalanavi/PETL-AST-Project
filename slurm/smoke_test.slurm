#!/bin/bash
# =============================================================================
# SLURM smoke test: 5 folds × 3 epochs (quick validation that everything works)
# Expected runtime: ~15-30 minutes on RTX 3090 / V100
# =============================================================================

#SBATCH --job-name=petl-smoke
#SBATCH --output=/home/yandex/APDL2526a/petl-ast/PETL-AST-Project/outputs/smoke_test_%j.out
#SBATCH --error=/home/yandex/APDL2526a/petl-ast/PETL-AST-Project/outputs/smoke_test_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
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

mkdir -p outputs

echo "=== PETL-AST Smoke Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
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
print('Set epochs_ESC=3 for smoke test')
"

# Run training (all 5 folds, 3 epochs each)
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

# Restore original yaml
cp hparams/train.yaml.bak hparams/train.yaml
rm hparams/train.yaml.bak

echo ""
echo "=== Smoke Test Verification ==="

# Check outputs
python -c "
import os, sys

output_dir = 'outputs'
errors = []

# Check log file
log = os.path.join(output_dir, 'training.log')
if not os.path.exists(log):
    errors.append('MISSING: training.log')
else:
    with open(log) as f:
        content = f.read()
    if 'nan' in content.lower().split('trainloss')[1] if 'Trainloss' in content else True:
        # Check for NaN more carefully
        nan_count = content.count('Trainloss at epoch 0: nan')
        if nan_count > 0:
            errors.append(f'NaN DETECTED in {nan_count} fold(s) - training is broken!')
    if 'Avg accuracy' in content:
        # Extract final accuracy
        for line in content.split('\n'):
            if 'Avg accuracy' in line:
                print(f'  {line.strip()}')
    if 'Folds accuracy' in content:
        for line in content.split('\n'):
            if 'Folds accuracy' in line:
                print(f'  {line.strip()}')
    if 'Training time' in content:
        for line in content.split('\n'):
            if 'Training time' in line:
                print(f'  {line.strip()}')

# Check checkpoints
ckpts = [f for f in os.listdir(output_dir) if f.startswith('bestmodel_fold')]
print(f'  Checkpoints saved: {len(ckpts)} / 5')
if len(ckpts) < 5:
    errors.append(f'Only {len(ckpts)}/5 checkpoints saved')

if errors:
    print('\nPROBLEMS:')
    for e in errors:
        print(f'  !! {e}')
    sys.exit(1)
else:
    print('\nAll checks passed! Ready for full training.')
"

echo ""
echo "=== Smoke test complete: $(date) ==="
