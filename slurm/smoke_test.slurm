#!/bin/bash
# =============================================================================
# Smoke test: 5 folds × 1 epoch — validates the full pipeline
# Expected: ~5-10 min on a single GPU
#
# Usage:  sbatch slurm/smoke_test.slurm
# =============================================================================

#SBATCH --job-name=petl-smoke
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
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

echo "=== PETL-AST Smoke Test ==="
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

# Override epochs to 3 for smoke test (restore afterwards)
cp hparams/train.yaml hparams/train.yaml.bak
python -c "
import yaml
with open('hparams/train.yaml') as f:
    hp = yaml.safe_load(f)
hp['epochs_ESC'] = 1
with open('hparams/train.yaml', 'w') as f:
    yaml.dump(hp, f, default_flow_style=False)
print('epochs_ESC set to 1')
"

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
nan_lines = [l for l in c.split('\n') if 'Trainloss' in l and 'nan' in l]
if nan_lines:
    print(f'!! NaN in {len(nan_lines)} lines - BROKEN')
else:
    print('OK: No NaN in losses')
for l in c.split('\n'):
    if any(k in l for k in ['Trainloss at epoch 0', 'Folds accuracy', 'Avg accuracy', 'Training time']):
        print(f'  {l.strip()}')
ckpts = [f for f in os.listdir(d) if f.startswith('bestmodel')]
print(f'  Checkpoints: {len(ckpts)}/5')
"
echo "=== Done: $(date) ==="
