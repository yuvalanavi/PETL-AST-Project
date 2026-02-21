#!/bin/bash
# Script that runs INSIDE the NGC container for full training
REPO_DIR="$HOME/petl-ast/PETL-AST-Project"
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
