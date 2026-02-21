#!/bin/bash
# =============================================================================
# One-time setup script for PETL-AST on the TAU CS SLURM cluster.
# Run this interactively on the client node (c-00X) after SSH-ing in.
#
# Usage:  bash ~/petl-ast/PETL-AST-Project/slurm/setup_cluster.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$HOME/petl-ast"
VENV_DIR="${PROJECT_DIR}/venv"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"

echo "=== PETL-AST Cluster Setup ==="
echo "Project dir: ${PROJECT_DIR}"
echo ""

# --- 1. Check repo exists ---
if [ ! -d "${REPO_DIR}" ]; then
    echo "[1/4] Cloning repository..."
    mkdir -p "${PROJECT_DIR}"
    git clone https://github.com/yuvalanavi/PETL-AST-Project.git "${REPO_DIR}"
else
    echo "[1/4] Repository exists. Pulling latest..."
    cd "${REPO_DIR}" && git pull && cd -
fi

# --- 2. Create Python 3.10 virtual environment ---
if [ -d "${VENV_DIR}" ]; then
    echo "[2/4] Virtual environment already exists."
else
    echo "[2/4] Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- 3. Install dependencies (authors' exact versions) ---
echo "[3/4] Installing PyTorch 1.13.1 + CUDA 11.7 and dependencies..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 -q

pip install transformers==4.28.1 numpy==1.22.3 librosa==0.9.2 -q
pip install soundfile wandb pyyaml matplotlib -q

echo "      Verifying..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import ASTModel; print('  transformers OK')"
python -c "import librosa; print('  librosa OK')"

# --- 4. Download ESC-50 dataset ---
echo "[4/4] Downloading ESC-50 dataset..."
cd "${REPO_DIR}"
bash download_data.sh

echo ""
echo "=== Setup complete! ==="
echo "Disk usage: $(du -sh ${PROJECT_DIR} | cut -f1)"
echo ""
echo "Next steps:"
echo "  cd ${REPO_DIR}"
echo "  sbatch slurm/smoke_test.slurm"
echo "  squeue --me"
