#!/bin/bash
# =============================================================================
# One-time setup on the TAU CS SLURM cluster.
# Uses NLP course storage (large volume, no home quota issues).
#
# Usage:  bash setup_cluster.sh
# =============================================================================
set -euo pipefail

STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/yuvalanavi"
PROJECT_DIR="${STORAGE}/petl-ast"
VENV_DIR="${PROJECT_DIR}/venv"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"

echo "=== PETL-AST Cluster Setup ==="
echo "Storage: ${STORAGE}"
echo ""

# --- 1. Create project directory ---
mkdir -p "${PROJECT_DIR}"

# --- 2. Clone / update repo ---
if [ ! -d "${REPO_DIR}" ]; then
    echo "[1/4] Cloning repository..."
    git clone https://github.com/yuvalanavi/PETL-AST-Project.git "${REPO_DIR}"
else
    echo "[1/4] Pulling latest..."
    cd "${REPO_DIR}" && git pull
fi

# --- 3. Create venv + install authors' exact requirements ---
if [ -f "${VENV_DIR}/bin/activate" ]; then
    echo "[2/4] Virtual environment already exists."
else
    echo "[2/4] Creating virtual environment..."
    rm -rf "${VENV_DIR}"
    python3 -m venv --without-pip "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    echo "      Bootstrapping pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

echo "[3/4] Installing authors' exact requirements..."
pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 -q

pip install --no-cache-dir \
    transformers==4.28.1 \
    numpy==1.22.3 \
    librosa==0.9.2 \
    soundfile wandb pyyaml matplotlib -q

echo "      Verifying..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import ASTModel; print('  transformers OK')"
python -c "import librosa; print('  librosa OK')"

# --- 4. Download ESC-50 + prepare directories ---
echo "[4/4] Downloading ESC-50 dataset..."
cd "${REPO_DIR}"
mkdir -p outputs slurm/logs
bash download_data.sh

echo ""
echo "=== Setup complete! ==="
echo "Disk usage: $(du -sh ${PROJECT_DIR} | cut -f1)"
echo ""
echo "Next steps:"
echo "  cd ${REPO_DIR}"
echo "  sbatch slurm/smoke_test.slurm"
echo "  squeue --me"
