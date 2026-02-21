#!/bin/bash
# =============================================================================
# One-time setup script for PETL-AST on the TAU CS SLURM cluster.
# Run this interactively on the client node (c-00X) after SSH-ing in.
#
# Usage:  bash setup_cluster.sh
# =============================================================================
set -euo pipefail

COURSE_DIR="/home/yandex/APDL2526a"
PROJECT_DIR="${COURSE_DIR}/petl-ast"
VENV_DIR="${PROJECT_DIR}/venv"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"

echo "=== PETL-AST Cluster Setup ==="
echo "Course storage: ${COURSE_DIR}"
echo "Project dir:    ${PROJECT_DIR}"
echo ""

# --- 1. Create project directory ---
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"
echo "[1/5] Project directory ready: ${PROJECT_DIR}"

# --- 2. Create Python 3.10 virtual environment ---
if [ -d "${VENV_DIR}" ]; then
    echo "[2/5] Virtual environment already exists at ${VENV_DIR}"
else
    echo "[2/5] Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    echo "      Created at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- 3. Install dependencies (authors' exact versions) ---
echo "[3/5] Installing PyTorch 1.13.1 + CUDA 11.7 and dependencies..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 -q

pip install transformers==4.28.1 -q
pip install numpy==1.22.3 librosa==0.9.2 soundfile wandb pyyaml matplotlib -q

echo "      Verifying installation..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import ASTModel; print('  transformers OK')"
python -c "import librosa; print('  librosa OK')"

# --- 4. Clone repository ---
if [ -d "${REPO_DIR}" ]; then
    echo "[4/5] Repository already exists. Pulling latest..."
    cd "${REPO_DIR}"
    git pull
else
    echo "[4/5] Cloning repository..."
    git clone https://github.com/yuvalanavi/PETL-AST-Project.git "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# --- 5. Download ESC-50 dataset ---
echo "[5/5] Downloading ESC-50 dataset..."
bash download_data.sh

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run smoke test:"
echo "  cd ${REPO_DIR}"
echo "  sbatch slurm/smoke_test.slurm"
echo ""
echo "To run full training:"
echo "  cd ${REPO_DIR}"
echo "  sbatch slurm/full_training.slurm"
echo ""
echo "Monitor jobs with:  squeue --me"
