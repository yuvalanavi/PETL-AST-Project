#!/bin/bash
# =============================================================================
# One-time setup: clone repo + download ESC-50 data.
# No venv needed — SLURM jobs use NGC containers (per cluster guidelines).
#
# Usage:  bash ~/petl-ast/PETL-AST-Project/slurm/setup_cluster.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$HOME/petl-ast"
REPO_DIR="${PROJECT_DIR}/PETL-AST-Project"

echo "=== PETL-AST Cluster Setup ==="

# --- 1. Repo ---
if [ ! -d "${REPO_DIR}" ]; then
    echo "[1/3] Cloning repository..."
    mkdir -p "${PROJECT_DIR}"
    git clone https://github.com/yuvalanavi/PETL-AST-Project.git "${REPO_DIR}"
else
    echo "[1/3] Pulling latest..."
    cd "${REPO_DIR}" && git pull
fi

cd "${REPO_DIR}"

# --- 2. Download ESC-50 ---
echo "[2/3] Downloading ESC-50 dataset..."
bash download_data.sh

# --- 3. Make scripts executable ---
echo "[3/3] Setting permissions..."
chmod +x slurm/run_smoke_test.sh slurm/run_full_training.sh

echo ""
echo "=== Setup complete! ==="
echo "Disk usage: $(du -sh ${PROJECT_DIR} | cut -f1)"
echo ""
echo "Submit smoke test:"
echo "  cd ${REPO_DIR}"
echo "  sbatch slurm/smoke_test.slurm"
echo "  squeue --me"
