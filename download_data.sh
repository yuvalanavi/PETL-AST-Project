#!/usr/bin/env bash
# Download and extract the ESC-50 dataset into data/ESC-50/
set -euo pipefail

DATA_DIR="data"
ESC_DIR="${DATA_DIR}/ESC-50"

if [ -d "${ESC_DIR}/audio" ] && [ -f "${ESC_DIR}/meta/esc50.csv" ]; then
    echo "ESC-50 already downloaded at ${ESC_DIR}. Skipping."
    exit 0
fi

echo "Downloading ESC-50 dataset..."
mkdir -p "${DATA_DIR}"

# Clone the ESC-50 repo (contains audio + metadata)
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git "${ESC_DIR}"

# Verify
if [ -f "${ESC_DIR}/meta/esc50.csv" ]; then
    echo "Download complete. Dataset at ${ESC_DIR}/"
    echo "  Audio files: $(ls ${ESC_DIR}/audio/*.wav 2>/dev/null | wc -l | tr -d ' ') files"
else
    echo "ERROR: Download may have failed. Check ${ESC_DIR}/"
    exit 1
fi
