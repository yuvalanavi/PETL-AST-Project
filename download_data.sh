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

# Download as tarball (works without git auth)
wget -q https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.tar.gz -O /tmp/esc50.tar.gz
tar -xzf /tmp/esc50.tar.gz -C "${DATA_DIR}"
mv "${DATA_DIR}/ESC-50-master" "${ESC_DIR}"
rm /tmp/esc50.tar.gz

# Verify
if [ -f "${ESC_DIR}/meta/esc50.csv" ]; then
    echo "Download complete. Dataset at ${ESC_DIR}/"
    AUDIO_COUNT=$(ls "${ESC_DIR}/audio/"*.wav 2>/dev/null | wc -l | tr -d ' ')
    echo "  Audio files: ${AUDIO_COUNT} files"
else
    echo "ERROR: Download may have failed. Check ${ESC_DIR}/"
    exit 1
fi
