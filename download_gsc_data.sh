#!/usr/bin/env bash
# Download and extract the Google Speech Commands v2 dataset
set -euo pipefail

DATA_DIR="data"
GSC_DIR="${DATA_DIR}/speech_commands_v0.02"

if [ -d "${GSC_DIR}/stop" ] && [ -f "${GSC_DIR}/validation_list.txt" ]; then
    echo "GSC v2 already downloaded at ${GSC_DIR}. Skipping."
    exit 0
fi

echo "Downloading Google Speech Commands v2 (this may take a while)..."
mkdir -p "${GSC_DIR}"

# Download the full dataset archive
wget -q http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O /tmp/gsc_v2.tar.gz

echo "Extracting..."
tar -xzf /tmp/gsc_v2.tar.gz -C "${GSC_DIR}"
rm /tmp/gsc_v2.tar.gz

# Verify
if [ -f "${GSC_DIR}/validation_list.txt" ]; then
    echo "Download complete. Dataset at ${GSC_DIR}/"
    # GSC has 35 folders for the 35 commands
    FOLDER_COUNT=$(find "${GSC_DIR}" -maxdepth 1 -type d | wc -l)
    echo "  Command folders: $((FOLDER_COUNT-1))"
else
    echo "ERROR: Download may have failed. Check ${GSC_DIR}/"
    exit 1
fi
