#!/usr/bin/env python3
"""
Collect audio samples from ESC-50 for project submission.
Picks representative samples from training and validation sets (fold 0).
"""

import os
import csv
import shutil
import random

random.seed(42)

DATA_PATH = "data"
ESC_DIR = os.path.join(DATA_PATH, "ESC-50")
AUDIO_DIR = os.path.join(ESC_DIR, "audio")
META_PATH = os.path.join(ESC_DIR, "meta", "esc50.csv")
OUTPUT_DIR = "samples"

TRAIN_FOLDS = [1, 2, 3]
VAL_FOLDS = [4]

SAMPLES_PER_SPLIT = 10


def collect():
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "validation"), exist_ok=True)

    with open(META_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train_rows = [r for r in rows if int(r['fold']) in TRAIN_FOLDS]
    val_rows = [r for r in rows if int(r['fold']) in VAL_FOLDS]

    # Pick diverse samples across different categories
    train_samples = _pick_diverse(train_rows, SAMPLES_PER_SPLIT)
    val_samples = _pick_diverse(val_rows, SAMPLES_PER_SPLIT)

    print(f"=== Audio Samples (fold 0 split) ===\n")

    print(f"Training samples ({len(train_samples)}):")
    for r in train_samples:
        src = os.path.join(AUDIO_DIR, r['filename'])
        dst = os.path.join(OUTPUT_DIR, "train", r['filename'])
        shutil.copy2(src, dst)
        print(f"  {r['filename']} — {r['category']} (fold {r['fold']})")

    print(f"\nValidation samples ({len(val_samples)}):")
    for r in val_samples:
        src = os.path.join(AUDIO_DIR, r['filename'])
        dst = os.path.join(OUTPUT_DIR, "validation", r['filename'])
        shutil.copy2(src, dst)
        print(f"  {r['filename']} — {r['category']} (fold {r['fold']})")

    print(f"\nSaved to {OUTPUT_DIR}/")


def _pick_diverse(rows, n):
    """Pick n samples spread across different categories."""
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r['category'], []).append(r)

    cats = sorted(by_cat.keys())
    picked = []
    i = 0
    while len(picked) < n and i < len(cats):
        cat_rows = by_cat[cats[i % len(cats)]]
        picked.append(random.choice(cat_rows))
        i += 1

    return picked[:n]


if __name__ == '__main__':
    collect()
