#!/usr/bin/env python3
"""
Generate convergence plots from training log output.
Parses the stdout log produced by main.py / train.py.

Usage:
    python utils/visualization.py --log outputs/training.log --output_dir outputs/plots
"""

import os
import re
import argparse
import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse training.log to extract per-epoch metrics grouped by fold."""
    folds = {}
    current_fold = -1

    with open(log_path, 'r') as f:
        lines = f.readlines()

    epoch = -1
    for line in lines:
        # Detect fold boundaries by counting model instantiations
        if 'Number of trainable params' in line:
            current_fold += 1
            folds[current_fold] = {'train_loss': [], 'train_acc': [], 'val_acc': []}
            epoch = -1

        m_train_loss = re.match(r'Trainloss at epoch (\d+): ([\d.enan+-]+)', line)
        if m_train_loss and current_fold >= 0:
            epoch = int(m_train_loss.group(1))
            try:
                loss = float(m_train_loss.group(2))
            except ValueError:
                loss = float('nan')
            folds[current_fold]['train_loss'].append(loss)

        m_train_acc = re.match(r'Train intent accuracy:\s+([\d.enan+-]+)', line)
        if m_train_acc and current_fold >= 0:
            try:
                acc = float(m_train_acc.group(1))
            except ValueError:
                acc = float('nan')
            folds[current_fold]['train_acc'].append(acc)

        m_val_acc = re.match(r'Valid intent accuracy:\s+([\d.enan+-]+)', line)
        if m_val_acc and current_fold >= 0:
            try:
                acc = float(m_val_acc.group(1))
            except ValueError:
                acc = float('nan')
            folds[current_fold]['val_acc'].append(acc)

    return folds


def plot_fold(fold_data, fold_idx, output_dir):
    """Generate loss and accuracy plots for a single fold."""
    epochs = list(range(len(fold_data['train_loss'])))
    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, fold_data['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss — Fold {fold_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, fold_data['train_acc'], label='Train Acc')
    ax2.plot(epochs, fold_data['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Accuracy — Fold {fold_idx}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'convergence_fold{fold_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_folds(log_path, output_dir):
    """Generate plots for all folds found in the log."""
    os.makedirs(output_dir, exist_ok=True)

    folds = parse_log(log_path)
    if not folds:
        print(f"No training data found in {log_path}")
        return

    for fold_idx, fold_data in folds.items():
        plot_fold(fold_data, fold_idx, output_dir)

    # Combined accuracy plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for fold_idx, fold_data in folds.items():
        epochs = list(range(len(fold_data['val_acc'])))
        if epochs:
            ax.plot(epochs, fold_data['val_acc'], label=f'Fold {fold_idx}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy — All Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'convergence_all_folds.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='outputs/training.log')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    args = parser.parse_args()
    plot_all_folds(args.log, args.output_dir)
