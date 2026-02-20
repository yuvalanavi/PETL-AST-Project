#!/usr/bin/env python3
"""
Generate convergence plots (loss and accuracy) from CSV training logs.

Usage:
    python utils/visualization.py --log_dir outputs --output_dir outputs/plots
"""

import os
import csv
import argparse
import glob
import matplotlib.pyplot as plt


def read_log(csv_path):
    """Read a training log CSV and return lists of metrics."""
    epochs, train_losses, val_losses = [], [], []
    train_accs, val_accs = [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            train_accs.append(float(row['train_acc']))
            val_accs.append(float(row['val_acc']))

    return epochs, train_losses, val_losses, train_accs, val_accs


def plot_fold(csv_path, output_dir, fold_idx):
    """Generate loss and accuracy plots for a single fold."""
    epochs, train_losses, val_losses, train_accs, val_accs = read_log(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss — Fold {fold_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in val_accs], label='Val Acc')
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


def plot_all_folds(log_dir, output_dir):
    """Generate plots for all folds found in log_dir."""
    os.makedirs(output_dir, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(log_dir, 'log_fold*.csv')))
    if not log_files:
        print(f"No log_fold*.csv files found in {log_dir}")
        return

    for log_path in log_files:
        fold_idx = int(log_path.split('fold')[-1].replace('.csv', ''))
        plot_fold(log_path, output_dir, fold_idx)

    # Combined accuracy plot across all folds
    fig, ax = plt.subplots(figsize=(8, 5))
    for log_path in log_files:
        fold_idx = int(log_path.split('fold')[-1].replace('.csv', ''))
        epochs, _, _, _, val_accs = read_log(log_path)
        ax.plot(epochs, [a * 100 for a in val_accs], label=f'Fold {fold_idx}')

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
    parser.add_argument('--log_dir', type=str, default='outputs')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    args = parser.parse_args()
    plot_all_folds(args.log_dir, args.output_dir)
