#!/usr/bin/env python3
"""
Generate convergence plots from training log output.
Supports both a single combined log and per-fold SLURM output files.

Usage:
    # Single log (all folds in one file):
    python utils/visualization.py --log outputs/training.log --output_dir outputs/plots

    # Per-fold SLURM logs (one file per fold):
    python utils/visualization.py --log_dir outputs/slurm_logs --output_dir outputs/plots
"""

import os
import re
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_single_log(log_path):
    """Parse a log file to extract per-epoch metrics grouped by fold."""
    folds = {}
    current_fold = -1

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'Number of trainable params' in line:
            current_fold += 1
            folds[current_fold] = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        m_train_loss = re.match(r'Trainloss at epoch (\d+): ([\d.enan+-]+)', line)
        if m_train_loss and current_fold >= 0:
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


def parse_fold_logs(log_dir):
    """Parse per-fold SLURM output files from a directory."""
    fold_files = sorted(glob.glob(os.path.join(log_dir, '*train*fold*_*.out')))
    if not fold_files:
        fold_files = sorted(glob.glob(os.path.join(log_dir, '*fold*_*.out')))
    if not fold_files:
        fold_files = sorted(glob.glob(os.path.join(log_dir, '*.out')))

    folds = {}
    for f in fold_files:
        m = re.search(r'fold(\d+)', os.path.basename(f))
        if m:
            fold_idx = int(m.group(1))
        else:
            fold_idx = len(folds)

        parsed = parse_single_log(f)
        if parsed:
            folds[fold_idx] = list(parsed.values())[0]

    return folds


def plot_fold(fold_data, fold_idx, output_dir):
    """Generate loss and accuracy plots for a single fold."""
    n = min(len(fold_data['train_loss']), len(fold_data['train_acc']), len(fold_data['val_acc']))
    if n == 0:
        return
    fold_data = {k: v[:n] for k, v in fold_data.items()}
    epochs = list(range(n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, fold_data['train_loss'], label='Train Loss', color='#e74c3c')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss — Fold {fold_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, fold_data['train_acc'], label='Train Acc', color='#3498db')
    ax2.plot(epochs, fold_data['val_acc'], label='Val Acc', color='#2ecc71')
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


def plot_all_folds_combined(folds, output_dir):
    """Generate combined plots across all folds."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for fold_idx in sorted(folds.keys()):
        data = folds[fold_idx]
        n = min(len(data['train_loss']), len(data['train_acc']), len(data['val_acc']))
        if n == 0:
            continue
        epochs = list(range(n))
        axes[0].plot(epochs, data['train_loss'][:n], label=f'Fold {fold_idx}', alpha=0.8)
        axes[1].plot(epochs, data['train_acc'][:n], label=f'Fold {fold_idx}', alpha=0.8)
        axes[2].plot(epochs, data['val_acc'][:n], label=f'Fold {fold_idx}', alpha=0.8)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss — All Folds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy — All Folds')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Validation Accuracy — All Folds')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'convergence_all_folds.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    print("\n=== Results Summary ===")
    best_accs = []
    for fold_idx in sorted(folds.keys()):
        data = folds[fold_idx]
        if data['val_acc']:
            best_val = max(data['val_acc'])
            final_loss = data['train_loss'][-1] if data['train_loss'] else float('nan')
            best_accs.append(best_val)
            print(f"  Fold {fold_idx}: best val acc = {best_val:.2f}%, final loss = {final_loss:.6f}")
    if best_accs:
        avg = np.mean(best_accs)
        std = np.std(best_accs)
        print(f"\n  Average best val accuracy: {avg:.2f}% ± {std:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None, help='Single combined log file')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory with per-fold SLURM output files')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    args = parser.parse_args()

    if args.log_dir:
        folds = parse_fold_logs(args.log_dir)
    elif args.log:
        folds = parse_single_log(args.log)
    else:
        if os.path.isdir('outputs/slurm_logs'):
            folds = parse_fold_logs('outputs/slurm_logs')
        elif os.path.isfile('outputs/training.log'):
            folds = parse_single_log('outputs/training.log')
        else:
            print("No log files found. Use --log or --log_dir.")
            return

    if not folds:
        print("No training data found in logs.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for fold_idx, fold_data in sorted(folds.items()):
        plot_fold(fold_data, fold_idx, args.output_dir)

    plot_all_folds_combined(folds, args.output_dir)


if __name__ == '__main__':
    main()
