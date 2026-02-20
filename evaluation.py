#!/usr/bin/env python3
"""
Standalone evaluation script for PETL-AST project.
Loads a trained model checkpoint and evaluates on the ESC-50 test fold.

Usage:
    python evaluation.py --data_path data --checkpoint outputs/bestmodel_fold0 --fold 0
    python evaluation.py --data_path data --checkpoint_dir outputs   # evaluates all folds
"""

import torch
import argparse
import numpy as np
import yaml
import os
import glob

from src.AST_adapters import AST_adapter
from dataset.esc_50 import ESC_50
from utils.engine import eval_one_epoch
from torch.utils.data import DataLoader


FOLDS_TRAIN = [[1,2,3], [2,3,4], [3,4,5], [4,5,1], [5,1,2]]
FOLDS_VALID = [[4], [5], [1], [2], [3]]
FOLDS_TEST  = [[5], [1], [2], [3], [4]]


def load_model(checkpoint_path, max_length, num_classes, device, reduction_rate=96,
               adapter_type='Pfeiffer', kernel_size=8):
    """Instantiate the model and load saved weights."""
    model = AST_adapter(
        max_length=max_length,
        num_classes=num_classes,
        final_output='ALL',
        reduction_rate=reduction_rate,
        adapter_type=adapter_type,
        seq_or_par='parallel',
        apply_residual=False,
        adapter_block='conformer',
        kernel_size=kernel_size,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def evaluate_fold(args, fold, checkpoint_path, train_params, device):
    """Evaluate a single fold and return test accuracy."""
    max_len = train_params['max_len_AST_ESC']
    num_classes = train_params['num_classes_ESC']
    batch_size = train_params['batch_size_ESC']

    test_data = ESC_50(
        args.data_path, max_len, 'test',
        train_fold_nums=FOLDS_TRAIN[fold],
        valid_fold_nums=FOLDS_VALID[fold],
        test_fold_nums=FOLDS_TEST[fold],
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = load_model(
        checkpoint_path, max_len, num_classes, device,
        reduction_rate=args.reduction_rate,
        kernel_size=args.kernel_size,
    )

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)

    print(f"Fold {fold}: test_loss={test_loss:.4f}, test_acc={test_acc*100:.2f}%")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate PETL-AST on ESC-50')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with bestmodel_fold* files')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (0-4) for single checkpoint eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--reduction_rate', type=int, default=96)
    parser.add_argument('--kernel_size', type=int, default=8)
    parser.add_argument('--hparams', type=str, default='hparams/train.yaml')
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.hparams, 'r') as f:
        train_params = yaml.safe_load(f)

    if args.checkpoint and args.fold is not None:
        evaluate_fold(args, args.fold, args.checkpoint, train_params, device)

    elif args.checkpoint_dir:
        ckpt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'bestmodel_fold*')))
        if not ckpt_files:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            return

        accuracies = []
        for ckpt_path in ckpt_files:
            fold_idx = int(ckpt_path.split('fold')[-1])
            acc = evaluate_fold(args, fold_idx, ckpt_path, train_params, device)
            accuracies.append(acc)

        print(f"\nAverage accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"Std:              {np.std(accuracies)*100:.2f}%")
    else:
        parser.error("Provide either --checkpoint + --fold, or --checkpoint_dir")


if __name__ == '__main__':
    main()
