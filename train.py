#!/usr/bin/env python3
"""
Wrapper around the authors' main.py that adds:
  - CSV logging per fold (for convergence plots)
  - --num_folds / --num_epochs overrides (for smoke testing)

Does NOT modify authors' code. Monkey-patches the training loop at runtime.

Usage:
  python train.py [all main.py args] [--num_folds N] [--num_epochs N]

Examples:
  # Smoke test (1 fold, 3 epochs)
  python train.py --data_path data --dataset_name ESC-50 --method adapter \
      --adapter_block conformer --adapter_type Pfeiffer --seq_or_par parallel \
      --reduction_rate_adapter 96 --kernel_size 8 --device cuda \
      --num_folds 1 --num_epochs 3 --save_best_ckpt True --output_path ./outputs

  # Full training (same args, drop --num_folds/--num_epochs)
  python train.py --data_path data --dataset_name ESC-50 --method adapter \
      --adapter_block conformer --adapter_type Pfeiffer --seq_or_par parallel \
      --reduction_rate_adapter 96 --kernel_size 8 --device cuda \
      --save_best_ckpt True --output_path ./outputs
"""

import sys
import os
import csv
import main as authors_main


def patch_and_run():
    parser = authors_main.argparse.ArgumentParser(
        'PETL-AST with logging wrapper',
        parents=[authors_main.get_args_parser()]
    )
    parser.add_argument('--num_folds', type=int, default=None,
                        help='Override fold count (e.g. 1 for smoke test)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override epoch count (e.g. 3 for smoke test)')
    args = parser.parse_args()

    # Extract our extra args, then delegate to the original main()
    num_folds_override = args.num_folds
    num_epochs_override = args.num_epochs
    # Remove our args so the original code doesn't see them
    delattr(args, 'num_folds')
    delattr(args, 'num_epochs')

    # Monkey-patch the main function to inject logging + overrides
    original_main = authors_main.main

    def patched_main(args):
        import yaml
        import numpy as np
        import time
        import datetime
        import copy
        import torch
        from torch.optim import AdamW
        from utils.engine import train_one_epoch, eval_one_epoch
        from torch.utils.data import DataLoader
        from dataset.esc_50 import ESC_50

        start_time = time.time()

        if args.use_wandb:
            import wandb
            wandb.init(project=args.project_name, name=args.exp_name, entity=args.entity)

        print(args)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        device = torch.device(args.device)

        if args.seed:
            seed = args.seed
            torch.manual_seed(seed)
            np.random.seed(seed)

        with open('hparams/train.yaml', 'r') as file:
            train_params = yaml.safe_load(file)

        if args.dataset_name == 'ESC-50':
            max_len_AST = train_params['max_len_AST_ESC']
            num_classes = train_params['num_classes_ESC']
            batch_size = train_params['batch_size_ESC']
            epochs = train_params['epochs_ESC']
        else:
            # For non-ESC-50 datasets, fall back to original main
            original_main(args)
            return

        if args.method == 'prompt-tuning':
            final_output = train_params['final_output_prompt_tuning']
        else:
            final_output = train_params['final_output']

        # Apply overrides
        if num_epochs_override is not None:
            epochs = num_epochs_override

        fold_number = 5
        folds_train = [[1,2,3], [2,3,4], [3,4,5], [4,5,1], [5,1,2]]
        folds_valid = [[4], [5], [1], [2], [3]]
        folds_test = [[5], [1], [2], [3], [4]]

        if num_folds_override is not None:
            fold_number = min(num_folds_override, fold_number)

        from src.AST_adapters import AST_adapter
        criterion = torch.nn.CrossEntropyLoss()
        output_dir = args.output_path.lstrip('./')
        accuracy_folds = []

        for fold in range(0, fold_number):
            # Skip folds that already have a completed checkpoint + log (crash recovery)
            existing_ckpt = os.path.join(output_dir, f'bestmodel_fold{fold}')
            existing_log = os.path.join(output_dir, f'log_fold{fold}.csv')
            if os.path.isfile(existing_ckpt) and os.path.isfile(existing_log):
                with open(existing_log) as _f:
                    _rows = list(csv.DictReader(_f))
                if len(_rows) >= epochs:
                    print(f"\n=== Fold {fold} already complete ({len(_rows)} epochs logged). Skipping. ===")
                    accuracy_folds.append(None)
                    continue

            print(f"\n{'='*60}")
            print(f"  FOLD {fold+1}/{fold_number}")
            print(f"{'='*60}")
            train_data = ESC_50(args.data_path, max_len_AST, 'train',
                                train_fold_nums=folds_train[fold],
                                valid_fold_nums=folds_valid[fold],
                                test_fold_nums=folds_test[fold],
                                apply_SpecAug=True)
            val_data = ESC_50(args.data_path, max_len_AST, 'valid',
                              train_fold_nums=folds_train[fold],
                              valid_fold_nums=folds_valid[fold],
                              test_fold_nums=folds_test[fold])
            test_data = ESC_50(args.data_path, max_len_AST, 'test',
                               train_fold_nums=folds_train[fold],
                               valid_fold_nums=folds_valid[fold],
                               test_fold_nums=folds_test[fold])

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

            lr = train_params['lr_adapter']
            model = AST_adapter(
                max_length=max_len_AST, num_classes=num_classes,
                final_output=final_output,
                reduction_rate=args.reduction_rate_adapter,
                adapter_type=args.adapter_type,
                seq_or_par=args.seq_or_par,
                apply_residual=args.apply_residual,
                adapter_block=args.adapter_block,
                kernel_size=args.kernel_size,
                model_ckpt=args.model_ckpt_AST
            ).to(device)

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of trainable params of the model:', n_parameters)

            optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                              eps=1e-6, weight_decay=train_params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader) * epochs)

            print(f"Start training for {epochs} epochs")
            best_acc = 0.

            # CSV logging
            output_dir = args.output_path.lstrip('./')
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f'log_fold{fold}.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.DictWriter(csv_file,
                fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
            csv_writer.writeheader()

            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, scheduler, device, criterion)
                print(f"Trainloss at epoch {epoch}: {train_loss}")

                val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = model.state_dict()
                    if args.save_best_ckpt:
                        os.makedirs(output_dir, exist_ok=True)
                        torch.save(best_params,
                                   os.path.join(output_dir, f'bestmodel_fold{fold}'))

                print("Train intent accuracy: ", train_acc * 100)
                print("Valid intent accuracy: ", val_acc * 100)

                current_lr = optimizer.param_groups[0]['lr']
                print('Learning rate after initialization: ', current_lr)

                csv_writer.writerow({
                    'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc, 'lr': current_lr
                })
                csv_file.flush()

                if args.use_wandb:
                    import wandb
                    wandb.log({"train_loss": train_loss, "valid_loss": val_loss,
                               "train_accuracy": train_acc, "val_accuracy": val_acc,
                               "lr": current_lr})

            csv_file.close()

            best_model = copy.copy(model)
            best_model.load_state_dict(best_params)
            test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)
            accuracy_folds.append(test_acc)

        # For skipped folds (crash recovery), re-evaluate to get test accuracy
        for i, acc in enumerate(accuracy_folds):
            if acc is None:
                print(f"Re-evaluating fold {i} from checkpoint...")
                ckpt_path = os.path.join(output_dir, f'bestmodel_fold{i}')
                _test_data = ESC_50(args.data_path, max_len_AST, 'test',
                                    train_fold_nums=folds_train[i],
                                    valid_fold_nums=folds_valid[i],
                                    test_fold_nums=folds_test[i])
                _test_loader = DataLoader(_test_data, batch_size=batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)
                _model = AST_adapter(
                    max_length=max_len_AST, num_classes=num_classes, final_output=final_output,
                    reduction_rate=args.reduction_rate_adapter, adapter_type=args.adapter_type,
                    seq_or_par=args.seq_or_par, apply_residual=args.apply_residual,
                    adapter_block=args.adapter_block, kernel_size=args.kernel_size,
                    model_ckpt=args.model_ckpt_AST).to(device)
                _model.load_state_dict(torch.load(ckpt_path, map_location=device))
                _, _acc = eval_one_epoch(_model, _test_loader, device, criterion)
                accuracy_folds[i] = _acc

        print("Folds accuracy: ", accuracy_folds)
        print(f"Avg accuracy over the {fold_number} fold(s): ", np.mean(accuracy_folds))
        print(f"Std accuracy over the {fold_number} fold(s): ", np.std(accuracy_folds))

        total_time = time.time() - start_time
        print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

        if args.use_wandb:
            import wandb
            wandb.finish()

    patched_main(args)


if __name__ == '__main__':
    patch_and_run()
