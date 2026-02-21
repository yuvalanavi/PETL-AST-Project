#!/usr/bin/env python3
"""
Thin wrapper around the authors' main.py.
- Captures stdout to a log file for later parsing into convergence plots
- Does NOT reimplement the training loop — calls main.main() directly

Usage:
  python train.py [all main.py args]

The log file is saved to outputs/training.log (or the --output_path directory).
"""

import sys
import os


class TeeStream:
    """Write to both a file and stdout simultaneously."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


if __name__ == '__main__':
    output_dir = 'outputs'
    for i, arg in enumerate(sys.argv):
        if arg == '--output_path' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1].lstrip('./')
            break

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training.log')

    tee = TeeStream(log_path)
    sys.stdout = tee

    import main as authors_main
    from argparse import ArgumentParser

    parser = ArgumentParser('PETL-AST', parents=[authors_main.get_args_parser()])
    args = parser.parse_args()
    authors_main.main(args)

    sys.stdout = tee.stdout
    tee.close()
    print(f"\nTraining log saved to: {log_path}")
