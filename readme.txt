================================================================================
PETL-AST Project — Conformer Adapter for Audio Spectrogram Transformer
================================================================================

Paper: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers"
       (Cappellazzo et al., 2024) — https://arxiv.org/abs/2312.03694
Task:  Environmental Sound Classification on ESC-50
Team:  Yuval Anavi (318677622), Eden Avrahami (207106444),
       Guy Yaffe (207253980), Tom Nouri (209402833)

================================================================================
ONBOARDING — READ THIS FIRST
================================================================================

PREREQUISITES:
  - Python 3.10 (required by course — verify with: python3.10 --version)
  - pip
  - git
  - ~2GB free disk (model weights + ESC-50 dataset)

SETUP (5 minutes):

  1. Clone the repo:
       git clone <REPO_URL>
       cd PETL-AST-Project

  2. Create and activate a virtual environment:
       python3.10 -m venv .venv
       source .venv/bin/activate        # Linux/Mac
       .venv\Scripts\activate           # Windows

  3. Install dependencies:
       pip install -r requirements.txt

  4. Download the ESC-50 dataset:
       bash download_data.sh
     This downloads ESC-50 into data/ESC-50/ (~600MB).
     Expected structure after download:
       data/ESC-50/
       ├── audio/          (2000 .wav files)
       └── meta/
           └── esc50.csv   (metadata with fold assignments)

  5. Verify setup:
       python -c "import torch; from transformers import ASTModel; print('OK')"

ORIENTATION:
  - Read docs/planning/execution-plan.md for the full sprint plan and your tasks.
  - Read configs/esc50_conformer.yaml for all hyperparameters.
  - The .cursor/rules/ file will auto-guide your Cursor AI agent.

================================================================================
PROJECT STRUCTURE
================================================================================

  PETL-AST-Project/
  ├── configs/
  │   └── esc50_conformer.yaml      All hyperparameters (single source of truth)
  ├── data/                         Downloaded ESC-50 (gitignored)
  ├── docs/
  │   ├── planning/                 Sprint plan
  │   └── project/                  Paper, guidelines, proposal
  ├── outputs/                      Checkpoints, logs, plots (gitignored)
  ├── samples/                      Audio samples for submission
  ├── src/
  │   ├── models/
  │   │   ├── conformer_adapter.py  Conformer Adapter nn.Module
  │   │   └── ast_with_adapter.py   AST + adapter injection + freeze logic
  │   ├── data/
  │   │   └── esc50_dataset.py      ESC-50 Dataset class (5-fold CV)
  │   └── utils/
  │       ├── engine.py             train_one_epoch, eval_one_epoch
  │       └── visualization.py      Convergence plot generation
  ├── train.py                      Training entry point
  ├── evaluation.py                 Evaluation entry point
  ├── download_data.sh              ESC-50 download script
  ├── requirements.txt              Python dependencies
  └── readme.txt                    This file

================================================================================
HOW TO TRAIN
================================================================================

Run 5-fold cross-validation training on ESC-50:

  python train.py --config configs/esc50_conformer.yaml

Optional overrides:

  python train.py --config configs/esc50_conformer.yaml \
                  --device cuda \
                  --output_dir outputs/run_01

The script will:
  - Train a Conformer Adapter (Pfeiffer, parallel) injected into a frozen AST
  - Run 5-fold CV as required by the ESC-50 protocol
  - Save checkpoints and training logs (CSV) to the output directory
  - Print per-fold and average accuracy at the end

================================================================================
HOW TO EVALUATE
================================================================================

Evaluate a trained model checkpoint:

  python evaluation.py --config configs/esc50_conformer.yaml \
                       --checkpoint outputs/run_01/fold_0/best_model.pt \
                       --fold 0

Or evaluate all 5 folds:

  python evaluation.py --config configs/esc50_conformer.yaml \
                       --checkpoint_dir outputs/run_01

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:
  - HuggingFace: MIT/ast-finetuned-audioset-10-10-0.4593
  - Used as frozen backbone (we wrote the adapter + training code ourselves)

Reference implementation (for understanding, not copy-paste):
  - https://github.com/umbertocappellazzo/PETL_AST

Libraries:
  - PyTorch (training framework)
  - HuggingFace Transformers (pre-trained AST model loading)
  - librosa (audio resampling)
  - matplotlib (convergence plots)

Dataset:
  - ESC-50: https://github.com/karolpiczak/ESC-50
  - 2000 clips, 50 classes, 5-fold CV
