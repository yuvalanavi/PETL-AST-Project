================================================================================
PETL-AST Project — Conformer Adapter for Audio Spectrogram Transformer
================================================================================

Paper: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers"
       (Cappellazzo et al., 2024) — https://arxiv.org/abs/2312.03694
Task:  Environmental Sound Classification on ESC-50
Team:  Yuval Anavi (318677622), Eden Avrahami (207106444),
       Guy Yaffe (207253980), Tom Nouri (209402833)

Codebase: Based on the authors' published code at
          https://github.com/umbertocappellazzo/PETL_AST
          See "WHAT WE WROTE VS AUTHORS' CODE" below for attribution.

================================================================================
ONBOARDING — READ THIS FIRST
================================================================================

PREREQUISITES:
  - Python 3.10 (required by course — verify with: python3.10 --version)
  - pip
  - git
  - ~2GB free disk (model weights + ESC-50 dataset)

  If you don't have Python 3.10:
    macOS (Homebrew):  brew install python@3.10
    Ubuntu/Debian:     sudo apt install python3.10 python3.10-venv
    Colab:             Already available — skip venv steps, pip install directly.

SETUP (5 minutes):

  1. Clone the repo:
       git clone git@github.com:yuvalanavi/PETL-AST-Project.git
       cd PETL-AST-Project

  2. Create and activate a virtual environment:
       python3.10 -m venv .venv
       source .venv/bin/activate

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
  - Read docs/planning/execution-plan.md for the full sprint plan.
  - The key file to understand: src/AST_adapters.py (Conformer Adapter).
  - Hyperparameters: hparams/train.yaml
  - The .cursor/rules/ file will auto-guide your Cursor AI agent.

================================================================================
PROJECT STRUCTURE
================================================================================

  PETL-AST-Project/
  ├── src/                            Authors' model implementations
  │   ├── AST_adapters.py             ★ Conformer + Bottleneck adapters
  │   ├── AST.py                      Base AST wrapper
  │   ├── AST_LoRA.py                 LoRA implementation
  │   ├── AST_prompt_tuning.py        Prompt/prefix tuning
  │   ├── MoA.py                      Mixture of Adapters
  │   └── Wav2Vec_adapter.py          Wav2Vec adapters
  ├── dataset/                        Authors' dataset classes
  │   ├── esc_50.py                   ★ ESC-50 (our focus)
  │   └── ...                         Other datasets
  ├── utils/engine.py                 Authors' train/eval loops
  ├── hparams/train.yaml              Authors' hyperparameters
  ├── main.py                         Authors' main training script
  ├── evaluation.py                   Our standalone evaluation script
  ├── download_data.sh                Our ESC-50 download script
  ├── docs/                           Our planning + project docs
  ├── requirements.txt                Updated dependencies
  └── readme.txt                      This file

================================================================================
HOW TO TRAIN
================================================================================

Run 5-fold cross-validation on ESC-50 with Conformer Adapter (Pfeiffer):

  python main.py \
      --data_path 'data' \
      --dataset_name 'ESC-50' \
      --method 'adapter' \
      --adapter_block 'conformer' \
      --adapter_type 'Pfeiffer' \
      --seq_or_par 'parallel' \
      --reduction_rate_adapter 96 \
      --kernel_size 8 \
      --apply_residual False \
      --is_AST True \
      --seed 10

Expected output:
  - Trains 5 folds × 50 epochs each
  - Prints per-fold accuracy and average (~88.30%)
  - Reports trainable parameter count (~271K)

Optional flags:
  --device cuda                       GPU device (default: cuda)
  --save_best_ckpt True               Save best checkpoint per fold
  --output_path ./outputs             Checkpoint save directory
  --use_wandb True                    Enable Weights & Biases logging
  --project_name 'PETL-AST'          W&B project name

================================================================================
HOW TO EVALUATE
================================================================================

Evaluate a saved checkpoint on a specific fold:

  python evaluation.py \
      --data_path 'data' \
      --checkpoint outputs/bestmodel_fold0 \
      --fold 0

================================================================================
WHAT WE WROTE VS AUTHORS' CODE
================================================================================

Authors' code (from github.com/umbertocappellazzo/PETL_AST):
  - src/            All model implementations
  - dataset/        All dataset classes
  - utils/          Training and evaluation loops
  - hparams/        Hyperparameter configs
  - main.py         Main training pipeline

Our additions:
  - evaluation.py   Standalone evaluation script (course requirement)
  - download_data.sh  Dataset download automation
  - readme.txt      This documentation
  - docs/           Sprint planning and project documentation
  - requirements.txt  Updated dependencies (torch 1.13→2.4, transformers 4.28→4.57)
  - Any modifications to authors' code are marked with # MODIFIED: <reason>

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:
  - HuggingFace: MIT/ast-finetuned-audioset-10-10-0.4593

Dataset:
  - ESC-50: https://github.com/karolpiczak/ESC-50
  - 2000 clips, 50 classes, 5-fold CV

Libraries:
  - PyTorch, HuggingFace Transformers, librosa, matplotlib
