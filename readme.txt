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
QUICK START (Google Colab — Recommended)
================================================================================

  1. Upload / clone this repo to Colab:
       !git clone <your-repo-url>
       %cd PETL-AST-Project

  2. Install dependencies (torch/torchaudio are pre-installed on Colab):
       !pip install -r requirements.txt

  3. Download the ESC-50 dataset:
       !bash download_data.sh

  4. Smoke test (single fold, 3 epochs) — takes ~5 min on T4:
       !python main.py \
           --data_path 'data' --dataset_name 'ESC-50' \
           --method 'adapter' --adapter_block 'conformer' \
           --adapter_type 'Pfeiffer' --seq_or_par 'parallel' \
           --reduction_rate_adapter 96 --kernel_size 8 \
           --device cuda --num_folds 1 --num_epochs 3 \
           --save_best_ckpt True --output_path './outputs'

  5. Full training (5-fold × 50 epochs) — ~2.5 hours on T4:
       !python main.py \
           --data_path 'data' --dataset_name 'ESC-50' \
           --method 'adapter' --adapter_block 'conformer' \
           --adapter_type 'Pfeiffer' --seq_or_par 'parallel' \
           --reduction_rate_adapter 96 --kernel_size 8 \
           --device cuda \
           --save_best_ckpt True --output_path './outputs'

  6. Generate convergence plots:
       !python utils/visualization.py --log_dir outputs --output_dir outputs/plots

  7. Evaluate saved checkpoints:
       !python evaluation.py --data_path data --checkpoint_dir outputs --device cuda

================================================================================
LOCAL SETUP (Optional — for development only)
================================================================================

PREREQUISITES:
  - Python 3.10 (verify: python3.10 --version)
  - pip, git

  Install Python 3.10 if needed:
    macOS (Homebrew):  brew install python@3.10
    Ubuntu/Debian:     sudo apt install python3.10 python3.10-venv

SETUP:
  1. Clone:    git clone <your-repo-url> && cd PETL-AST-Project
  2. Venv:     python3.10 -m venv .venv && source .venv/bin/activate
  3. Deps:     pip install -r requirements.txt
  4. Data:     bash download_data.sh
  5. Verify:   python -c "import torch; from transformers import ASTModel; print('OK')"

Note: Training locally on CPU/MPS is slow. Use Colab T4 for actual runs.

================================================================================
PROJECT STRUCTURE
================================================================================

  PETL-AST-Project/
  ├── main.py                         Training script (train.py)
  ├── evaluation.py                   Standalone evaluation script
  ├── src/                            Model implementations
  │   ├── AST_adapters.py             ★ Conformer + Bottleneck adapters
  │   ├── AST.py                      Base AST wrapper
  │   ├── AST_LoRA.py                 LoRA implementation
  │   ├── AST_prompt_tuning.py        Prompt/prefix tuning
  │   ├── MoA.py                      Mixture of Adapters
  │   └── Wav2Vec_adapter.py          Wav2Vec adapters
  ├── dataset/                        Dataset classes
  │   ├── esc_50.py                   ★ ESC-50 (our focus)
  │   └── ...                         Other datasets
  ├── utils/
  │   ├── engine.py                   Train/eval loops
  │   └── visualization.py            Convergence plot generation
  ├── hparams/train.yaml              Hyperparameters
  ├── download_data.sh                ESC-50 download script
  ├── requirements.txt                Dependencies
  ├── docs/                           Planning + project docs
  └── readme.txt                      This file

================================================================================
WHAT WE WROTE VS AUTHORS' CODE
================================================================================

Authors' code (from github.com/umbertocappellazzo/PETL_AST):
  - src/            All model implementations
  - dataset/        All dataset classes
  - utils/engine.py Training and evaluation loops
  - hparams/        Hyperparameter configs
  - main.py         Main training pipeline

Our additions:
  - evaluation.py           Standalone evaluation script (course requirement)
  - utils/visualization.py  Convergence plot generation
  - download_data.sh        Dataset download automation
  - readme.txt              This documentation
  - requirements.txt        Updated for Colab/modern torch compatibility
  - docs/                   Sprint planning and project documentation

Modifications to authors' code (all marked with "# MODIFIED:"):
  - main.py:        Added CSV logging per fold for convergence plots;
                    fixed output_path concatenation bug
  - dataset/esc_50.py: torch.as_tensor() conversion for torch 2.x compat
  - hparams/train.yaml: Added epochs_ESC key (was missing; main.py expects it)

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:  MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace)
Dataset:            ESC-50 — 2000 clips, 50 classes, 5-fold CV
Libraries:          PyTorch, HuggingFace Transformers, librosa, matplotlib
