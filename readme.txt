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
       Open PETL_AST_Colab.ipynb and run cells in order.

  2. Or manually:
       !pip install -r requirements.txt
       !bash download_data.sh

  3. Smoke test (1 fold, 3 epochs — ~5 min on T4):
       !python train.py \
           --data_path data --dataset_name ESC-50 \
           --method adapter --adapter_block conformer \
           --adapter_type Pfeiffer --seq_or_par parallel \
           --reduction_rate_adapter 96 --kernel_size 8 \
           --device cuda --num_folds 1 --num_epochs 3 \
           --save_best_ckpt True --output_path ./outputs

  4. Full training (5 folds × 50 epochs — ~2.5 hours on T4):
       !python train.py \
           --data_path data --dataset_name ESC-50 \
           --method adapter --adapter_block conformer \
           --adapter_type Pfeiffer --seq_or_par parallel \
           --reduction_rate_adapter 96 --kernel_size 8 \
           --device cuda \
           --save_best_ckpt True --output_path ./outputs

  5. Generate convergence plots:
       !python utils/visualization.py --log_dir outputs --output_dir outputs/plots

  6. Evaluate saved checkpoints:
       !python evaluation.py --data_path data --checkpoint_dir outputs --device cuda

================================================================================
LOCAL SETUP (Optional — for development only)
================================================================================

PREREQUISITES:
  - Python 3.10 (verify: python3.10 --version)
  - pip, git

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
  ├── main.py                         Authors' training script (unmodified)
  ├── train.py                        Our wrapper: adds logging + smoke test flags
  ├── evaluation.py                   Our standalone evaluation script
  ├── src/                            Authors' model implementations (unmodified)
  │   ├── AST_adapters.py             ★ Conformer + Bottleneck adapters
  │   ├── AST.py                      Base AST wrapper
  │   ├── AST_LoRA.py                 LoRA implementation
  │   ├── AST_prompt_tuning.py        Prompt/prefix tuning
  │   ├── MoA.py                      Mixture of Adapters
  │   └── Wav2Vec_adapter.py          Wav2Vec adapters
  ├── dataset/                        Authors' dataset classes
  │   ├── esc_50.py                   ★ ESC-50 (1-line bugfix for torch 2.x)
  │   └── ...                         Other datasets
  ├── utils/
  │   ├── engine.py                   Authors' train/eval loops (unmodified)
  │   └── visualization.py            Our convergence plot generation
  ├── hparams/train.yaml              Authors' hyperparams (1-line bugfix)
  ├── download_data.sh                Our ESC-50 download script
  ├── PETL_AST_Colab.ipynb            Our Colab notebook
  ├── requirements.txt                Dependencies
  ├── docs/                           Planning + project docs
  └── readme.txt                      This file

================================================================================
WHAT WE WROTE VS AUTHORS' CODE
================================================================================

Authors' code (from github.com/umbertocappellazzo/PETL_AST, unmodified):
  - src/            All model implementations
  - utils/engine.py Training and evaluation loops
  - main.py         Main training pipeline

Authors' code with minimal bugfixes:
  - dataset/esc_50.py   1-line fix: torch.as_tensor() for torch 2.x compat
  - hparams/train.yaml  1-line fix: added epochs_ESC key (main.py expects it)

Our additions:
  - train.py                Our wrapper script (CSV logging, --num_folds/--num_epochs)
  - evaluation.py           Standalone evaluation script (course requirement)
  - utils/visualization.py  Convergence plot generation
  - download_data.sh        Dataset download automation
  - PETL_AST_Colab.ipynb    Ready-to-run Colab notebook
  - readme.txt              This documentation
  - requirements.txt        Dependencies for Colab compatibility
  - docs/                   Sprint planning and project documentation

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:  MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace)
Dataset:            ESC-50 — 2000 clips, 50 classes, 5-fold CV
Libraries:          PyTorch, HuggingFace Transformers 4.44.0, librosa, matplotlib
