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
QUICK START (SLURM Cluster — Recommended)
================================================================================

Prerequisites: SSH access to TAU CS SLURM cluster, Python 3.10, CUDA GPU.

  1. SSH into the cluster:
       ssh <username>@slurm-client.cs.tau.ac.il

  2. One-time setup (installs venv + deps + downloads ESC-50):
       cd <repo-dir>
       bash slurm/setup_cluster.sh

  3. Smoke test (5 folds × 3 epochs — ~15-30 min):
       sbatch slurm/smoke_test.slurm
       squeue --me                          # monitor
       cat slurm/logs/petl-smoke_<jobid>.out  # view output

  4. Full training (5 folds × 50 epochs — ~3-5 hours):
       sbatch slurm/full_training.slurm

  5. Generate convergence plots:
       python utils/visualization.py --log_dir outputs --output_dir outputs/plots

  6. Evaluate saved checkpoints:
       python evaluation.py --data_path data --checkpoint_dir outputs --device cuda

================================================================================
ALTERNATIVE: Google Colab
================================================================================

  1. Open PETL_AST_Colab.ipynb and run cells in order.
     Note: Colab uses Python 3.12 + PyTorch 2.x, which differs from the authors'
     original environment. The SLURM cluster approach is preferred as it matches
     the authors' exact setup (Python 3.10, PyTorch 1.13.1, transformers 4.28.1).

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

Note: Training locally on CPU/MPS is slow. Use the SLURM cluster for actual runs.

================================================================================
PROJECT STRUCTURE
================================================================================

  PETL-AST-Project/
  ├── main.py                         Authors' training script (unmodified)
  ├── train.py                        Our wrapper: adds stdout logging
  ├── evaluation.py                   Our standalone evaluation script
  ├── src/                            Authors' model implementations (unmodified)
  │   ├── AST_adapters.py             ★ Conformer + Bottleneck adapters
  │   ├── AST.py                      Base AST wrapper
  │   ├── AST_LoRA.py                 LoRA implementation
  │   ├── AST_prompt_tuning.py        Prompt/prefix tuning
  │   ├── MoA.py                      Mixture of Adapters
  │   └── Wav2Vec_adapter.py          Wav2Vec adapters
  ├── dataset/                        Authors' dataset classes (unmodified)
  │   ├── esc_50.py                   ★ ESC-50 loader
  │   └── ...                         Other datasets
  ├── utils/
  │   ├── engine.py                   Authors' train/eval loops (unmodified)
  │   └── visualization.py            Our convergence plot generation
  ├── hparams/train.yaml              Authors' hyperparams (1-line bugfix)
  ├── download_data.sh                Our ESC-50 download script
  ├── slurm/                          SLURM cluster job scripts
  │   ├── setup_cluster.sh            One-time venv + data setup
  │   ├── smoke_test.slurm            Quick validation job
  │   └── full_training.slurm         Full 50-epoch training job
  ├── PETL_AST_Colab.ipynb            Colab notebook (alternative)
  ├── requirements.txt                Dependencies (authors' original versions)
  ├── docs/                           Planning + project docs
  └── readme.txt                      This file

================================================================================
WHAT WE WROTE VS AUTHORS' CODE
================================================================================

Authors' code (from github.com/umbertocappellazzo/PETL_AST, unmodified):
  - src/            All model implementations
  - utils/engine.py Training and evaluation loops
  - dataset/        Dataset loader classes

Authors' code with minimal modifications:
  - main.py              3 lines added: START_FOLD/END_FOLD/EPOCHS_OVERRIDE
                         env vars for per-fold parallelism (no change when unset)
  - hparams/train.yaml   Added epochs_ESC key (bug: main.py reads this key)
                         Reduced batch_size_ESC to 16 (TITAN Xp 12GB OOM at 32)

Our additions:
  - train.py                Wrapper script (stdout logging to file)
  - evaluation.py           Standalone evaluation script (course requirement)
  - utils/visualization.py  Convergence plot generation
  - download_data.sh        Dataset download automation
  - slurm/                  SLURM cluster scripts (setup, smoke test, full training)
  - PETL_AST_Colab.ipynb    Ready-to-run Colab notebook
  - readme.txt              This documentation
  - requirements.txt        Dependencies (authors' original versions)
  - docs/challenges.md      Detailed log of all reproduction challenges

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:  MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace)
Dataset:            ESC-50 — 2000 clips, 50 classes, 5-fold CV
Libraries:          PyTorch 1.13.1, HuggingFace Transformers 4.28.1, librosa,
                    torchaudio, matplotlib
Environment:        Python 3.10, CUDA 11.7
