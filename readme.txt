================================================================================
PETL-AST Project — Conformer Adapter for Audio Spectrogram Transformer
================================================================================

Paper: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers"
       (Cappellazzo et al., 2024) — https://arxiv.org/abs/2312.03694
Task:  Environmental Sound Classification (ESC-50) and Keyword Spotting (GSC)
Team:  Yuval Anavi (318677622), Eden Avrahami (207106444),
       Guy Yaffe (207253980), Tom Nouri (209402833)

Codebase: Based on the authors' published code at
          https://github.com/umbertocappellazzo/PETL_AST
          See "WHAT WE WROTE VS AUTHORS' CODE" below for attribution.

================================================================================
SETUP & TRAINING (SLURM Cluster)
================================================================================

Prerequisites: SSH access to TAU CS SLURM cluster, Python 3.10, CUDA GPU.

  1. SSH into the cluster:
       ssh <username>@slurm-client.cs.tau.ac.il

  2. One-time setup (installs venv + deps + downloads ESC-50 / GSC):
       cd <repo-dir>
       bash slurm/setup_cluster.sh

  3. Full training 
     — For ESC-50 (5 folds × 50 epochs, ~1 hour per fold): 
          sbatch slurm/full_training_esc50.slurm
     - For GSC (1 fold × 75 epochs):
          sbatch slurm/full_training_gsc.slurm

  4. Evaluate saved checkpoints:
       python evaluation.py --data_path data --checkpoint_dir outputs --device cuda

  5. Generate convergence plots:
       - For ESC-50: python utils/visualization.py --log_dir slurm/logs --output_dir results/esc50
       - For GSC: python utils/visualization.py --log_dir slurm/logs --output_dir results/gsc

================================================================================
LOCAL SETUP (Optional)
================================================================================
ESC-50:
  1. Clone:    git clone <your-repo-url> && cd PETL-AST-Project
  2. Venv:     python3.10 -m venv .venv && source .venv/bin/activate
  3. Deps:     pip install -r requirements.txt
  4. Data:     bash download_esc50_data.sh
  5. Train:    python main.py --dataset_name esc50 --data_path data \
                 --method adapter --adapter_type conformer \
                 --location pfeiffer_parallel --output_path /outputs \
                 --save_best_ckpt True

GSC:
  1. Clone:    git clone <your-repo-url> && cd PETL-AST-Project
  2. Venv:     python3.10 -m venv .venv && source .venv/bin/activate
  3. Deps:     pip install -r requirements.txt
  4. Data:     bash download_gsc_data.sh
  5. Train:    python main.py --dataset_name GSC --data_path data \
                 --method adapter --adapter_type conformer \
                 --location pfeiffer_parallel --output_path /outputs \
                 --save_best_ckpt True

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
  - train.py                       Wrapper script (stdout logging to file)
  - evaluation.py                  Standalone evaluation script (course requirement)
  - utils/visualization.py         Convergence plot generation
  - download_data.sh               Dataset download automation
  - slurm/                         SLURM cluster scripts (setup, smoke test, full training)
  - readme.txt                     This documentation
  - requirements.txt               Dependencies (authors' original versions + CUDA index)
  - docs/challenges.md             Detailed log of all reproduction challenges
  - samples/                       Audio samples from ESC-50 and GSC train and validation sets
  - google_speech_commands_v2.py   Modified for lazy loading support

================================================================================
TOOLS AND REFERENCES
================================================================================

Pre-trained model:  MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace)
Datasets:           ESC-50 (2000 clips, 50 classes, 5-fold CV)
                    Google Speech Commands V2 (105K clips, 35 classes)
Libraries:          PyTorch 1.13.1, HuggingFace Transformers 4.28.1, librosa,
                    torchaudio, matplotlib
Environment:        Python 3.10, CUDA 11.7
