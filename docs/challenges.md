# Challenges Reproducing Results from the Authors' Repository

**Paper**: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" (Cappellazzo et al., 2024)
**Authors' repo**: https://github.com/umbertocappellazzo/PETL_AST

---

## Phase 1: Google Colab Attempts (Failed)

We initially targeted Google Colab (T4 GPU, free tier) as our compute environment. This led to a chain of incompatibilities that ultimately forced us to abandon Colab entirely.

### 1. Dependency Version Incompatibility

**Problem**: The authors developed their code with `torch==1.13.1` and `transformers==4.28.1` (circa early 2023). Google Colab ships with CUDA 12.8, `torch 2.10+`, and Python 3.12. The authors' `torch==1.13.1` has no compatible wheels for CUDA 12.8, and `transformers==4.28.1` depends on an old `tokenizers` library that fails to build on Python 3.12.

**What we tried**: We attempted to bridge the gap by using newer, API-compatible versions of transformers (4.36.0, 4.44.0) while keeping Colab's torch 2.10. This appeared to work initially but led to the NaN issue described below.

### 2. `torch.transpose()` NumPy Incompatibility (torch 2.x only)

**File**: `dataset/esc_50.py`, line 56

**Problem**: The authors call `torch.transpose(self.x[index], 0, 1)` on a NumPy array. In torch 1.13, this worked via implicit conversion. In torch 2.0+, this raises:

```
TypeError: transpose() received an invalid combination of arguments
- got (numpy.ndarray, int, int), but expected (Tensor input, int dim0, int dim1)
```

**Fix (Colab only, later reverted)**: Wrapped with `torch.as_tensor()`. After migrating to SLURM with torch 1.13.1, this fix was reverted — the file is now unmodified.

### 3. NaN Losses with Modern PyTorch / Transformers

**Problem**: This was the most persistent and ultimately unsolvable issue on Colab. Every combination of modern torch + transformers produced `nan` training losses and 2% accuracy (random chance for 50 classes).

**Strategies attempted** (all failed):

1. **`transformers==4.44.0`**: Auto-selected `ASTSdpaAttention` (Scaled Dot-Product Attention) — produced NaN on every epoch.
2. **`transformers==4.36.0`** (last version with eager `ASTAttention`): Initially appeared to work, but after a Colab runtime restart, also produced NaN.
3. **Disabling SDPA backends** (`torch.backends.cuda.enable_flash_sdp(False)`, `enable_mem_efficient_sdp(False)`): Only controls the kernel used inside `F.scaled_dot_product_attention`, doesn't prevent the `ASTSdpaAttention` class from being selected by transformers.
4. **Monkey-patching `ASTPreTrainedModel._supports_sdpa = False`**: Successfully forced the eager `ASTAttention` class, confirmed via `model.print()`. But NaN persisted — proving the issue was deeper than the attention implementation.

**Conclusion**: The 2+ year version gap between the authors' environment and Colab's environment causes numerical instability that cannot be patched around. The only solution was matching the authors' exact versions.

### 4. Google Colab Operational Limitations

Beyond NaN, Colab presented multiple operational challenges:

- **T4 GPU (16 GB VRAM)**: OOM with batch_size=32 when using eager attention (higher memory than SDPA). Required reduction to 16.
- **DataLoader deadlocks**: `num_workers=4` caused intermittent multiprocessing deadlocks under GPU memory pressure. Required reduction to 2.
- **Kernel disconnects**: Colab disconnects after ~90 min of inactivity or ~12 hours total, wiping all runtime state mid-training. We implemented a Google Drive mount + symlink workaround for checkpoint persistence, but kernel disconnects still required full re-initialization.
- **No persistent environment**: Every runtime restart required reinstalling all dependencies.

---

## Phase 2: SLURM Cluster Migration

After exhausting all Colab workarounds, we migrated to the TAU CS SLURM cluster, which allows installing the authors' exact environment.

### 5. Missing `epochs_ESC` Key in Hyperparameter YAML

**File**: `hparams/train.yaml`

**Problem**: The authors' `main.py` reads `train_params['epochs_ESC']` for the ESC-50 dataset, but the YAML only defines `epochs_ESC_AST: 50` — no `epochs_ESC` key exists:

```
KeyError: 'epochs_ESC'
```

**Solution**: Added `epochs_ESC: 50` to `hparams/train.yaml`. This is a bug in the authors' repo — the naming convention is inconsistent across datasets.

### 6. `argparse` `type=bool` Bug

**File**: `main.py`, argument definitions

**Problem**: Several arguments use `type=bool` (e.g., `--save_best_ckpt`, `--apply_residual`). Python's `bool()` converts any non-empty string to `True`, so `--apply_residual False` actually sets it to `True`. Similarly, `--seed` lacks `type=int`.

**Solution**: We rely on default values (which are correct) and avoid passing these arguments explicitly. No code modification needed.

### 7. Checkpoint Save Path Bug

**File**: `main.py`, line 327

**Problem**: Checkpoint path is constructed as `os.getcwd() + args.output_path + f'/bestmodel_fold{fold}'`. With `--output_path './outputs'`, this produces a malformed path (`.../Project./outputs/...`).

**Solution**: We pass `--output_path '/outputs'` (leading `/`, no `.`) so concatenation produces a valid path. Our `train.py` wrapper also calls `os.makedirs()` to ensure the directory exists.

### 8. Unconditional `wandb` Import

**Problem**: `main.py` imports `wandb` at module level regardless of `--use_wandb`. Crashes if `wandb` is not installed.

**Solution**: Include `wandb` in `requirements.txt`.

### 9. SLURM Cluster Setup: Home Directory Quota

**Problem**: The cluster home directory has a 4GB quota. PyTorch 1.13.1 alone is ~1.8GB, making it impossible to install all dependencies via `pip install --user` or a venv in `~`.

**What we tried**:
- `pip install --user`: Exceeded 4GB home quota during torch installation.
- `python3 -m venv`: Failed because `python3.10-venv` package was not installed, and `apt install` requires sudo.
- `python3 -m venv --without-pip` + bootstrap pip via `curl`: Created the venv but still hit quota when installing into `~`.
- NGC containers (`easy_ngc`): Requires `sudo` internally, which fails in non-interactive SLURM batch jobs.

**Solution**: Used the NLP course storage at `/vol/joberant_nobck/data/NLP_368307701_2526a/<username>/` — a large, persistent volume with no practical quota limits. Created the venv and cloned the repo there.

### 10. `pkg_resources` Missing in setuptools 82.x

**Problem**: `librosa==0.9.2` imports `from pkg_resources import resource_filename`. The latest `setuptools==82.0.0` (installed by default via `get-pip.py`) has removed or broken `pkg_resources` as a top-level module:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

Force-reinstalling setuptools 82.0.0 did not help — the module is genuinely absent.

**Solution**: Downgraded to `setuptools==69.5.1`, which still ships `pkg_resources`.

### 11. CUDA OOM on TITAN Xp (12GB VRAM)

**Problem**: The `studentkillable` SLURM partition only provides TITAN Xp (12GB) and RTX 2080 (8GB) GPUs. The authors used A40 (48GB). With `batch_size_ESC=32`, the model OOMs during the attention score computation:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 510.00 MiB
(GPU 0; 11.90 GiB total capacity; 10.98 GiB already allocated)
```

The `killable` partition (which has A5000, RTX 3090, V100, A6000) is not accessible to student accounts.

**Solution**: Reduced `batch_size_ESC` from 32 to 16 in `hparams/train.yaml`. This fits comfortably in 12GB and should not significantly affect final accuracy — only convergence speed.

### 12. 25-Hour Training Exceeds SLURM Time Limits

**Problem**: Full training is 5 folds × 50 epochs. At ~6 min per fold-epoch on TITAN Xp, this totals ~25 hours — far exceeding any reasonable SLURM time limit for `studentkillable`. The authors' code runs all 5 folds sequentially in a single process with no checkpoint-resume support.

**Solution**: Added 3 lines to `main.py` to support fold selection via environment variables (`START_FOLD`, `END_FOLD`, `EPOCHS_OVERRIDE`). Default behavior is unchanged when the env vars are not set. Combined with SLURM job arrays (`sbatch --array=0-4`), each fold runs as a separate parallel job (~5 hours each), fitting within the time limit. This also gives a ~5x wall-clock speedup.

---

## Summary of All Modifications

### Authors' files modified:

| File | Change | Reason |
|---|---|---|
| `hparams/train.yaml` | Added `epochs_ESC: 50` | Missing key bug in authors' repo |
| `hparams/train.yaml` | `batch_size_ESC: 32` → `16` | TITAN Xp (12GB) OOM |
| `main.py` | 3 lines: `START_FOLD`, `END_FOLD`, `EPOCHS_OVERRIDE` env vars | Per-fold parallelism for SLURM time limits |

### Our additions:

| File | Purpose |
|---|---|
| `train.py` | Wrapper: captures stdout to log file for convergence plots |
| `evaluation.py` | Standalone evaluation script (course requirement) |
| `utils/visualization.py` | Convergence plot generation |
| `download_data.sh` | ESC-50 dataset download automation |
| `slurm/setup_cluster.sh` | One-time cluster setup (venv, deps, data) |
| `slurm/smoke_test.slurm` | Quick validation job (1 epoch per fold, parallel) |
| `slurm/full_training.slurm` | Full 50-epoch training job (parallel folds) |
| `PETL_AST_Colab.ipynb` | Colab notebook (alternative, has NaN issues) |
| `requirements.txt` | Dependencies (authors' original versions) |

### Files unmodified from authors:

- `src/` — All model implementations (`AST_adapters.py`, `AST.py`, etc.)
- `utils/engine.py` — Training and evaluation loops
- `dataset/` — All dataset loader classes
