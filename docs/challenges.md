# Challenges Reproducing Results from the Authors' Repository

**Paper**: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" (Cappellazzo et al., 2024)
**Authors' repo**: https://github.com/umbertocappellazzo/PETL_AST

---

## 1. Dependency Version Incompatibility

**Problem**: The authors developed their code with `torch==1.13.1` and `transformers==4.28.1` (circa early 2023). Our compute environment (Google Colab with T4 GPU) ships with CUDA 12.8 and `torch 2.10+`. The authors' `torch==1.13.1` cannot run on CUDA 12.8 — there are no compatible wheel builds, and building from source is impractical.

Additionally, `transformers==4.28.1` depends on an old version of the `tokenizers` library that requires Rust compilation. On Colab's Python 3.12 environment, this build fails because no pre-built wheels are available.

**Solution**: We use `transformers==4.44.0`, which is the latest version that retains the same `ASTAttention` and `ASTLayer` forward-pass signatures used in the authors' code (accepting `head_mask` and `output_attentions` arguments). Later versions (≥4.49) refactored the AST model's attention interface, removing these parameters entirely, which would break the authors' adapter injection code. For PyTorch, we use whatever version Colab provides (torch 2.x with CUDA 12.x), which is forward-compatible.

---

## 2. `torch.transpose()` NumPy Incompatibility (torch 2.x)

**File**: `dataset/esc_50.py`, line 56

**Problem**: The authors' ESC-50 dataset class stores pre-processed audio features as NumPy arrays (`self.x`). In the `__getitem__` method, they call `torch.transpose(self.x[index], 0, 1)` directly on a NumPy array. In `torch 1.13`, this worked because `torch.transpose()` implicitly converted NumPy arrays to tensors. In `torch 2.0+`, this implicit conversion was removed, producing:

```
TypeError: transpose() received an invalid combination of arguments
- got (numpy.ndarray, int, int), but expected (Tensor input, int dim0, int dim1)
```

**Solution**: Wrapped the array with `torch.as_tensor()` before transposing:

```python
# Original (line 56):
fbank = torch.transpose(self.x[index], 0, 1)

# Fixed:
fbank = torch.transpose(torch.as_tensor(self.x[index]), 0, 1)
```

This is a zero-copy operation when the data types are compatible, so it does not affect performance or numerical results.

---

## 3. Missing `epochs_ESC` Key in Hyperparameter YAML

**File**: `hparams/train.yaml`

**Problem**: The authors' `main.py` (line 138) reads `train_params['epochs_ESC']` for the ESC-50 dataset. However, the authors' `hparams/train.yaml` only defines `epochs_ESC_AST: 50` — there is no `epochs_ESC` key. Running the training script as-is produces:

```
KeyError: 'epochs_ESC'
```

This appears to be a bug in the authors' repository, as the key naming convention is inconsistent across datasets (e.g., `epochs_FSC_AST` has a corresponding `epochs_FSC_WAV`, but ESC-50 only has the AST variant and `main.py` expects the shorter key).

**Solution**: Added the missing key to `hparams/train.yaml`:

```yaml
epochs_ESC: 50  # BUGFIX: main.py reads this key, not epochs_ESC_AST
```

---

## 4. `argparse` `type=bool` Bug

**File**: `main.py`, argument definitions

**Problem**: Several arguments in the authors' `main.py` use `type=bool` with argparse (e.g., `--save_best_ckpt`, `--apply_residual`, `--is_AST`). Python's `bool()` constructor converts *any non-empty string* to `True`, including the string `"False"`. This means passing `--apply_residual False` on the command line sets `apply_residual=True` — the opposite of what is intended.

Similarly, `--seed` lacks a `type=int` annotation, so it is parsed as a string, which later crashes when used as `torch.manual_seed(seed)`.

**Solution**: We do not pass these arguments on the command line and instead rely on their default values (which are correct):
- `apply_residual` defaults to `False`
- `is_AST` defaults to `True`
- `seed` defaults to `10`

This is a known Python argparse pitfall. We did not modify the authors' code to fix it — we simply avoid triggering it.

---

## 5. Checkpoint Save Path Bug

**File**: `main.py`, line 327

**Problem**: The authors construct the checkpoint save path via string concatenation:

```python
torch.save(best_params, os.getcwd() + args.output_path + f'/bestmodel_fold{fold}')
```

When `args.output_path` is `'./outputs'` (a relative path), this produces a malformed path like `.../PETL-AST-Project./outputs/bestmodel_fold0` (note the `Project.` instead of `Project/`). The parent directory does not exist, causing a `RuntimeError`.

**Solution**: In our `train.py` wrapper, we use `os.path.join()` and `os.makedirs()` to construct paths correctly. We do not modify the authors' `main.py`. When using `main.py` directly (without our wrapper), the `--output_path` argument must start with `/` to produce a valid absolute path via concatenation.

---

## 6. Unconditional `wandb` Import

**File**: `main.py`, line 24

**Problem**: The authors' `main.py` has `import wandb` at module level, regardless of whether `--use_wandb` is set. If `wandb` is not installed, the script crashes immediately with `ModuleNotFoundError`, even when wandb logging is not requested.

**Solution**: We include `wandb` in our `requirements.txt` to ensure it is always available. This adds ~10 seconds to installation but avoids the crash.

---

## Summary of Modifications

| File | Change | Reason |
|---|---|---|
| `dataset/esc_50.py` | 1 line: `torch.as_tensor()` wrap | torch 2.x removed implicit numpy→tensor conversion |
| `hparams/train.yaml` | 1 line: added `epochs_ESC: 50` | Key expected by `main.py` was missing |

All other authors' files (`main.py`, `src/`, `utils/engine.py`) remain **unmodified**. Our additions (`train.py`, `evaluation.py`, `utils/visualization.py`) are separate files that import from the authors' code without altering it.
