# PETL-AST Project: 2-Day Sprint Execution Plan

**Paper**: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" (Cappellazzo et al., 2024)
**Scope**: Reproduce the Conformer Adapter results on ESC-50 using the authors' code
**Team**: Yuval, Eden, Guy, Tom
**Compute**: Google Colab T4 GPU (free tier)
**Authors' repo**: https://github.com/umbertocappellazzo/PETL_AST

---

## Approach

We use the authors' published codebase as our foundation. We run their code to train the Conformer Adapter on ESC-50, reproduce the reported results (~88.30% accuracy), and document the process. Per course guidelines, this is expected — we must understand the code, train it ourselves, cite what's theirs, and document challenges.

**Runtime**: Training on Colab T4 is the default. Local machines are for code editing only.

---

## 1. Repository Structure

```
PETL-AST-Project/
├── .cursor/rules/                    # Our: Cursor agent rules
├── docs/
│   ├── planning/                     # Our: this plan
│   └── project/                      # Our: paper, guidelines, proposal
├── data/                             # Gitignored — ESC-50 dataset
├── outputs/                          # Gitignored — checkpoints, logs, plots
├── samples/                          # Our: audio samples for submission
│
│  --- Authors' code (from github.com/umbertocappellazzo/PETL_AST) ---
│
├── src/                              # Authors': model implementations
│   ├── AST.py                        #   Base AST (full FT, linear probing)
│   ├── AST_adapters.py               #   ★ Bottleneck + Conformer adapters
│   ├── AST_LoRA.py                   #   LoRA implementation
│   ├── AST_prompt_tuning.py          #   Prompt/prefix tuning
│   ├── MoA.py                        #   Mixture of Adapters
│   └── Wav2Vec_adapter.py            #   Wav2Vec 2.0 adapters
├── dataset/                          # Authors': dataset classes
│   ├── esc_50.py                     #   ★ ESC-50 (our focus)
│   └── ...                           #   Other datasets
├── utils/
│   ├── engine.py                     # Authors': train/eval one epoch
│   └── visualization.py              # Our: convergence plot generation
├── hparams/
│   └── train.yaml                    # Authors': all hyperparameters
├── main.py                           # Authors': main training entry point
│
│  --- Our additions ---
│
├── evaluation.py                     # Our: standalone eval script (required)
├── download_data.sh                  # Our: ESC-50 download script
├── requirements.txt                  # Updated for Colab compatibility
├── readme.txt                        # Our: submission README
└── .gitignore
```

---

## 2. Key Files to Understand (priority order)

Everyone on the team must be able to explain these. The professor may ask.

### Must understand deeply:

| File | What it does | Key things to know |
|---|---|---|
| `src/AST_adapters.py` | Defines Conformer Adapter module + AST wrapper | `Conformer_adapter` class (the novel contribution), `AST_adapter` class (injection + freeze logic), `ASTLayer_adapter` (modified forward pass with parallel adapter) |
| `main.py` | Full training pipeline | Arg parsing, model instantiation, optimizer/scheduler setup, 5-fold CV loop for ESC-50, checkpoint saving |
| `utils/engine.py` | Train/eval loops | `train_one_epoch()`, `eval_one_epoch()` — simple and short |
| `dataset/esc_50.py` | ESC-50 dataset class | Fold splitting, audio loading via `AutoFeatureExtractor`, SpecAugment, resampling 44.1kHz→16kHz |
| `hparams/train.yaml` | Hyperparameters | ESC-50: max_len=500, 50 classes, batch_size=32, 50 epochs, lr=0.005, weight_decay=0.1 |

### Good to understand:

| File | Why |
|---|---|
| `src/AST.py` | Base AST wrapper — helps understand what `AST_adapter` extends |
| The other PETL methods in `src/` | Context for the paper's comparisons, may come up in oral exam |

---

## 3. Training Commands

### Smoke test (1 fold, 3 epochs — ~5 min on T4):

```bash
python main.py \
    --data_path 'data' \
    --dataset_name 'ESC-50' \
    --method 'adapter' \
    --adapter_block 'conformer' \
    --adapter_type 'Pfeiffer' \
    --seq_or_par 'parallel' \
    --reduction_rate_adapter 96 \
    --kernel_size 8 \
    --device cuda \
    --num_folds 1 --num_epochs 3 \
    --save_best_ckpt True --output_path './outputs'
```

### Full training (5 folds × 50 epochs — ~2.5 hours on T4):

```bash
python main.py \
    --data_path 'data' \
    --dataset_name 'ESC-50' \
    --method 'adapter' \
    --adapter_block 'conformer' \
    --adapter_type 'Pfeiffer' \
    --seq_or_par 'parallel' \
    --reduction_rate_adapter 96 \
    --kernel_size 8 \
    --device cuda \
    --save_best_ckpt True --output_path './outputs'
```

Expected: ~88.30% average accuracy over 5 folds, ~271K trainable parameters.

### After training — generate plots:

```bash
python utils/visualization.py --log_dir outputs --output_dir outputs/plots
```

### After training — evaluate checkpoints:

```bash
python evaluation.py --data_path data --checkpoint_dir outputs --device cuda
```

---

## 4. Phase Breakdown

### Phase 1: Setup & Understanding (Day 1, first half)

| Task | Who | Detail |
|---|---|---|
| Verify setup | Everyone | Pull repo, install deps on Colab, download ESC-50 |
| Read the code together | Everyone | Walk through `src/AST_adapters.py` and `main.py` as a team |
| Understand the Conformer Adapter | Everyone | Trace the forward pass: pointwise conv → GLU → depthwise conv → batchnorm → swish → pointwise conv. Map to paper Fig. 1 and Section 2.2 |
| Understand adapter injection | Everyone | How `ASTLayer_adapter.forward()` adds adapter output parallel to FFN. What gets frozen vs unfrozen |

### Phase 2: Smoke Test & First Run (Day 1, second half)

| Task | Who | Detail |
|---|---|---|
| Smoke test on Colab | 1 person | Run `--num_folds 1 --num_epochs 3`. Verify loss decreases, param count ~271K |
| Fix any Colab issues | Team | Path issues, CUDA issues, dependency mismatches |
| Launch full training | 1 person | Start 5-fold × 50 epochs on Colab. Monitor first fold |
| Read remaining code | Others | While training runs, everyone reads the code they haven't covered yet |

### Phase 3: Evaluation & Deliverables (Day 2)

| Task | Who | Detail |
|---|---|---|
| Generate convergence plots | 1 person | From CSV logs produced by training |
| Run `evaluation.py` | 1 person | Verify standalone eval matches training results |
| Collect audio samples | 1 person | Copy representative clips from ESC-50 into `samples/` |
| Polish `readme.txt` | 1 person | Final run instructions matching actual CLI |
| Document challenges | Everyone | What broke, what we fixed — goes into report |

### Phase 4: Report (Day 2 + after sprint)

| Section | Notes |
|---|---|
| Abstract | What we did, our results |
| Introduction | PETL problem, AST model, conformer adapter solution |
| Related Work | PETL methods (LoRA, adapters, prompt tuning) — paper covers this well |
| Method | Conformer Adapter architecture, Pfeiffer placement, training setup |
| Results + Discussion | Our 5-fold accuracy vs paper's 88.30%, convergence plots, observations |
| Challenges | Torch version updates, the LayerNorm bug, Pfeiffer placement ambiguity |
| Future Work | Suggest improvement (do NOT implement) |
| Limitations & Broader Impact | Model biases, environmental sound classification risks |

---

## 5. Potential Challenges to Watch For

| Issue | What to do |
|---|---|
| **torch version on Colab** | Colab has torch 2.8+ / CUDA 12.8. Authors used torch 1.13. We pin transformers==4.28.1 (authors' version) for API compat. Code mods for torch 2.x are already done. |
| **ESC-50 path structure** | Dataset class expects `data_path/ESC-50/meta/esc50.csv` and `audio/*.wav`. Our download puts it at `data/ESC-50/`. Use `--data_path 'data'`. |
| **`type=bool` argparse bug** | Authors use `type=bool` which doesn't work correctly. Don't pass `--apply_residual False` or `--is_AST True` from CLI — let defaults apply. |
| **LayerNorm bug in Conformer Adapter** | The `forward()` computes LayerNorm but overwrites the result. This is in the published code and produced the reported numbers. Don't "fix" it. Note it in report. |
| **Pfeiffer placement ambiguity** | Paper text says "parallel to MHSA" but code places adapter parallel to FFN. Follow the code. Note this in report. |

---

## 6. What We Wrote vs What's Theirs

**Authors' code (cited):**
- `src/` — all model implementations
- `dataset/` — all dataset classes
- `utils/engine.py` — train/eval loops
- `hparams/train.yaml` — hyperparameters
- `main.py` — training pipeline

**Our additions:**
- `evaluation.py` — standalone evaluation script
- `utils/visualization.py` — convergence plot generation
- `download_data.sh` — dataset download automation
- `readme.txt` — submission documentation
- `requirements.txt` — updated for Colab torch 2.x compatibility

**Modifications to authors' code (all marked with `# MODIFIED:`):**
- `main.py`: CSV logging, output path fix, `--num_folds`/`--num_epochs` overrides
- `dataset/esc_50.py`: `torch.as_tensor()` for torch 2.x compatibility
- `hparams/train.yaml`: Added `epochs_ESC` key (missing in original)
