# PETL-AST Project: 2-Day Sprint Execution Plan

**Paper**: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" (Cappellazzo et al., 2024)
**Scope**: Reproduce the Conformer Adapter results on ESC-50 using the authors' code
**Team**: Yuval, Eden, Guy, Tom
**Compute**: Single consumer GPU (Colab T4 or similar)
**Authors' repo**: https://github.com/umbertocappellazzo/PETL_AST

---

## Approach

We use the authors' published codebase as our foundation. We run their code to train the Conformer Adapter on ESC-50, reproduce the reported results (~88.30% accuracy), and document the process. Per course guidelines, this is expected — we must understand the code, train it ourselves, cite what's theirs, and document challenges.

---

## 1. Repository Structure

```
PETL-AST-Project/
├── .cursor/rules/                    # Our: Cursor agent rules
├── docs/
│   ├── planning/                     # Our: this plan
│   └── project/                      # Our: paper, guidelines, proposal
├── data/                             # Gitignored — ESC-50 dataset
├── outputs/                          # Gitignored — checkpoints, logs
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
│   ├── fluentspeech.py               #   FSC
│   ├── google_speech_commands_v2.py  #   GSC
│   ├── urban_sound_8k.py             #   US8K
│   └── iemocap.py                    #   IEMOCAP
├── utils/
│   └── engine.py                     # Authors': train/eval one epoch
├── hparams/
│   └── train.yaml                    # Authors': all hyperparameters
├── main.py                           # Authors': main training entry point
│
│  --- Our additions ---
│
├── evaluation.py                     # Our: standalone eval script (required by course)
├── download_data.sh                  # Our: ESC-50 download script
├── requirements.txt                  # Updated from authors' (modern versions)
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

## 3. The Training Command

For our specific experiment (Conformer Adapter, Pfeiffer, parallel, ESC-50):

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
    --apply_residual False \
    --is_AST True \
    --seed 10
```

Expected: ~88.30% average accuracy over 5 folds, ~271K trainable parameters.

---

## 4. Phase Breakdown

### Phase 1: Setup & Understanding (Day 1, first half)

| Task | Who | Detail |
|---|---|---|
| Verify setup | Everyone | Pull repo, install deps, download ESC-50, run the verify command |
| Read the code together | Everyone | Walk through `src/AST_adapters.py` and `main.py` as a team. Understand every class and function. |
| Understand the Conformer Adapter | Everyone | Trace the forward pass: input → pointwise conv → GLU → depthwise conv → batchnorm → swish → pointwise conv → output. Map it to paper Fig. 1 and Section 2.2. |
| Understand adapter injection | Everyone | How `ASTLayer_adapter.forward()` adds the adapter output parallel to FFN. What gets frozen vs unfrozen. |

### Phase 2: First Training Run (Day 1, second half)

| Task | Who | Detail |
|---|---|---|
| Quick smoke test | 1 person | Run 1 fold, 2-3 epochs. Verify loss decreases, no crashes, param count is ~271K. |
| Fix any issues | Team | Dependency mismatches, path issues, CUDA/MPS issues |
| Launch full training | 1 person | Start 5-fold × 50 epochs. Monitor first fold. |
| Add logging for plots | 1-2 people | The authors use wandb. We need convergence plots for the report — either enable wandb or add CSV logging to `engine.py`. |

### Phase 3: Evaluation & Deliverables (Day 2)

| Task | Who | Detail |
|---|---|---|
| Write `evaluation.py` | 1 person | Standalone eval script (course requirement). Load checkpoint, run eval on test fold, print accuracy. |
| Generate convergence plots | 1 person | Loss + accuracy curves per fold. Use matplotlib. |
| Collect audio samples | 1 person | Copy representative clips from ESC-50 train/val into `samples/`. |
| Polish `readme.txt` | 1 person | Final run instructions matching the actual CLI. |
| Document challenges | Everyone | What broke, what we fixed, any deviations from the paper. This goes into the report. |

### Phase 4: Report (Day 2 + after sprint)

| Section | Notes |
|---|---|
| Abstract | What we did, our results |
| Introduction | PETL problem, AST model, conformer adapter solution |
| Related Work | PETL methods (LoRA, adapters, prompt tuning) — paper covers this well |
| Method | Conformer Adapter architecture, Pfeiffer placement, training setup |
| Results + Discussion | Our 5-fold accuracy vs paper's 88.30%, convergence plots, observations |
| Challenges | Dependency updates, any modifications, the LayerNorm bug, Pfeiffer placement ambiguity |
| Future Work | Suggest improvement (do NOT implement) |
| Limitations & Broader Impact | Model biases, environmental sound classification risks |

---

## 5. Potential Challenges to Watch For

| Issue | What to do |
|---|---|
| **Dependency versions** | Authors use torch 1.13 / transformers 4.28. We use modern versions. Internal HuggingFace class imports (`ASTLayer`, `ASTEncoder`, `ASTOutput`) might have changed. Already verified they work with transformers 4.57. |
| **ESC-50 path structure** | The dataset class expects `data_path/ESC-50/meta/esc50.csv` and `data_path/ESC-50/audio/*.wav`. Our download script puts it at `data/ESC-50/`. So `--data_path 'data'` should work. |
| **No separate eval script** | Authors' `main.py` trains + evaluates in one run. We need to write a standalone `evaluation.py` for the course requirement. |
| **Convergence plots** | Authors use wandb (optional). We need plots for the report. Either use wandb, tensorboard, or add CSV logging. |
| **LayerNorm bug in Conformer Adapter** | The `forward()` computes LayerNorm but overwrites the result. This is in the published code and produced the reported numbers. Don't "fix" it. Note it as an observation in the report. |
| **Pfeiffer placement ambiguity** | Paper text says "parallel to MHSA" but code places adapter parallel to FFN. Follow the code. Note this in the report. |

---

## 6. What We Wrote vs What's Theirs

For the report's citation section:

**Authors' code (cited):**
- `src/` — all model implementations
- `dataset/` — all dataset classes
- `utils/engine.py` — train/eval loops
- `hparams/train.yaml` — hyperparameters
- `main.py` — training pipeline

**Our additions:**
- `evaluation.py` — standalone evaluation script
- `download_data.sh` — dataset download automation
- `readme.txt` — submission documentation
- Convergence plot generation
- Any bug fixes or modifications (documented in report)
- Dependency updates from torch 1.13 → 2.4
