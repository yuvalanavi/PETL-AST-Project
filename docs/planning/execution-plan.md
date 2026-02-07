# PETL-AST Project: 2-Day Sprint Execution Plan

**Paper**: "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" (Cappellazzo et al., 2024)
**Scope**: Conformer Adapter (Pfeiffer, parallel) on frozen AST, evaluated on ESC-50
**Team**: Yuval, Eden, Guy, Tom
**Compute**: Single consumer GPU (Colab T4 or similar)

---

## 1. Repository Structure

```
PETL-AST-Project/
├── .cursor/
│   └── rules/
│       └── project-conventions.mdc   # Coding agent rules
├── configs/
│   └── esc50_conformer.yaml          # All hyperparameters (single source of truth)
├── data/                             # .gitignored — downloaded ESC-50 goes here
├── docs/
│   ├── planning/                     # This plan and sprint tracking
│   └── project/                      # Paper, guidelines, proposal
├── outputs/                          # .gitignored — checkpoints, logs, plots
├── samples/                          # Audio samples for submission (from train/val)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conformer_adapter.py      # Conformer Adapter nn.Module
│   │   └── ast_with_adapter.py       # AST wrapper: loads pretrained, injects adapters, freezes
│   ├── data/
│   │   ├── __init__.py
│   │   └── esc50_dataset.py          # ESC-50 Dataset class (5-fold CV aware)
│   └── utils/
│       ├── __init__.py
│       ├── engine.py                 # train_one_epoch, eval_one_epoch
│       └── visualization.py          # Convergence plot generation
├── train.py                          # Main training entry point (standalone, argparse)
├── evaluation.py                     # Standalone evaluation script
├── download_data.sh                  # One-liner to download + extract ESC-50
├── requirements.txt                  # Pinned deps, Python 3.10
├── readme.txt                        # Submission README with exact run instructions
└── .gitignore
```

**Design rationale**:
- `train.py` and `evaluation.py` live at the root as required by the course guidelines.
- All reusable code lives under `src/` to keep scripts thin.
- `configs/` holds YAML so hyperparameters are never hardcoded in Python.
- `data/` and `outputs/` are gitignored to keep the repo lightweight.

---

## 2. Critical Technical Decisions (Locked In)

These are extracted from the paper's Section 3.1 and the reference implementation. **Do not deviate** unless a decision is explicitly revisited.

| Parameter | Value | Source |
|---|---|---|
| Pre-trained model | `MIT/ast-finetuned-audioset-10-10-0.4593` | Paper §3.1, HuggingFace |
| Hidden dim (d) | 768 | AST architecture |
| Num layers | 12 | AST architecture |
| Adapter config | **Pfeiffer** (adapter parallel to MHSA only) | Paper Table 1 — best for audio tasks |
| Adapter placement | **Parallel** (no residual) | Paper §3.1 + reference code |
| Adapter block | **Conformer** | Our project scope |
| Reduction rate (RR) | 96 → bottleneck dim r = 8 | Paper §3.1 |
| Kernel size (k) | 8 | Paper §3.3 — optimal for ESC-50 |
| Expected trainable params | ~271K (0.29% of 85.5M) | Paper Table 1 |
| Optimizer | AdamW (betas=0.9,0.98, eps=1e-6) | Reference code |
| Learning rate | 0.005 | Paper §3.1 |
| Weight decay | 0.1 | Paper §3.1 |
| Scheduler | CosineAnnealingLR (step-level) | Reference code |
| Epochs | 50 | Reference hparams/train.yaml |
| Batch size | 32 | Reference hparams/train.yaml |
| Max spectrogram length | 500 | Reference hparams/train.yaml |
| ESC-50 evaluation | 5-fold cross-validation | ESC-50 protocol |
| SpecAugment (train) | freq_mask=24, time_mask=80 | Reference code, ESC-50 dataset |
| Classification | Mean pooling over all tokens (final_output='ALL') | Reference code |
| Target accuracy | ~88.30% (paper reports for Pfeiffer Conformer) | Paper Table 1 |
| Audio resampling | 44.1kHz → 16kHz via librosa | Reference dataset code |

### Conformer Adapter Architecture (from paper Fig. 1 + Section 2.2)

```
Input x: [B, N, d=768]
    │
    ├── transpose to [B, d, N]
    ├── Pointwise Conv1d (d → 2r=16, kernel=1)
    ├── GLU (dim=1) → [B, r=8, N]
    ├── Depthwise Conv1d (r → r, kernel=8, groups=r, padding='same')
    ├── BatchNorm1d(r)
    ├── SiLU (Swish)
    ├── Pointwise Conv1d (r → d=768, kernel=1)
    ├── Dropout(0.0)
    └── transpose to [B, N, d]
Output: [B, N, d=768]
```

The adapter output is **added** to the MHSA output (parallel placement).

### Reference Code Bug (Replicate Exactly)

In the reference `Conformer_adapter.forward()`:
```python
out = self.lnorm(x)       # LayerNorm is computed...
out = x.transpose(-1, -2) # ...but then overwritten with raw x
```
The LayerNorm result is **discarded**. The LN module exists and its parameters are trained, but its output is never used. We replicate this behavior exactly to match the paper's reported results. Note this in the report as an observation.

### What We Train (Unfrozen Parameters)

Per the reference code's `_unfreeze_adapters()`:
1. All adapter modules (conformer adapter in each of 12 layers, applied to FFN position per Pfeiffer)
2. `layernorm_after` in each transformer layer
3. The final `layernorm` of the model
4. The classification head (linear: 768 → 50)

**Wait — important clarification from the reference code**: Despite "Pfeiffer = MHSA only" in the paper text, the reference code for Pfeiffer actually places the adapter parallel to the **FFN** block (after `layernorm_after`), not the MHSA. The paper's Table description says Pfeiffer adapters are "added in parallel to only the MHSA layer" but the code does `adapter_module_FFN` placed parallel to the FFN output. We follow the **code** (which produced the reported numbers), not the ambiguous paper text.

---

## 3. Phase Breakdown

### Phase 0: Pre-Sprint Infrastructure (Yuval — NOW, before team meeting)

**Goal**: When teammates open the repo, they can immediately start writing model/pipeline code.

| Task | Detail |
|---|---|
| Init git repo | `git init`, create `.gitignore` |
| Create directory scaffold | All folders from the structure above with `__init__.py` files |
| Write `requirements.txt` | Pinned versions, tested on Python 3.10 |
| Write `configs/esc50_conformer.yaml` | All hyperparameters from the table above |
| Write `download_data.sh` | Script to download and extract ESC-50 |
| Write `.cursor/rules/project-conventions.mdc` | Team coding rules for AI agents |
| Write onboarding section in `readme.txt` | Setup instructions for teammates |
| Write `src/data/esc50_dataset.py` | ESC-50 dataset class (5-fold CV, SpecAug) |
| Verify data pipeline | Load dataset, print shapes, confirm spectrogram extraction works |

### Phase 1: Model Implementation (Day 1, ~2-3 hours)

| Task | Detail | Who |
|---|---|---|
| `src/models/conformer_adapter.py` | Implement the Conformer Adapter `nn.Module` exactly per the architecture above | 1 person |
| `src/models/ast_with_adapter.py` | Load pretrained AST, inject adapters into each layer, freeze backbone, unfreeze adapter params + LNs | 1 person |
| Param count verification | Instantiate model, assert trainable params ≈ 271K | Same person |
| Unit smoke test | Forward pass with random tensor [2, 500, 128], verify output shape [2, 50] | Same person |

### Phase 2: Training Pipeline (Day 1, ~2-3 hours — can run in parallel with Phase 1)

| Task | Detail | Who |
|---|---|---|
| `src/utils/engine.py` | `train_one_epoch()` and `eval_one_epoch()` functions | 1 person |
| `train.py` | Argparse CLI, 5-fold loop, optimizer/scheduler setup, logging, checkpoint save, CSV logging for plots | 1 person |
| `evaluation.py` | Load checkpoint, run eval on test fold, print accuracy | Same or another person |
| `src/utils/visualization.py` | Read CSV logs, generate loss/accuracy convergence plots (matplotlib) | 1 person |

### Phase 3: Integration, Training & Debugging (Day 1 evening → Day 2 morning)

| Task | Detail |
|---|---|
| End-to-end test | Run 1 fold, 2-3 epochs — verify loss decreases, no crashes |
| Fix bugs | Whatever breaks during integration |
| Full training run | Launch 5-fold × 50 epochs on GPU (this takes hours — start ASAP) |
| Monitor training | Watch for divergence, NaN, OOM |

### Phase 4: Deliverables & Reporting (Day 2)

| Task | Detail | Who |
|---|---|---|
| Generate convergence plots | Loss + accuracy curves from CSV logs | 1 person |
| Copy audio samples | Pick representative clips from train/val sets into `samples/` | 1 person |
| Polish `readme.txt` | Exact run instructions for `train.py` and `evaluation.py` | 1 person |
| Start report outline | LaTeX template on Overleaf with sections filled in | 1 person |
| Package `project_code.zip` | Everything except `data/`, `outputs/`, `.git/` | 1 person |

---

## 4. Task Parallelization Strategy

With 4 people and coding agents, the implementation is fast. The bottleneck is **GPU training time** (5 folds × 50 epochs ≈ several hours on T4). So the strategy is: **merge code fast, start training early, use training time for report/polish work**.

### Day 1

```
Morning (Yuval solo — pre-meeting prep):
  └── Phase 0: Infrastructure, data pipeline, config

Afternoon (Full team):
  ├── Person A + Person B: Phase 1 (Model)
  │   ├── A: conformer_adapter.py
  │   └── B: ast_with_adapter.py (depends on A finishing the adapter module)
  │
  └── Person C + Person D: Phase 2 (Training pipeline)
      ├── C: engine.py + train.py
      └── D: evaluation.py + visualization.py

Evening:
  └── Everyone: Phase 3 — integrate, debug, launch first full training run
```

### Day 2

```
Morning:
  ├── Training running on GPU (launched night before or early morning)
  ├── Fix any issues from overnight run
  └── If training crashed: debug and relaunch

Afternoon:
  ├── Person A: Generate plots from training logs
  ├── Person B: Polish readme.txt + package code zip
  ├── Person C + D: Start report on Overleaf
  └── Everyone reviews final deliverables
```

---

## 5. Estimated Training Time

Back-of-envelope for T4 GPU:
- ESC-50: 2000 clips, 5 folds → train split ≈ 1200 samples per fold
- Batch size 32 → ~38 steps/epoch
- 50 epochs → ~1900 steps per fold
- AST forward pass on T4 ≈ ~0.15s/batch (small model, small adapter)
- Per fold: ~1900 × 0.15 ≈ ~5 minutes
- 5 folds: ~25 minutes total

This is **very fast**. ESC-50 is a small dataset. We can afford multiple runs if needed.

---

## 6. Immediate Next Steps (Yuval — Right Now)

1. **Initialize the git repo** and push the skeleton.
2. **Create all directories** and placeholder files.
3. **Write `requirements.txt`** — install and verify it works on Python 3.10.
4. **Write `configs/esc50_conformer.yaml`** with all locked hyperparameters.
5. **Write `download_data.sh`** and download ESC-50 yourself to test.
6. **Write `src/data/esc50_dataset.py`** — the data loading is mechanical and well-understood from the reference.
7. **Write the Cursor rules file** so teammates' agents follow the same conventions.
8. **Write the onboarding guide** (setup instructions) in `readme.txt`.

When teammates arrive, they clone → `pip install -r requirements.txt` → run `download_data.sh` → open Cursor → start working on their assigned module immediately. Zero config time.

---

## 7. Teammate Onboarding Guide

```
Quick Start (5 minutes):
1. Clone the repo
2. Create a Python 3.10 venv:
     python3.10 -m venv venv && source venv/bin/activate
3. Install dependencies:
     pip install -r requirements.txt
4. Download ESC-50:
     bash download_data.sh
5. Verify setup:
     python -c "from src.data.esc50_dataset import ESC50Dataset; print('OK')"
6. Open in Cursor — the .cursor/rules/ file will guide your AI agent.
7. Check this plan for your assigned task.
```

---

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| OOM on T4 (16GB) | Batch size 32 with frozen backbone + tiny adapter should be fine. If OOM, reduce to 16. |
| Accuracy far below 88.3% | Verify: (1) correct hyperparams, (2) adapter is parallel not sequential, (3) SpecAug enabled, (4) all 5 folds averaged, (5) correct LN unfreezing. |
| Training diverges | LR 0.005 is high — if divergence, try 0.001. Check gradient norms. |
| ESC-50 download issues | Dataset is ~600MB from GitHub. Mirror link in download script. |
| Reference code uses old transformers (4.28) | We'll use a recent version. May need to adjust AST layer class imports. Test early. |
| 5-fold CV takes too long | Unlikely (see §5). But if needed, start with 1 fold to validate, then run all 5. |

---

## 9. Definition of Done

- [ ] `train.py` runs end-to-end on ESC-50 with 5-fold CV
- [ ] `evaluation.py` loads a checkpoint and prints fold accuracy
- [ ] Trainable parameter count ≈ 271K
- [ ] Final accuracy is within reasonable range of 88.3% (±3%)
- [ ] Convergence plots (loss + accuracy) generated and saved
- [ ] Audio samples saved in `samples/`
- [ ] `readme.txt` has exact run instructions
- [ ] `requirements.txt` installs cleanly on Python 3.10
- [ ] Code is clean enough that any team member can explain any function
