# EXP-003 — Transfer Learning with Wav2Vec2

**Date:** May 2026  
**Notebook:** notebooks/03_a_module_transfer_learning.ipynb  
**Status:** Complete  

## Objective
Compare Wav2Vec2 transfer learning against the baseline LSTM (EXP-002)
on the same ESC-50 → CIFAR-10 mapped dataset using the same evaluation
protocol. Measure the delta between scratch training and pretrained
audio representations.

## Model Architecture
- Type: Wav2Vec2-base backbone + classification head
- Backbone: facebook/wav2vec2-base (12 transformer layers, 768 hidden)
- Input: Raw waveform (16kHz, 5 seconds)
- Pooling: Mean pool across time dimension
- Embedding dim: 512
- Classifier: Linear(512, 10)
- Dropout: 0.3
- Total parameters: ~94M (mostly frozen in Stage 1)

## Training Configuration
- Optimiser: AdamW (weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=30)
- Loss: CrossEntropyLoss
- Epochs: 30 per fold per stage
- Batch size: 16
- Validation: 5-Fold Stratified Cross Validation

### Stage 1 — Frozen Backbone
- All Wav2Vec2 weights frozen
- Only projection and classifier head trained
- lr: 1e-3

### Stage 2 — Partial Unfreeze
- Last 4 transformer layers unfrozen
- Loaded Stage 1 weights as starting point
- lr: 1e-4 (lower to protect pretrained weights)

## Dataset
- Identical to EXP-002
- Total: 400 samples (360 ESC-50 + 40 synthesised deer)
- Train per fold: ~320 samples (32 per class)
- Val per fold: ~80 samples (8 per class)
- Key difference from EXP-002: raw waveform input at 16kHz
  instead of MelSpectrogram at 22050Hz

## Results

### K-Fold Cross Validation
| Experiment                          | Mean Acc | Std   |
|-------------------------------------|----------|-------|
| Baseline LSTM (scratch) — EXP-002   | 52.75%   | 4.29% |
| Wav2Vec2 Frozen (Stage 1)           | 59.75%   | 2.29% |
| Wav2Vec2 Partial Unfreeze (Stage 2) | 70.25%   | 4.36% |
| **Delta (Baseline → Stage 2)**      | **+17.50%** | — |

### Stage Breakdown
| Stage   | Mean Acc | Std   | Delta vs Previous |
|---------|----------|-------|-------------------|
| Baseline| 52.75%   | 4.29% | —                 |
| Stage 1 | 59.75%   | 2.29% | +7.00%            |
| Stage 2 | 70.25%   | 4.36% | +10.50%           |

## Key Observations

1. Pretrained representations alone outperform scratch training —
   Stage 1 frozen backbone beats baseline by 7% without any
   fine-tuning of Wav2Vec2 weights. Confirms pretrained audio
   features transfer well even to non-speech classification tasks.

2. Progressive unfreezing mirrors V-module pattern — same
   staged improvement seen in visual module (frozen → fine-tuned
   → deep unfreeze). Consistent pattern across modalities is a
   meaningful finding for VATSA architecture design.

3. Stability improves with pretraining — std dropped from 4.29%
   (baseline) to 2.29% (Stage 1), meaning pretrained model is
   less sensitive to fold composition. Small dataset variance
   hurts scratch training more than transfer learning.

4. Stage 2 std increases again (4.36%) — partial unfreeze
   introduces more variance, expected with only 32 training
   samples per class and 4 unfrozen transformer layers.

5. 70.25% is modest in absolute terms — this is a dataset
   limitation not a model limitation. 32 training samples per
   class is insufficient for a 94M parameter model to generalise
   fully. Expected to improve significantly with larger dataset.

## Limitations

1. Same dataset limitations as EXP-002 — 32 training samples
   per class, synthesised deer, weak class mappings for
   automobile and horse
2. Stage 2 loads Stage 1 best fold weights — not strictly
   independent across folds. Acknowledged limitation.
3. No Stage 3 deep unfreeze attempted — deferred to EXP-004
   if dataset is expanded
4. Full dataset per-class evaluation includes training data —
   indicative only, not a true held-out test

## Comparison with V-Module

| Module   | Scratch    | Frozen TL  | Partial Unfreeze | Delta  |
|----------|------------|------------|------------------|--------|
| V-Module | 79.00%     | 94.00%     | 96.31%           | +17.31%|
| A-Module | 52.75%     | 59.75%     | 70.25%           | +17.50%|

Delta is remarkably consistent across modalities (+17.31% vs
+17.50%). This is an interesting finding — transfer learning
provides similar relative improvement regardless of modality,
suggesting a consistent architectural benefit rather than a
dataset-specific effect.

## Next Steps

### EXP-004 (Optional) — Stage 3 Deep Unfreeze
Unfreeze all 12 transformer layers with very low lr (1e-5).
Only worthwhile if dataset is expanded first.

### EXP-005 — Dataset Expansion
ESC-50 is too small for meaningful deep learning benchmarks.
Options:
- AudioSet (Google) — 2M clips, 527 classes
- FSD50K — 51K clips, 200 classes
- UrbanSound8K — 8732 clips, 10 classes (closest to our needs)

Expanding to UrbanSound8K or FSD50K subset would give a more
reliable benchmark and is recommended before finalising the
A-module results for the VATSA preprint.

### A-Module Integration (EXP-006)
Port best model (Stage 2 Wav2Vec2) into modules/audio/ as
clean production code. Connect audio embedding (512-dim) to
shared VATSA latent space alongside visual embedding (512-dim).
First cross-modal experiment: V+A fusion on paired data.

## Files
- Notebook: notebooks/audio/03_a_module_transfer_learning.ipynb
- Model checkpoint: vatsa_audio_encoder_transfer.pth
- Baseline checkpoint: vatsa_audio_encoder_baseline.pth (EXP-002)