# EXP-002 — Baseline RNN Classifier (Single LSTM)

**Date:** May 2026  
**Notebook:** notebooks/audio/01_02_a_module_baseline_RNN_LSTM_classifier.ipynb  (renamed)
**Status:** Complete  

## Objective
Establish a baseline RNN classifier on the ESC-50 → CIFAR-10 mapped 
dataset using a single layer LSTM trained from scratch.
This baseline will be compared against transfer learning (Wav2Vec2) 
in EXP-003.

## Model Architecture
- Type: Single layer LSTM
- Input: MelSpectrogram (64 mel bands, 5 seconds, 22050Hz)
- Input shape: (batch, time_steps, 64)
- Hidden size: 128
- Embedding dim: 512
- Classifier: Linear(512, 10)
- Dropout: 0.3
- Total parameters: ~400K (to be confirmed)

## Training Configuration
- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=30)
- Loss: CrossEntropyLoss
- Epochs: 30 per fold
- Batch size: 16
- Validation: 5-Fold Stratified Cross Validation

## Dataset
- Total samples: 400 (360 ESC-50 + 40 synthesised deer)
- Train per fold: ~320 samples (32 per class)
- Val per fold: ~80 samples (8 per class)
- See EXP-001 for full dataset preparation details

## Results

### K-Fold Cross Validation (Honest Metric)
| Fold | Val Accuracy |
|------|-------------|
| 1    | 50.00%      |
| 2    | 58.75%      |
| 3    | 53.75%      |
| 4    | 55.00%      |
| 5    | 46.25%      |
| **Mean** | **52.75%** |
| **Std**  | **4.29%**  |

### Per-Class Accuracy (Full Dataset — Indicative Only)
| Class      | Accuracy | Match Quality | Notes                    |
|------------|----------|---------------|--------------------------|
| airplane   | 90.00%   | Strong        | Distinct audio signature |
| automobile | 82.50%   | Weak          | Car horn only            |
| bird       | 85.00%   | Strong        | Chirping birds           |
| cat        | 67.50%   | Strong        | Underperformed           |
| deer       | 90.00%   | Synthetic     | ⚠️ Likely memorised      |
| dog        | 72.50%   | Strong        | Underperformed           |
| frog       | 95.00%   | Strong        | Best performer           |
| horse      | 75.00%   | Weak          | Farm animals used        |
| ship       | 77.50%   | Moderate      | Sea waves + engine       |
| truck      | 87.50%   | Moderate      | Train sounds used        |

**Note:** Full dataset accuracy (82.25%) is inflated — model has seen 
training samples. K-Fold mean (52.75%) is the honest baseline figure.

## Key Observations
1. 52.75% mean accuracy vs 10% random chance — 5x better than random 
   with only 32 training samples per class
2. Frog and airplane are strongest — both have very distinct 
   audio signatures with little ambiguity
3. Cat and dog underperformed despite being direct ESC-50 matches — 
   possibly confused with each other
4. Deer at 90% on full dataset is suspicious — synthesised samples 
   likely have consistent musicgen signature that model memorised
5. Weak mapped classes (automobile, horse) performed reasonably — 
   suggests LSTM is learning audio features not just class identity

## Limitations
1. Very small dataset — 32 training samples per class is minimal
2. Deer class is fully synthesised — out-of-distribution risk
3. Horse mapped to farm animals — not true horse sounds
4. Full dataset per-class evaluation includes training data — 
   not a true held-out test

## Next Experiment
EXP-003 — Transfer Learning with Wav2Vec2
Same dataset, same evaluation protocol, direct comparison against 
this baseline. Expected significant improvement given pretrained 
audio representations.

## Benchmark Target
| Experiment | Mean Accuracy | Notes              |
|------------|---------------|--------------------|
| EXP-002    | 52.75%        | Baseline LSTM      |
| EXP-003    | TBD           | Wav2Vec2 fine-tune |