# EXP-001 — Audio Module Data Preparation

**Date:** May 2026  
**Status:** In Progress  

## Objective
Build a balanced 10-class audio dataset mapped to CIFAR-10 categories
for the VATSA Audio Encoder baseline experiment.

## Dataset
- Source: ESC-50 (2000 samples, 50 classes, 5 seconds each, 44.1kHz)
- Target: 10 classes matching CIFAR-10 labels

## Class Mapping Decisions

| CIFAR-10   | ESC-50 Source           | Match Quality | Notes                        |
|------------|-------------------------|---------------|------------------------------|
| airplane   | airplane, helicopter    | Strong        | Both aerial vehicles         |
| automobile | car_horn                | Weak          | Sound effect not engine      |
| bird       | chirping_birds          | Strong        | Direct match                 |
| cat        | cat                     | Strong        | Direct match                 |
| deer       | synthesised             | Synthetic     | Not in ESC-50 - see below    |
| dog        | dog                     | Strong        | Direct match                 |
| frog       | frog                    | Strong        | Direct match                 |
| horse      | cow, pig, sheep         | Weak          | Farm animals, not horse      |
| ship       | sea_waves, engine       | Moderate      | Combined for better coverage |
| truck      | train                   | Moderate      | Heavy vehicle sound          |

## Synthesis Decision — Deer
- ESC-50 has no deer class
- Attempted: facebook/audiogen-medium — blocked by transformers compatibility
- Used: facebook/musicgen-small via HuggingFace pipeline
- Note: musicgen is a music model not environmental sound model
  — synthesis quality is unknown and a potential variable in results
- Fallback available: pitch shift augmentation of farm animal sounds
  if musicgen quality is poor after listening test

## Balancing
- All classes capped at 40 samples
- Total dataset: 400 samples (360 ESC-50 + 40 synthesised deer)
- Train/Val split: 80/20 stratified

## Known Weaknesses — To Discuss in Preprint
1. automobile mapped to car_horn — atypical automobile sound
2. horse mapped to farm animals — not true horse sounds
3. deer fully synthesised — out-of-distribution risk
4. Small dataset — 40 samples per class is very limited for RNN training
   (ESC-50 is designed for classification research not as a large-scale dataset)

## Next Experiment
EXP-002 — Baseline LSTM from scratch on this dataset