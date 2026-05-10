# EXP-006 — AudioSet Training: Baseline LSTM and Wav2Vec2

**Date:** May 2026  
**Notebook:** notebooks/audio/06_a_module_audioset_training.ipynb  
**Status:** Complete  

## Objective
Retrain baseline LSTM and Wav2Vec2 transfer learning models on the
AudioSet dataset (EXP-005) to measure the impact of larger, more
diverse training data. Direct comparison against ESC-50 experiments
(EXP-002, EXP-003).

## Results

### K-Fold Cross Validation Summary
| Experiment                        | Dataset  | Mean Acc | Std   |
|-----------------------------------|----------|----------|-------|
| Baseline LSTM (scratch)           | ESC-50   | 52.75%   | 4.29% |
| Wav2Vec2 Frozen (Stage 1)         | ESC-50   | 59.75%   | 2.29% |
| Wav2Vec2 Partial Unfreeze (Stg 2) | ESC-50   | 70.25%   | 4.36% |
| Baseline LSTM (scratch)           | AudioSet | 28.30%   | 2.47% |
| Wav2Vec2 Frozen (Stage 1)         | AudioSet | 30.41%   | 3.37% |
| Wav2Vec2 Partial Unfreeze (Stg 2) | AudioSet | 34.54%   | 2.70% |

### AudioSet Per-Class Accuracy — Baseline LSTM
| Class      | Accuracy | Notes                        |
|------------|----------|------------------------------|
| airplane   | 17.00% ⚠️ | Poor — noisy clips           |
| automobile | 42.00% ⚠️ | Moderate                     |
| bird       | 37.00% ⚠️ | Expected strong — underperformed |
| cat        | 19.00% ⚠️ | Poor                         |
| deer       | 83.00%   | ⚠️ Inflated — augmentation memorisation |
| dog        | 18.00% ⚠️ | Poor despite direct match    |
| frog       | 34.41% ⚠️ | Moderate                     |
| horse      | 18.00% ⚠️ | Poor                         |
| ship       | 2.00%  ⚠️ | Near-zero — model learned nothing |
| truck      | 50.00%   | Best performer                |

### AudioSet Per-Class Accuracy — Wav2Vec2 Stage 2
| Class      | Accuracy | Notes                        |
|------------|----------|------------------------------|
| airplane   | 29.00% ⚠️ | Poor — far from speech domain |
| automobile | 46.00% ⚠️ | Moderate                     |
| bird       | 61.00%   | Best — tonal, closer to speech |
| cat        | 45.00% ⚠️ | Moderate                     |
| deer       | 60.00%   | ⚠️ Augmentation memorisation  |
| dog        | 11.00% ⚠️ | Worse than baseline          |
| frog       | 55.91%   | Moderate                     |
| horse      | 40.00% ⚠️ | Poor                         |
| ship       | 44.00% ⚠️ | Poor                         |
| truck      | 15.00% ⚠️ | Poor — far from speech domain |

## Key Findings

### Finding 1 — AudioSet underperformed ESC-50 despite more data
AudioSet baseline (28.30%) is significantly worse than ESC-50
baseline (52.75%) despite having 2.5x more samples per class.
More data did not help — data quality was the limiting factor.

### Finding 2 — Dataset quality matters more than dataset size
ESC-50 is carefully curated — every clip is predominantly the
target sound for its full 5 second duration. AudioSet clips are
10 seconds of real-world YouTube audio where the target sound is
often brief, background, or mixed with other sounds. The model
cannot reliably learn class features from weakly labelled clips.

### Finding 3 — Transfer learning domain mismatch is amplified
On ESC-50, Wav2Vec2 gave +17.50% over baseline despite speech
pretraining mismatch. On AudioSet, Wav2Vec2 gave only +6.24%
over baseline — and both are far below ESC-50 results.
Hypothesis: on very small datasets (ESC-50, 32 train samples)
any structured pretrained representation helps. On larger noisy
datasets the domain mismatch (speech vs environmental sound)
dominates and limits transfer learning benefit.

### Finding 4 — Deer augmentation inflates results
Deer scores 83% (baseline) and 60% (Wav2Vec2) — significantly
above other classes. Augmented samples are highly similar to
each other, making deer an easy class to memorise rather than
generalise. This confirms augmented classes should be treated
with caution in per-class analysis.

### Finding 5 — Ship near-zero in baseline
Ship scored 2% in baseline LSTM — essentially random. AudioSet
ship clips likely contain highly variable content (ocean sounds,
engine noise, music on boats) making it impossible for a simple
LSTM to find consistent features with limited training data.

## Analysis — Why ESC-50 > AudioSet

| Factor              | ESC-50             | AudioSet           |
|---------------------|--------------------|--------------------|
| Clip duration       | 5 seconds          | 10 seconds         |
| Label quality       | Carefully curated  | Weak YouTube labels|
| Sound prominence    | Dominant throughout| Often background   |
| Background noise    | Minimal            | Real-world noise   |
| Train samples/class | 32 (after split)   | ~80 (after split)  |
| Result              | 52.75%             | 28.30%             |

## Implications for VATSA A-Module

The ESC-50 Wav2Vec2 Stage 2 model (70.25%) remains the best
performing A-module checkpoint. For VATSA cross-modal fusion
experiments, this model will be used as the audio encoder.

AudioSet results demonstrate that for the A-module to improve
meaningfully, a properly curated audio dataset with CIFAR-10
aligned classes is needed — not just more data. This is a
known challenge in cross-modal research and is documented as
an open research question for the VATSA architecture.

## Recommended Next Steps

### Option A — Use ESC-50 Wav2Vec2 model for fusion
Accept 70.25% as current A-module benchmark. Proceed to
cross-modal V+A fusion experiment. Document dataset limitation
openly in preprint.

### Option B — Curate a clean AudioSet subset
Manually filter AudioSet clips to keep only those where target
sound is prominent. Labour intensive but would produce a cleaner
dataset. Estimated 20-30% of clips are high quality.

### Option C — Use a domain-matched pretrained model
Replace Wav2Vec2 (speech) with PANNs or AudioCLIP —
pretrained on environmental sounds not speech. Expected to
significantly improve transfer learning results on both datasets.

Recommended: Option A now, Option C in next experiment version.

## Files
- Notebook: notebooks/audio/06_a_module_audioset_training.ipynb
- Baseline model: vatsa_audio_encoder_audioset_baseline.pth
- Transfer model: vatsa_audio_encoder_audioset_transfer.pth
- Best overall: vatsa_audio_encoder_transfer.pth (ESC-50, EXP-003)