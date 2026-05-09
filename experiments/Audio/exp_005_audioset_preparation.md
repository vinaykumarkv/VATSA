# EXP-005 — AudioSet Dataset Preparation

**Date:** May 2026  
**Notebook:** notebooks/05_a_module_audioset_preparation.ipynb  
**Status:** Complete  

## Objective
Clean, balance, and finalise the AudioSet downloads into a training
ready dataset. Cap all classes at 100 samples for balance. Synthesise
deer class via audio augmentation.

## Balancing Decision
Raw downloads were uneven (93 to 176 per class). Two options considered:

| Option | Approach | Decision |
|--------|----------|----------|
| Cap at lowest (93) | Perfect balance, loses data | Rejected |
| Cap at 100 | Near-perfect balance, minimal loss | Selected |
| Weighted loss | Keep natural counts | Deferred to future experiment |

Capped at 100 — round number, above frog minimum for most classes,
keeps dataset balanced for fair benchmarking.

## Deer Synthesis — Augmentation Approach
MusicGen synthesis used in EXP-001 produced uncertain quality audio
and introduces an external model dependency that could affect
reproducibility in future experiments. Switched to deterministic
augmentation pipeline.

### Source Audio
Horse, dog, and cat clips from the capped AudioSet dataset.
More appropriate animal sound sources than ESC-50 farm animals
(cow, pig, sheep) used in EXP-001.

### Augmentation Pipeline
Applied in sequence to each generated sample:
1. Pitch shift — random uniform (-3, +3) semitones
2. Time stretch — random uniform (0.85, 1.15) rate
3. Low level Gaussian noise — mean 0, std 0.005
4. Normalise to peak amplitude

Random seed fixed at 42 for full reproducibility.

## Final Dataset

| Class      | Count | Source              |
|------------|-------|---------------------|
| airplane   | 100   | AudioSet            |
| automobile | 100   | AudioSet            |
| bird       | 100   | AudioSet            |
| cat        | 100   | AudioSet            |
| deer       | 100   | Augmentation        |
| dog        | 100   | AudioSet            |
| frog       | 93    | AudioSet (all available) |
| horse      | 100   | AudioSet            |
| ship       | 100   | AudioSet            |
| truck      | 100   | AudioSet            |
| **Total**  | **993**| —                  |

Note: frog has 93 samples — all available clips used, below 100 cap.
Total is 993 not 1000 for this reason.

## Audio Specifications
- Sample rate: 16000Hz (16kHz)
- Duration: 10 seconds per clip
- Format: WAV mono
- MelSpectrogram: 64 mel bands, fmax 8000Hz (for training)

## Comparison with ESC-50 Dataset

| Property          | ESC-50 Experiment  | AudioSet Experiment |
|-------------------|--------------------|---------------------|
| Samples per class | 40                 | ~100                |
| Total samples     | 400                | 993                 |
| Clip duration     | 5 seconds          | 10 seconds          |
| Sample rate       | 22050Hz            | 16000Hz             |
| Source            | Curated dataset    | Real-world YouTube  |
| Class coverage    | 7/10 native        | 9/10 native         |
| Deer source       | MusicGen synthesis | Augmentation        |

## Quality Checks
- All 993 files verified to load without errors
- Min/max duration within expected range
- Visual inspection of waveforms and MelSpectrograms per class

## Known Limitations
1. Deer is still synthesised — out-of-distribution risk remains
2. Frog has only 93 samples — slight class imbalance
3. AudioSet weak labels — target sound may not be prominent
   in full 10 second clip, unlike ESC-50 which is carefully curated
4. Augmented deer sourced from horse, dog, cat sounds —
   not true deer audio

## Next Experiment
EXP-006 — Retrain baseline LSTM and Wav2Vec2 on AudioSet dataset.
Direct comparison against EXP-002 and EXP-003 to measure impact
of larger, more diverse training data on accuracy.

## Files
- Final manifest: AudioSet/final_dataset.csv
- Synthesised deer: AudioSet/synthesised/deer/
- Notebook: notebooks/05_a_module_audioset_preparation.ipynb