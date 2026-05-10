# VATSA — Audio Module (A-Module)

## Overview
The Audio Module is the second completed component of the VATSA
unified five-modality architecture. It projects audio input into
a 512-dimensional shared latent space — the same space used by
the Visual Module — enabling cross-modal fusion.

## Architecture
- Backbone: facebook/wav2vec2-base (pretrained on speech)
- Input: Raw waveform at 16kHz
- Pooling: Mean pooling across time dimension
- Projection: Linear(768 → 512) + LayerNorm
- Output: 512-dim embedding in shared VATSA latent space

## Benchmark Results

| Experiment                        | Dataset  | Mean Acc | Std   |
|-----------------------------------|----------|----------|-------|
| Baseline LSTM (scratch)           | ESC-50   | 52.75%   | 4.29% |
| Wav2Vec2 Frozen                   | ESC-50   | 59.75%   | 2.29% |
| Wav2Vec2 Partial Unfreeze (best)  | ESC-50   | 70.25%   | 4.36% |
| Baseline LSTM (scratch)           | AudioSet | 28.30%   | 2.47% |
| Wav2Vec2 Frozen                   | AudioSet | 30.41%   | 3.37% |
| Wav2Vec2 Partial Unfreeze         | AudioSet | 34.54%   | 2.70% |

**Best checkpoint:** ESC-50 Wav2Vec2 Partial Unfreeze — 70.25%

## Key Research Finding
Dataset quality matters more than dataset size. ESC-50 (40 samples,
carefully curated) outperformed AudioSet (100 samples, weak YouTube
labels) across all experiments. Documented in EXP-006.

## Usage

```python
from modules.audio import VATSA_AudioPipeline

pipeline  = VATSA_AudioPipeline(checkpoint_path="path/to/checkpoint.pth")
result    = pipeline.run("audio.wav")
embedding = result["embedding"]  # (1, 512) — shared latent space
```

## Interface Contract
Matches VATSA_VisualEncoder exactly:
- Output dict key: `"embedding"`
- Embedding dim: 512
- Ready for cross-modal attention fusion

## Experiments
Full research logs in `experiments/audio/`:
- EXP-001: Data preparation (ESC-50)
- EXP-002: Baseline LSTM
- EXP-003: Transfer learning Wav2Vec2
- EXP-004: AudioSet download
- EXP-005: AudioSet preparation
- EXP-006: AudioSet training

## Status
✅ Complete — ready for V+A cross-modal fusion