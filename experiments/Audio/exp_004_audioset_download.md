# EXP-004 — AudioSet Download and Class Mapping

**Date:** May 2026  
**Notebook:** notebooks/04_a_module_audioset_download.ipynb  
**Status:** Complete  

## Objective
Download AudioSet clips for 10 CIFAR-10 aligned classes to expand
the training dataset beyond ESC-50's 40 samples per class.
AudioSet provides a significantly larger and more diverse audio
corpus sourced from real-world YouTube content.

## Why AudioSet over UrbanSound8K
UrbanSound8K was evaluated first but rejected — only 2 out of 10
CIFAR-10 classes mapped cleanly (automobile, dog). AudioSet has
direct matches for 9 out of 10 classes with only deer missing.
Cross-modal fusion with CIFAR-10 visual classes is the core VATSA
objective — dataset choice must preserve that alignment.

## AudioSet Ontology IDs Used

| CIFAR-10   | AudioSet ID  | Display Name              | Match Quality |
|------------|--------------|---------------------------|---------------|
| airplane   | /m/0cmf2     | Fixed-wing aircraft       | Strong        |
| automobile | /m/0k4j      | Car                       | Strong        |
| bird       | /m/015p6     | Bird                      | Strong        |
| cat        | /m/01yrx     | Cat                       | Strong        |
| deer       | None         | —                         | Synthesis     |
| dog        | /m/068hy     | Domestic animals, pets    | Strong        |
| frog       | /m/09ld4     | Frog                      | Strong        |
| horse      | /m/03k3r     | Horse                     | Strong        |
| ship       | /m/06q74     | Ship                      | Strong        |
|            | /m/019jd     | Boat, Water vehicle       | Strong        |
| truck      | /m/07r04     | Truck                     | Strong        |

## ID Corrections During Mapping
Initial IDs were incorrect for three classes — identified and fixed
by searching class_labels_indices.csv directly:
- frog: /m/0lt4_ → /m/09ld4 (original mapped to wrong class)
- horse: /m/03vt0 → /m/03k3r (original mapped to Insect)
- ship: /m/04rlf → /m/06q74 (original mapped to Music)

## Download Results

| Class      | Segments Available | Downloaded | Failed |
|------------|--------------------|------------|--------|
| airplane   | 127                | 108        | 19     |
| automobile | 200                | 176        | 24     |
| bird       | 200                | 165        | 35     |
| cat        | 200                | 167        | 33     |
| deer       | 0                  | 0          | —      |
| dog        | 200                | 167        | 33     |
| frog       | 123                | 93         | 30     |
| horse      | 163                | 125        | 38     |
| ship       | 200                | 178        | 22     |
| truck      | 175                | 150        | 25     |
| **Total**  | **1588**           | **1329**   | **259**|

**Success rate: 83.7%** — failures due to YouTube video takedowns,
geo-restrictions, or unavailable content. Expected for YouTube-sourced
datasets.

## Technical Setup
- Tool: yt-dlp 2026.03.17
- Audio conversion: ffmpeg 8.1.1
- Format: WAV, 16kHz, mono
- Clip duration: 10 seconds (AudioSet standard)
- Download capped at 200 per class

## Known Limitations
1. YouTube-sourced — reproducibility depends on video availability
   at time of download. Failed IDs saved to failed_downloads.txt
2. Deer class unavailable in AudioSet ontology — requires synthesis
3. Some classes below 200 cap due to limited AudioSet coverage
   (frog: 93, airplane: 108, horse: 125)
4. Clip labels are weak — AudioSet uses segment-level labels,
   target sound may not be prominent throughout entire clip

## Files
- Download manifest: AudioSet/download_list.csv
- Failed IDs: AudioSet/failed_downloads.txt
- Raw clips: AudioSet/clips/<class>/
- Notebook: notebooks/audio/04_a_module_audioset_download.ipynb