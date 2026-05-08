# Experiment 006 — Bench Marking Visual Module

**Date:** 23 April 2026  
**Status:** ✅ Complete  
**Phase:** 1 — V-Module  

---

## Objective
Bench marking the visual module which is using Yolo model and finetuned embeddings to detect objects in a video stream

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | coco dataset trained|
| Model  | Yolo |
| Started from checkpoint | Yes — Exp_006 |

---

## Result

| Metric | Value |
|--------|-------|
| Status | Success |


---

## Observations
- FPS:              22.71
- Embed time:       43.25ms   ← includes YOLO + encoder per frame
- Detections/frame: 3.3 avg

- Detection mean:   18.3ms   p95: 21.4ms
- Embedding mean:   0.0ms    ← synthetic images had no detections so encoder never ran
- Implied FPS:      54.6     ← detection only, no embedding cost

- Batch  1:    89 emb/sec
- Batch  4:   361 emb/sec
- Batch  8:   695 emb/sec
- Batch 16:  1336 emb/sec  ← sweet spot
- Batch 32:  1196 emb/sec  ← VRAM limit, slight drop

- Consistency:          1.0000  ✅ perfect — deterministic
- Augmentation robust:  0.2883  ⚠️  low — expected, encoder trained on classification not contrastive learning
- Inter-image sep:      0.3810  ✅ good — embeddings are distinct

- Allocated:  63.7 MB   ← very lean
- Peak:       73.7 MB
- Reserved:  826.0 MB   ← PyTorch pre-reserves, not actually used

**One note on augmentation robustness (0.2883)**: This is low but not a problem right now. It means the same image cropped differently produces fairly different embeddings. This would matter for contrastive learning (Phase 5 fusion) but is fine for Phase 1 detection. It will improve naturally when we will train with contrastive objectives in Phase 5.


---

## Bugs / Issues
- no bugs

---

## Key Lesson
Learnt benchmarking metrics for visual encoder and detection model

---

## Next Step
→ Experiment 008 : Summarizing Visual model and planning next steps in Audio Encoder and dection model
