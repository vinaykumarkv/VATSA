# Experiment 005 — Object Detection

**Date:** 23 April 2026  
**Status:** ✅ Complete  
**Phase:** 1 — V-Module  

---

## Objective
Using Yolo model and finetuned embeddings to detect objects

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | coco dataset trained|
| Model  | Yolo |
| Started from checkpoint | Yes — Exp_004 |

---

## Result

| Metric | Value |
|--------|-------|
| Status | Success |


---

## Observations
- Yolo is trained on 80 classess, so we may need to use more robust model in final fusion
- we were able to detect the objects successfully

---

## Bugs / Issues
- no bugs were found just model path was not detected, so we used os and sys packages to direct notebook to right path

---

## Key Lesson
Learnt how embedding model we built from another model of 10 classess can be used in embeddings of 80 classess, its quite fascinating.

---

## Next Step
→ Experiment 006: application of objection detection on video streaming
