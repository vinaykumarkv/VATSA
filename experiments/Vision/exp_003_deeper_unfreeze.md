# Experiment 003 - Deeper Unfreeze

**Date:** 22 April 2026  
**Status:** ✅ Complete  
**Phase:** 1 — V-Module  

---

## Objective
Deeper unfreeze the bit early layers of EfficientNet-B0 (features[4:]) so they adapt to een more CIFAR-10 patterns. Continue training from the Experiment 002 checkpoint with a lower learning rate to avoid destroying pretrained weights. This answers: *"How much does domain-specific deeper fine-tuning improve over lighter unfreezed features?"*

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (same as Exp 002) |
| Model class | `VATSA_VisualEncoder(embedding_dim=512, num_classes=10)` |
| Backbone | `efficientnet_b0` — continued from Exp 002 checkpoint |
| Projection layer | `nn.Linear(1280, 512)` |
| Classifier | `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))` |
| Frozen layers | `backbone.features[0:5]` frozen |
| Trainable layers | `backbone.features[6:]` + Projection + Classifier |
| Optimizer | `AdamW`, lr=**5e-5** (reduced from 1e-4), weight_decay=1e-4 |
| Scheduler | `CosineAnnealingLR`, T_max=20 |
| Epochs | 20 (continued from Exp 002) |
| Batch size | 256 |
| num_workers | 2 |
| Mixed precision | Yes (GradScaler) |
| Dropout | `Dropout(0.3)` before classifier |

---

## Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 96.31% |
| vs Previous best | 94.46% |
| Training time | Not monitored (away from screen)|
| Saved checkpoint | vasta_visual_encoder_cifar10_deeper_unfreeze.pth |

---

## Observations
- Loss curve was reduced slower and best models were shown after couple of epochs
- Model worked as expected and produced bit better performance

---

## Bugs / Issues
- found a bug in testing script since weights were not matched.
- I was unable to run the script on my laptop so used kaggle notebook which offers higher GPU and memory

---

## Key Lesson
Learnt how deeper unfreezing can unlocked slighly better performance

---

## Next Step
→ Experiment 004: will perform final improvement with higher epoches on top of checkpoint.
