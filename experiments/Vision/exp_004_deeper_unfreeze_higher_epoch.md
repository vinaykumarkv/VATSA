# Experiment 003 - Deeper Unfreeze

**Date:** 22 April 2026  
**Status:** 🔄 Running  
**Phase:** 1 — V-Module  

---

## Objective
Deeper unfreeze the bit early layers of EfficientNet-B0 (features[4:]) so they adapt to een more CIFAR-10 patterns. Continue training from the Experiment 002 checkpoint with a lower learning rate to avoid destroying pretrained weights. This answers: *"How much does domain-specific deeper fine-tuning improve over lighter unfreezed features?"*

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (same as Exp 003) |
| Model class | `VATSA_VisualEncoder(embedding_dim=512, num_classes=10)` |
| Backbone | `efficientnet_b0` — continued from Exp 003 checkpoint |
| Projection layer | `nn.Linear(1280, 512)` |
| Classifier | `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))` |
| Frozen layers | `backbone.features[0:5]` frozen |
| Trainable layers | `backbone.features[6:]` + Projection + Classifier |
| Optimizer | `AdamW`, lr=**5e-5** (reduced from 1e-4), weight_decay=1e-4 |
| Scheduler | `CosineAnnealingLR`, T_max=20 |
| Epochs | 40 (continued from Exp 003) |
| Batch size | 256 |
| num_workers | 2 |
| Mixed precision | Yes (GradScaler) |
| Dropout | `Dropout(0.3)` before classifier |

---

## Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.69%** |
| vs Previous best | 96.31% |
| Training time | Not monitored (away from screen)|
| Saved checkpoint | vasta_visual_encoder_cifar10_deeper_unfreeze_epoch.pth |

---

## Observations
- Loss curve was reduced slower and best models were shown after couple of epochs
- Model worked as expected and produced bit better performance

---

## Bugs / Issues
- I was unable to run the script on my laptop so used kaggle notebook which offers higher GPU and memory

---

## Key Lesson
Learnt how higher epoch may lead to slighly better performance, but no significant change.

---

## Next Step
→ Experiment 005: object detection using Yolo with our embeddings
