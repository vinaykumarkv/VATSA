# Experiment 002 — Fine-tune Last Layers

**Date:** 22 April 2026  
**Status:** ✅ Complete  
**Phase:** 1 — V-Module  

---

## Objective
Unfreeze the later layers of EfficientNet-B0 (features[6:]) so they adapt to CIFAR-10 patterns. Continue training from the Experiment 001 checkpoint with a lower learning rate to avoid destroying pretrained weights. This answers: *"How much does domain-specific fine-tuning improve over frozen features?"*

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (same as Exp 001) |
| Model class | `VATSA_VisualEncoder(embedding_dim=512, num_classes=10)` |
| Backbone | `efficientnet_b0` — continued from Exp 001 checkpoint |
| Projection layer | `nn.Linear(1280, 512)` |
| Classifier | `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))` |
| Frozen layers | `backbone.features[0:5]` frozen |
| Trainable layers | `backbone.features[6:]` + Projection + Classifier |
| Optimizer | `AdamW`, lr=**1e-4** (reduced from 1e-3), weight_decay=1e-4 |
| Scheduler | `CosineAnnealingLR`, T_max=20 |
| Epochs | 20 (continued from Exp 001) |
| Batch size | 256 |
| num_workers | 2 |
| Mixed precision | Yes (GradScaler) |
| Dropout | `Dropout(0.3)` before classifier |

---

## Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.46%** |
| vs Experiment 001 | **+15.27%** |
| Saved checkpoint | `vatsa_visual_encoder_cifar10.pth` (overwritten with best) |

---

## Observations
- Massive gain from unfreezing last layers confirms that earlier layers (edges, textures) are universal, while later layers (semantic concepts) need domain adaptation
- Lower LR (1e-4 vs 1e-3) was critical — prevents catastrophic forgetting of pretrained weights
- Dropout(0.3) helped regularisation — model generalised better to test set
- Loss was still declining at epoch 20, suggesting more epochs could yield further gains

---

## Bugs Fixed This Session
- Evaluation code used a separate fresh `nn.Linear` not loaded from `.pth` → rebuilt `VATSA_VisualEncoder` class with `self.classifier` matching training code, loaded from checkpoint

---

## Key Lesson
Fine-tuning later layers unlocks domain-specific pattern learning. The early frozen layers act as a universal feature detector (edges, colours, textures from ImageNet). The unfrozen late layers adapt those features to CIFAR-10-specific concepts. This is the core mechanism of transfer learning in practice.

---

## Next Step
→ Experiment 003: Deeper unfreeze — try `features[4:]` with lr=5e-5 to push toward 95–96%
