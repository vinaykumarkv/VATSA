# Experiment 001 — Frozen Backbone Baseline

**Date:** 21 April 2026  
**Status:** ✅ Complete  
**Phase:** 1 — V-Module  

---

## Objective
Establish a baseline using EfficientNet-B0 as a pure feature extractor with all backbone weights frozen. Only the projection head and classifier are trained. This answers: *"How much does ImageNet pretraining alone give us on CIFAR-10?"*

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Model class | `VATSA_VisualEncoder(embedding_dim=512, num_classes=10)` |
| Backbone | `efficientnet_b0` — `IMAGENET1K_V1` weights |
| Projection layer | `nn.Linear(1280, 512)` |
| Classifier | `nn.Linear(512, 10)` |
| Frozen layers | All `backbone.features` frozen |
| Trainable layers | Projection + Classifier only |
| Optimizer | `AdamW`, lr=1e-3, weight_decay=1e-4 |
| Scheduler | `CosineAnnealingLR`, T_max=20 |
| Epochs | 20 |
| Batch size | 256 |
| num_workers | 2 (Windows limitation) |
| Mixed precision | Yes (GradScaler) |
| Dropout | None |

---

## Result

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **79.19%** |
| Training time | (not monitored was away) |
| Saved checkpoint | `vat sa_visual_encoder_cifar10.pth` |

---

## Observations
- ImageNet features alone transfer well to CIFAR-10 even with no fine-tuning
- 79% is well above basic CNN from scratch (~70–75%), confirming transfer learning value
- Loss was still decreasing at epoch 20 — model had not converged fully

---

## Bugs Fixed This Session
- `nn.Linear(512,10)` was inside the training loop → moved to `__init__` as `self.classifier`
- `num_workers=8` caused Windows error 1455 → reduced to `num_workers=2` + `persistent_workers=True`
- wandb `ConnectionResetError` on Windows → removed wandb, replaced with `print()` logging

---

## Key Lesson
Frozen pretrained backbone = pure feature extractor. The 1280-dim EfficientNet output encodes rich ImageNet-learned patterns. The projection layer (1280→512) learns to compress these into VATSA's shared latent space. Even without domain adaptation, these features are highly transferable.

---

## Next Step
→ Experiment 002: Unfreeze last layers for domain-specific fine-tuning
