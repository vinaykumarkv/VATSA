import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VATSA_VisualEncoder(nn.Module):
    """
    VATSA Visual Module — V-Module
    Projects images into 512-dim shared latent space.
    Mirrors VATSA_AudioEncoder interface for multimodal fusion.

    Backbone: EfficientNet-B0 (pretrained on ImageNet)
    Input:    Image tensor (batch, 3, 224, 224)
    Output:   {"embedding": tensor (batch, 512)}
    """

    def __init__(
        self,
        embedding_dim  : int   = 512,
        backbone_name  : str   = "efficientnet_b0",
        pretrained     : bool  = True,
        freeze_backbone: bool  = True,
    ):
        super().__init__()

        if backbone_name == "efficientnet_b0":
            weights  = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        num_features            = backbone.classifier[1].in_features
        backbone.classifier     = nn.Identity()
        self.backbone           = backbone

        self.projection         = nn.Linear(num_features, embedding_dim)
        self.norm               = nn.LayerNorm(embedding_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> dict:
        features  = self.backbone(x)
        embedding = self.projection(features)
        embedding = self.norm(embedding)
        return {"embedding": embedding}

    def unfreeze_top_layers(self, num_layers: int = 4) -> None:
        """Unfreeze top N EfficientNet blocks for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = list(self.backbone.features.children())
        for block in blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable parameters after unfreeze: {trainable:,}")