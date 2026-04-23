import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VATSA_VisualEncoder(nn.Module):
    """
    VATSA Visual Module
    Projects images into 512-dim shared latent space.
    Future-ready for multimodal fusion.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()

        if backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        else:
            raise ValueError("Unsupported backbone")

        num_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Linear(num_features, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 10)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        embedding = self.projection(features)
        embedding = self.norm(embedding)
        return {"embedding": embedding}