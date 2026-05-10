import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from .config import AUDIO_CONFIG


class VATSA_AudioEncoder(nn.Module):
    """
    VATSA Audio Module — A-Module
    Projects audio into 512-dim shared latent space.
    Mirrors VATSA_VisualEncoder interface for multimodal fusion.

    Backbone: facebook/wav2vec2-base
    Input:    Raw waveform tensor (batch, time_steps) at 16kHz
    Output:   {"embedding": tensor (batch, 512)}
    """

    def __init__(
        self,
        embedding_dim   : int  = AUDIO_CONFIG["embedding_dim"],
        backbone_name   : str  = AUDIO_CONFIG["backbone"],
        freeze_backbone : bool = True,
        dropout         : float = AUDIO_CONFIG["dropout"],
    ):
        super().__init__()

        if backbone_name == "wav2vec2-base":
            self.backbone = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base"
            )
            hidden_size = self.backbone.config.hidden_size  # 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.norm       = nn.LayerNorm(embedding_dim)
        self.dropout    = nn.Dropout(dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch, time_steps) — raw waveform at 16kHz
        outputs    = self.backbone(x)
        hidden     = outputs.last_hidden_state   # (batch, seq, 768)
        pooled     = hidden.mean(dim=1)          # (batch, 768)
        pooled     = self.dropout(pooled)
        embedding  = self.projection(pooled)     # (batch, 512)
        embedding  = self.norm(embedding)        # normalise

        return {"embedding": embedding}

    def unfreeze_top_layers(self, num_layers: int = 4) -> None:
        """Unfreeze top N transformer layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        encoder_layers = self.backbone.encoder.layers
        for layer in encoder_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable parameters after unfreeze: {trainable:,}")