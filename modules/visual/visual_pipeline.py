import torch
from vatsa.encoder import VATSA_VisualEncoder
from vatsa.detector import VATSA_Detector


class VATSA_VisualPipeline:
    """
    Full V-Module pipeline:
    Image → YOLO detection → Region crops → 512-dim embeddings
    """

    def __init__(self, encoder_path, device="cuda"):
        self.device = device

        # Load your trained encoder
        self.encoder = VATSA_VisualEncoder(
            embedding_dim=512,
            freeze_backbone=True
        ).to(device)

        checkpoint = torch.load(encoder_path, map_location=device)
        # Load only backbone + projection (no classifier needed for embeddings)
        state = {k: v for k, v in checkpoint['model_state'].items()
                 if not k.startswith('classifier')}
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.eval()

        self.detector = VATSA_Detector()

    @torch.no_grad()
    def process_image(self, pil_image):
        """
        Input:  PIL Image
        Output: list of {label, confidence, bbox, embedding (512-dim)}
        """
        regions = self.detector.detect_and_crop(pil_image)

        if not regions:
            return []

        # Batch all crops through encoder at once
        crops = torch.stack([r["crop_tensor"] for r in regions]).to(self.device)
        embeddings = self.encoder(crops)["embedding"]  # (N, 512)

        results = []
        for i, region in enumerate(regions):
            results.append({
                "label": region["label"],
                "confidence": region["confidence"],
                "bbox": region["bbox"],
                "embedding": embeddings[i]  # 512-dim tensor
            })

        return results