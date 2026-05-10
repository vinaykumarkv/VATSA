import torch
from modules.visual.encoder  import VATSA_VisualEncoder
from modules.visual.detector import VATSA_Detector
from modules.visual.config   import VISUAL_CONFIG


class VATSA_VisualPipeline:
    """
    Full V-Module pipeline:
    Image → YOLO detection → Region crops → 512-dim embeddings

    Usage:
        pipeline  = VATSA_VisualPipeline()
        results   = pipeline.process_image(pil_image)
        embedding = results[0]["embedding"]  # (512,)
    """

    def __init__(
        self,
        encoder_path: str = VISUAL_CONFIG["checkpoint"],
        device      : str = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load trained encoder — strip classifier head
        self.encoder = VATSA_VisualEncoder(
            embedding_dim   = VISUAL_CONFIG["embedding_dim"],
            freeze_backbone = True
        ).to(self.device)

        checkpoint = torch.load(
            encoder_path, map_location=self.device
        )
        state = {
            k: v for k, v in checkpoint["model_state"].items()
            if not k.startswith("classifier")
        }
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.eval()

        self.detector = VATSA_Detector()
        print(f"VATSA VisualPipeline ready — device: {self.device}")

    @torch.no_grad()
    def process_image(self, pil_image) -> list:
        """
        Input:  PIL Image
        Output: list of dicts with keys:
                  label, confidence, bbox, embedding (512-dim tensor)
        """
        regions = self.detector.detect_and_crop(pil_image)

        if not regions:
            return []

        crops      = torch.stack(
            [r["crop_tensor"] for r in regions]
        ).to(self.device)
        embeddings = self.encoder(crops)["embedding"]  # (N, 512)

        results = []
        for i, region in enumerate(regions):
            results.append({
                "label"     : region["label"],
                "confidence": region["confidence"],
                "bbox"      : region["bbox"],
                "embedding" : embeddings[i]
            })

        return results