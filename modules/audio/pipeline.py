import torch
import librosa
import numpy as np
from pathlib import Path
from .encoder import VATSA_AudioEncoder
from .config import AUDIO_CONFIG


class VATSA_AudioPipeline:
    """
    VATSA Audio Pipeline
    End-to-end inference from audio file to 512-dim embedding.
    Mirrors VATSA_VisualPipeline interface.

    Usage:
        pipeline = VATSA_AudioPipeline()
        result   = pipeline.run("path/to/audio.wav")
        embedding = result["embedding"]  # (1, 512)
    """

    def __init__(
        self,
        checkpoint_path : str  = AUDIO_CONFIG["checkpoint"],
        device          : str  = None,
        freeze_backbone : bool = True,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.n_samples   = (
            self.sample_rate * AUDIO_CONFIG["duration"]
        )

        # Load encoder
        self.encoder = VATSA_AudioEncoder(
            embedding_dim   = AUDIO_CONFIG["embedding_dim"],
            freeze_backbone = freeze_backbone
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location = self.device
        )
        # Load only encoder-compatible keys
        encoder_state = {
            k: v for k, v in checkpoint["model_state"].items()
            if not k.startswith("classifier")
        }
        self.encoder.load_state_dict(encoder_state, strict=False)
        self.encoder.eval()

        print(f"VATSA AudioPipeline ready — device: {self.device}")

    def preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file to waveform tensor."""
        y, _ = librosa.load(
            audio_path,
            sr       = self.sample_rate,
            duration = self.n_samples / self.sample_rate
        )

        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        else:
            y = y[:self.n_samples]

        y = (y - y.mean()) / (y.std() + 1e-9)

        return torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    def run(self, audio_path: str) -> dict:
        """
        Run full pipeline on a single audio file.

        Args:
            audio_path: path to .wav file

        Returns:
            dict with keys:
                embedding — (1, 512) tensor in shared latent space
                audio_path — input path for reference
        """
        audio_tensor = self.preprocess(audio_path).to(self.device)

        with torch.no_grad():
            result = self.encoder(audio_tensor)

        return {
            "embedding"  : result["embedding"],
            "audio_path" : audio_path
        }

    def run_batch(self, audio_paths: list) -> dict:
        """
        Run pipeline on a list of audio files.

        Args:
            audio_paths: list of paths to .wav files

        Returns:
            dict with keys:
                embeddings — (N, 512) tensor
                audio_paths — input paths for reference
        """
        tensors = torch.stack([
            self.preprocess(p).squeeze(0) for p in audio_paths
        ]).to(self.device)

        with torch.no_grad():
            result = self.encoder(tensors)

        return {
            "embeddings" : result["embedding"],
            "audio_paths": audio_paths
        }