import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from .config import AUDIO_CONFIG


class VATSA_AudioDataset(Dataset):
    """
    VATSA Audio Module Dataset
    Loads audio files and returns raw waveform tensors
    for Wav2Vec2 input at 16kHz.
    """

    def __init__(
        self,
        dataframe,
        sample_rate : int            = AUDIO_CONFIG["sample_rate"],
        duration    : int            = AUDIO_CONFIG["duration"],
        label_col   : Optional[str]  = "cifar_label",
    ):
        self.df          = dataframe.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.n_samples   = sample_rate * duration
        self.label_col   = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        path = row["filepath"]

        y, _ = librosa.load(
            path,
            sr       = self.sample_rate,
            duration = self.n_samples / self.sample_rate
        )

        # pad or trim to fixed length
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        else:
            y = y[:self.n_samples]

        # normalise
        y = (y - y.mean()) / (y.std() + 1e-9)

        audio_tensor = torch.tensor(y, dtype=torch.float32)

        if self.label_col and self.label_col in row:
            label = torch.tensor(row[self.label_col], dtype=torch.long)
            return audio_tensor, label

        return audio_tensor