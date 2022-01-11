import torch.nn as nn
from wav2vec_speech_to_text.config.config import hparams

IN_CHANNELS = hparams["model"]["in_channels"]
OUT_CHANNELS = hparams["model"]["out_channels"]

class FeatureEncoder(nn.Module):
    """
    Feature Encoder for Wav2Vec model.
    This module processes the raw audio waveform input and extracts high-level features.
    It consists of several convolutional layers that progressively downsample the input.
    The output shape is (B, C, T'), where B is the batch size,  
    C is the number of channels, and T' is the reduced time dimension.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, OUT_CHANNELS, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(IN_CHANNELS, OUT_CHANNELS, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(IN_CHANNELS, OUT_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)  # shape: (B, C, T')