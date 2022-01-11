import torch.nn as nn
from wav2vec_speech_to_text.config.config import hparams

DIM = hparams["model"]["dim"]
NUM_LAYERS = hparams["model"]["num_layers"]
NHEAD = hparams["model"]["nhead"]

class ContextNetwork(nn.Module):
    """ 
    Context Network for Wav2Vec model.
    This network uses a Transformer encoder to process the latent features extracted from the audio input.  
    It is designed to capture long-range dependencies in the audio signal.
    """
    def __init__(self, dim=DIM, num_layers=NUM_LAYERS, nhead=NHEAD):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        return x.permute(1, 2, 0)