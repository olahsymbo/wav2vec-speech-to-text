import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

from wav2vec_speech_to_text.model.cnn import FeatureEncoder
from wav2vec_speech_to_text.model.transformer import ContextNetwork
from wav2vec_speech_to_text.data.contants import index_to_label
from wav2vec_speech_to_text.config.config import hparams

DIM = hparams["model"]["dim"]


class Wav2Vec(nn.Module):
    """
    Wav2Vec model that encodes audio features and computes context representations.
    This model consists of a feature encoder and a context network.
    The feature encoder extracts latent features from the input audio,
    and the context network processes these features to produce context representations.
    """

    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.context = ContextNetwork()
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)
        return z, c

class Wav2VecCTC(nn.Module):
    """ Wav2Vec model with a CTC head for speech-to-text tasks.
    This model uses the Wav2Vec base model to extract features and context,
    and adds a convolutional layer to project the context output to the vocabulary size.
    The output is a log-softmax tensor suitable for CTC loss computation.
    """
    def __init__(self, base_model, vocab_size):
        super().__init__()
        self.base = base_model
        self.ctc_head = nn.Sequential(
            nn.Conv1d(DIM, vocab_size, kernel_size=1)
        )

    def forward(self, x):
        _, c = self.base(x)
        logits = self.ctc_head(c)
        return logits.log_softmax(dim=1)

class Wav2VecClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            _, c = self.base(x) 
        pooled = c.mean(dim=2) 
        return self.classifier(pooled)


def contrastive_loss(c, z, mask_indices):
    """
    c: context output (B, C, T)
    z: latent features (B, C, T)
    mask_indices: indices of masked timesteps (B, T)
    """
    B, C, T = c.shape
    losses = []

    for b in range(B):
        for t in range(T):
            if mask_indices[b, t]:
                true = z[b, :, t]  
                pred = c[b, :, t]  
                negatives = z[b].permute(1, 0)  # all other z's in this sample

                pos_sim = F.cosine_similarity(pred, true, dim=0)
                neg_sim = F.cosine_similarity(pred.unsqueeze(0), negatives, dim=1)

                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.tensor(0, dtype=torch.long).to(z.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
                losses.append(loss)

    return torch.stack(losses).mean()

def greedy_decoder(logits, labels):
    """
    Greedy decoder for CTC outputs.
    Args:
        logits: Tensor of shape (T, C) where T is the number of timesteps and C is the number of classes.
        labels: List of labels corresponding to the classes.
    Returns:
        Decoded string from the logits.
    """
    pred_ids = logits.argmax(dim=1)
    pred_tokens = [labels[i] for i in pred_ids]
    return ''.join(pred_tokens)

def decode(logits):
    
    pred = torch.argmax(logits, dim=1)
    pred = pred.cpu().numpy()
    prev = -1
    output = []
    for p in pred:
        if p != prev and p != 0:
            output.append(index_to_label[p])
        prev = p
    return ''.join(output)