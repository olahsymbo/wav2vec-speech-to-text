import os

import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
from wav2vec_speech_to_text.config.config import hparams
from wav2vec_speech_to_text.utils.preprocess import Preprocessor
from wav2vec_speech_to_text.config.config import hparams

SAMPLE_RATE = hparams["dataset"]["sample_rate"]

class AudioDataset(Dataset):
    """
    Load audio data from the LibriSpeech dataset.
    It applies mel spectrogram transformation and resampling to the audio data.
    """
    def __init__(self, root="artifacts/librispeech", subset="train-clean-100", n_mel=80):
        os.makedirs(root, exist_ok=True)

        should_download = not os.path.exists(os.path.join(root, subset))
        
        self.dataset = LIBRISPEECH(root=root, url=subset, download=should_download)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel,
            normalized=True,
        )
        self.preprocessor = Preprocessor()

    def __len__(self):
        return 10 #len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        waveform = self.preprocessor.resample(waveform)
        return waveform, transcript.lower()
