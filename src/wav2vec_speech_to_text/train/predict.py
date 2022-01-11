import torch
import torchaudio

from wav2vec_speech_to_text.model.wav2vec import Wav2Vec, Wav2VecCTC, decode
from wav2vec_speech_to_text.utils.model_state import ModelState
from wav2vec_speech_to_text.data.contants import labels
from wav2vec_speech_to_text.utils.preprocess import Preprocessor
from wav2vec_speech_to_text.config.config import hparams

MODEL_PATH = hparams["training"]["model_path"]

def stt_predict(filepath, model_path="checkpoints/best_wave2vec_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform, _ = torchaudio.load(filepath)
    waveform = Preprocessor().resample(waveform).squeeze(0).unsqueeze(0).unsqueeze(0).to(device)

    base = Wav2Vec()
    model = Wav2VecCTC(base, vocab_size=len(labels)).to(device)
    ModelState(model).load(path=MODEL_PATH, map_location=device)
    model.eval()

    with torch.no_grad():
        logits = model(waveform)[0]
        return decode(logits)
