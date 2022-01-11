import torch

from wav2vec_speech_to_text.data.splitter import DataSplitter
from wav2vec_speech_to_text.data.loader import AudioDataset
from wav2vec_speech_to_text.model.wav2vec import Wav2Vec, Wav2VecCTC, decode
from wav2vec_speech_to_text.utils.metrics import compute_metrics
from wav2vec_speech_to_text.utils.model_state import ModelState
from wav2vec_speech_to_text.data.contants import labels
from wav2vec_speech_to_text.config.config import hparams

MODEL_PATH = hparams["training"]["model_path"]

@torch.no_grad()
def test_wave2vec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = AudioDataset(subset="test-clean")
    test_dl = DataSplitter(test_ds, batch_size=1, shuffle=False)

    base = Wav2Vec()
    model = Wav2VecCTC(base, vocab_size=len(labels)).to(device)
    ModelState(model).load(path=MODEL_PATH, map_location=device)
    model.eval()

    refs, hyps = [], []

    for waveform, transcript in test_dl:
        waveform = waveform.to(device).unsqueeze(1)  # shape: (B, 1, T)
        logits = model(waveform)  # (B, vocab, T)
        pred_str = decode(logits[0])
        refs.append(transcript[0].lower())
        hyps.append(pred_str)

    wer, cer = compute_metrics(hyps, refs)
    print(f"Test WER: {wer:.3f} | CER: {cer:.3f}")


if __name__ == "__main__":
    test_wave2vec()
