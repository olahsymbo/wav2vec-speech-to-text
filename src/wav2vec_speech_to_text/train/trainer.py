import torch
from wav2vec_speech_to_text.data.loader import AudioDataset
from wav2vec_speech_to_text.data.splitter import DataSplitter
from wav2vec_speech_to_text.utils.metrics import compute_metrics
from wav2vec_speech_to_text.utils.model_state import ModelState
from wav2vec_speech_to_text.utils.logger import logger
from wav2vec_speech_to_text.config.config import hparams
from wav2vec_speech_to_text.data.contants import labels, label_to_index
from wav2vec_speech_to_text.model.wav2vec import Wav2Vec, Wav2VecCTC, decode

BATCH_SIZE = hparams["training"]["batch_size"]
EPOCHS = hparams["training"]["epochs"]

def train_wave2vec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("Loading training data...")
    dataset = AudioDataset(subset="train-clean-100")
    dataloader = DataSplitter(dataset, batch_size=BATCH_SIZE)
    
    val_set = AudioDataset(subset="test-clean")
    val_loader = DataSplitter(val_set, batch_size=BATCH_SIZE, shuffle=False)

    vocab_size = len(labels)

    logger.info("Initializing model: Wav2Vec")
    base = Wav2Vec()
    model = Wav2VecCTC(base, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    best_wer = float("inf")

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info("Training...")
        model.train()
        total_loss = 0
        
        for x, y in dataloader:
            print(x.shape)
            x = x.to(device)
            targets = [torch.tensor([label_to_index[c] for c in t if c in label_to_index]) for t in y]
            if any([len(t) == 0 for t in targets]):
                continue 

            target_lengths = torch.tensor([len(t) for t in targets])
            targets = torch.cat(targets)

            logits = model(x) 
            input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(2), dtype=torch.long)
            logits = logits.permute(2, 0, 1) 

            loss = ctc_loss(logits, targets.to(device), input_lengths, target_lengths.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss:.4f}")

        model.eval()

        refs, hyps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).unsqueeze(1)
                logits = model(x)
                hyp = decode(logits[0].detach().cpu())
                refs.append(y[0])
                hyps.append(hyp)

        wer, cer = compute_metrics(hyps, refs)
        print(f"Epoch {epoch} — WER: {wer:.3f}, CER: {cer:.3f}")

        if wer < best_wer:
            best_wer = wer
            model_state = ModelState(save_dir=hparams["training"]["save_dir"])
            model_state.save("best_wave2vec_model.pth")
            print("Best model saved.")


if __name__ == "__main__":
    train_wave2vec()
    logger.info("Training complete.")
