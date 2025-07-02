# wav2vec-speech-to-text

End-to-end speech-to-text system using a custom Wav2Vec-style model built in PyTorch, with CTC decoding for transcription from raw waveform input.

---

## Overview

This project implements a simplified Wav2Vec architecture for **automatic speech recognition (ASR)**. It processes raw audio waveforms and produces text transcriptions using:

- A convolutional **feature encoder**
- A **Transformer-based context network**
- A **CTC projection head** with log-softmax output

---

## Model Architecture

### Feature Encoder (`FeatureEncoder`)

Extracts latent representations from raw waveforms using 1D convolutions:


## Usage

### Training
```
python app/cli.py --mode train
```

Trains the model using raw audio files and paired transcripts. Uses CTC loss.

### Inference
```
python app/cli.py --mode predict --audio path/to/audio.wav
```

Prints decoded transcription using greedy CTC decoding.

### Optional: Contrastive Pretraining
You can pretrain the model using unlabeled data and a contrastive loss:

```
contrastive_loss(context, features, mask_indices)
```

Encourages context features to match the correct latent feature


## Features

✅ Wav2Vec-style encoder + transformer context

✅ CTC decoding for ASR

✅ Greedy decoder (CTC collapse)

✅ Modular PyTorch design

🧪 Optional contrastive pretraining support

