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

Raw Audio → FeatureEncoder (CNN) → ContextNetwork (Transformer) → CTC Head → Transcription


### Components:

- **FeatureEncoder**  
  Extracts latent representations from raw waveforms using 1D convolutions. Downsamples and projects input into compact feature maps.

- **ContextNetwork**  
  A stack of Transformer encoder layers that capture long-range dependencies and contextual information in the latent feature sequence.

- **CTC Head**  
  A 1x1 convolution followed by log-softmax to map transformer outputs to a vocabulary distribution at each timestep, enabling CTC loss.

The components perform: 

- End-to-end training with CTC Loss
- Greedy decoding
- Contrastive loss pretraining
- Classification head for auxiliary tasks

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

