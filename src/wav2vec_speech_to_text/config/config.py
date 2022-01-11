import yaml

with open("src/wav2vec_speech_to_text/config/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
