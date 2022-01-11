import argparse
import os

import yaml

from wav2vec_speech_to_text.config import config
from wav2vec_speech_to_text.train import predict, test, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test", "predict"])
    parser.add_argument(
        "--config",
        type=str,
        default="src/wav2vec_speech_to_text/config/hparams.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("--audio_path", type=str, help="Required for predict mode")

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            custom_hparams = yaml.safe_load(f)
        config.hparams.update(custom_hparams)
    else:
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.mode == "train":
        trainer.train_wave2vec()
    elif args.mode == "test":
        test.test_wave2vec()
    elif args.mode == "predict":
        if not args.audio_path:
            raise ValueError("Must pass --audio_path for predict mode")
        predict(args.audio_path)


if __name__ == "__main__":
    main()
