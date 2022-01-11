import os

import torch

from wav2vec_speech_to_text.utils.logger import logger


class ModelState:
    def __init__(self, model, save_dir="checkpoints") -> None:
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, name="model.pth", optimizer=None, epoch=None, extra=None) -> None:
        save_path = os.path.join(self.save_dir, name)
        state = {"model_state_dict": self.model.state_dict()}
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        if extra:
            state.update(extra)
        torch.save(state, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path, optimizer=None, map_location=None) -> int:
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch")
        logger.info(f"Model loaded from {path} (epoch {epoch})")
        return epoch
