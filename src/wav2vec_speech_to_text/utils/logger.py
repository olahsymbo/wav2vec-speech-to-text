import logging

logger = logging.getLogger("wav2vec_speech_to_text")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
