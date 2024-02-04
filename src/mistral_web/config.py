import logging
import os

logger = logging.getLogger("mistral-web")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    fmt="[%(asctime)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(ch)


def get_model_checkpoint() -> str:
    return os.getenv("model_dir", "X:/Mistral-7B-Instruct-v0.2")
