"""Logging, seeding and small helpers."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    return logging.getLogger("moaa")


def check_gpu(logger: logging.Logger):
    logger.info("*" * 60)
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU available: %s (%.2f GB)", torch.cuda.get_device_name(0), mem_gb)
    else:
        logger.warning("No GPU available. Using CPU instead.")
    logger.info("*" * 60)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_default(name: str, fallback=None):
    """Read a default value from an environment variable (set by configs/paths.sh)."""
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def env_list(name: str):
    """Colon-separated list from an environment variable (like $PATH)."""
    value = os.environ.get(name)
    if not value:
        return None
    return [v for v in value.split(":") if v]


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def log_clf_metrics(logger, metrics: dict, prefix: str = "test"):
    logger.info("WER: %.4f", metrics.get(f"{prefix}_wer", float("nan")))
    logger.info("CER: %.4f", metrics.get(f"{prefix}_cer", float("nan")))
    logger.info("Accented-or-not   | acc: %.4f | loss: %.4f",
                metrics.get(f"{prefix}_acc_accented_or_not", float("nan")),
                metrics.get(f"{prefix}_loss_accented_or_not", float("nan")))
    logger.info("Gender (ALL)      | acc: %.4f | loss: %.4f",
                metrics.get(f"{prefix}_acc_gender", float("nan")),
                metrics.get(f"{prefix}_loss_gender", float("nan")))
    logger.info("Accent (KNOWN)    | acc: %.4f | loss: %.4f | n_known_accent=%s",
                metrics.get(f"{prefix}_acc_accent", float("nan")),
                metrics.get(f"{prefix}_loss_accent", float("nan")),
                metrics.get(f"{prefix}_n_known_accent", "NA"))
