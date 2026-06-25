"""Centralized logging setup.

Import `get_logger(__name__)` anywhere in the project instead of
calling `logging.getLogger` directly, so every module logs in the
same format and to the same place. __name__ automatically gives the
module path (e.g. src.services.training_service) so you always know 
which file logged which message.

"""

import logging
import os
import sys
from datetime import datetime

_LOG_DIR = "artifacts/reports"
os.makedirs(_LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(_LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d')}.log")

_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger that writes to both console and a daily log file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoids duplicate handlers on re-import)
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMATTER)

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(_FORMATTER)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger