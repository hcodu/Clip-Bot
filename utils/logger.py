import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "clip_bot",
    log_file: str | Path | None = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger. Safe to call multiple times —
    returns the existing logger if already configured.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "clip_bot") -> logging.Logger:
    """Return a child logger under the clip_bot namespace."""
    return logging.getLogger(f"clip_bot.{name}")
