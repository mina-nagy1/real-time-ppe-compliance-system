"""
logger.py
---------
Structured JSON logging for the PPE compliance system.
All modules import from here — never use print() in production code.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }

        # Attach exception info when present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Attach any extra fields passed via extra={...}
        reserved = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved:
                payload[key] = value

        return json.dumps(payload, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Returns a named logger that writes:
      • JSON to  outputs/logs/<name>.log
      • Human-readable to stdout

    Usage
    -----
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("detector ready", extra={"model": "yolo11n-seg.pt"})
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured — return as-is
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── JSON file handler ────────────────────────────────────────────────
    log_file = LOG_DIR / f"{name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # ── Human-readable stdout handler ───────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def log_exception(logger: logging.Logger, msg: str = "Unhandled exception") -> None:
    """Call inside an except block to log full traceback as structured JSON."""
    logger.error(msg, extra={"traceback": traceback.format_exc()})
