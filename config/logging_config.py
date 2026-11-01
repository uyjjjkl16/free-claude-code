"""Loguru-based structured logging configuration.

All logs are written to server.log and the console as JSON lines
for full traceability.
Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize()
are included at top level for easy grep/filter.
"""

import json
import logging
import sys
from pathlib import Path

from loguru import logger

_configured = False

# Context keys we promote to top-level JSON for traceability
_CONTEXT_KEYS = ("request_id", "node_id", "chat_id")


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level."""
    extra = record.get("extra", {})

    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]

    record["_json"] = json.dumps(out, default=str)
    return "{_json}\n"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2

        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


def configure_logging(
    log_file: str,
    *,
    level: str = "INFO",
    force: bool = False,
) -> None:
    """Configure loguru with JSON output to file and console."""

    global _configured

    if _configured and not force:
        return

    _configured = True

    # Remove default loguru handler
    logger.remove()

    # Create or truncate log file
    Path(log_file).write_text("", encoding="utf-8")

    # File sink: save logs to server.log
    logger.add(
        log_file,
        level=level,
        enqueue=True,
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
    )

    # Console sink: make logs visible in Render
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        format=_serialize_with_context,
    )

    # Intercept stdlib logging
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(level)
