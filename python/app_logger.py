"""
A simple app_logger
"""

import json
import logging
from logging.config import dictConfig
import os

LOGGING_LEVEL = "INFO"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# LOGGER
LOGGING_DEFINITIONS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"verbose": {"format": "{levelname}|{asctime}|{module}|{funcName}|{message}", "style": "{"}},
    "handlers": {
        "log_to_file": {
            "level": "INFO",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "d",
            "interval": 30,
            "backupCount": 30,
            "filename": os.path.join(LOG_DIR, "intelidb_updater.log"),
            "formatter": "verbose",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "IntelidbUpdaterLogger": {"handlers": ["log_to_file", "console"], "level": LOGGING_LEVEL, "propagate": True}
    },
}


def reset_logging() -> None:
    """Reset Logging"""
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    loggers = [
        logger for logger_name, logger in manager.loggerDict.items() if logger_name.startswith("IntelidbUpdater")
    ]
    for logger in loggers:
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                # Copied from `logging.shutdown`.
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)


reset_logging()
log = logging.getLogger("IntelidbUpdaterLogger")
dictConfig(LOGGING_DEFINITIONS)
