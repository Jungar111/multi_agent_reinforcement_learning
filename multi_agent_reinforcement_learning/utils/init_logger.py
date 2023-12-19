"""Module for logging."""
import logging

import colorlog


def init_logger() -> logging.Logger:
    """Set up logger."""
    log_format = "%(asctime)s - " "%(name)s - " "%(funcName)s - " "%(message)s"

    bold_seq = "\033[1m"
    colorlog_format = f"{bold_seq} " "%(log_color)s " f"{log_format}"
    colorlog.basicConfig(format=colorlog_format)

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    return logger
