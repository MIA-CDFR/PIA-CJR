# ============================================================
# FILE: src/utils/logger.py
# ============================================================

import logging


def setup_logger(name: str):

    logger = logging.getLogger(name)

    # Avoid duplicated handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger