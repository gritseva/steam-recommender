# Logging Utils module
import logging
from config.config import LOGGING_LEVEL


def setup_logging():
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
