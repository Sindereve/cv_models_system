import logging
import sys
import json
from .config import LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT


def get_logger(
        name: str
    ) -> logging.Logger:
    """
    Create logger
    
    :param name: Name logger
    :type name: str
    
    :return: Logger
    :rtype: Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)

    if LOG_FORMAT == "JSON":
        formatter = JSONLogFormatter()
    else:
        formatter = TextLogFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


# -------- FORMAT --------

class TextLogFormatter(logging.Formatter):
    LEVEL_STYLES = {
        logging.DEBUG:   "🔎 [ DEBUG ]",
        logging.INFO:    "💙 [ INFO  ]",
        logging.WARNING: "💛 [WARNING]",
        logging.ERROR:   "💔 [ ERROR ]",
        logging.CRITICAL:"🔥 [CRITIC ]",
    }

    def format(self, record):
        level = self.LEVEL_STYLES.get(record.levelno, record.levelname)
        fmt = f"%(asctime)s | {level} | %(name)s | %(message)s"
        self._style._fmt = fmt
        return super().format(record)


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record, LOG_DATEFMT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log)
