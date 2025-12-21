import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "TEXT")
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"