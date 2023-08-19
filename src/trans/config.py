"""Project Configuration."""
import os
import sys
from logging.config import dictConfig

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))


class Config:
    """Provide default Configuration."""

    # logging
    LOG_LEVEL = "INFO"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    @classmethod
    def configure_logger(cls, root_module_name: str) -> None:
        """Configure logging."""
        dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "stdout_formatter": {
                    "format": cls.LOG_LINE_FORMAT,
                    "datefmt": cls.LOG_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "stdout_handler": {
                    "level": cls.LOG_LEVEL,
                    "formatter": "stdout_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["stdout_handler"],
                    "level": cls.LOG_LEVEL,
                    "propagate": True,
                },
            },
        })

    # folders
    MODELDIR = os.path.join(rootdir, "model_config")

    VOCAB_SIZE = 32000  # vocabulary size
    SEQ_LEN = 128  # maximum length of a sequence
    N_LAYER = 6  # number of layers
    N_HEAD = 8  # number of heads
    D_MODEL = 512  # embedding size
    D_FF = D_MODEL * 4  # dimension of feed forward
    D_MHA = D_MODEL // N_HEAD  # dimension of multi-head attention
    DROP_RATE = 0.1  # dropout rate


class TestConfig(Config):
    """Provide Testing Configuration."""
    LOG_LEVEL = "DEBUG"
