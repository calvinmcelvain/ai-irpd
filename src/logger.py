import sys
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

logs_path = Path("src").resolve().parents[1] / "logs"
logs_path.mkdir(exist_ok=True, parents=True)


def setup_logger():
    today = datetime.now()
    app_log_file = logs_path / "app-{}.log".format(today.strftime("%Y_%m_%d"))
    debug_log_file = logs_path / "debug.log"

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "detailed",
                "stream": sys.stdout
            },
            "app_file": {
                "class": "logging.FileHandler",
                "filename": str(app_log_file),
                "level": "INFO",
                "formatter": "detailed",
                "encoding": "utf-8",
            },
            "debug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(debug_log_file),
                "level": "DEBUG",
                "formatter": "detailed",
                "encoding": "utf-8",
                "maxBytes": 5 * 1024 * 1024,  # 5 MB
                "backupCount": 1,
            }
        },
        "loggers": {
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "app_file", "debug_file"],
                "propagate": False
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "app_file", "debug_file"]
        }
    })

    logger = logging.getLogger("app")
    logger.info("Application logger initialized.")