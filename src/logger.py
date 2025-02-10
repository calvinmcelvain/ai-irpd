import sys
import logging
import logging.config
from pathlib import Path
from logging.handlers import RotatingFileHandler

logs_path = Path(__file__).resolve().parents[1] / "logs"
logs_path.mkdir(exist_ok=True, parents=True)


def setup_logger():
    app_log_file = logs_path / "app.log"

    open(app_log_file, 'w').close()

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "stream": sys.stdout
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(app_log_file),
                "level": "INFO",
                "formatter": "detailed",
                "encoding": "utf-8",
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 5
            }
        },
        "loggers": {
            "app": {
                "level": "INFO",
                "handlers": ["app_file"],
                "propagate": False
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "app_file"]
        }
    })

    logger = logging.getLogger("app")
    logger.info("Application logger initialized.")