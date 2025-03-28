import sys
import logging
import logging.config
from pathlib import Path


logs_path = Path("src").resolve().parents[1] / "logs"
app_log_file = logs_path / "app.log"
debug_log_file = logs_path / "debug.log"
logs_path.mkdir(exist_ok=True, parents=True)



def clear_logger(app: bool = True, debug: bool = True):
    if app:
        open(app_log_file, 'w').close()
    if debug:
        open(debug_log_file, 'w').close()


def setup_logger():
    clear_logger()

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
                "class": "logging.FileHandler",
                "filename": str(debug_log_file),
                "level": "DEBUG",
                "formatter": "detailed",
                "encoding": "utf-8",
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