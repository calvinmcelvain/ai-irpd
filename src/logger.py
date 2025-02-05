import sys
import logging
import logging.config
from pathlib import Path
from utils import regex_group
sys.path.append(Path().absolute().parent)

logs_path = Path(__file__).resolve().parents[1] / "logs"


def next_log():
    logs_path.mkdir(exist_ok=True)
    logs = [child.name for child in logs_path.glob("*.log")]
    log_numbers = [int(regex_group(log, r"log_(\d+).log")) for log in logs]
    return logs_path / f"log_{max(log_numbers, default=1000) + 1}.log"


def setup_logger(clear_logs: bool = False):
    if clear_logs and logs_path.exists():
        for log in logs_path.iterdir():
            log.unlink()
    
    log_filename = next_log()
    
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
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_filename),
                "level": "INFO",
                "formatter": "detailed",
                "encoding": "utf-8",
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        }
    })

    logger = logging.getLogger()
    logger.propagate = True
    logger.handlers[0].flush = sys.stdout.flush
    
    if clear_logs:
        logger.debug("Previous logs cleared.")
    logger.info(f"Log '{log_filename.name}' initialized.")
    