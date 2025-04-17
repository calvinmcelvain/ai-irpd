"""
Module for logger.

Contains the LoggerManager model.
"""
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict
from pathlib import Path

from helpers.utils import load_config



class LoggerManager:
    """
    LoggerManager model.
    
    Contains methods to clear and setup logger based on logger settings.
    
    Note: Log files are stored in './logs/'.
    """
    def __init__(
        self,
        config_file: str = "logger.json",
        debug_file: bool = True
    ):
        self.logs_path = Path("src").resolve().parents[1] / "logs"
        self.logs_path.mkdir(exist_ok=True)
        self.config_file = config_file
        self.debug_file = debug_file
        self.log_files: Dict[str, Path] = {}

    def clear_logs(self):
        """
        Clears all logs by truncating the log files.
        """
        for log_file in self.logs_path.glob("*.log"):
            try:
                with log_file.open("w"):
                    pass
                logging.info(f"Truncated log file: {log_file}")
            except Exception as e:
                logging.error(f"Error truncating log file '{log_file}': {e}")
                raise
        return None

    def setup_logger(self, logger_name: str = "app"):
        config = load_config(self.config_file)
        logging.config.dictConfig(config)

        log = logging.getLogger(logger_name)
        log_file_path = self.logs_path / f"{logger_name}.log"
        self.log_files[logger_name] = log_file_path

        # Ensure debug file is created with a rotating handler
        if self.debug_file:
            debug_file_path = self.logs_path / "debug.log"
            rotating_handler = RotatingFileHandler(
                debug_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            rotating_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            rotating_handler.setFormatter(formatter)
            log.addHandler(rotating_handler)

        log.info(f"{logger_name.capitalize()} logger initialized.")
        return None
