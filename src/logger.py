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
        debug_file: bool = False
    ):
        self.logs_path = Path("src").resolve().parents[1] / "logs"
        self.logs_path.mkdir(exist_ok=True)
        self.config_file = config_file
        self.debug_file = debug_file
        self.log_files: Dict[str, Path] = {}
    
    def _sequential_debug_file(self):
        """
        Generate a sequential debug file name to avoid overwriting.
        """
        index = 1
        while True:
            debug_file_path = self.logs_path / f"debug_{index}.log"
            if not debug_file_path.exists():
                return debug_file_path
            index += 1

    def clear_logs(self):
        """
        Clears all logs from logs directory.
        """
        for log_file in self.logs_path.glob("*.log"):
            log_file.unlink(missing_ok=True)
        return None

    def setup_logger(self, logger_name: str = "app"):
        config = load_config(self.config_file)
        logging.config.dictConfig(config)

        logger = logging.getLogger(logger_name)
        log_file_path = self.logs_path / f"{logger_name}.log"
        self.log_files[logger_name] = log_file_path

        # Ensure debug file is created with a rotating handler
        if self.debug_file:
            debug_file_path = self._sequential_debug_file()
            rotating_handler = RotatingFileHandler(
                debug_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            rotating_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            rotating_handler.setFormatter(formatter)
            logger.addHandler(rotating_handler)

        logger.info(f"{logger_name.capitalize()} logger initialized.")
        return None
