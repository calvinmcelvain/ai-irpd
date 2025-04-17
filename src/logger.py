"""
Module for logger.

Contains the LoggerManager model.
"""
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict
from pathlib import Path

from helpers.utils import load_config


LOGGER_CONFIGS = load_config("logger.json")



class LoggerManager:
    """
    LoggerManager model.
    
    Contains methods to clear logger based on logger settings.
    
    Note: Log files are stored in './logs/'.
    """
    def __init__(self, debug: bool = True):
        self.logs_path = Path("src").resolve().parents[1] / "logs"
        self.logs_path.mkdir(exist_ok=True)
        self.log_files: Dict[str, Path] = {}
        
        # Loading logger configs.
        logger_name = LOGGER_CONFIGS["name"]
        logging.config.dictConfig(LOGGER_CONFIGS["config"])
        log_file_name = LOGGER_CONFIGS["log_file_names"]["main"]
        log_file_path = self.logs_path / log_file_name
        self.log_files[logger_name] = log_file_path
        
        # Log.
        self.log = logging.getLogger(logger_name)
        
        # Setting up debug file if specified.
        if debug:
            debug_file_path = self.logs_path / "debug.log"
            rotating_handler = RotatingFileHandler(
                debug_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
            )
            rotating_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            rotating_handler.setFormatter(formatter)
            self.log.addHandler(rotating_handler)

    def clear_logs(self):
        """
        Clears all logs by truncating the log files.
        """
        for log_file in self.logs_path.glob("*.log"):
            try:
                with log_file.open("w"):
                    pass
                self.log.info(f"Truncated log file: {log_file}")
            except Exception as e:
                self.log.error(f"Error truncating log file '{log_file}': {e}")
                raise
        return None
