import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from core.utils.static_variable import LOG_PATH_FILE

class Logger:
    def __init__(self, module=None):
        self.module = module
        self.log_file_path = LOG_PATH_FILE
        self.log_level = self._get_log_level()
        self._setup_logging()
        self._create_log_files()  # Create log files if they don't exist

    def _get_log_level(self):
        log_levels = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET
        }
        log_level = os.environ.get("DEEPFACE_LOG_LEVEL", "INFO").upper()
        if log_level in log_levels:
            return log_levels[log_level]
        else:
            self._dump_log(
                f"Invalid $DEEPFACE_LOG_LEVEL value. "
                "Setting app log level to INFO."
            )
            return logging.INFO

    def _setup_logging(self):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Ensure log directory exists
        if not os.path.exists(self.log_file_path):
            print(f"Log directory created at: {self.log_file_path}")
            os.makedirs(self.log_file_path, exist_ok=True)

        # Create a file handler for each log level
        self.info_handler = RotatingFileHandler(
            os.path.join(self.log_file_path, "info.log"), maxBytes=10_000_000, backupCount=5
        )
        self.info_handler.setLevel(logging.INFO)
        self.info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        self.info_handler.setFormatter(logging.Formatter(log_format))

        self.debug_handler = RotatingFileHandler(
            os.path.join(self.log_file_path, "debug.log"), maxBytes=10_000_000, backupCount=5
        )
        self.debug_handler.setLevel(logging.DEBUG)
        self.debug_handler.setFormatter(logging.Formatter(log_format))

        self.warning_handler = RotatingFileHandler(
            os.path.join(self.log_file_path, "warning.log"), maxBytes=10_000_000, backupCount=5
        )
        self.warning_handler.setLevel(logging.WARNING)
        self.warning_handler.setFormatter(logging.Formatter(log_format))

        self.error_handler = RotatingFileHandler(
            os.path.join(self.log_file_path, "error.log"), maxBytes=10_000_000, backupCount=5
        )
        self.error_handler.setLevel(logging.ERROR)
        self.error_handler.setFormatter(logging.Formatter(log_format))

        # Set up the root logger
        logging.basicConfig(level=self.log_level, format=log_format)
        logger = logging.getLogger()
        logger.addHandler(self.info_handler)
        logger.addHandler(self.debug_handler)
        logger.addHandler(self.warning_handler)
        logger.addHandler(self.error_handler)


    def _create_log_files(self):
        """Create empty log files if they don't exist"""
        for level in ["info.log", "debug.log", "warning.log", "error.log"]:
            file_path = os.path.join(self.log_file_path, level)
            if not os.path.exists(file_path):
                try:
                    open(file_path, 'w').close()  # Create an empty file
                    self.info(f"Created log file: {file_path}")
                except Exception as e:
                    self.error(f"Failed to create log file {file_path}: {e}")

    def info(self, message):
        logging.info(message)

    def debug(self, message):
        logging.debug(message)

    def warn(self, message):
        logging.warning(message)

    def error(self, message):
        logging.error(message)

    def critical(self, message):
        logging.critical(message)

    def _dump_log(self, message):
        # Write log message to the file
        log_file_path = os.path.join(self.log_file_path, "error.log")  # Default to error log
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{str(datetime.now())[2:-7]} - {message}\n")
        except Exception as e:
            print(f"Failed to dump log message: {e}")

def get_logger():
    return Logger()
