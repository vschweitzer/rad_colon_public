import logging
from typing import Optional
import threading
import radiomics
import datetime
import sys


global_logger: Optional[logging.Logger] = None


def setup_logging() -> None:
    global global_logger
    if global_logger is not None:
        return
    global_logger = logging.getLogger("rad_colon")

    timestamp_string: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    simple_formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s - %(threadName)s: %(message)s", datefmt="%H:%M:%S"
    )

    detailed_formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s - %(threadName)s - %(levelname)s: %(message)s"
    )

    file_handler: logging.FileHandler = logging.FileHandler(
        filename=f"rad_colon_{timestamp_string}.log", mode="w"
    )
    radiomics_file_handler: logging.FileHandler = logging.FileHandler(
        filename=f"rad_colon_radiomics_{timestamp_string}.log", mode="w"
    )
    stdout_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

    global_logger.setLevel(logging.DEBUG)

    # Don't spam console, write details to log file
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(simple_formatter)

    global_logger.handlers = []
    global_logger.addHandler(file_handler)
    global_logger.addHandler(stdout_handler)

    radiomics_logger: logging.Logger = radiomics.logger
    radiomics.setVerbosity(logging.DEBUG)
    radiomics_logger.setLevel(logging.NOTSET)
    radiomics_logger.addHandler(radiomics_file_handler)


def log_wrapper(message: str, loglevel: int = logging.INFO) -> None:
    global global_logger
    assert global_logger is not None
    global_logger.log(loglevel, message)


if __name__ == "__main__":
    setup_logging()
    log_wrapper("NOTSET", logging.NOTSET)
    log_wrapper("DEBUG", logging.DEBUG)
    log_wrapper("INFO", logging.INFO)
    log_wrapper("WARNING", logging.WARNING)
    log_wrapper("ERROR", logging.ERROR)
    log_wrapper("CRITICAL", logging.CRITICAL)
