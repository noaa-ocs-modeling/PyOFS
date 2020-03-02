import logging
import sys

DEFAULT_LOG_FORMAT = '[%(asctime)s] %(name)-4s %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S', format=DEFAULT_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # check if logger is already configured
    if logger.level == logging.NOTSET and len(logger.handlers) == 0:
        # check if logger has a parent
        if '.' in name:
            logger.parent = get_logger(name.rsplit('.', 1)[0])
        else:
            logger = create_logger(name)

    return logger


def create_logger(name: str, log_filename: str = None, file_level: int = logging.DEBUG, console_level: int = logging.INFO,
                  log_format: str = None) -> logging.Logger:
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # remove handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    log_formatter = logging.Formatter(log_format)

    if console_level != logging.NOTSET:
        if console_level <= logging.INFO:
            console_output = logging.StreamHandler(sys.stdout)
            console_output.setFormatter(log_formatter)
            console_output.setLevel(console_level)
            console_output.addFilter(LoggingOutputFilter())
            logger.addHandler(console_output)

        console_errors = logging.StreamHandler(sys.stderr)
        console_errors.setFormatter(log_formatter)
        console_errors.setLevel(max((console_level, logging.WARNING)))
        logger.addHandler(console_errors)

    if log_filename is not None:
        log_file = logging.FileHandler(log_filename)
        log_file.setFormatter(log_formatter)
        log_file.setLevel(file_level)
        logger.addHandler(log_file)

    return logger


class LoggingOutputFilter(logging.Filter):
    """ class to filter output from a logger to only INFO or DEBUG """

    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)
