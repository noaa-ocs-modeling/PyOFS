import datetime
import logging
import os
import sys

import numpy

CRS_EPSG = 4326

if 'OFS_DATA' in os.environ:
    DATA_DIRECTORY = os.environ['OFS_DATA']
else:
    DATA_DIRECTORY = r"C:\data\OFS"

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999.0

# # for development branch
# DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'develop')

# using relative values with the PREDICTOR option will break rendering in Leaflet.CanvasLayer.Field
TIFF_CREATION_OPTIONS = {
    'TILED': 'YES',
    'COMPRESS': 'DEFLATE',
    'NUM_THREADS': 'ALL_CPUS',
    'BIGTIFF': 'IF_SAFER'
}


class NoDataError(Exception):
    """ Error for no data found. """
    pass


DEFAULT_LOG_FORMAT = '[%(asctime)s] %(name)-11s %(levelname)-8s: %(message)s'


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


def create_logger(name: str, log_filename: str = None, file_level: int = logging.DEBUG, console_level: int = logging.INFO, log_format: str = None) -> logging.Logger:
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # remove handlers
    for handler in [handler for handler in logger.handlers]:
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


def round_to_hour(datetime_object: datetime.datetime, direction: str = None) -> datetime.datetime:
    """
    Return given datetime rounded to the nearest hour.

    :param datetime_object: datetime to round
    :param direction: either 'ceiling' or 'floor', optional
    :return: rounded datetime
    """

    start_of_hour = datetime_object.replace(minute=0, second=0, microsecond=0)
    half_hour = datetime_object.replace(minute=30, second=0, microsecond=0)

    if direction == 'ceiling' or datetime_object >= half_hour:
        datetime_object = start_of_hour + datetime.timedelta(hours=1)
    elif direction == 'floor' or datetime_object < half_hour:
        datetime_object = start_of_hour

    return datetime_object


def round_to_ten_minutes(datetime_object: datetime.datetime) -> datetime.datetime:
    """
    Return given datetime rounded to the nearest ten minutes.

    :param datetime_object: datetime to round
    :return: rounded datetime
    """

    return datetime_object.replace(minute=int(round(datetime_object.minute, -1)), second=0, microsecond=0)


def range_daily(start_time: datetime.datetime, end_time: datetime.datetime) -> list:
    """
    Generate range of times between given times at day intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    days = duration.days
    stride = 1 if days > 0 else -1

    return [start_time + datetime.timedelta(days=day) for day in range(0, days, stride)]


def range_hourly(start_time: datetime.datetime, end_time: datetime.datetime) -> list:
    """
    Generate range of times between given times at hour intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    hours = int(duration.total_seconds() / 3600)
    stride = 1 if duration.days > 0 else -1

    return [start_time + datetime.timedelta(hours=hour) for hour in range(0, hours, stride)]


def ten_minute_range(start_time: datetime.datetime, end_time: datetime.datetime) -> list:
    """
    Generate range of times between given times at ten minute intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    minutes = int(duration.total_seconds() / 60)
    stride = 10

    return [start_time + datetime.timedelta(minutes=minute) for minute in range(0, minutes + 1, stride)]


def overview_levels(shape: (int, int)) -> [int]:
    levels = []
    shape = numpy.array(shape)
    factor = 2
    while numpy.all(shape / factor > 1):
        levels.append(factor)
        factor *= 2
    return levels
