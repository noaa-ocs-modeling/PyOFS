from datetime import datetime, timedelta
import logging
import os
from os import PathLike
from pathlib import Path
import sys
from typing import Union

import numpy

CRS_EPSG = 4326

DATA_DIRECTORY = Path(os.getenv('OFS_DATA', r'C:\data\OFS'))
AZURE_CREDENTIALS_FILENAME = Path(os.getenv('AZURE_CRED', r'C:\data\azure_credentials.txt'))

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999.0

# using relative values with the PREDICTOR option will break rendering in Leaflet.CanvasLayer.Field
TIFF_CREATION_OPTIONS = {
    'TILED': 'YES',
    'COMPRESS': 'DEFLATE',
    'NUM_THREADS': 'ALL_CPUS',
    'BIGTIFF': 'IF_SAFER',
}


class NoDataError(Exception):
    """ Error for no data found. """

    pass


def get_logger(
    name: str,
    log_filename: PathLike = None,
    file_level: int = None,
    console_level: int = None,
    log_format: str = None,
) -> logging.Logger:
    if file_level is None:
        file_level = logging.DEBUG
    if console_level is None:
        console_level = logging.INFO
    logger = logging.getLogger(name)

    # check if logger is already configured
    if logger.level == logging.NOTSET and len(logger.handlers) == 0:
        # check if logger has a parent
        if '.' in name:
            logger.parent = get_logger(name.rsplit('.', 1)[0])
        else:
            # otherwise create a new split-console logger
            logger.setLevel(logging.DEBUG)
            if console_level != logging.NOTSET:
                if console_level <= logging.INFO:
                    class LoggingOutputFilter(logging.Filter):
                        def filter(self, rec):
                            return rec.levelno in (logging.DEBUG, logging.INFO)

                    console_output = logging.StreamHandler(sys.stdout)
                    console_output.setLevel(console_level)
                    console_output.addFilter(LoggingOutputFilter())
                    logger.addHandler(console_output)

                console_errors = logging.StreamHandler(sys.stderr)
                console_errors.setLevel(max((console_level, logging.WARNING)))
                logger.addHandler(console_errors)

    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(file_level)
        for existing_file_handler in [
            handler for handler in logger.handlers if type(handler) is logging.FileHandler
        ]:
            logger.removeHandler(existing_file_handler)
        logger.addHandler(file_handler)

    if log_format is None:
        log_format = '[%(asctime)s] %(name)-15s %(levelname)-8s: %(message)s'
    log_formatter = logging.Formatter(log_format)
    for handler in logger.handlers:
        handler.setFormatter(log_formatter)

    return logger


def split_layer_filename(filename: PathLike) -> (Path, Union[str, int]):
    if not isinstance(filename, Path):
        filename = Path(filename)
    name = filename.name
    if ':' in name:
        name, layer = str(filename).rsplit(':', 1)
        filename = filename.parent / name
        try:
            layer = int(layer)
        except ValueError:
            pass
    else:
        layer = None

    return filename, layer


def repository_root(path: PathLike = None) -> Path:
    if path is None:
        path = __file__
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_file():
        path = path.parent
    if '.git' in (child.name for child in path.iterdir()) or path == path.parent:
        return path
    else:
        return repository_root(path.parent)


def round_to_hour(datetime_object: datetime, direction: str = None) -> datetime:
    """
    Return given datetime rounded to the nearest hour.

    :param datetime_object: datetime to round
    :param direction: either 'ceiling' or 'floor', optional
    :return: rounded datetime
    """

    start_of_hour = datetime_object.replace(minute=0, second=0, microsecond=0)
    half_hour = datetime_object.replace(minute=30, second=0, microsecond=0)

    if direction == 'ceiling' or datetime_object >= half_hour:
        datetime_object = start_of_hour + timedelta(hours=1)
    elif direction == 'floor' or datetime_object < half_hour:
        datetime_object = start_of_hour

    return datetime_object


def round_to_ten_minutes(datetime_object: datetime) -> datetime:
    """
    Return given datetime rounded to the nearest ten minutes.

    :param datetime_object: datetime to round
    :return: rounded datetime
    """

    return datetime_object.replace(
        minute=int(round(datetime_object.minute, -1)), second=0, microsecond=0
    )


def range_daily(start_time: datetime, end_time: datetime) -> list:
    """
    Generate range of times between given times at day intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    days = duration.days
    stride = 1 if days > 0 else -1

    return [start_time + timedelta(days=day) for day in range(0, days, stride)]


def range_hourly(start_time: datetime, end_time: datetime) -> list:
    """
    Generate range of times between given times at hour intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    hours = int(duration / timedelta(hours=1))
    stride = 1 if duration.days > 0 else -1

    return [start_time + timedelta(hours=hour) for hour in range(0, hours, stride)]


def ten_minute_range(start_time: datetime, end_time: datetime) -> list:
    """
    Generate range of times between given times at ten minute intervals.

    :param start_time: beginning of time interval
    :param end_time: end of time interval
    :return: range of datetimes
    """

    duration = end_time - start_time
    minutes = int(duration / timedelta(minutes=1))
    stride = 10

    return [start_time + timedelta(minutes=minute) for minute in range(0, minutes + 1, stride)]


def overview_levels(shape: (int, int)) -> [int]:
    levels = []
    shape = numpy.array(shape)
    factor = 2
    while numpy.all(shape / factor > 1):
        levels.append(factor)
        factor *= 2
    return levels
