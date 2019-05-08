# coding=utf-8
"""
Data utility functions.

Created on Jun 13, 2018

@author: zachary.burnett
"""

import datetime
import os

import fiona
import numpy
import rasterio
import xarray


def copy_xarray(input_path: str, output_path: str) -> xarray.Dataset:
    """
    Copy given xarray observation to a local file at the given path.

    :param input_path: path to observation to copy
    :param output_path: path to output file
    :return: copied observation at given path
    """

    print(f'Reading observation from {input_path}')

    input_dataset = xarray.open_dataset(input_path, decode_times=False)

    print(f'Copying observation to local memory...')

    # deep copy of xarray observation
    output_dataset = input_dataset.copy(deep=True)

    print(f'Writing to {output_path}')

    # save observation to file
    output_dataset.to_netcdf(output_path)

    return output_dataset


def round_to_day(datetime_object: datetime.datetime, direction: str = None) -> datetime.datetime:
    """
    Return given datetime rounded to the nearest day.

    :param datetime_object: datetime to round
    :param direction: either 'ceiling' or 'floor', optional
    :return: rounded datetime
    """

    start_of_day = datetime_object.replace(hour=0, minute=0, second=0, microsecond=0)
    half_day = datetime_object.replace(hour=12, minute=0, second=0, microsecond=0)

    if direction == 'ceiling' or (direction is None and datetime_object >= half_day):
        datetime_object = start_of_day + datetime.timedelta(days=1)
    elif direction == 'floor' or (direction is None and datetime_object < half_day):
        datetime_object = start_of_day

    return datetime_object


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


# return numpy.arange(start_time, end_time, dtype='datetime64[D]')


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


# return numpy.arange(start_time, end_time, dtype='datetime64[h]')


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


def get_masked_data(masked_constant: numpy.ma.core.MaskedConstant) -> object:
    """
    Wrapper to make sure we don't call .data on a regular constant, which will cause memory problems.

    :param masked_constant: input constant
    :return: data held within masked constant
    """

    if 'MaskedConstant' in str(type(masked_constant)) or 'MaskedArray' in str(type(masked_constant)):
        return masked_constant.data
    else:
        return masked_constant


def write_gpkg_subdataset(input_data: numpy.ndarray, output_filename: str, layer_name: str, height: int, width: int,
                          dtype: str, crs: rasterio.crs.CRS, transform: rasterio.Affine, nodata: float,
                          overwrite: bool = False, **kwargs):
    """
    Write input array to a raster layer in a geopackage.
    If the layer exists in the given geopackage, the entire file must be overwritten to replace it.

    :param input_data: array of data to write to raster
    :param output_filename: geopackage filename
    :param layer_name: name of output layer
    :param height: numbers of rows of the raster observation
    :param width: number of columns of the raster observation
    :param dtype: data type for bands
    :param crs: coordinate reference system
    :param transform: affine transformation mapping the pixel space to geographic space
    :param nodata: pixel value to be interpreted as invalid data
    :param overwrite: whether to erase entire geopackage, if replacing existing layer
    :raises Exception: raised GDAL error
    """

    with rasterio.Env(OGR_GPKG_FOREIGN_KEY_CHECK='NO'):
        try:
            with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1, dtype=dtype,
                               crs=crs, transform=transform, nodata=nodata, raster_table=layer_name,
                               raster_identifier=layer_name, raster_description=layer_name,
                               append_subdataset='YES', **kwargs) as output_raster:
                output_raster.write(input_data.astype(dtype), 1)

            print(f'Writing {output_filename}:{layer_name}')

        except rasterio._err.CPLE_AppDefinedError:
            print(f'Subdataset already exists at {output_filename}:{layer_name}')

    if overwrite:
        print(f'Erasing {output_filename}')

    # if error with appending, erase entire observation and append as new
    with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1, dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata, raster_table=layer_name, raster_identifier=layer_name,
                       raster_description=layer_name, **kwargs) as output_raster:
        output_raster.write(input_data.astype(dtype), 1)


def datetime64_to_time(datetime64: numpy.datetime64) -> datetime.datetime:
    """
    Convert numpy.datetime64 object to native Python datetime.datetime object

    :param datetime64: numpy datetime
    :return: python datetime
    """

    return datetime.datetime.fromtimestamp(datetime64.values.astype(datetime.datetime) * 1e-9)


def get_first_record(vector_dataset_filename: str):
    fiona_kwargs = {}

    if ':' in os.path.split(vector_dataset_filename)[-1]:
        vector_dataset_filename, layer_name = vector_dataset_filename.rsplit(':', 1)
        fiona_kwargs['layer'] = layer_name

    with fiona.open(vector_dataset_filename, **fiona_kwargs) as vector_layer:
        return next(iter(vector_layer))


class NoDataError(Exception):
    """
    Error for no data found.
    """
    pass


def rotate_coordinates(longitude: float, latitude: float, pole: numpy.array) -> tuple:
    """
    Convert longitude and latitude to rotated pole coordinates.

    :param longitude: longitude
    :param latitude: latitude
    :param pole: rotated pole
    :return: coordinates in rotated pole system
    """

    if type(pole) is not numpy.array:
        pole = numpy.array(pole)

    # convert degrees to radians
    longitude = longitude * numpy.pi / 180
    latitude = latitude * numpy.pi / 180
    pole = pole * numpy.pi / 180

    # precalculate sin / cos
    latitude_sine = numpy.sin(latitude)
    latitude_cosine = numpy.cos(latitude)
    pole_latitude_sine = numpy.sin(pole[1])
    pole_latitude_cosine = numpy.cos(pole[1])

    # calculate rotation transformation
    phi_1 = longitude - pole[0]

    phi_1_sine = numpy.sin(phi_1)
    phi_1_cosine = numpy.cos(phi_1)

    phi_2 = numpy.arctan2(phi_1_sine * latitude_cosine,
                          phi_1_cosine * latitude_cosine * pole_latitude_sine - latitude_sine * pole_latitude_cosine)
    theta_2 = numpy.arcsin(phi_1_cosine * latitude_cosine * pole_latitude_cosine + latitude_sine * pole_latitude_sine)

    # convert radians to degrees
    phi_2 = phi_2 * 180 / numpy.pi
    theta_2 = theta_2 * 180 / numpy.pi

    return phi_2, theta_2


def unrotate_coordinates(phi: float, theta: float, pole: numpy.array) -> tuple:
    """
    Convert rotated pole coordinates to longitude and latitude.

    :param phi: rotated pole longitude
    :param theta: rotated pole latitude
    :param pole: rotated pole
    :return: coordinates in rotated pole system
    """

    # convert degrees to radians
    phi = phi * numpy.pi / 180
    theta = theta * numpy.pi / 180
    pole = pole * numpy.pi / 180

    # precalculate sin / cos
    phi_sine = numpy.sin(phi)
    phi_cosine = numpy.cos(phi)
    theta_sine = numpy.sin(theta)
    theta_cosine = numpy.cos(theta)
    pole_latitude_sine = numpy.sin(pole[1])
    pole_latitude_cosine = numpy.cos(pole[1])

    # calculate rotation transformation
    longitude = pole[0] + numpy.arctan2(phi_sine * theta_cosine,
                                        phi_cosine * theta_cosine * pole_latitude_sine + theta_sine * pole_latitude_cosine)
    latitude = numpy.arcsin(-phi_cosine * theta_cosine * pole_latitude_cosine + theta_sine * pole_latitude_sine)

    # convert radians to degrees
    longitude = longitude * 180 / numpy.pi
    latitude = latitude * 180 / numpy.pi

    return longitude, latitude


if __name__ == '__main__':
    import PyOFS

    copy_xarray('https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_6km',
                os.path.join(PyOFS.DATA_DIR, 'output', 'test', 'hfradar_uswc_6km.nc'))

    print('done')
