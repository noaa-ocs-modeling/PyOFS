# coding=utf-8
"""
Data utility functions.

Created on Jun 13, 2018

@author: zachary.burnett
"""

import datetime

import numpy
import rasterio
import xarray


def copy_xarray(input_path: str, output_path: str) -> xarray.Dataset:
    """
    Copy given xarray dataset to a local file at the given path.

    :param input_path: Path to dataset to copy.
    :param output_path: Path to output file.
    :return: Copied dataset, now at local filename.
    """

    print(f'Copying dataset to {output_path}')

    input_dataset = xarray.open_dataset(input_path, decode_times=False)

    # deep copy of xarray dataset
    output_dataset = input_dataset.copy(deep=True)

    # save dataset to file
    output_dataset.to_netcdf(output_path)

    return output_dataset


def round_to_day(datetime_object: datetime.datetime, direction: str = None) -> datetime.datetime:
    """
    Return given datetime rounded to the nearest day.

    :param datetime_object: Datetime to round.
    :param direction: Either 'ceiling' or 'floor', optional.
    :return: Rounded datetime.
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

    :param datetime_object: Datetime to round.
    :param direction: Either 'ceiling' or 'floor', optional.
    :return: Rounded datetime.
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

    :param datetime_object: Datetime to round.
    :return: Rounded datetime.
    """

    return datetime_object.replace(minute=int(round(datetime_object.minute, -1)), second=0, microsecond=0)


def day_range(start_datetime: datetime.datetime, end_datetime: datetime.datetime) -> list:
    """
    Generate range of times between given times at day intervals.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :return: Range of datetimes.
    """

    duration = end_datetime - start_datetime
    return [start_datetime + datetime.timedelta(days=day) for day in range(duration.days)]

    # return numpy.arange(start_datetime, end_datetime, dtype='datetime64[D]')


def hour_range(start_datetime: datetime.datetime, end_datetime: datetime.datetime) -> list:
    """
    Generate range of times between given times at hour intervals.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :return: Range of datetimes.
    """

    duration = end_datetime - start_datetime
    return [start_datetime + datetime.timedelta(hours=hour) for hour in
            range(int((duration.days * 24) + (duration.seconds / 3600)))]

    # return numpy.arange(start_datetime, end_datetime, dtype='datetime64[h]')


def ten_minute_range(start_datetime: datetime.datetime, end_datetime: datetime.datetime) -> list:
    """
    Generate range of times between given times at ten minute intervals.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :return: Range of datetimes.
    """

    duration = end_datetime - start_datetime
    return [start_datetime + datetime.timedelta(minutes=minute) for minute in
            range(0, int((duration.days * 24 * 60) + (duration.seconds / 60)) + 1, 10)]


def get_masked_data(masked_constant: numpy.ma.core.MaskedConstant) -> object:
    """
    Wrapper to make sure we don't call .data on a regular constant, which will cause memory problems.

    :param masked_constant: Input constant.
    :return: Data held within masked constant.
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

    :param input_data: Array of data to write to raster.
    :param output_filename: Geopackage filename.
    :param layer_name: Name of output layer.
    :param height: The numbers of rows of the raster dataset.
    :param width: The number of columns of the raster dataset.
    :param dtype: The data type for bands.
    :param crs: The coordinate reference system.
    :param transform: Affine transformation mapping the pixel space to geographic space.
    :param nodata: Defines the pixel value to be interpreted as not valid data.
    :param overwrite: Whether to erase entire geopackage, if replacing existing layer.
    :raises Exception: raised GDAL error.
    """

    with rasterio.Env(OGR_GPKG_FOREIGN_KEY_CHECK='NO') as env:
        try:
            with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1, dtype=dtype,
                               crs=crs, transform=transform, nodata=nodata, raster_table=layer_name,
                               raster_identifier=layer_name, raster_description=layer_name,
                               append_subdataset='YES') as output_raster:
                output_raster.write(input_data.astype(dtype), 1)

            print(f'Writing {output_filename}:{layer_name}')

        except rasterio._err.CPLE_AppDefinedError as exception:
            print(f'Subdataset already exists at {output_filename}:{layer_name}')

            # if overwrite:  #     print(f'Erasing {output_filename}')  #     # if error with appending, erase entire dataset and append as new  #     with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1,  #                        dtype=dtype, crs=crs, transform=transform, nodata=nodata, raster_table=layer_name,  #                        raster_identifier=layer_name, raster_description=layer_name) as output_raster:  #         output_raster.write(input_data.astype(dtype), 1)


class NoDataError(Exception):
    pass


if __name__ == '__main__':
    print('done')
