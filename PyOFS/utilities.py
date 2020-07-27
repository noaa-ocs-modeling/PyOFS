# coding=utf-8
"""
Data utility functions.

Created on Jun 13, 2018

@author: zachary.burnett
"""

from datetime import datetime, timedelta
from functools import partial
import os

import fiona
import fiona.crs
import numpy
import pyproj
import rasterio
import shapely
from shapely.geometry import shape
from shapely.ops import transform
import xarray

from PyOFS import get_logger

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WEB_MERCATOR = pyproj.Proj('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs')

GRAVITATIONAL_ACCELERATION = 9.80665  # meters per second squared
SIDEREAL_ROTATION_PERIOD = timedelta(hours=23, minutes=56, seconds=4.1)

LOGGER = get_logger('PyOFS.utili')


def copy_xarray(input_path: str, output_path: str) -> xarray.Dataset:
    """
    Copy given xarray observation to a local file at the given path.

    :param input_path: path to observation to copy
    :param output_path: path to output file
    :return: copied observation at given path
    """

    LOGGER.info(f'Reading observation from {input_path}')

    input_dataset = xarray.open_dataset(input_path, decode_times=False)

    LOGGER.info(f'Copying observation to local memory...')

    # deep copy of xarray observation
    output_dataset = input_dataset.copy(deep=True)

    LOGGER.info(f'Writing to {output_path}')

    # save observation to file
    output_dataset.to_netcdf(output_path)

    return output_dataset


def round_to_day(datetime_object: datetime, direction: str = None) -> datetime:
    """
    Return given datetime rounded to the nearest day.

    :param datetime_object: datetime to round
    :param direction: either 'ceiling' or 'floor', optional
    :return: rounded datetime
    """

    start_of_day = datetime_object.replace(hour=0, minute=0, second=0, microsecond=0)
    half_day = datetime_object.replace(hour=12, minute=0, second=0, microsecond=0)

    if direction == 'ceiling' or (direction is None and datetime_object >= half_day):
        datetime_object = start_of_day + timedelta(days=1)
    elif direction == 'floor' or (direction is None and datetime_object < half_day):
        datetime_object = start_of_day

    return datetime_object


# return numpy.arange(start_time, end_time, dtype='datetime64[D]')


# return numpy.arange(start_time, end_time, dtype='datetime64[h]')


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


def write_gpkg_subdataset(input_data: numpy.array, output_filename: str, layer_name: str, height: int, width: int, dtype: str, crs: rasterio.crs.CRS, transform: rasterio.Affine, nodata: float,
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
            with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata, raster_table=layer_name,
                               raster_identifier=layer_name, raster_description=layer_name, append_subdataset='YES', **kwargs) as output_raster:
                output_raster.write(input_data.astype(dtype), 1)

            LOGGER.info(f'Writing {output_filename}:{layer_name}')

        except rasterio._err.CPLE_AppDefinedError:
            LOGGER.info(f'Subdataset already exists at {output_filename}:{layer_name}')

    if overwrite:
        LOGGER.info(f'Erasing {output_filename}')

    # if error with appending, erase entire observation and append as new
    with rasterio.open(output_filename, 'w', driver='GPKG', height=height, width=width, count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata, raster_table=layer_name,
                       raster_identifier=layer_name, raster_description=layer_name, **kwargs) as output_raster:
        output_raster.write(input_data.astype(dtype), 1)


def datetime64_to_time(datetime64: numpy.datetime64) -> datetime:
    """
    Convert numpy.datetime64 object to native Python datetime object

    :param datetime64: numpy datetime
    :return: python datetime
    """

    return datetime.fromtimestamp(datetime64.values.astype(datetime) * 1e-9)


def get_first_record(vector_dataset_filename: str):
    fiona_kwargs = {}

    if ':' in os.path.split(vector_dataset_filename)[-1]:
        vector_dataset_filename, layer_name = vector_dataset_filename.rsplit(':', 1)
        fiona_kwargs['layer'] = layer_name

    with fiona.open(vector_dataset_filename, **fiona_kwargs) as vector_layer:
        return next(iter(vector_layer))


class RotatedPoleCoordinateSystem:
    def __init__(self, pole: (float, float)):
        """
        New rotated pole coordinate system.

        :param pole: unrotated coordinates of pole
        """

        self.pole = pole if type(pole) is numpy.array else numpy.array(pole)

    def rotate_coordinates(self, point: numpy.array, projection: pyproj.Proj = None) -> numpy.array:
        """
        Convert longitude and latitude to rotated pole coordinates.

        :param point: unrotated coordinates
        :param projection: projection
        :return: coordinates rotated around pole
        """

        if type(point) is not numpy.array:
            point = numpy.array(point)

        if projection is not None:
            point = numpy.array(pyproj.transform(projection, WGS84, point[0], point[1]))

        # convert degrees to radians
        point = point * numpy.pi / 180
        pole = self.pole * numpy.pi / 180

        # calculate sine and cosine
        local_longitude = point[0] - pole[0]
        sine_longitude = numpy.sin(local_longitude)
        cosine_longitude = numpy.cos(local_longitude)
        sine_latitude = numpy.sin(point[1])
        cosine_latitude = numpy.cos(point[1])
        sine_pole_latitude = numpy.sin(pole[1])
        cosine_pole_latitude = numpy.cos(pole[1])

        # precalculate rotation transformation
        rotated_longitude = numpy.arctan2(sine_longitude * cosine_latitude, cosine_longitude * cosine_latitude * sine_pole_latitude - sine_latitude * cosine_pole_latitude)
        rotated_latitude = numpy.arcsin(cosine_longitude * cosine_latitude * cosine_pole_latitude + sine_latitude * sine_pole_latitude)

        # convert radians to degrees
        return numpy.array((rotated_longitude * 180 / numpy.pi, rotated_latitude * 180 / numpy.pi))

    def unrotate_coordinates(self, rotated_point: numpy.array) -> numpy.array:
        """
        Convert rotated pole coordinates to longitude and latitude.

        :param rotated_point: rotated coordinates
        :return: coordinates unrotated around pole
        """

        if type(rotated_point) is not numpy.array:
            rotated_point = numpy.array(rotated_point)

        # convert degrees to radians
        rotated_point = rotated_point * numpy.pi / 180
        pole = self.pole * numpy.pi / 180

        # precalculate sine and cosine
        sine_rotated_longitude = numpy.sin(rotated_point[0])
        cosine_rotated_longitude = numpy.cos(rotated_point[0])
        sine_rotated_latitude = numpy.sin(rotated_point[1])
        cosine_rotated_latitude = numpy.cos(rotated_point[1])
        sine_pole_latitude = numpy.sin(pole[1])
        cosine_pole_latitude = numpy.cos(pole[1])

        # calculate rotation transformation
        longitude = pole[0] + numpy.arctan2(sine_rotated_longitude * cosine_rotated_latitude,
                                            cosine_rotated_longitude * cosine_rotated_latitude * sine_pole_latitude + sine_rotated_latitude * cosine_pole_latitude)
        latitude = numpy.arcsin(-cosine_rotated_longitude * cosine_rotated_latitude * cosine_pole_latitude + sine_rotated_latitude * sine_pole_latitude)

        # convert radians to degrees
        return numpy.array((longitude * 180 / numpy.pi, latitude * 180 / numpy.pi))

    @staticmethod
    def find_pole(points: numpy.array, starting_pole: numpy.array, samples: int = 10, sample_radius: float = 1) -> numpy.array:
        """
        Find pole given points with the same rotated latitude.

        :param points: array of points with a constant distance to the pole
        :param starting_pole: approximate pole to start from
        :param samples: number of samples to take
        :param sample_radius: radius in degrees around starting pole to search
        :return: central point
        """

        deltas = numpy.linspace(-sample_radius, sample_radius, numpy.sqrt(samples))

        discrepancies = numpy.empty((len(deltas), len(deltas)))

        for lon_index, lon_delta in enumerate(deltas):
            for lat_index, lat_delta in enumerate(deltas):
                current_pole_candidate = starting_pole + numpy.array((lon_delta, lat_delta))

                current_rotated_pole = RotatedPoleCoordinateSystem(current_pole_candidate)
                _, rotated_latitudes = current_rotated_pole.rotate_coordinates(points)

                discrepancies[lon_index, lat_index] = numpy.abs(numpy.diff(rotated_latitudes)).max()

        discrepancies = xarray.DataArray(discrepancies, coords={'lon': starting_pole[0] + deltas, 'lat': starting_pole[1] + deltas}, dims=('lon', 'lat'))

        from matplotlib import pyplot
        discrepancies.plot.imshow()
        pyplot.show()

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.pole)})'


def xarray_to_geopackage(input_path: str, output_path: str, epsg: int = 4326):
    polygons = {}

    with xarray.open_dataset(input_path) as input_dataset:
        starting_index = 0
        for contour_index in range(len(input_dataset['time'])):
            ending_index = int(input_dataset['EndPoint'][contour_index].values.item()) if contour_index <= len(input_dataset['EndPoint']) else -1

            contour_datetime = datetime64_to_time(input_dataset['time'][contour_index])
            longitude = input_dataset['longitude'][starting_index:ending_index]
            latitude = input_dataset['latitude'][starting_index:ending_index]
            polygons[contour_datetime] = shapely.geometry.Polygon(zip(longitude, latitude))

            starting_index = ending_index + 1

    # define schema
    schema = {'geometry': 'Polygon', 'properties': {'datetime': 'datetime'}}

    # add value fields to schema
    schema['properties'].update({'area': 'float', 'perimeter': 'float'})

    records = [{
        'geometry': shapely.geometry.mapping(polygon),
        'properties': {
            'datetime': contour_time,
            'area': transform(partial(pyproj.transform, pyproj.Proj(init=f'epsg:{epsg}'), pyproj.Proj(init='epsg:3857')), polygon).area,
            'perimeter': polygon.length
        }
    } for contour_time, polygon in polygons.items()]

    with fiona.open(output_path, 'w', 'GPKG', schema=schema, crs=fiona.crs.from_epsg(epsg)) as output_file:
        output_file.writerecords(records)


def geodetic_radius(latitude: float) -> float:
    """
    Get the approximate radius of the Earth, using the WGS84 ellipsoid at a given latitude.

    :param latitude: latitude in degrees
    :return: WGS84 ellipsoid radius
    """

    semimajor_radius = 6378137
    semiminor_radius = 6356752.314245

    sine_latitude = numpy.sin(latitude * numpy.pi / 180) * 180 / numpy.pi
    cosine_latitude = numpy.cos(latitude * numpy.pi / 180) * 180 / numpy.pi

    return numpy.sqrt(
        ((semimajor_radius ** 2 * cosine_latitude) ** 2 + (semiminor_radius ** 2 + sine_latitude) ** 2) / ((semimajor_radius * cosine_latitude) ** 2 + (semiminor_radius + sine_latitude) ** 2))


def rossby_deformation_radius(latitude: float) -> float:
    """
    Get the Rossby deformation radius given a latitude and water depth.

    :param latitude: latitude in degrees
    :return: Rossby radius
    """

    oceanic_speed_of_sound = 1500  # meters per second
    oceanic_potential_density = 1025  # kilograms per cubic meter

    coriolis_frequency = numpy.sin(latitude * numpy.pi / 180) * 4 * numpy.pi / SIDEREAL_ROTATION_PERIOD / timedelta(seconds=1)

    brunt_vaisala_frequency = numpy.sqrt(GRAVITATIONAL_ACCELERATION / oceanic_potential_density)
    scale_height = oceanic_speed_of_sound ** 2 / GRAVITATIONAL_ACCELERATION
    return brunt_vaisala_frequency * scale_height / (numpy.pi * coriolis_frequency)


def translate_geographic_coordinates(point: numpy.array, offset: numpy.array) -> numpy.array:
    """
    Add meter offset to geographic coordinates using WebMercator.

    :param point: geographic coordinates
    :param offset: translation vector in meters
    :return: new geographic coordinates
    """

    if type(point) is not numpy.array:
        point = numpy.array(point)

    if type(offset) is not numpy.array:
        offset = numpy.array(offset)

    return numpy.array(pyproj.transform(WEB_MERCATOR, WGS84, *(pyproj.transform(WGS84, WEB_MERCATOR, *point)) + offset))


if __name__ == '__main__':
    # xarray_to_geopackage(#     input_path=r"C:\Users\zachary.burnett\Downloads\lagr_contour_Exp24_r20Clon-125Clat40dr500m_20100601-20100604.nc",#     output_path=r"C:\Users\zachary.burnett\Downloads\alex_contour.gpkg")

    geopackage_path = r"C:\Data\develop\output\test\contours.gpkg"
    layer_names = ['wcofs_qck_20160925T010000_20160929T010000_1h', 'wcofs_qck_20160925T010000_20160929T010000_24h', 'wcofs_qck_geostrophic_20160925T010000_20160929T010000_1h']

    for layer_name in layer_names:
        records = []

        with fiona.open(geopackage_path, layer=layer_name) as input_file:
            for record in input_file:
                polygon = shapely.geometry.Polygon(record['geometry']['coordinates'][0])
                record['properties'].update({'area': polygon.area, 'perimeter': polygon.length})
                records.append(record)

        # define schema
        schema = {'geometry': 'Polygon', 'properties': {'contour': 'str', 'datetime': 'datetime'}}

        # add value fields to schema
        schema['properties'].update({'area': 'float', 'perimeter': 'float'})

        with fiona.open(geopackage_path, 'w', 'GPKG', schema=schema, crs=fiona.crs.from_epsg(3857), layer=f'{layer_name}_copy') as output_file:
            output_file.writerecords(records)

    print('done')
