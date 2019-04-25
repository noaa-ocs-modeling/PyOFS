# coding=utf-8
"""
National Data Buoy Center moorings.

Created on Aug 1, 2018

@author: zachary.burnett
"""

import datetime
import logging
import os
import re
from concurrent import futures

import fiona
import fiona.crs
import numpy
import rasterio
import requests
import shapely
import shapely.geometry
import xarray

from PyOFS import CRS_EPSG, DATA_DIR, utilities

MEASUREMENT_VARIABLES = ['water_temperature', 'conductivity', 'salinity', 'o2_saturation', 'dissolved_oxygen',
                         'chlorophyll_concentration', 'turbidity', 'water_ph', 'water_eh']

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})
FIONA_CRS = fiona.crs.from_epsg(CRS_EPSG)

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")
WCOFS_NDBC_STATIONS_FILENAME = os.path.join(DATA_DIR, r"reference\ndbc_stations.txt")

SOURCE_URL = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/ocean/catalog.html'


class NDBCStation:
    """
    Buoy data of ocean variables within a time interval.
    """

    def __init__(self, station):
        """
        Creates new observation object.

        :param str station: station name
        :raises NoDataError: if observation does not exist
        """

        self.valid = False
        self.station_name = station
        self.url = f'https://dods.ndbc.noaa.gov/thredds/dodsC/data/ocean/{self.station_name}/{self.station_name}o9999.nc'

        try:
            self.netcdf_dataset = xarray.open_dataset(self.url)
            self.longitude = self.netcdf_dataset['longitude'].values.item()
            self.latitude = self.netcdf_dataset['latitude'].values.item()
            self.valid = True
        except:
            raise utilities.NoDataError(f'No NDBC observation found at {self.url}')

    def geometry(self) -> shapely.geometry.Point:
        """
        Get geometry as a point.

        :return: point
        """

        return shapely.geometry.point.Point(self.longitude, self.latitude)

    def data(self, start_time, end_time) -> dict:
        """
        Collects data from given station in the given time interval.

        :param datetime.datetime start_time: beginning of time interval
        :param datetime.datetime end_time: end of time interval
        :return: dictionary of data from the given station over the given time interval
        """

        logging.info(
            f'Collecting NDBC data of station {self.station_name} from {start_time} to {end_time}...')

        output_data = {variable: None for variable in MEASUREMENT_VARIABLES}

        time_data = self.netcdf_dataset['time'].values

        start_index = numpy.searchsorted(time_data, numpy.datetime64(start_time))
        end_index = numpy.searchsorted(time_data, numpy.datetime64(end_time))

        if end_index - start_index > 0:
            # concurrently populate dictionary with data for each variable
            for variable in MEASUREMENT_VARIABLES:
                output_data[variable] = self.netcdf_dataset[variable][start_index:end_index, :, :].values

        return output_data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.station_name})'


class NDBCRange:
    """
    Buoy data of ocean variables within a time interval.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, stations: list = None):
        """
        Creates new observation object.

        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param stations: list of station names
        :raises NoDataError: if data does not exist
        """

        self.start_time = start_time
        self.end_time = end_time

        if stations is None:
            with requests.get(SOURCE_URL) as catalog:
                self.station_names = re.findall("href='(.*?)/catalog.html'", catalog.text)
        else:
            self.station_names = stations

        self.stations = {}

        # concurrently populate dictionary with datasets for each station within given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(NDBCStation, station_name): station_name for station_name in
                               self.station_names}

            for completed_future in futures.as_completed(running_futures):
                station_name = running_futures[completed_future]

                if type(completed_future.exception()) is not utilities.NoDataError:
                    result = completed_future.result()
                    logging.info(f'Collecting NDBC data from station {station_name}...')
                    self.stations[station_name] = result

            del running_futures

        if len(self.stations) == 0:
            raise utilities.NoDataError(
                f'No NDBC datasets found between {self.start_time} and {self.end_time}.')

    def write_vector(self, output_filename: str, layer_name: str, start_time: datetime.datetime = None,
                     end_time: datetime.datetime = None):
        """
        Write average of buoy data for all hours in the given time interval to a single layer of the provided output file.

        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param output_filename: path to output file
        :param layer_name: name of layer to write
        """

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        station_data = {}

        # concurrently populate dictionary with data for each station within given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(station.data, start_time, end_time): station_name for
                               station_name, station in self.stations.items()}

            for completed_future in futures.as_completed(running_futures):
                result = completed_future.result()

                if result is not None:
                    station_name = running_futures[completed_future]
                    station_data[station_name] = result

        schema = {
            'geometry': 'Point', 'properties': {
                'name': 'str', 'longitude': 'float', 'latitude': 'float',
                'water_temperature': 'float', 'conductivity': 'float', 'salinity': 'float',
                'o2_saturation': 'float', 'dissolved_oxygen': 'float', 'chlorophyll_concentration': 'float',
                'turbidity': 'float', 'water_ph': 'float', 'water_eh': 'float'
            }
        }

        with fiona.open(output_filename, 'w', 'GPKG', schema, FIONA_CRS, layer=layer_name) as layer:
            logging.debug('Creating features...')

            layer_records = []

            for station_name, station_data in station_data.items():
                station = self.stations[station_name]
                longitude = float(station.longitude)
                latitude = float(station.latitude)

                record = {
                    'geometry': {'type': 'Point', 'coordinates': (longitude, latitude)}, 'properties': {
                        'name': station_name, 'longitude': longitude, 'latitude': latitude,
                        'water_temperature': float(numpy.ma.mean(station_data['water_temperature'])),
                        'conductivity': float(numpy.ma.mean(station_data['conductivity'])),
                        'salinity': float(numpy.ma.mean(station_data['salinity'])),
                        'o2_saturation': float(
                            numpy.ma.mean(station_data['o2_saturation'])),
                        'dissolved_oxygen': float(numpy.ma.mean(station_data['dissolved_oxygen'])),
                        'chlorophyll_concentration': float(
                            numpy.ma.mean(station_data['chlorophyll_concentration'])),
                        'turbidity': float(numpy.ma.mean(station_data['turbidity'])),
                        'water_ph': float(numpy.ma.mean(station_data['water_ph'])),
                        'water_eh': float(
                            numpy.ma.mean(station_data['water_eh']))
                    }
                }

                layer_records.append(record)

            logging.info(f'Writing {output_filename}:{layer_name}')
            layer.writerecords(layer_records)

    def __repr__(self):
        used_params = [self.start_time.__repr__(), self.end_time.__repr__()]
        optional_params = [self.station_names]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({", ".join(used_params)})'


def check_station(dataset: xarray.Dataset, study_area_polygon_filename: str) -> bool:
    """
    Check whether station exists within the given study area.

    :param dataset: NetCDF Dataset
    :param study_area_polygon_filename: vector file containing study area boundary
    :return: whether station is within study area
    """

    # construct polygon from the first record in the layer
    study_area_polygon = shapely.geometry.Polygon(
        utilities.get_first_record(study_area_polygon_filename)['geometry']['coordinates'][0])

    lon = dataset['longitude'][:]
    lat = dataset['latitude'][:]

    point = shapely.geometry.point.Point(lon, lat)

    return point.intersects(study_area_polygon)


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test')
    wcofs_stations = list(numpy.genfromtxt(WCOFS_NDBC_STATIONS_FILENAME, dtype='str'))

    start_time = datetime.datetime(2018, 7, 14)
    end_time = datetime.datetime.now()
    date_interval_string = f'{start_time.strftime("%m%d%H")}_{end_time.strftime("%m%d%H")}'

    ndbc_range = NDBCRange(start_time, end_time, stations=wcofs_stations)
    ndbc_range.write_vector(os.path.join(output_dir, 'ndbc.gpkg'), f'NDBC_{date_interval_string}')

    print('done')
