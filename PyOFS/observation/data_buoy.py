# coding=utf-8
"""
National Data Buoy Center moorings.

Created on Aug 1, 2018

@author: zachary.burnett
"""

from concurrent import futures
from datetime import datetime
import os
import re

import fiona
import fiona.crs
import numpy
import requests
import shapely
import shapely.geometry
import xarray

import PyOFS
from PyOFS import CRS_EPSG, DATA_DIRECTORY, utilities, get_logger

LOGGER = get_logger('PyOFS.NDBC')

MEASUREMENT_VARIABLES = ['water_temperature', 'conductivity', 'salinity', 'o2_saturation', 'dissolved_oxygen', 'chlorophyll_concentration', 'turbidity', 'water_ph', 'water_eh']

OUTPUT_CRS = fiona.crs.from_epsg(CRS_EPSG)

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIRECTORY, r"reference\wcofs.gpkg:study_area")
WCOFS_NDBC_STATIONS_FILENAME = os.path.join(DATA_DIRECTORY, r"reference\ndbc_stations.txt")

CATALOG_URL = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/ocean/catalog.html'
SOURCE_URL = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/ocean'


class DataBuoyDataset:
    """
    Dataset of NDBC buoy.
    """

    def __init__(self, station_name: str):
        """
        NDBC data buoy

        :param station_name: station name
        :raises NoDataError: if observation does not exist
        """

        self.station_name = station_name
        self.url = f'{SOURCE_URL}/{self.station_name}/{self.station_name}o9999.nc'

        try:
            self.dataset = xarray.open_dataset(self.url)
            self.longitude = self.dataset['longitude'].values.item()
            self.latitude = self.dataset['latitude'].values.item()
        except:
            raise PyOFS.NoDataError(f'No NDBC observation found at {self.url}')

    def geometry(self) -> shapely.geometry.Point:
        """
        Get geometry as a point.

        :return: point
        """

        return shapely.geometry.point.Point(self.longitude, self.latitude)

    def data(self, variable: str, start_time: datetime, end_time: datetime) -> dict:
        """
        Collects data from given station in the given time interval.

        :param variables: list of variable names
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :return: dictionary of data from the given station over the given time interval
        """

        return self.dataset[variable].sel(time=slice(start_time, end_time))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.station_name})'


class DataBuoyRange:
    """
    Data from multiple NDBC buoys.
    """

    def __init__(self, stations: [str] = None):
        """
        Collection of NDBC data buoys

        :param stations: list of station names
        :raises NoDataError: if data does not exist
        """

        if stations is None:
            with requests.get(CATALOG_URL) as station_catalog:
                self.station_names = re.findall("href='(.*?)/catalog.html'", station_catalog.text)
        elif type(stations) is str:
            self.station_names = list(numpy.genfromtxt(WCOFS_NDBC_STATIONS_FILENAME, dtype='str'))
        else:
            self.station_names = stations

        self.stations = {}

        LOGGER.debug(f'Collecting NDBC data from {len(self.station_names)} station...')

        # concurrently populate dictionary with datasets for each station
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(DataBuoyDataset, station_name): station_name for station_name in self.station_names}

            for completed_future in futures.as_completed(running_futures):
                station_name = running_futures[completed_future]

                if type(completed_future.exception()) is not PyOFS.NoDataError:
                    result = completed_future.result()
                    self.stations[station_name] = result

            del running_futures

        if len(self.stations) == 0:
            raise PyOFS.NoDataError(f'No NDBC datasets found in {self.stations}')

    def data(self, variables: [str], start_time: datetime, end_time: datetime) -> {str: {str: xarray.DataArray}}:
        """
        Get data of given variables within given time interval.

        :param variables:
        :param start_time: Start of time interval.
        :param end_time: End of time interval.
        :return:
        """

        output_data = {}

        for station_name, station in self.stations.items():
            output_data[station_name] = {}

            for variable in variables:
                output_data[station_name][variable] = station.data(variable, start_time, end_time)

        return output_data

    def data_average(self, variables: [str], start_time: datetime, end_time: datetime) -> {str: {str: float}}:
        """
        Get data of given variables within given time interval.

        :param variables:
        :param start_time: Start of time interval.
        :param end_time: End of time interval.
        :return:
        """

        output_data = {}

        for station_name, station in self.stations.items():
            output_data[station_name] = {}

            for variable in variables:
                output_data[station_name][variable] = float(station.data(variable, start_time, end_time).mean('time', skipna=True))

        return output_data

    def write_vector(self, output_filename: str, start_time: datetime, end_time: datetime, variables: [str] = None):
        """
        Write average of buoy data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: path to output file
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param variables: list of variable names
        """

        layer_name = None

        if ':' in os.path.split(output_filename)[-1]:
            output_filename, layer_name = output_filename.rsplit(':', 1)

        if variables is None:
            variables = MEASUREMENT_VARIABLES

        station_data = self.data_average(variables, start_time, end_time)

        # # concurrently populate dictionary with data for each station within given time interval
        # with futures.ThreadPoolExecutor() as concurrency_pool:
        #     running_futures = {
        #         station_name: {
        #             variable: concurrency_pool.submit(station.data, variable, start_time, end_time)
        #             for variable in variables}
        #         for station_name, station in self.stations.items()
        #     }
        #
        #     for station_name, station_running_futures in running_futures:
        #         station_data[station_name] = {}
        #
        #         for completed_future in futures.as_completed(station_running_futures):
        #             result = completed_future.result()
        #
        #             if result is not None:
        #                 station_data[station_name][station_running_futures[completed_future]] = result

        schema = {
            'geometry': 'Point', 'properties': {
                'name': 'str',
                'longitude': 'float',
                'latitude': 'float',
                'water_temperature': 'float',
                'conductivity': 'float',
                'salinity': 'float',
                'o2_saturation': 'float',
                'dissolved_oxygen': 'float',
                'chlorophyll_concentration': 'float',
                'turbidity': 'float',
                'water_ph': 'float',
                'water_eh': 'float'
            }
        }

        LOGGER.debug('Creating features...')

        layer_records = []

        for station_name, station_data in station_data.items():
            station = self.stations[station_name]

            record = {
                'geometry': {
                    'type': 'Point',
                    'coordinates': (station.longitude, station.latitude)
                },
                'properties': {
                    'name': station_name,
                    'longitude': station.longitude,
                    'latitude': station.latitude,
                    'water_temperature': station_data['water_temperature'],
                    'conductivity': station_data['conductivity'],
                    'salinity': station_data['salinity'],
                    'o2_saturation': station_data['o2_saturation'],
                    'dissolved_oxygen': station_data['dissolved_oxygen'],
                    'chlorophyll_concentration': station_data['chlorophyll_concentration'],
                    'turbidity': station_data['turbidity'],
                    'water_ph': station_data['water_ph'],
                    'water_eh': station_data['water_eh']
                }
            }

            layer_records.append(record)

        LOGGER.info(f'Writing to {output_filename}{":" + layer_name if layer_name is not None else ""}')
        with fiona.open(output_filename, 'w', 'GPKG', schema, OUTPUT_CRS, layer=layer_name) as output_layer:
            output_layer.writerecords(layer_records)

    def __repr__(self):
        used_params = []
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
    study_area_polygon = shapely.geometry.Polygon(utilities.get_first_record(study_area_polygon_filename)['geometry']['coordinates'][0])

    lon = dataset['longitude'][:]
    lat = dataset['latitude'][:]

    point = shapely.geometry.point.Point(lon, lat)

    return point.intersects(study_area_polygon)


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIRECTORY, r'output\test')
    wcofs_stations = list(numpy.genfromtxt(WCOFS_NDBC_STATIONS_FILENAME, dtype='str'))

    start_time = datetime(2018, 7, 14)
    end_time = datetime.now()
    date_interval_string = f'{start_time:%m%d%H}_{end_time:%m%d%H}'

    ndbc_range = DataBuoyRange(wcofs_stations)
    ndbc_range.write_vector(os.path.join(output_dir, f'ndbc.gpkg:NDBC_{date_interval_string}'), start_time=start_time, end_time=end_time)

    print('done')
