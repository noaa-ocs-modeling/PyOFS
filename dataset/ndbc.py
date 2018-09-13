"""
National Data Buoy Center moorings.

Created on Aug 1, 2018

@author: zachary.burnett
"""

import concurrent.futures
import datetime
import os
import re

import fiona
import fiona.crs
import netCDF4
import numpy
import rasterio
import requests
import shapely
import shapely.geometry

from dataset import _utilities

MEASUREMENT_VARIABLES = ['water_temperature', 'conductivity', 'salinity', 'o2_saturation', 'dissolved_oxygen',
                         'chlorophyll_concentration', 'turbidity', 'water_ph', 'water_eh']

FIONA_WGS84 = fiona.crs.from_epsg(4326)
RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})

DATA_DIR = os.environ['OFS_DATA']

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")
WCOFS_NDBC_STATIONS_FILENAME = os.path.join(DATA_DIR, r"reference\ndbc_stations.txt")

NDBC_CATALOG_URL = f'https://dods.ndbc.noaa.gov/thredds/catalog/data/ocean/catalog.html'


class NDBC_Station:
    """
    Buoy data of ocean variables within a time interval.
    """

    def __init__(self, station):
        """
        Creates new dataset object.

        :param str station: Station name.
        :raises NoDataError: if dataset does not exist.
        """

        self.valid = False

        self.station_name = station

        self.url = f'https://dods.ndbc.noaa.gov/thredds/dodsC/data/ocean/{self.station_name}/{self.station_name}o9999.nc'

        try:
            self.netcdf_dataset = netCDF4.Dataset(self.url)
            self.longitude = self.netcdf_dataset['longitude'][:].filled()[0]
            self.latitude = self.netcdf_dataset['latitude'][:].filled()[0]
            self.valid = True
        except:
            raise _utilities.NoDataError(f'No NDBC dataset found at {self.url}')

    def geometry(self):
        return shapely.geometry.point.Point(self.longitude, self.latitude)

    def data(self, start_datetime, end_datetime):
        """
        Collects data from given station in the given time interval.

        :param datetime.datetime start_datetime: Beginning of time interval.
        :param datetime.datetime end_datetime: End of time interval.
        :return: Dictionary of data from the given station over the given time interval.
        :rtype: dict(str, numpy.ma.MaskedArray)
        """

        print(f'Collecting NDBC data of station {self.station_name} from {start_datetime} to {end_datetime}...')

        output_data = {current_variable: None for current_variable in MEASUREMENT_VARIABLES}

        time_var = self.netcdf_dataset['time']

        # parse datetime objects from "seconds since 1970-01-01 00:00:00 UTC" using netCDF4 date parser
        start_hour, end_hour = netCDF4.date2num([start_datetime, end_datetime], time_var.units)

        time_data = time_var[:]

        start_index = numpy.searchsorted(time_data, start_hour)
        end_index = numpy.searchsorted(time_data, end_hour)

        if end_index - start_index > 0:
            # concurrently populate dictionary with data for each variable
            for current_variable in MEASUREMENT_VARIABLES:
                output_data[current_variable] = self.netcdf_dataset[current_variable][start_index:end_index, :, :]

        return output_data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.station_name})'


class NDBC_Range:
    """
    Buoy data of ocean variables within a time interval.
    """

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, stations: list = None):
        """
        Creates new dataset object.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param stations: List of station names.
        :raises NoDataError: if data does not exist.
        """

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.station_names = stations

        if self.station_names is None:
            with requests.get(NDBC_CATALOG_URL) as catalog:
                self.station_names = re.findall("href='(.*?)/catalog.html'", catalog.text)

        self.stations = {}

        # concurrently populate dictionary with datasets for each station within given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            station_futures = {concurrency_pool.submit(NDBC_Station, current_station_name): current_station_name for
                               current_station_name in self.station_names}

            for current_future in concurrent.futures.as_completed(station_futures):
                current_station_name = station_futures[current_future]

                if type(current_future.exception()) is not _utilities.NoDataError:
                    current_result = current_future.result()
                    print(f'Collecting NDBC data from station {current_station_name}...')
                    self.stations[current_station_name] = current_result

        if len(self.stations) == 0:
            raise _utilities.NoDataError(
                    f'No NDBC datasets found between {self.start_datetime} and {self.end_datetime}.')

    def write_vector(self, output_filename: str, layer_name: str, start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None):
        """
        Write average of buoy data for all hours in the given time interval to a single layer of the provided output file.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        station_data = {}

        # concurrently populate dictionary with data for each station within given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {
                concurrency_pool.submit(current_station.data, start_datetime, end_datetime): current_station_name for
                current_station_name, current_station in self.stations.items()}

            for current_future in concurrent.futures.as_completed(data_futures):
                current_result = current_future.result()

                if current_result is not None:
                    current_station_name = data_futures[current_future]
                    station_data[current_station_name] = current_result

        schema = {
            'geometry': 'Point', 'properties': {
                'name':                      'str', 'longitude': 'float', 'latitude': 'float',
                'water_temperature':         'float', 'conductivity': 'float', 'salinity': 'float',
                'o2_saturation':             'float', 'dissolved_oxygen': 'float', 'chlorophyll_concentration': 'float',
                'turbidity':                 'float', 'water_ph': 'float', 'water_eh': 'float'
            }
        }

        with fiona.open(output_filename, 'w', 'GPKG', schema, FIONA_WGS84, layer=layer_name) as current_layer:
            print('Creating features...')

            current_layer_records = []

            for current_station_name, current_station_data in station_data.items():
                current_station = self.stations[current_station_name]
                current_lon = float(current_station.longitude)
                current_lat = float(current_station.latitude)

                water_temperature = numpy.ma.mean(current_station_data['water_temperature'])
                conductivity = numpy.ma.mean(current_station_data['conductivity'])
                salinity = numpy.ma.mean(current_station_data['salinity'])
                o2_saturation = numpy.ma.mean(current_station_data['o2_saturation'])
                dissolved_oxygen = numpy.ma.mean(current_station_data['dissolved_oxygen'])
                chlorophyll_concentration = numpy.ma.mean(current_station_data['chlorophyll_concentration'])
                turbidity = numpy.ma.mean(current_station_data['turbidity'])
                water_ph = numpy.ma.mean(current_station_data['water_ph'])
                water_eh = numpy.ma.mean(current_station_data['water_eh'])

                current_record = {
                    'geometry': {'type': 'Point', 'coordinates': (current_lon, current_lat)}, 'properties': {
                        'name':                      current_station_name, 'longitude': current_lon,
                        'latitude':                  current_lat,
                        'water_temperature':         water_temperature if water_temperature is not numpy.ma.masked else 0,
                        'conductivity':              conductivity if conductivity is not numpy.ma.masked else 0,
                        'salinity':                  salinity if salinity is not numpy.ma.masked else 0,
                        'o2_saturation':             o2_saturation if o2_saturation is not numpy.ma.masked else 0,
                        'dissolved_oxygen':          dissolved_oxygen if dissolved_oxygen is not numpy.ma.masked else 0,
                        'chlorophyll_concentration': chlorophyll_concentration if chlorophyll_concentration is not numpy.ma.masked else 0,
                        'turbidity':                 turbidity if turbidity is not numpy.ma.masked else 0,
                        'water_ph':                  water_ph if water_ph is not numpy.ma.masked else 0,
                        'water_eh':                  water_eh if water_eh is not numpy.ma.masked else 0
                    }
                }

                current_layer_records.append(current_record)

            print(f'Writing {output_filename}:{layer_name}')
            current_layer.writerecords(current_layer_records)

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.station_names]

        for current_param in optional_params:
            if current_param is not None:
                if 'str' in str(type(current_param)):
                    current_param = f'"{current_param}"'
                else:
                    current_param = str(current_param)

                used_params.append(current_param)

        return f'{self.__class__.__name__}({", ".join(used_params)})'


def check_station(dataset: netCDF4.Dataset, study_area_polygon_filename: str) -> bool:
    """
    Check whether station exists within the given study area.

    :param dataset: NetCDF Dataset
    :param study_area_polygon_filename:
    :return: Whether station is within study area.
    """

    study_area_polygon_filename, layer_name = study_area_polygon_filename.split(':')

    if layer_name == '':
        layer_name = None

    # construct polygon from the first record in the layer
    with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
        study_area_polygon = shapely.geometry.Polygon(vector_layer.next()['geometry']['coordinates'][0])

    current_lon = dataset['longitude'][:]
    current_lat = dataset['latitude'][:]

    current_point = shapely.geometry.point.Point(current_lon, current_lat)

    return current_point.intersects(study_area_polygon)


if __name__ == '__main__':
    start_datetime = datetime.datetime(2018, 7, 14)
    end_datetime = datetime.datetime.now()

    output_dir = r'C:\Data\ndbc'

    wcofs_stations = list(numpy.genfromtxt(WCOFS_NDBC_STATIONS_FILENAME, dtype='str'))

    # get dataset from source
    ndbc_dataset = NDBC_Range(start_datetime, end_datetime, stations=wcofs_stations)

    date_interval_string = f'{start_datetime.strftime("%m%d%H")}_{end_datetime.strftime("%m%d%H")}'

    # write average vector
    ndbc_dataset.write_vector(os.path.join(output_dir, 'ndbc.gpkg'), f'NDBC_{date_interval_string}')

    print('done')
