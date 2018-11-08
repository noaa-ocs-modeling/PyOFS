# coding=utf-8
"""
National Data Buoy Center moorings.

Created on Aug 1, 2018

@author: zachary.burnett
"""

from concurrent import futures
import datetime
import os
import re

import fiona
import fiona.crs
import numpy
import rasterio
import requests
import shapely
import shapely.geometry
import xarray

from main import DATA_DIR
from dataset import _utilities

MEASUREMENT_VARIABLES = ['water_temperature', 'conductivity', 'salinity', 'o2_saturation', 'dissolved_oxygen',
                         'chlorophyll_concentration', 'turbidity', 'water_ph', 'water_eh']

FIONA_WGS84 = fiona.crs.from_epsg(4326)
RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})

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
            self.netcdf_dataset = xarray.open_dataset(self.url)
            self.longitude = self.netcdf_dataset['longitude'].values.item()
            self.latitude = self.netcdf_dataset['latitude'].values.item()
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
        
        output_data = {variable: None for variable in MEASUREMENT_VARIABLES}
        
        time_data = self.netcdf_dataset['time'].values
        
        start_index = numpy.searchsorted(time_data, numpy.datetime64(start_datetime))
        end_index = numpy.searchsorted(time_data, numpy.datetime64(end_datetime))
        
        if end_index - start_index > 0:
            # concurrently populate dictionary with data for each variable
            for variable in MEASUREMENT_VARIABLES:
                output_data[variable] = self.netcdf_dataset[variable][start_index:end_index, :, :].values
        
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
        with futures.ThreadPoolExecutor() as concurrency_pool:
            station_futures = {concurrency_pool.submit(NDBC_Station, station_name): station_name for station_name in
                               self.station_names}
            
            for completed_future in futures.as_completed(station_futures):
                station_name = station_futures[completed_future]
                
                if type(completed_future.exception()) is not _utilities.NoDataError:
                    result = completed_future.result()
                    print(f'Collecting NDBC data from station {station_name}...')
                    self.stations[station_name] = result
        
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
        with futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {concurrency_pool.submit(station.data, start_datetime, end_datetime): station_name for
                            station_name, station in self.stations.items()}
            
            for completed_future in futures.as_completed(data_futures):
                result = completed_future.result()
                
                if result is not None:
                    station_name = data_futures[completed_future]
                    station_data[station_name] = result
        
        schema = {
            'geometry': 'Point', 'properties': {
                'name': 'str', 'longitude': 'float', 'latitude': 'float',
                'water_temperature': 'float', 'conductivity': 'float', 'salinity': 'float',
                'o2_saturation': 'float', 'dissolved_oxygen': 'float', 'chlorophyll_concentration': 'float',
                'turbidity': 'float', 'water_ph': 'float', 'water_eh': 'float'
            }
        }
        
        with fiona.open(output_filename, 'w', 'GPKG', schema, FIONA_WGS84, layer=layer_name) as layer:
            print('Creating features...')
            
            layer_records = []
            
            for station_name, station_data in station_data.items():
                station = self.stations[station_name]
                longitude = float(station.longitude)
                latitude = float(station.latitude)
                
                water_temperature = float(numpy.ma.mean(station_data['water_temperature']))
                conductivity = float(numpy.ma.mean(station_data['conductivity']))
                salinity = float(numpy.ma.mean(station_data['salinity']))
                o2_saturation = float(numpy.ma.mean(station_data['o2_saturation']))
                dissolved_oxygen = float(numpy.ma.mean(station_data['dissolved_oxygen']))
                chlorophyll_concentration = float(numpy.ma.mean(station_data['chlorophyll_concentration']))
                turbidity = float(numpy.ma.mean(station_data['turbidity']))
                water_ph = float(numpy.ma.mean(station_data['water_ph']))
                water_eh = float(numpy.ma.mean(station_data['water_eh']))
                
                record = {
                    'geometry': {'type': 'Point', 'coordinates': (longitude, latitude)}, 'properties': {
                        'name': station_name, 'longitude': longitude, 'latitude': latitude,
                        'water_temperature': water_temperature, 'conductivity': conductivity,
                        'salinity': salinity, 'o2_saturation': o2_saturation,
                        'dissolved_oxygen': dissolved_oxygen,
                        'chlorophyll_concentration': chlorophyll_concentration, 'turbidity': turbidity,
                        'water_ph': water_ph, 'water_eh': water_eh
                    }
                }
                
                layer_records.append(record)
            
            print(f'Writing {output_filename}:{layer_name}')
            layer.writerecords(layer_records)
    
    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
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
    :param study_area_polygon_filename:
    :return: Whether station is within study area.
    """
    
    study_area_polygon_filename, layer_name = study_area_polygon_filename.split(':')
    
    if layer_name == '':
        layer_name = None
    
    # construct polygon from the first record in the layer
    with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
        study_area_polygon = shapely.geometry.Polygon(next(iter(vector_layer))['geometry']['coordinates'][0])
    
    lon = dataset['longitude'][:]
    lat = dataset['latitude'][:]
    
    point = shapely.geometry.point.Point(lon, lat)
    
    return point.intersects(study_area_polygon)


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test')
    
    start_datetime = datetime.datetime(2018, 7, 14)
    end_datetime = datetime.datetime.now()
    
    wcofs_stations = list(numpy.genfromtxt(WCOFS_NDBC_STATIONS_FILENAME, dtype='str'))
    
    # get dataset from source
    ndbc_dataset = NDBC_Range(start_datetime, end_datetime, stations=wcofs_stations)
    
    date_interval_string = f'{start_datetime.strftime("%m%d%H")}_{end_datetime.strftime("%m%d%H")}'
    
    # write average vector
    ndbc_dataset.write_vector(os.path.join(output_dir, 'ndbc.gpkg'), f'NDBC_{date_interval_string}')
    
    print('done')
