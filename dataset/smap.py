# coding=utf-8
"""
Sea surface salinity rasters from ESA SMOS.

Created on Feb 6, 2019

@author: zachary.burnett
"""

from collections import OrderedDict
import datetime
import os

import fiona
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.wkt
import xarray

from dataset import _utilities
from main import DATA_DIR

try:
    from logbook import Logger
except ImportError:
    class Logger(object):
        def __init__(self, name, level=0):
            self.name = name
            self.level = level
        
        debug = info = warn = warning = notice = error = exception = \
            critical = log = lambda *a, **kw: None

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})

SOURCE_URLS = OrderedDict({
    'OpenDAP': OrderedDict({
        'JPL': 'https://thredds.jpl.nasa.gov/thredds/dodsC/ncml_aggregation/SalinityDensity/smap/aggregate__SMAP_JPL_L3_SSS_CAP_MONTHLY_V42.ncml',
    })
})


class SMAP_Dataset:
    study_area_transform = None
    study_area_extent = None
    study_area_bounds = None
    study_area_coordinates = None

    def __init__(self, study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, logger: Logger = None):
        """
        Retrieve VIIRS NetCDF dataset from NOAA with given datetime.

        :param study_area_polygon_filename: filename of vector file containing study area boundary
        :param logger: logbook logger
        :raises NoDataError: if dataset does not exist.
        """

        self.logger = logger
        
        self.study_area_polygon_filename, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)
        
        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None
        
        for source, source_url in SOURCE_URLS['OpenDAP'].items():
            try:
                self.netcdf_dataset = xarray.open_dataset(source_url)
                break
            except Exception as error:
                if self.logger is not None:
                    self.logger.error(f'Error collecting dataset from {source}: {error}')
        
        # construct rectangular polygon of granule extent
        lon_min = float(self.netcdf_dataset.geospatial_lon_min)
        lon_max = float(self.netcdf_dataset.geospatial_lon_max)
        lat_min = float(self.netcdf_dataset.geospatial_lat_min)
        lat_max = float(self.netcdf_dataset.geospatial_lat_max)
        
        if lon_min < lon_max:
            self.data_extent = shapely.geometry.Polygon(
                [(lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min), (lon_min, lat_min)])
        else:
            # geospatial bounds cross the antimeridian, so we create a multipolygon
            self.data_extent = shapely.geometry.MultiPolygon([
                shapely.geometry.Polygon(
                    [(lon_min, lat_max), (180, lat_max), (180, lat_min), (lon_min, lat_min)]),
                shapely.geometry.Polygon(
                    [(-180, lat_max), (lon_max, lat_max), (lon_max, lat_min), (-180, lat_min)])])
        
        lon_pixel_size = numpy.mean(numpy.diff(self.netcdf_dataset['longitude'].values))
        lat_pixel_size = numpy.mean(numpy.diff(self.netcdf_dataset['latitude'].values))
        
        if SMAP_Dataset.study_area_extent is None:
            # get first record in layer
            with fiona.open(self.study_area_polygon_filename, layer=study_area_polygon_layer_name) as vector_layer:
                SMAP_Dataset.study_area_extent = shapely.geometry.MultiPolygon(
                    [shapely.geometry.Polygon(polygon[0]) for polygon in
                     next(iter(vector_layer))['geometry']['coordinates']])
            
            SMAP_Dataset.study_area_bounds = SMAP_Dataset.study_area_extent.bounds
            SMAP_Dataset.study_area_transform = rasterio.transform.from_origin(SMAP_Dataset.study_area_bounds[0],
                                                                               SMAP_Dataset.study_area_bounds[3],
                                                                               lon_pixel_size, lat_pixel_size)
        
        if SMAP_Dataset.study_area_bounds is not None:
            self.netcdf_dataset = self.netcdf_dataset.sel(longitude=slice(SMAP_Dataset.study_area_bounds[0],
                                                                          SMAP_Dataset.study_area_bounds[2]),
                                                          latitude=slice(SMAP_Dataset.study_area_bounds[3],
                                                                         SMAP_Dataset.study_area_bounds[1]))
        
        if SMAP_Dataset.study_area_coordinates is None:
            SMAP_Dataset.study_area_coordinates = {
                'lon': self.netcdf_dataset['longitude'], 'lat': self.netcdf_dataset['latitude']
            }
    
    def bounds(self) -> tuple:
        """
        Get coordinate bounds of dataset.

        :return: Tuple of bounds (west, south, east, north)
        """
        
        return self.data_extent.bounds
    
    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: Tuple of cell sizes (x_size, y_size)
        """
        
        return self.netcdf_dataset.geospatial_lon_resolution, self.netcdf_dataset.geospatial_lat_resolution
    
    def data(self, data_datetime: datetime.datetime, variable: str = 'sss') -> numpy.ndarray:
        """
        Retrieve SMOS SSS data.
        
        :param data_datetime: datetime to retrieve (only uses month)
        :param variable: SMOS variable to retrieve
        :return: array of data
        """
        
        output_data = None
        
        if variable == 'sss':
            output_data = self._sss(data_datetime)
        
        return output_data
    
    def _sss(self, data_datetime: datetime.datetime) -> numpy.ndarray:
        """
        Retrieve SMOS SSS data.

        :param data_datetime: datetime to retrieve (only uses month)
        :return: array of data
        """
        
        # SMOS has data on month-long resolution
        data_datetime = datetime.datetime(data_datetime.year, data_datetime.month, 16)
        
        if numpy.datetime64(data_datetime) in self.netcdf_dataset['times']:
            return self.netcdf_dataset['smap_sss'].sel(times=data_datetime).values
        else:
            raise _utilities.NoDataError('No data exists for that time.')
    
    def write_rasters(self, output_dir: str, data_datetime: datetime.datetime, variables: list = ['sss'],
                      filename_prefix: str = 'smos', fill_value: float = -9999.0, drivers: list = ['GTiff']):
        """
        Write SMOS rasters to file using data from given variables.

        :param output_dir: path to output directory
        :param data_datetime: datetime to retrieve (only uses month)
        :param variables: list of variable names to write
        :param filename_prefix: prefix for output filenames
        :param fill_value: desired fill value of output
        :param drivers: list of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        """
        
        for variable in variables:
            input_data = self.data(data_datetime, variable)
            
            if input_data is not None and not numpy.isnan(input_data).all():
                if fill_value is not -9999.0:
                    input_data[numpy.isnan(input_data)] = fill_value
                
                gdal_args = {
                    'height': input_data.shape[0], 'width': input_data.shape[1], 'count': 1, 'dtype': rasterio.float32,
                    'crs': RASTERIO_WGS84, 'transform': SMAP_Dataset.study_area_transform, 'nodata': fill_value
                }
                
                for driver in drivers:
                    if driver == 'AAIGrid':
                        file_extension = 'asc'
                        gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                    elif driver == 'GPKG':
                        file_extension = 'gpkg'
                    else:
                        file_extension = 'tiff'
                    
                    output_filename = os.path.join(output_dir, f'{filename_prefix}_{variable}.{file_extension}')
                    
                    # use rasterio to write to raster with GDAL args
                    if self.logger is not None:
                        self.logger.notice(f'Writing to {output_filename}')
                    with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                        output_raster.write(input_data, 1)
    
    def __repr__(self):
        used_params = []
        optional_params = [self.study_area_polygon_filename]
        
        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)
                
                used_params.append(param)
        
        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test')
    
    smap_dataset = SMAP_Dataset()
    smap_dataset.write_rasters(output_dir, datetime.datetime(2018, 12, 1))
    
    print('done')
