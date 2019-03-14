# coding=utf-8
"""
Sea surface salinity rasters from ESA SMOS.

Created on Feb 6, 2019

@author: zachary.burnett
"""

import datetime
import logging
import os
from collections import OrderedDict
from typing import Collection

import fiona
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.wkt
import xarray

from PyOFS import CRS_EPSG, DATA_DIR
from PyOFS.dataset import _utilities

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})

SOURCE_URLS = OrderedDict({
    'OpenDAP': OrderedDict({
        'JPL': 'https://thredds.jpl.nasa.gov/thredds/dodsC/ncml_aggregation/SalinityDensity/smap/aggregate__SMAP_JPL_L3_SSS_CAP_MONTHLY_V42.ncml',
    })
})


class SMAPDataset:
    """
    Soil Moisture Active Passive (SMAP) satellite sea-surface salinity.
    """

    study_area_transform = None
    study_area_extent = None
    study_area_bounds = None
    study_area_coordinates = None

    def __init__(self, study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME):
        """
        Retrieve VIIRS NetCDF dataset from NOAA with given datetime.

        :param study_area_polygon_filename: filename of vector file containing study area boundary
        :raises NoDataError: if dataset does not exist
        """

        self.study_area_polygon_filename, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        for source, source_url in SOURCE_URLS['OpenDAP'].items():
            try:
                self.netcdf_dataset = xarray.open_dataset(source_url)
                break
            except Exception as error:
                logging.error(f'Error collecting dataset from {source}: {error}')

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

        if SMAPDataset.study_area_extent is None:
            # get first record in layer
            with fiona.open(self.study_area_polygon_filename, layer=study_area_polygon_layer_name) as vector_layer:
                SMAPDataset.study_area_extent = shapely.geometry.MultiPolygon(
                    [shapely.geometry.Polygon(polygon[0]) for polygon in
                     next(iter(vector_layer))['geometry']['coordinates']])

            SMAPDataset.study_area_bounds = SMAPDataset.study_area_extent.bounds
            SMAPDataset.study_area_transform = rasterio.transform.from_origin(SMAPDataset.study_area_bounds[0],
                                                                              SMAPDataset.study_area_bounds[3],
                                                                              lon_pixel_size, lat_pixel_size)

        if SMAPDataset.study_area_bounds is not None:
            self.netcdf_dataset = self.netcdf_dataset.sel(longitude=slice(SMAPDataset.study_area_bounds[0],
                                                                          SMAPDataset.study_area_bounds[2]),
                                                          latitude=slice(SMAPDataset.study_area_bounds[3],
                                                                         SMAPDataset.study_area_bounds[1]))

        if SMAPDataset.study_area_coordinates is None:
            SMAPDataset.study_area_coordinates = {
                'lon': self.netcdf_dataset['longitude'], 'lat': self.netcdf_dataset['latitude']
            }

    def bounds(self) -> tuple:
        """
        Get coordinate bounds of dataset.

        :return: tuple of bounds (west, south, east, north)
        """

        return self.data_extent.bounds

    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: tuple of cell sizes (x_size, y_size)
        """

        return self.netcdf_dataset.geospatial_lon_resolution, self.netcdf_dataset.geospatial_lat_resolution

    def data(self, data_time: datetime.datetime, variable: str = 'sss') -> numpy.ndarray:
        """
        Retrieve SMOS SSS data.

        :param data_time: datetime to retrieve (only uses month)
        :param variable: SMOS variable to retrieve
        :return: array of data
        """

        output_data = None

        if variable == 'sss':
            output_data = self._sss(data_time)

        return output_data

    def _sss(self, data_time: datetime.datetime) -> numpy.ndarray:
        """
        Retrieve SMOS SSS data.

        :param data_time: datetime to retrieve (only uses month)
        :return: array of data
        """

        # SMOS has data on month-long resolution
        data_time = datetime.datetime(data_time.year, data_time.month, 16)

        if numpy.datetime64(data_time) in self.netcdf_dataset['times'].values:
            return self.netcdf_dataset['smap_sss'].sel(times=data_time).values
        else:
            raise _utilities.NoDataError(f'No data exists for {data_time.strftime("%Y%m%dT%H%M%S")}.')

    def write_rasters(self, output_dir: str, data_time: datetime.datetime,
                      variables: Collection[str] = tuple(['sss']),
                      filename_prefix: str = 'smos', fill_value: float = -9999.0, driver: str = 'GTiff'):
        """
        Write SMOS rasters to file using data from given variables.

        :param output_dir: path to output directory
        :param data_time: datetime to retrieve (only uses month)
        :param variables: variable names to write
        :param filename_prefix: prefix for output filenames
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        """

        for variable in variables:
            input_data = self.data(data_time, variable)

            if input_data is not None and not numpy.isnan(input_data).all():
                if fill_value is not -9999.0:
                    input_data[numpy.isnan(input_data)] = fill_value

                gdal_args = {
                    'height': input_data.shape[0], 'width': input_data.shape[1], 'count': 1,
                    'dtype': rasterio.float32,
                    'crs': RASTERIO_CRS, 'transform': SMAPDataset.study_area_transform, 'nodata': fill_value
                }

                if driver == 'AAIGrid':
                    file_extension = 'asc'
                    gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                elif driver == 'GPKG':
                    file_extension = 'gpkg'
                else:
                    file_extension = 'tiff'
                    gdal_args.update({
                        'TILED': 'YES'
                    })

                output_filename = os.path.join(output_dir, f'{filename_prefix}_{variable}.{file_extension}')

                # use rasterio to write to raster with GDAL args
                logging.info(f'Writing to {output_filename}')
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

    smap_dataset = SMAPDataset()
    smap_dataset.write_rasters(output_dir, datetime.datetime(2018, 12, 1))

    print('done')
