# coding=utf-8
"""
RTOFS model output data collection and transformation by interpolation onto Cartesian grid.

Created on Jun 25, 2018

@author: zachary.burnett
"""

from collections import OrderedDict
from datetime import datetime, timedelta, date
import os
import threading
from typing import Collection

import fiona.crs
import numpy
import rasterio.control
from rasterio.crs import CRS
import rasterio.features
import rasterio.mask
import rasterio.warp
from shapely import geometry
import xarray

from PyOFS import CRS_EPSG, DATA_DIRECTORY, LEAFLET_NODATA_VALUE, utilities
from PyOFS.utilities import get_logger

LOGGER = get_logger('PyOFS.RTOFS')

OUTPUT_CRS = fiona.crs.from_epsg(CRS_EPSG)

COORDINATE_VARIABLES = ['time', 'lev', 'lat', 'lon']

DATASET_STRUCTURE = {
    '2ds': {
        'nowcast': {
            'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'], 'diag': ['ssh', 'ice_coverage', 'ice_thickness']
        }, 'forecast': {
            'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'], 'diag': ['ssh', 'ice_coverage', 'ice_thickness']
        }
    }, '3dz': {
        'nowcast': {'salt': ['salinity'], 'temp': ['temperature'], 'uvel': ['u'], 'vvel': ['v']}, 'forecast': {'salt': ['salinity'], 'temp': ['temperature'], 'uvel': ['u'], 'vvel': ['v']}
    }
}

DATA_VARIABLES = {
    'sst': {'2ds': {'prog': 'sst'}, '3dz': {'temp': 'temperature'}},
    'sss': {'2ds': {'prog': 'sss'}, '3dz': {'salt': 'salinity'}},
    'ssu': {'2ds': {'prog': 'u_velocity'}, '3dz': {'uvel': 'u'}},
    'ssv': {'2ds': {'prog': 'v_velocity'}, '3dz': {'vvel': 'v'}},
    'ssh': {'2ds': {'diag': 'ssh'}}
}

TIME_DELTAS = {'daily': range(-3, 8 + 1)}

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIRECTORY, r"reference\wcofs.gpkg:study_area")

SOURCE_URL = 'https://nomads.ncep.noaa.gov:9090/dods/rtofs'

GLOBAL_LOCK = threading.Lock()


class RTOFSDataset:
    """
    Real-Time Ocean Forecasting System (RTOFS) NetCDF observation.
    """

    def __init__(self, model_date: datetime = None, source: str = '2ds', time_interval: str = 'daily', study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME):
        """
        Creates new observation object from datetime and given model parameters.

        :param model_date: model run date
        :param source: rither '2ds' or '3dz'
        :param time_interval: time interval of model output
        :param study_area_polygon_filename: filename of vector file containing study area boundary
        """

        if model_date is None:
            model_date = datetime.now()

        if type(model_date) is date:
            self.model_time = datetime.combine(model_date, datetime.min.time())
        else:
            self.model_time = model_date.replace(hour=0, minute=0, second=0, microsecond=0)

        self.source = source
        self.time_interval = time_interval

        self.study_area_polygon_filename = study_area_polygon_filename
        self.study_area_geojson = utilities.get_first_record(self.study_area_polygon_filename)['geometry']

        self.datasets = {}
        self.dataset_locks = {}

        date_string = f'{self.model_time:%Y%m%d}'

        if self.time_interval == 'daily':
            for forecast_direction, datasets in DATASET_STRUCTURE[self.source].items():
                self.datasets[forecast_direction] = {}
                self.dataset_locks[forecast_direction] = {}

                date_dir = f'rtofs_global{date_string}'

                for dataset_name in datasets:
                    filename = f'rtofs_glo_{self.source}_{forecast_direction}_{self.time_interval}_{dataset_name}'
                    url = f'{SOURCE_URL}/{date_dir}/{filename}'

                    try:
                        dataset = xarray.open_dataset(url)
                        self.datasets[forecast_direction][dataset_name] = dataset
                        self.dataset_locks[forecast_direction][dataset_name] = threading.Lock()
                    except OSError as error:
                        LOGGER.warning(f'{error.__class__.__name__}: {error}')

        if (len(self.datasets['nowcast']) + len(self.datasets['forecast'])) > 0:
            if len(self.datasets['nowcast']) > 0:
                sample_dataset = next(iter(self.datasets['nowcast'].values()))
            else:
                sample_dataset = next(iter(self.datasets['forecast'].values()))

            # for some reason RTOFS has longitude values shifted by 360
            self.raw_lon = sample_dataset['lon'].values
            self.lon = self.raw_lon - 180 - numpy.min(self.raw_lon)
            self.lat = sample_dataset['lat'].values

            lon_pixel_size = sample_dataset['lon'].resolution
            lat_pixel_size = sample_dataset['lat'].resolution

            self.global_west = numpy.min(self.lon)
            self.global_north = numpy.max(self.lat)

            self.global_grid_transform = rasterio.transform.from_origin(self.global_west, self.global_north, lon_pixel_size, lat_pixel_size)

            self.study_area_west, self.study_area_south, self.study_area_east, self.study_area_north = geometry.shape(self.study_area_geojson).bounds

            self.study_area_transform = rasterio.transform.from_origin(self.study_area_west, self.study_area_north, lon_pixel_size, lat_pixel_size)
        else:
            raise utilities.NoDataError(f'No RTOFS datasets found for {self.model_time}.')

    def data(self, variable: str, time: datetime, crop: bool = True) -> xarray.DataArray:
        """
        Get data of specified variable at specified hour.

        :param variable: name of variable to retrieve
        :param time: time from which to retrieve data
        :param crop: whether to crop to study area extent
        :return: array of data
        """

        if time >= self.model_time:
            direction = 'forecast'
        else:
            direction = 'nowcast'

        if self.time_interval == 'daily':
            time = time.replace(hour=0, minute=0, second=0, microsecond=0)

        if direction in DATASET_STRUCTURE[self.source]:
            if len(self.datasets[direction]) > 0:
                if variable in DATA_VARIABLES:
                    datasets = DATA_VARIABLES[variable][self.source]
                    dataset_name, variable_name = next(iter(datasets.items()))

                    with self.dataset_locks[direction][dataset_name]:
                        data_variable = self.datasets[direction][dataset_name][DATA_VARIABLES[variable][self.source][dataset_name]]

                        # TODO study areas that cross over longitude +74.16 may have problems here
                        if crop:
                            selection = data_variable.sel(time=time, method='nearest').sel(lon=slice(self.study_area_west + 360, self.study_area_east + 360),
                                                                                           lat=slice(self.study_area_south, self.study_area_north))
                        else:
                            western_selection = data_variable.sel(time=time, method='nearest').sel(lon=slice(180, numpy.max(self.raw_lon)), lat=slice(numpy.min(self.lat), numpy.max(self.lat)))
                            eastern_selection = data_variable.sel(time=time, method='nearest').sel(lon=slice(numpy.min(self.raw_lon), 180), lat=slice(numpy.min(self.lat), numpy.max(self.lat)))
                            selection = numpy.concatenate((western_selection, eastern_selection), axis=1)

                        selection = numpy.flip(selection.squeeze(), axis=0)
                        return selection
                else:
                    raise ValueError(f'Variable must be not one of {list(DATA_VARIABLES.keys())}.')
            else:
                LOGGER.warning(f'{direction} does not exist in RTOFS observation for {self.model_time:%Y%m%d}.')
        else:
            raise ValueError(f'Direction must be one of {list(DATASET_STRUCTURE[self.source].keys())}.')

    def write_rasters(self, output_dir: str, variables: list, time: datetime, filename_prefix: str = None, filename_suffix: str = None, fill_value=LEAFLET_NODATA_VALUE, driver: str = 'GTiff',
                      crop: bool = True):
        """
        Write averaged raster data of given variables to given output directory.

        :param output_dir: path to directory
        :param variables: variable names to use
        :param time: time from which to retrieve data
        :param filename_prefix: prefix for filenames
        :param filename_suffix: suffix for filenames
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param crop: whether to crop to study area extent
        """

        if variables is None:
            variables = DATA_VARIABLES[self.source]

        if filename_prefix is None:
            filename_prefix = 'rtofs'
        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        if self.time_interval == 'daily':
            time = time.replace(hour=0, minute=0, second=0, microsecond=0)

        time_delta = int((time - self.model_time).total_seconds() / (24 * 60 * 60))
        direction = 'forecast' if time_delta >= 0 else 'nowcast'
        time_delta_string = f'{direction[0]}{abs(time_delta) + 1 if direction == "forecast" else abs(time_delta):03}'

        variable_means = {variable: self.data(variable, time, crop).values for variable in variables if variable not in ['dir', 'mag']}

        if 'dir' in variables or 'mag' in variables:
            u_name = 'ssu'
            v_name = 'ssv'

            if u_name not in variable_means:
                u_data = self.data(u_name, time, crop).values
            else:
                u_data = variable_means[u_name]

            if v_name not in variable_means:
                v_data = self.data(v_name, time, crop).values
            else:
                v_data = variable_means[v_name]

            if u_data is not None and v_data is not None:
                variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

        # write interpolated grids to raster files
        for variable, variable_mean in variable_means.items():
            if variable_mean is not None:
                if crop:
                    transform = self.study_area_transform
                else:
                    transform = self.global_grid_transform

                if fill_value is not None:
                    variable_mean[numpy.isnan(variable_mean)] = fill_value

                gdal_args = {
                    'transform': transform,
                    'height': variable_mean.shape[0],
                    'width': variable_mean.shape[1],
                    'count': 1,
                    'dtype': rasterio.float32,
                    'crs': CRS.from_dict(OUTPUT_CRS),
                    'nodata': numpy.array([fill_value]).astype(variable_mean.dtype).item()
                }

                output_filename = f'{filename_prefix}_{variable}_{self.model_time:%Y%m%d}_{time_delta_string}{filename_suffix}'
                output_filename = os.path.join(output_dir, output_filename)

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

                output_filename = f'{os.path.splitext(output_filename)[0]}.{file_extension}'

                LOGGER.info(f'Writing {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(variable_mean, 1)

    def write_raster(self, output_filename: str, variable: str, time: datetime, fill_value=LEAFLET_NODATA_VALUE, driver: str = 'GTiff', crop: bool = True):
        """
        Writes interpolated raster of given variable to output path.

        :param output_filename: path of raster file to create
        :param variable: name of variable
        :param time: time from which to retrieve data
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param crop: whether to crop to study area extent
        """

        output_data = self.data(variable, time, crop).values

        if output_data is not None:
            if crop:
                transform = self.study_area_transform
            else:
                transform = self.global_grid_transform

            gdal_args = {
                'transform': transform,
                'height': output_data.shape[0],
                'width': output_data.shape[1],
                'count': 1,
                'dtype': rasterio.float32,
                'crs': CRS.from_dict(OUTPUT_CRS),
                'nodata': numpy.array([fill_value]).astype(output_data.dtype).item()
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

            output_filename = f'{os.path.splitext(output_filename)[0]}.{file_extension}'

            LOGGER.info(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(output_data, 1)

    def to_xarray(self, variables: Collection[str] = None, mean: bool = True) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: variables to use
        :param mean: whether to average all time indices
        :return: xarray observation of given variables
        """

        if variables is None:
            variables = ('sst', 'sss', 'ssu', 'ssv', 'ssh')

        output_dataset = xarray.Dataset()

        if self.time_interval == 'daily':
            times = [self.model_time + timedelta(days=day_delta) for day_delta in TIME_DELTAS['daily']]
        else:
            times = None

        coordinates = None
        variables_data = {}

        for variable in variables:
            variable_data = [self.data(variable=variable, time=time) for time in times]

            if mean:
                coordinates = OrderedDict({
                    'lat': variable_data[0]['lat'].values, 'lon': variable_data[0]['lon'].values
                })
            else:
                coordinates = OrderedDict({
                    'time': times, 'lat': variable_data[0]['lat'].values, 'lon': variable_data[0]['lon'].values
                })

            variable_data = numpy.stack(variable_data, axis=0)
            variable_data = numpy.squeeze(variable_data)

            if mean:
                variable_data = numpy.nanmean(variable_data, axis=0)

            variables_data[variable] = variable_data

        for variable, variable_data in variables_data.items():
            output_dataset.update({variable: xarray.DataArray(variable_data, coords=coordinates, dims=tuple(coordinates.keys()))}, inplace=True)

        return output_dataset

    def to_netcdf(self, output_file: str, variables: Collection[str] = None, mean: bool = True):
        """
        Writes to NetCDF file.

        :param output_file: output file to write
        :param mean: whether to average all time indices
        :param variables: variables to use
        """

        self.to_xarray(variables, mean).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.model_time.__repr__()]
        optional_params = [self.source, self.time_interval, self.study_area_polygon_filename]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIRECTORY, r'output\test')

    rtofs_dataset = RTOFSDataset(datetime.now())
    rtofs_dataset.write_raster(os.path.join(output_dir, 'rtofs_ssh.tiff'), 'ssh', datetime.now())

    print('done')
