# coding=utf-8
"""
WCOFS model output data collection and transformation by interpolation onto Cartesian grid.

Created on Jun 25, 2018

@author: zachary.burnett
"""

import datetime
import logging
import os
import threading
from concurrent import futures
from typing import Collection

import fiona
import fiona.crs
import numpy
import rasterio.control
import rasterio.mask
import rasterio.warp
import shapely.geometry
import xarray
from rasterio.io import MemoryFile
from scipy import interpolate

from PyOFS import CRS_EPSG, DATA_DIR, utilities

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})
FIONA_CRS = fiona.crs.from_epsg(CRS_EPSG)

GRID_LOCATIONS = {'face': 'rho', 'edge1': 'u', 'edge2': 'v', 'node': 'psi'}
COORDINATE_VARIABLES = ['grid', 'ocean_time', 'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v', 'lon_psi',
                        'lat_psi',
                        'angle', 'pm', 'pn']
STATIC_VARIABLES = ['h', 'f', 'mask_rho', 'mask_u', 'mask_v', 'mask_psi']
DATA_VARIABLES = {
    'sst': {'2ds': 'temp_sur', 'avg': 'temp'},
    'ssu': {'2ds': 'u_sur', 'avg': 'u'},
    'ssv': {'2ds': 'v_sur', 'avg': 'v'},
    'sss': {'2ds': 'salt_sur', 'avg': 'salt'},
    'ssh': {'2ds': 'zeta', 'avg': 'zeta'}
}

WCOFS_MODEL_HOURS = {'n': -24, 'f': 72}
WCOFS_MODEL_RUN_HOUR = 3

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs.gpkg:study_area')
WCOFS_4KM_GRID_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs_4km_grid.nc')
WCOFS_2KM_GRID_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs_2km_grid.nc')
VALID_SOURCE_STRINGS = ['stations', 'fields', 'avg', '2ds']

GLOBAL_LOCK = threading.Lock()

SOURCE_URLS = ['https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS',
               os.path.join(DATA_DIR, 'input/wcofs/avg')]


class WCOFSDataset:
    """
    West Coast Ocean Forecasting System (WCOFS) NetCDF dataset.
    """

    grid_transforms = None
    grid_shapes = None
    grid_bounds = None
    data_coordinates = None
    variable_grids = None
    masks = None
    angle = None

    def __init__(self, model_date: datetime.datetime = None, source: str = None,
                 time_deltas: list = None, x_size: float = None, y_size: float = None, grid_filename: str = None,
                 source_url: str = None, wcofs_string: str = 'wcofs'):
        """
        Creates new dataset object from datetime and given model parameters.

        :param model_date: model run date
        :param source: one of 'stations', 'fields', 'avg', or '2ds'
        :param time_deltas: integers of times from model start for which to retrieve data (days for avg, hours for others)
        :param x_size: size of cell in X direction
        :param y_size: size of cell in Y direction
        :param grid_filename: filename of NetCDF containing WCOFS grid coordinates
        :param source_url: directory containing NetCDF files
        :param wcofs_string: WCOFS string in filename
        :raises ValueError: if source is not valid
        :raises NoDataError: if no datasets exist for the given model run
        """

        if model_date is None:
            model_date = datetime.datetime.now()

        # set start time to WCOFS model run time (0300 UTC)
        if type(model_date) is datetime.date:
            self.model_time = datetime.datetime.combine(model_date,
                                                        datetime.datetime.min.time()) + datetime.timedelta(
                hours=3)
        else:
            self.model_time = model_date.replace(hour=3, minute=0, second=0, microsecond=0)

        if source is None:
            source = 'avg'
        elif source not in VALID_SOURCE_STRINGS:
            raise ValueError(f'Location must be one of {VALID_SOURCE_STRINGS}')

        self.source = source

        # 'avg' is daily average datasets
        if time_deltas is None:
            if self.source == 'avg':
                self.time_deltas = [-1, 0, 1, 2]
            else:
                self.time_deltas = list(range(WCOFS_MODEL_HOURS['n'], WCOFS_MODEL_HOURS['f'] + 1))
        else:
            self.time_deltas = time_deltas

        if source_url is None:
            source_url = SOURCE_URLS[0]

        if grid_filename is None:
            grid_filename = WCOFS_4KM_GRID_FILENAME

        self.x_size = x_size
        self.y_size = y_size
        self.grid_filename = grid_filename
        self.wcofs_string = wcofs_string

        month_string = self.model_time.strftime('%Y%m')
        date_string = self.model_time.strftime('%Y%m%d')

        self.netcdf_datasets = {}

        for source in SOURCE_URLS:
            if self.source == 'avg':
                for day in self.time_deltas:
                    if (day < 0 and -1 in self.netcdf_datasets.keys()) or (
                            day >= 0 and 1 in self.netcdf_datasets.keys()):
                        continue

                    model_type = 'nowcast' if day < 0 else 'forecast'
                    url = f'{source}/{month_string}/nos.{self.wcofs_string}.avg.{model_type}.{date_string}.t{WCOFS_MODEL_RUN_HOUR:02}z.nc'

                    try:
                        self.netcdf_datasets[-1 if day < 0 else 1] = xarray.open_dataset(url, decode_times=False)
                        self.source_url = source
                    except OSError:
                        logging.warning(f'No WCOFS dataset found at {url}')
            else:
                for hour in self.time_deltas:
                    model_type = 'n' if hour <= 0 else 'f'
                    url = f'{source}/{month_string}/nos.{self.wcofs_string}.{self.source}.{model_type}' + \
                          f'{abs(hour):03}.{date_string}.t{WCOFS_MODEL_RUN_HOUR:02}z.nc'

                    try:
                        self.netcdf_datasets[hour] = xarray.open_dataset(url, decode_times=False)
                        self.source_url = source
                    except OSError:
                        logging.warning(f'No WCOFS dataset found at {url}')

        if len(self.netcdf_datasets) > 0:
            self.dataset_locks = {time_delta: threading.Lock() for time_delta in self.netcdf_datasets.keys()}

            sample_dataset = next(iter(self.netcdf_datasets.values()))

            with GLOBAL_LOCK:
                if WCOFSDataset.variable_grids is None:
                    WCOFSDataset.variable_grids = {}

                    variable_names = {}

                    for data_variable, source_variables in DATA_VARIABLES.items():
                        variable_names[source_variables[self.source]] = data_variable

                    for netcdf_variable_name, netcdf_variable in sample_dataset.data_vars.items():
                        if 'location' in netcdf_variable.attrs:
                            grid_name = GRID_LOCATIONS[netcdf_variable.location]

                            variable_name = netcdf_variable_name

                            for data_variable, source_variables in DATA_VARIABLES.items():
                                if source_variables[self.source] == netcdf_variable_name:
                                    variable_name = data_variable
                                    break

                            WCOFSDataset.variable_grids[variable_name] = grid_name
                        elif netcdf_variable_name in variable_names:
                            grid_name = netcdf_variable_name if netcdf_variable_name in ['u', 'v'] else 'rho'
                            WCOFSDataset.variable_grids[variable_names[netcdf_variable_name]] = grid_name

            with GLOBAL_LOCK:
                if WCOFSDataset.data_coordinates is None:
                    WCOFSDataset.data_coordinates = {}
                    WCOFSDataset.masks = {}

                    wcofs_grid = xarray.open_dataset(self.grid_filename, decode_times=False)

                    for grid_name in GRID_LOCATIONS.values():
                        WCOFSDataset.masks[grid_name] = ~(wcofs_grid[f'mask_{grid_name}'].values.astype(bool))

                        lon = wcofs_grid[f'lon_{grid_name}'].values
                        lat = wcofs_grid[f'lat_{grid_name}'].values

                        WCOFSDataset.data_coordinates[grid_name] = {}
                        WCOFSDataset.data_coordinates[grid_name]['lon'] = lon
                        WCOFSDataset.data_coordinates[grid_name]['lat'] = lat

                    WCOFSDataset.angle = wcofs_grid['angle'].values

            with GLOBAL_LOCK:
                if WCOFSDataset.grid_shapes is None:
                    WCOFSDataset.grid_shapes = {}

                    for grid_name in GRID_LOCATIONS.values():
                        WCOFSDataset.grid_shapes[grid_name] = WCOFSDataset.data_coordinates[grid_name]['lon'].shape

            # set pixel resolution if not specified
            if self.x_size is None:
                self.x_size = numpy.max(numpy.diff(WCOFSDataset.data_coordinates['psi']['lon']))
            if self.y_size is None:
                self.y_size = numpy.max(numpy.diff(WCOFSDataset.data_coordinates['psi']['lat']))

            with GLOBAL_LOCK:
                if WCOFSDataset.grid_transforms is None:
                    WCOFSDataset.grid_transforms = {}

                    for grid_name in GRID_LOCATIONS.values():
                        lon = WCOFSDataset.data_coordinates[grid_name]['lon']
                        lat = WCOFSDataset.data_coordinates[grid_name]['lat']

                        west = numpy.min(lon)
                        south = numpy.min(lat)

                        WCOFSDataset.grid_transforms[grid_name] = rasterio.transform.from_origin(west=west,
                                                                                                 north=south,
                                                                                                 xsize=self.x_size,
                                                                                                 ysize=-self.y_size)

            with GLOBAL_LOCK:
                if WCOFSDataset.grid_bounds is None and 'wcofs_grid' in locals():
                    WCOFSDataset.grid_bounds = {}

                    for grid_name in GRID_LOCATIONS.values():
                        lon = WCOFSDataset.data_coordinates[grid_name]['lon']
                        lat = WCOFSDataset.data_coordinates[grid_name]['lat']

                        west = numpy.min(lon)
                        north = numpy.max(lat)
                        east = numpy.max(lon)
                        south = numpy.min(lat)

                        WCOFSDataset.grid_bounds[grid_name] = (west, north, east, south)

        else:
            raise utilities.NoDataError(
                f'No WCOFS datasets found for {self.model_time} at the given time deltas ({self.time_deltas}).')

    def bounds(self, variable: str = 'psi') -> tuple:
        """
        Returns bounds of grid of given variable.

        :param variable: variable name
        :return: tuple of (west, north, east, south)
        """

        grid_name = WCOFSDataset.variable_grids[variable]
        return WCOFSDataset.grid_bounds[grid_name]

    def data(self, variable: str, time_delta: int) -> numpy.ndarray:
        """
        Get data of specified variable at specified hour.

        :param variable: name of variable to retrieve
        :param time_delta: time index to retrieve (days for avg, hours for others)
        :return: array of data
        """

        output_data = None

        if time_delta in self.time_deltas:
            if self.source == 'avg':
                if time_delta >= 0:
                    dataset_index = 1
                    day_index = time_delta
                else:
                    dataset_index = -1
                    day_index = 0

                if dataset_index in self.dataset_locks:
                    with self.dataset_locks[dataset_index]:
                        # get surface layer; the last layer (of 40) at dimension 1
                        if variable in ['ssu', 'ssv']:
                            # correct for angles
                            raw_u = self.netcdf_datasets[dataset_index][DATA_VARIABLES['ssu'][self.source]][day_index,
                                    -1, :-1, :].values
                            raw_v = self.netcdf_datasets[dataset_index][DATA_VARIABLES['ssv'][self.source]][day_index,
                                    -1, :, :-1].values
                            theta = WCOFSDataset.angle[:-1, :-1]

                            if variable == 'ssu':
                                output_data = raw_u * numpy.cos(theta) - raw_v * numpy.sin(theta)
                                extra_row = numpy.empty((1, output_data.shape[1]), dtype=output_data.dtype)
                                extra_row[:] = numpy.nan
                                output_data = numpy.concatenate((output_data, extra_row), axis=0)
                            elif variable == 'ssv':
                                output_data = raw_u * numpy.sin(theta) + raw_v * numpy.cos(theta)
                                extra_column = numpy.empty((output_data.shape[0], 1), dtype=output_data.dtype)
                                extra_column[:] = numpy.nan
                                output_data = numpy.concatenate((output_data, extra_column), axis=1)
                        else:
                            data_variable = self.netcdf_datasets[dataset_index][DATA_VARIABLES[variable][self.source]]
                            if len(data_variable.shape) == 3:
                                output_data = data_variable[day_index, :, :].values
                            if len(data_variable.shape) == 4:
                                output_data = data_variable[day_index, -1, :, :].values
            else:
                with self.dataset_locks[time_delta]:
                    output_data = self.netcdf_datasets[time_delta][DATA_VARIABLES[variable][self.source]][0, :,
                                  :].values

        return output_data

    def data_average(self, variable: str, time_deltas: list = None) -> numpy.ndarray:
        """
        Gets average of data from given time deltas.

        :param variable: variable to use
        :param time_deltas: integers of time indices to use in average (days for avg, hours for others)
        :return: array of data
        """

        time_deltas = time_deltas if time_deltas is not None else self.netcdf_datasets.keys()

        variable_data = []

        # concurrently populate array with data for every hour
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(self.data, variable, time_delta): time_delta for time_delta in
                               time_deltas}

            for completed_future in futures.as_completed(running_futures):
                result = completed_future.result()

                if result is not None:
                    variable_data.append(result)

            del running_futures

        if len(variable_data) > 0:
            variable_data = numpy.mean(numpy.stack(variable_data), axis=0)
        else:
            variable_data = None

        return variable_data

    def write_rasters(self, output_dir: str, variables: Collection[str] = None, filename_suffix: str = None,
                      time_deltas: list = None, study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                      driver: str = 'GTiff'):
        """
        Write averaged raster data of given variables to given output directory.

        :param output_dir: path to directory
        :param variables: variable names to use
        :param time_deltas: time indices to write
        :param filename_suffix: suffix for filenames
        :param study_area_polygon_filename: path to vector file containing study area boundary
        :param x_size: cell size of output grid in X direction
        :param y_size: cell size of output grid in Y direction
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        """

        start_time = datetime.datetime.now()

        if variables is None:
            variables = list(DATA_VARIABLES.keys())

        study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
            study_area_geojson = next(iter(vector_layer))['geometry']

        if x_size is None or y_size is None:
            sample_dataset = next(iter(self.netcdf_datasets.values()))

            # if cell sizes are not specified, get maximum coordinate differences between cell points on psi grid
            if x_size is None:
                x_size = numpy.max(numpy.diff(sample_dataset['lon_psi'][:]))
            if y_size is None:
                y_size = numpy.max(numpy.diff(sample_dataset['lat_psi'][:]))

        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        grid_variables = list(variables)

        if 'dir' in variables or 'mag' in variables:
            self.variable_grids['dir'] = 'rho'
            self.variable_grids['mag'] = 'rho'
            grid_variables += ['dir', 'mag']

            for required_variable in ['ssu', 'ssv']:
                if required_variable not in grid_variables:
                    grid_variables.append(required_variable)

        output_grid_coordinates = {}

        for variable in grid_variables:
            output_grid_coordinates[variable] = {}

            grid_name = self.variable_grids[variable]

            west = self.grid_bounds[grid_name][0]
            north = self.grid_bounds[grid_name][1]
            east = self.grid_bounds[grid_name][2]
            south = self.grid_bounds[grid_name][3]

            # WCOFS grid starts at southwest corner
            output_grid_coordinates[variable]['lon'] = numpy.arange(west, east, x_size)
            output_grid_coordinates[variable]['lat'] = numpy.arange(south, north, y_size)

        variable_means = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(self.data_average, variable, time_deltas): variable for variable
                               in variables if variable not in ['dir', 'mag']}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                result = completed_future.result()

                if result is not None:
                    variable_means[variable] = result

            del running_futures

        if 'dir' in variables or 'mag' in variables:
            u_name = 'ssu'
            v_name = 'ssv'

            if u_name not in variable_means:
                u_data = self.data_average(u_name, time_deltas)

                if u_data is not None:
                    variable_means[u_name] = u_data

            if v_name not in variable_means:
                v_data = self.data_average(v_name, time_deltas)

                if v_data is not None:
                    variable_means[v_name] = v_data

        logging.debug('parallel data aggregation took ' +
                      f'{(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        start_time = datetime.datetime.now()

        interpolated_data = {}

        if len(variable_means) > 0:
            # concurrently populate dictionary with interpolated data in given grid for each variable
            with futures.ThreadPoolExecutor() as concurrency_pool:
                running_futures = {}

                for variable, variable_data in variable_means.items():
                    if variable_data is not None:
                        logging.debug(f'Starting {variable} interpolation...')

                        grid_lon = output_grid_coordinates[variable]['lon']
                        grid_lat = output_grid_coordinates[variable]['lat']

                        grid_name = self.variable_grids[variable]

                        lon = self.data_coordinates[grid_name]['lon']
                        lat = self.data_coordinates[grid_name]['lat']

                        if len(grid_lon) > 0:
                            running_future = concurrency_pool.submit(interpolate_grid, lon, lat, variable_data,
                                                                     grid_lon, grid_lat)
                            running_futures[running_future] = variable

                for completed_future in futures.as_completed(running_futures):
                    variable = running_futures[completed_future]
                    result = completed_future.result()

                    if result is not None:
                        interpolated_data[variable] = result

                del running_futures

            if 'dir' in variables or 'mag' in variables:
                if 'ssu' in interpolated_data and 'ssv' in interpolated_data:
                    u_name = 'ssu'
                    v_name = 'ssv'

                    interpolated_data['dir'] = {}
                    interpolated_data['mag'] = {}

                    u_data = interpolated_data[u_name]
                    v_data = interpolated_data[v_name]

                    # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                    interpolated_data['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                    interpolated_data['mag'] = numpy.sqrt(u_data ** 2 + v_data ** 2)

                    if u_name not in variables:
                        del interpolated_data[u_name]

                    if v_name not in variables:
                        del interpolated_data[v_name]

            logging.debug(f'parallel grid interpolation took ' +
                          f'{(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        # write interpolated grids to raster files
        for variable, variable_data in interpolated_data.items():
            west = numpy.min(output_grid_coordinates[variable]['lon'])
            north = numpy.max(output_grid_coordinates[variable]['lat'])

            # WCOFS grid starts from southwest corner, but we'll be flipping the data in a moment so lets use the northwest point
            # TODO NOTE: You cannot use a negative (northward) y_size here, otherwise GDALSetGeoTransform will break at line 1253 of rasterio/_io.pyx
            if x_size is not None and y_size is not None:
                grid_transform = rasterio.transform.from_origin(west, north, x_size, y_size)
            else:
                grid_name = self.variable_grids[variable]
                grid_transform = self.grid_transforms[grid_name]

            # flip the data to ensure northward y_size (see comment above)
            raster_data = numpy.flip(variable_data.astype(rasterio.float32), axis=0)

            gdal_args = {
                'width': raster_data.shape[1], 'height': raster_data.shape[0], 'count': 1,
                'crs': RASTERIO_CRS, 'transform': grid_transform, 'dtype': raster_data.dtype,
                'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
            }

            with MemoryFile() as memory_file:
                with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                    memory_raster.write(raster_data, 1)

                with memory_file.open() as memory_raster:
                    masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

            masked_data = masked_data[0, :, :]

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

            output_filename = os.path.join(output_dir,
                                           f'wcofs_{variable}_{self.model_time.strftime("%Y%m%d")}' +
                                           f'{filename_suffix}.{file_extension}')

            if os.path.isfile(output_filename):
                os.remove(output_filename)

            logging.info(f'Writing to {output_filename}')
            with rasterio.open(output_filename, mode='w', driver=driver, **gdal_args) as output_raster:
                output_raster.write(masked_data, 1)

    def write_vector(self, output_filename: str, layer_name: str = None, time_deltas: list = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to the provided output file.

        :param output_filename: path to output file
        :param layer_name: name of layer to write
        :param time_deltas: integers of hours to use in average
        """

        variables = list(DATA_VARIABLES.keys())

        start_time = datetime.datetime.now()

        variable_means = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(self.data_average, variable, time_deltas): variable for variable
                               in variables}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                variable_means[variable] = completed_future.result()

            del running_futures

        logging.debug('parallel data aggregation took ' +
                      f'{(datetime.datetime.now() - start_time).total_seconds(): .2f} seconds')

        schema = {
            'geometry': 'Point', 'properties': {
                'row': 'int', 'col': 'int', 'rho_lon': 'float', 'rho_lat': 'float'
            }
        }

        for variable in variables:
            schema['properties'][variable] = 'float'

        start_time = datetime.datetime.now()

        logging.debug('Creating records...')

        # create features
        layer_records = []

        grid_height, grid_width = WCOFSDataset.data_coordinates['psi']['lon'].shape

        with futures.ThreadPoolExecutor() as concurrency_pool:
            feature_index = 1
            running_futures = []

            for col in range(grid_width):
                for row in range(grid_height):
                    if WCOFSDataset.masks['psi'][row, col] == 0:
                        # check if current record is unmasked
                        running_futures.append(
                            concurrency_pool.submit(self._create_fiona_record, variable_means, row, col,
                                                    feature_index))
                        feature_index += 1

            for completed_future in futures.as_completed(running_futures):
                result = completed_future.result()

                if result is not None:
                    layer_records.append(result)

        logging.debug(
            f'creating records took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        start_time = datetime.datetime.now()

        logging.info(f'Writing {output_filename}:{layer_name}')

        # create layer
        with fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_CRS,
                        layer=layer_name) as output_vector_file:
            output_vector_file.writerecords(layer_records)

        logging.debug(
            f'writing records took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

    def _create_fiona_record(self, variable_means, row, col, feature_index):
        # get coordinates of cell center
        rho_lon = WCOFSDataset.data_coordinates['rho']['lon'][row, col]
        rho_lat = WCOFSDataset.data_coordinates['rho']['lat'][row, col]

        record = {
            'geometry': {
                'id': feature_index, 'type': 'Point', 'coordinates': (float(rho_lon), float(rho_lat))
            }, 'properties': {
                'row': row, 'col': col, 'rho_lon': float(rho_lon), 'rho_lat': float(rho_lat)
            }
        }

        for variable in variable_means.keys():
            record['properties'][variable] = float(variable_means[variable][row, col])

        return record

    def to_xarray(self, variables: Collection[str] = None) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: variables to use
        :return: xarray dataset of given variables
        """

        output_dataset = xarray.Dataset()

        if variables is None:
            variables = list(DATA_VARIABLES.keys())

        for variable in variables:
            if self.source is 'avg':
                times = [self.model_time + datetime.timedelta(days=time_delta) for time_delta in self.time_deltas]
            else:
                times = [self.model_time + datetime.timedelta(hours=time_delta) for time_delta in self.time_deltas]

            grid = self.variable_grids[variable]
            data_stack = numpy.stack([self.data(variable, time_delta) for time_delta in self.time_deltas], axis=0)
            data_array = xarray.DataArray(data_stack,
                                          coords={
                                              'time': times,
                                              'lon': ((f'{grid}_eta', f'{grid}_xi'),
                                                      self.data_coordinates[grid]['lon']),
                                              'lat': ((f'{grid}_eta', f'{grid}_xi'),
                                                      self.data_coordinates[grid]['lat'])
                                          },
                                          dims=('time', f'{grid}_eta', f'{grid}_xi'))

            data_array.attrs['grid'] = grid
            output_dataset = output_dataset.update({variable: data_array})

        return output_dataset

    def to_netcdf(self, output_file: str, variables: Collection[str] = None):
        """
        Writes to NetCDF file.

        :param output_file: output file to write
        :param variables: variables to use
        """

        self.to_xarray(variables).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.model_time.__repr__()]
        optional_params = [self.source, self.time_deltas, self.x_size, self.y_size, self.grid_filename, self.source_url,
                           self.wcofs_string]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class WCOFSRange:
    """
    Range of WCOFS datasets.
    """

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, source: str = None,
                 time_deltas: list = None, x_size: float = None, y_size: float = None, grid_filename: str = None,
                 source_url: str = None, wcofs_string: str = 'wcofs'):
        """
        Create range of WCOFS datasets from the given time interval.

        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param source: one of 'stations', 'fields', 'avg', or '2ds'
        :param time_deltas: time indices (nowcast or forecast) to retrieve
        :param x_size: size of cell in X direction
        :param y_size: size of cell in Y direction
        :param grid_filename: filename of NetCDF containing WCOFS grid coordinates
        :param source_url: directory containing NetCDF files
        :param wcofs_string: WCOFS string in filename
        :raises NoDataError: if data does not exist
        """

        if source is None:
            source = 'avg'
        elif source not in VALID_SOURCE_STRINGS:
            raise ValueError(f'Location must be one of {VALID_SOURCE_STRINGS}')

        self.source = source
        self.time_deltas = time_deltas
        self.x_size = x_size
        self.y_size = y_size
        self.grid_filename = grid_filename
        self.source_url = source_url
        self.wcofs_string = wcofs_string

        if self.source == 'avg':
            self.start_time = utilities.round_to_day(start_time)
            self.end_time = utilities.round_to_day(end_time)
        else:
            self.start_time = utilities.round_to_hour(start_time)
            self.end_time = utilities.round_to_hour(end_time)

        logging.info(f'Collecting WCOFS stack between {self.start_time} and {self.end_time}...')

        # get all possible model dates that could overlap with the given time interval
        overlapping_start_time = self.start_time - datetime.timedelta(hours=WCOFS_MODEL_HOURS['f'] - 24)
        overlapping_end_time = self.end_time + datetime.timedelta(hours=-WCOFS_MODEL_HOURS['n'] - 24)
        model_dates = utilities.range_daily(utilities.round_to_day(overlapping_start_time, 'floor'),
                                            utilities.round_to_day(overlapping_end_time, 'ceiling'))

        self.datasets = {}

        # concurrently populate dictionary with WCOFS dataset objects for every time in the given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}

            for model_date in model_dates:
                if self.source == 'avg':
                    # construct start and end days from given time interval
                    start_duration = self.start_time - model_date
                    end_duration = self.end_time - model_date

                    start_day = round(start_duration.total_seconds() / (60 * 60 * 24))
                    end_day = round(end_duration.total_seconds() / (60 * 60 * 24))

                    if start_day <= WCOFS_MODEL_HOURS['n'] / 24:
                        start_day = round(WCOFS_MODEL_HOURS['n'] / 24)
                    elif start_day >= WCOFS_MODEL_HOURS['f'] / 24:
                        start_day = round(WCOFS_MODEL_HOURS['f'] / 24)
                    if end_day <= WCOFS_MODEL_HOURS['n'] / 24:
                        end_day = round(WCOFS_MODEL_HOURS['n'] / 24)
                    elif end_day >= WCOFS_MODEL_HOURS['f'] / 24:
                        end_day = round(WCOFS_MODEL_HOURS['f'] / 24)

                    overlapping_days = list(range(start_day, end_day))

                    if self.time_deltas is not None:
                        time_deltas = []

                        for day in self.time_deltas:
                            if day in overlapping_days:
                                time_deltas.append(day)
                    else:
                        time_deltas = overlapping_days
                else:
                    model_time = model_date + datetime.timedelta(hours=3)

                    # construct start and end hours from given time interval
                    start_duration = self.start_time - model_time
                    end_duration = self.end_time - model_time

                    start_hour = round(start_duration.total_seconds() / (60 * 60))
                    if start_hour <= WCOFS_MODEL_HOURS['n']:
                        start_hour = WCOFS_MODEL_HOURS['n']
                    elif start_hour >= WCOFS_MODEL_HOURS['f']:
                        start_hour = WCOFS_MODEL_HOURS['f']

                    end_hour = round(end_duration.total_seconds() / (60 * 60))
                    if end_hour <= WCOFS_MODEL_HOURS['n']:
                        end_hour = WCOFS_MODEL_HOURS['n']
                    elif end_hour >= WCOFS_MODEL_HOURS['f']:
                        end_hour = WCOFS_MODEL_HOURS['f']

                    overlapping_hours = list(range(start_hour, end_hour))

                    if self.time_deltas is not None:
                        time_deltas = []

                        for overlapping_hour in overlapping_hours:
                            if overlapping_hour in self.time_deltas:
                                time_deltas.append(overlapping_hour)
                    else:
                        time_deltas = overlapping_hours

                # get dataset for the current hours (usually all hours)
                if time_deltas is None or len(time_deltas) > 0:
                    running_future = concurrency_pool.submit(WCOFSDataset, model_date=model_date, source=self.source,
                                                             time_deltas=self.time_deltas, x_size=self.x_size,
                                                             y_size=self.y_size, grid_filename=self.grid_filename,
                                                             source_url=self.source_url, wcofs_string=self.wcofs_string)

                    running_futures[running_future] = model_date

            for completed_future in futures.as_completed(running_futures):
                model_date = running_futures[completed_future]

                if type(completed_future.exception()) is not utilities.NoDataError:
                    result = completed_future.result()
                    self.datasets[model_date] = result

            del running_futures

        if len(self.datasets) > 0:
            self.grid_transforms = WCOFSDataset.grid_transforms
            self.grid_shapes = WCOFSDataset.grid_shapes
            self.grid_bounds = WCOFSDataset.grid_bounds
            self.data_coordinates = WCOFSDataset.data_coordinates
            self.variable_grids = WCOFSDataset.variable_grids
        else:
            raise utilities.NoDataError(
                f'No WCOFS datasets found between {self.start_time} and {self.end_time}.')

    def data(self, variable: str, model_time: datetime.datetime, time_delta: int) -> numpy.ndarray:
        """
        Return data from given model run at given variable and hour.

        :param variable: name of variable to use
        :param model_time: datetime of start of model run
        :param time_delta: index of time to retrieve (days for avg, hours for others)
        :return: matrix of data
        """

        return self.datasets[model_time].data(variable, time_delta)

    def data_stack(self, variable: str, input_time: datetime.datetime) -> dict:
        """
        Return dictionary of data for each model run within the given variable and datetime.

        :param variable: name of variable to use
        :param input_time: datetime from which to retrieve data
        :return: dictionary of data for every model in the given datetime
        """

        output_data = {}

        # concurrently populate dictionary with dictionaries for each model intersection with the given datetime
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}

            for day, dataset in self.datasets.items():
                if self.source == 'avg':
                    # get current day index
                    time_difference = input_time - day
                    time_delta = round(time_difference.total_seconds() / (60 * 60 * 24))
                else:
                    # get current hour index
                    time_difference = input_time - day.replace(hour=3, minute=0, second=0)
                    time_delta = round(time_difference.total_seconds() / (60 * 60))

                if time_delta in dataset.time_deltas:
                    future = concurrency_pool.submit(dataset.data, variable, time_delta)
                    if time_delta < 0:
                        time_delta_string = f'n{abs(time_delta):03}'
                    else:
                        time_delta_string = f'f{abs(time_delta) + 1:03}'

                    running_futures[future] = f'{day.strftime("%Y%m%d")}_{time_delta_string}'

            for completed_future in futures.as_completed(running_futures):
                model_string = running_futures[completed_future]
                result = completed_future.result()

                if result is not None and len(result) > 0:
                    output_data[model_string] = result

            del running_futures
        return output_data

    def data_stacks(self, variable: str, start_time: datetime.datetime = None,
                    end_time: datetime.datetime = None) -> dict:
        """
        Return dictionary of data for each model run within the given variable and datetime.

        :param variable: name of variable to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :return: dictionary of data for every model for every time in the given time interval
        """

        logging.debug(f'Aggregating {variable} data...')

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        if self.source == 'avg':
            time_range = utilities.range_daily(utilities.round_to_day(start_time),
                                               utilities.round_to_day(end_time))
        else:
            time_range = utilities.range_hourly(utilities.round_to_hour(start_time),
                                                utilities.round_to_hour(end_time))

        output_data = {}

        # concurrently populate dictionary with data stack for each time in given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(self.data_stack, variable, data_time): data_time for
                               data_time in time_range}

            for completed_future in futures.as_completed(running_futures):
                data_time = running_futures[completed_future]
                result = completed_future.result()

                if len(result) > 0:
                    output_data[data_time] = result

            del running_futures

        return output_data

    def data_averages(self, variable: str, start_time: datetime.datetime = None,
                      end_time: datetime.datetime = None) -> dict:
        """
        Collect averaged data for every time index in given time interval.

        :param variable: name of variable to average
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :return: dictionary of data for every model in the given datetime
        """

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        input_data = self.data_stacks(variable, start_time, end_time)
        stacked_arrays = {}

        for _, data_stack in input_data.items():
            for model_time, model_data in data_stack.items():
                if model_time in stacked_arrays.keys():
                    stacked_arrays[model_time] = numpy.dstack([stacked_arrays[model_time], model_data])
                else:
                    stacked_arrays[model_time] = model_data

        output_data = {}

        # concurrently populate dictionary with average of data stack for each time in given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}

            for model_time, stacked_array in stacked_arrays.items():
                if len(stacked_array.shape) > 2:
                    future = concurrency_pool.submit(numpy.mean, stacked_array, axis=2)
                    running_futures[future] = model_time
                else:
                    output_data[model_time] = stacked_array

            for completed_future in futures.as_completed(running_futures):
                model_time = running_futures[completed_future]
                output_data[model_time] = completed_future.result()

            del running_futures
        return output_data

    def write_rasters(self, output_dir: str, variables: Collection[str] = None, filename_suffix: str = None,
                      study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      start_time: datetime.datetime = None, end_time: datetime.datetime = None,
                      x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999, driver: str = 'GTiff'):
        """
        Write raster data of given variables to given output directory, averaged over given time interval.

        :param output_dir: path to output directory
        :param variables: variable names to use
        :param filename_suffix: suffix for filenames
        :param study_area_polygon_filename: path to vector file containing study area boundary
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param x_size: cell size of output grid in X direction
        :param y_size: cell size of output grid in Y direction
        :param fill_value: desired fill value of output
        :param driver: string of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        """

        if variables is None:
            variables = list(DATA_VARIABLES.keys())

        study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
            study_area_geojson = next(iter(vector_layer))['geometry']

        if x_size is None or y_size is None:
            sample_dataset = next(iter(next(iter(self.datasets.values())).datasets.values()))

            # if cell sizes are not specified, get maximum coordinate differences between cell points on psi grid
            if x_size is None:
                x_size = numpy.max(numpy.diff(sample_dataset['lon_psi'][:]))
            if y_size is None:
                y_size = numpy.max(numpy.diff(sample_dataset['lat_psi'][:]))

        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        grid_variables = list(variables)

        if 'dir' in variables or 'mag' in variables:
            self.variable_grids['dir'] = 'rho'
            self.variable_grids['mag'] = 'rho'
            grid_variables += ['dir', 'mag']

            for required_variable in ['ssu', 'ssv']:
                if required_variable not in grid_variables:
                    grid_variables.append(required_variable)
        else:
            grid_variables = variables

        output_grid_coordinates = {}

        for variable in grid_variables:
            output_grid_coordinates[variable] = {}

            grid_name = self.variable_grids[variable]

            west = self.grid_bounds[grid_name][0]
            north = self.grid_bounds[grid_name][1]
            east = self.grid_bounds[grid_name][2]
            south = self.grid_bounds[grid_name][3]

            # WCOFS grid starts at southwest corner
            output_grid_coordinates[variable]['lon'] = numpy.arange(west, east, x_size)
            output_grid_coordinates[variable]['lat'] = numpy.arange(south, north, y_size)

        if start_time is None:
            start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_time, end_time): variable for
                variable in variables if variable not in ['dir', 'mag']}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del running_futures

        if 'dir' in variables or 'mag' in variables:
            u_name = 'ssu'
            v_name = 'ssv'

            if u_name not in variable_data_stack_averages:
                variable_data_stack_averages[u_name] = self.data_averages(u_name, start_time, end_time)

            if v_name not in variable_data_stack_averages:
                variable_data_stack_averages[v_name] = self.data_averages(v_name, start_time, end_time)

        logging.debug('parallel data aggregation took ' +
                      f'{(datetime.datetime.now() - start_time).total_seconds(): .2=f} seconds')

        start_time = datetime.datetime.now()

        interpolated_data = {}

        # concurrently populate dictionary with interpolated data in given grid for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}

            for variable, variable_data_stack in variable_data_stack_averages.items():
                logging.debug(f'Starting {variable} interpolation...')

                grid_lon = output_grid_coordinates[variable]['lon']
                grid_lat = output_grid_coordinates[variable]['lat']

                grid_name = self.variable_grids[variable]

                lon = self.data_coordinates[grid_name]['lon']
                lat = self.data_coordinates[grid_name]['lat']

                if len(grid_lon) > 0:
                    running_futures[variable] = {}

                    for model_string, model_data in variable_data_stack.items():
                        future = concurrency_pool.submit(interpolate_grid, lon, lat, model_data, grid_lon, grid_lat)

                        running_futures[variable][future] = model_string

            for variable, interpolation_futures in running_futures.items():
                interpolated_data[variable] = {}

                for completed_future in futures.as_completed(interpolation_futures):
                    model_string = interpolation_futures[completed_future]
                    interpolated_data[variable][model_string] = completed_future.result()

            del running_futures

        if 'dir' in variables or 'mag' in variables:
            u_name = 'ssu'
            v_name = 'ssv'

            interpolated_data['dir'] = {}
            interpolated_data['mag'] = {}

            u_data_stack = interpolated_data[u_name]
            v_data_stack = interpolated_data[v_name]

            for model_string in u_data_stack:
                u_data = u_data_stack[model_string]
                v_data = v_data_stack[model_string]

                # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                interpolated_data['dir'][model_string] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                interpolated_data['mag'][model_string] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

            if u_name not in variables:
                del interpolated_data[u_name]

            if v_name not in variables:
                del interpolated_data[v_name]

        logging.debug('parallel grid interpolation took ' +
                      f'{(datetime.datetime.now() - start_time).total_seconds(): .2f} seconds')

        # write interpolated grids to raster files
        for variable, variable_data_stack in interpolated_data.items():
            if len(variable_data_stack) > 0:
                west = numpy.min(output_grid_coordinates[variable]['lon'])
                north = numpy.max(output_grid_coordinates[variable]['lat'])

                # WCOFS grid starts from southwest corner, but we'll be flipping the data in a moment so lets use the northwest point
                # TODO NOTE: You cannot use a negative (northward) y_size here, otherwise GDALSetGeoTransform will break at line 1253 of rasterio/_io.pyx
                if x_size is not None and y_size is not None:
                    grid_transform = rasterio.transform.from_origin(west, north, x_size, y_size)
                else:
                    grid_name = self.variable_grids[variable]
                    grid_transform = self.grid_transforms[grid_name]

                for model_string, data in variable_data_stack.items():
                    # flip the data to ensure northward y_size (see comment above)
                    raster_data = numpy.flip(data.astype(rasterio.float32), axis=0)

                    gdal_args = {
                        'width': raster_data.shape[1], 'height': raster_data.shape[0], 'count': 1,
                        'crs': RASTERIO_CRS, 'transform': grid_transform, 'dtype': raster_data.dtype,
                        'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
                    }

                    with MemoryFile() as memory_file:
                        with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                            memory_raster.write(raster_data, 1)

                        with memory_file.open() as memory_raster:
                            masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

                    masked_data = masked_data[0, :, :]

                    if driver == 'AAIGrid':
                        file_extension = 'asc'
                        gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                    elif driver == 'GPKG':
                        file_extension = 'gpkg'
                    else:
                        file_extension = 'tiff'

                    output_filename = os.path.join(output_dir,
                                                   f'wcofs_{variable}_{model_string}{filename_suffix}.{file_extension}')

                    if os.path.isfile(output_filename):
                        os.remove(output_filename)

                    logging.info(f'Writing to {output_filename}')
                    with rasterio.open(output_filename, mode='w', driver=driver, **gdal_args) as output_raster:
                        output_raster.write(masked_data, 1)

    def write_vector(self, output_filename: str, variables: Collection[str] = None,
                     start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: path to output file
        :param variables: variable names to write to vector file
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        """

        if variables is None:
            variables = list(DATA_VARIABLES.keys())

        if start_time is None:
            start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_time, end_time): variable for
                variable in variables}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del running_futures

        logging.debug('parallel data aggregation took ' +
                      f'{(datetime.datetime.now() - start_time).total_seconds(): .2f} seconds')

        model_time_strings = [model_time for model_time in
                              next(iter(variable_data_stack_averages.values())).keys()]

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {'lon': 'float', 'lat': 'float'}
        }

        for variable in variables:
            schema['properties'][variable] = 'float'

        logging.debug('Creating records...')

        # create features
        layers = {model_time: [] for model_time in model_time_strings}

        layer_feature_indices = {layer_name: 1 for layer_name in layers.keys()}

        grid_height, grid_width = self.data_coordinates['psi']['lon'].shape

        for col in range(grid_width):
            for row in range(grid_height):
                # get coordinates of cell center
                rho_lon = self.data_coordinates['rho']['lon'][row, col]
                rho_lat = self.data_coordinates['rho']['lat'][row, col]

                for model_time_string in model_time_strings:
                    data = [float(variable_data_stack_averages[variable][model_time_string][row, col]) for variable
                            in variables]

                    if not numpy.isnan(data).all():
                        record = {
                            'id': layer_feature_indices[model_time_string],
                            'geometry': {'type': 'Point', 'coordinates': (rho_lon, rho_lat)}, 'properties': {
                                'lon': float(rho_lon), 'lat': float(rho_lat)
                            }
                        }

                        record['properties'].update(dict(zip(variables, data)))

                        layers[model_time_string].append(record)
                        layer_feature_indices[model_time_string] += 1

        # write queued features to layer
        for layer_name, layer_records in layers.items():
            logging.info(f'Writing {output_filename}:{layer_name}')
            with fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_CRS,
                            layer=layer_name) as layer:
                layer.writerecords(layer_records)

    def to_xarray(self, variables: Collection[str] = None, mean: bool = True) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: variables to use
        :param mean: whether to average all time indices
        :return: xarray dataset of given variables
        """

        data_arrays = {}

        if variables is None:
            variables = list(DATA_VARIABLES.keys())

        if mean:
            for variable in variables:
                grid = self.variable_grids[variable]

                data = self.data_averages(variable)
                data_stack = numpy.stack(
                    [time_delta_data for time_delta, time_delta_data in sorted(data.items(), reverse=True)], axis=0)

                data_array = xarray.DataArray(data_stack,
                                              coords={
                                                  'time_delta': sorted(data.keys(), reverse=True),
                                                  'lon': (
                                                      (f'{grid}_eta', f'{grid}_xi'),
                                                      self.data_coordinates[grid]['lon']),
                                                  'lat': (
                                                      (f'{grid}_eta', f'{grid}_xi'),
                                                      self.data_coordinates[grid]['lat'])
                                              },
                                              dims=('time_delta', f'{grid}_eta', f'{grid}_xi'))

                data_array.attrs['grid'] = grid
                data_arrays[variable] = data_array
        else:
            for variable in variables:
                grid = self.variable_grids[variable]

                if self.source == 'avg':
                    model_times = utilities.range_daily(utilities.round_to_day(self.start_time),
                                                        utilities.round_to_day(self.end_time))
                else:
                    model_times = utilities.range_hourly(utilities.round_to_hour(self.start_time),
                                                         utilities.round_to_hour(self.end_time))

                data = {}
                time_deltas = None

                for model_time in model_times:
                    model_time_data = self.data_stack(variable, model_time)
                    data[model_time] = numpy.stack(
                        [time_delta_data for time_delta, time_delta_data in
                         sorted(model_time_data.items(), reverse=True)], axis=0)

                    if time_deltas is None:
                        time_deltas = sorted(model_time_data.keys(), reverse=True)

                data_array = xarray.concat([time_delta_data for time_delta, time_delta_data in sorted(data.items())],
                                           'time')

                # data_stack = numpy.stack([time_delta_data for time_delta, time_delta_data in sorted(data.items())],
                #                          axis=0)
                # data_array = xarray.DataArray(data_stack,
                #                               coords={
                #                                   'time': sorted(data.keys()),
                #                                   'time_delta': time_deltas,
                #                                   'lon': ((f'{grid}_eta', f'{grid}_xi'),
                #                                           self.data_coordinates[grid]['lon']),
                #                                   'lat': ((f'{grid}_eta', f'{grid}_xi'),
                #                                           self.data_coordinates[grid]['lat'])
                #                               },
                #                               dims=('time', 'time_delta', f'{grid}_eta', f'{grid}_xi'))

                data_array.attrs['grid'] = grid
                data_arrays[variable] = data_array

        del data_stack

        output_dataset = xarray.Dataset(data_vars=data_arrays)

        del data_arrays

        return output_dataset

    def to_netcdf(self, output_file: str, variables: Collection[str] = None, mean: bool = True):
        """
        Writes to NetCDF file.

        :param output_file: output file to write
        :param variables: variables to use
        :param mean: whether to average all time indices
        """

        self.to_xarray(variables, mean).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.start_time.__repr__(), self.end_time.__repr__()]
        optional_params = [self.source, self.time_deltas, self.x_size, self.y_size, self.grid_filename, self.source_url,
                           self.wcofs_string]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


def interpolate_grid(input_lon: numpy.ndarray, input_lat: numpy.ndarray, input_data: numpy.ndarray,
                     output_lon: numpy.ndarray, output_lat: numpy.ndarray, method: str = 'nearest') -> numpy.ndarray:
    """
    Interpolate the given data onto a coordinate grid.

    :param input_lon: matrix of X coordinates in original grid
    :param input_lat: matrix of Y coordinates in original grid
    :param input_data: masked array of data values in original grid
    :param output_lon: longitude values of output grid
    :param output_lat: latitude values of output grid
    :param method: interpolation method
    :return: interpolated values within output grid
    """

    # get unmasked values only
    input_lon = input_lon[~numpy.isnan(input_data)]
    input_lat = input_lat[~numpy.isnan(input_data)]
    input_data = input_data[~numpy.isnan(input_data)]

    # force empty dimensions onto one-dimensional output coordinates
    if output_lon.ndim == 1:
        output_lon = output_lon[None, :]
    if output_lat.ndim == 1:
        output_lat = output_lat[:, None]

    # get grid interpolation
    interpolated_grid = interpolate.griddata((input_lon, input_lat), input_data, (output_lon, output_lat),
                                             method=method)

    return interpolated_grid


def reset_dataset_grid():
    """
    Reset all WCOFS Dataset grid variables to None. Useful when changing model output resolution.
    """
    WCOFSDataset.grid_transforms = None
    WCOFSDataset.grid_shapes = None
    WCOFSDataset.grid_bounds = None
    WCOFSDataset.data_coordinates = None
    WCOFSDataset.variable_grids = None
    WCOFSDataset.masks = None
    WCOFSDataset.angle = None


def write_convex_hull(netcdf_dataset: xarray.Dataset, output_filename: str, grid_name: str = 'psi'):
    """
    Extract the convex hull from the coordinate values of the given WCOFS NetCDF dataset, and write it to a file.

    :param netcdf_dataset: WCOFS format NetCDF dataset object
    :param output_filename: path to output file
    :param grid_name: name of grid. One of ('psi', 'rho', 'u', 'v')
    """

    output_filename, layer_name = output_filename.rsplit(':', 1)

    if layer_name == '':
        layer_name = None

    points = []

    lon_var = netcdf_dataset[f'lon_{grid_name}']
    lat_var = netcdf_dataset[f'lat_{grid_name}']

    height, width = lon_var.shape

    # rightwards over top row
    for col in range(width):
        points.append((lon_var[0, col].values, lat_var[0, col].values))

    # downwards over right col
    for row in range(height - 1):
        points.append((lon_var[row, width - 1].values, lat_var[row, width - 1].values))

    # leftward over bottom row
    for col in range(width - 1, -1, -1):
        points.append((lon_var[height - 1, col].values, lat_var[height - 1, col].values))

    # upwards over left col
    for row in range(height - 1, -1, -1):
        points.append((lon_var[row, 0].values, lat_var[row, 0].values))

    polygon = shapely.geometry.Polygon(points)

    schema = {'geometry': 'Polygon', 'properties': {'name': 'str'}}

    with fiona.open(output_filename, 'w', 'GPKG', schema=schema, crs=FIONA_CRS, layer=layer_name) as vector_file:
        vector_file.write({'properties': {'name': layer_name}, 'geometry': shapely.geometry.mapping(polygon)})


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test')

    start_time = datetime.datetime(2018, 11, 8)
    end_time = start_time + datetime.timedelta(days=1)

    wcofs_range = WCOFSRange(start_time, end_time, source='avg')
    wcofs_range.write_rasters(output_dir, variables=['sss', 'sst', 'ssu', 'ssv', 'zeta'])

    print('done')
