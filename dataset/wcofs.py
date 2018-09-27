# coding=utf-8
"""
PyOFS model output data collection and transformation by interpolation onto Cartesian grid.

Created on Jun 25, 2018

@author: zachary.burnett
"""

import concurrent.futures
import datetime
import os
import threading

import fiona
import fiona.crs
import numpy
import pyproj
import rasterio.control
import rasterio.mask
import rasterio.warp
import scipy.interpolate
import scipy.ndimage
import shapely.geometry
import xarray
from qgis.core import QgsFeature, QgsGeometry, QgsPoint, QgsVectorLayer
from rasterio.io import MemoryFile

from dataset import _utilities

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})
FIONA_WGS84 = fiona.crs.from_epsg(4326)
WCOFS_PROJ4 = pyproj.Proj('+proj=ob_tran +o_proj=longlat +o_lat_p=37.4 +o_lon_p=-57.6')

GRID_LOCATIONS = {'face': 'rho', 'edge1': 'u', 'edge2': 'v', 'node': 'psi'}
COORDINATE_VARIABLES = ['grid', 'ocean_time', 'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v', 'lon_psi',
                        'lat_psi', 'angle', 'pm', 'pn']
STATIC_VARIABLES = ['h', 'f', 'mask_rho', 'mask_u', 'mask_v', 'mask_psi']
MEASUREMENT_VARIABLES_2DS = ['u_sur', 'v_sur', 'temp_sur', 'salt_sur']
MEASUREMENT_VARIABLES = ['u', 'v', 'w', 'temp', 'salt']
WCOFS_MODEL_HOURS = {'n': -24, 'f': 72}
WCOFS_MODEL_RUN_HOUR = 3

DATA_DIR = os.environ['OFS_DATA']
STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

GLOBAL_LOCK = threading.Lock()


class WCOFS_Dataset:
    """
    West Coast Ocean Forecasting System (PyOFS) NetCDF dataset.
    """

    grid_transforms = None
    grid_shapes = None
    grid_bounds = None
    data_coordinates = None
    variable_grids = None
    masks = {}

    def __init__(self, model_date: datetime.datetime, source: str = None, time_indices: list = None,
                 x_size: float = None, y_size: float = None, uri: str = None):
        """
        Creates new dataset object from datetime and given model parameters.

        :param model_date: Model run date.
        :param source: One of 'stations', 'fields', 'avg', or '2ds'.
        :param time_indices: List of integers of times from model start for which to retrieve data (days for avg, hours for others).
        :param x_size: Size of cell in X direction.
        :param y_size: Size of cell in Y direction.
        :param uri: URI of local resource. Will override fetching hours, and only store a single dataset.
        :raises ValueError: if source is not valid.
        :raises NoDataError: if no datasets exist for the given model run.
        """

        valid_source_strings = ['stations', 'fields', 'avg', '2ds']

        if source is None:
            source = 'avg'
        elif source not in valid_source_strings:
            raise ValueError(f'Location must be one of {valid_source_strings}')

        self.source = source

        # 'avg' location is daily average datasets
        if time_indices is not None:
            self.time_indices = time_indices
        else:
            if self.source == 'avg':
                self.time_indices = [-1, 1, 2, 3]
            else:
                self.time_indices = range(WCOFS_MODEL_HOURS['n'], WCOFS_MODEL_HOURS['f'] + 1)

        # set start time to PyOFS model run time (0300 UTC)
        self.model_datetime = model_date.replace(hour=3, minute=0, second=0)
        self.x_size = x_size
        self.y_size = y_size
        self.uri = uri

        month_string = self.model_datetime.strftime('%Y%m')
        date_string = self.model_datetime.strftime('%Y%m%d')

        self.netcdf_datasets = {}

        if self.uri is not None:
            self.netcdf_datasets[0] = xarray.open_dataset(self.uri, decode_times=False)
        elif self.source == 'avg':
            for day in self.time_indices:
                model_type = 'forecast' if day > 0 else 'nowcast'
                url = f'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/{month_string}/nos.wcofs.avg.{model_type}.{date_string}.t{WCOFS_MODEL_RUN_HOUR:02}z.nc'

                try:
                    self.netcdf_datasets[day if day in [-1, 1] else 1] = xarray.open_dataset(url, decode_times=False)
                except OSError:
                    print(f'No WCOFS dataset found at {url}')
        else:
            # concurrently populate dictionary with NetCDF datasets for every hour in the given list of hours
            with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
                netcdf_futures = {}

                for hour in self.time_indices:
                    url = f'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/{month_string}/nos.wcofs.{self.location}.{"n" if hour <= 0 else "f"}{abs(hour):03}.{date_string}.t{self.cycle:02}z.nc'

                    future = concurrency_pool.submit(xarray.open_dataset, url, decode_times=False)
                    netcdf_futures[future] = hour

                for completed_future in concurrent.futures.as_completed(netcdf_futures):
                    try:
                        hour = netcdf_futures[completed_future]
                        result = completed_future.result()

                        if result is not None:
                            self.netcdf_datasets[hour] = result
                    except OSError:
                        # print(f'No WCOFS dataset found at {url}')
                        pass

                del netcdf_futures

        if len(self.netcdf_datasets) > 0:
            self.sample_netcdf_dataset = next(iter(self.netcdf_datasets.values()))

            self.dataset_locks = {hour: threading.Lock() for hour in self.netcdf_datasets.keys()}

            # get maximum pixel resolution that preserves data
            if self.x_size is None:
                self.x_size = numpy.max(numpy.diff(self.sample_netcdf_dataset[f'lon_psi']))
            if self.y_size is None:
                self.y_size = numpy.max(numpy.diff(self.sample_netcdf_dataset[f'lat_psi']))

            with GLOBAL_LOCK:
                if WCOFS_Dataset.grid_transforms is None:
                    WCOFS_Dataset.grid_transforms = {}
                    WCOFS_Dataset.grid_shapes = {}
                    WCOFS_Dataset.grid_bounds = {}
                    WCOFS_Dataset.data_coordinates = {}
                    WCOFS_Dataset.variable_grids = {}

                    for variable_name, variable in self.sample_netcdf_dataset.data_vars.items():
                        if 'location' in variable.attrs:
                            grid_name = GRID_LOCATIONS[variable.location]
                            WCOFS_Dataset.variable_grids[variable_name] = grid_name

                            # if grid_name not in WCOFS_Dataset.grid_names.keys():  #     WCOFS_Dataset.grid_names[grid_name] = []  #  # WCOFS_Dataset.grid_names[grid_name].append(variable_name)

                    for grid_name in GRID_LOCATIONS.values():
                        lon = self.sample_netcdf_dataset[f'lon_{grid_name}'].values
                        lat = self.sample_netcdf_dataset[f'lat_{grid_name}'].values

                        WCOFS_Dataset.data_coordinates[grid_name] = {}
                        WCOFS_Dataset.data_coordinates[grid_name]['lon'] = lon
                        WCOFS_Dataset.data_coordinates[grid_name]['lat'] = lat

                        WCOFS_Dataset.grid_shapes[grid_name] = self.sample_netcdf_dataset[f'lon_{grid_name}'].shape

                        west = numpy.min(lon)
                        north = numpy.max(lat)
                        east = numpy.max(lon)
                        south = numpy.min(lat)

                        WCOFS_Dataset.grid_transforms[grid_name] = rasterio.transform.from_origin(west=west,
                                                                                                  north=south,
                                                                                                  xsize=self.x_size,
                                                                                                  ysize=-self.y_size)

                        WCOFS_Dataset.grid_bounds[grid_name] = (west, north, east, south)

                        WCOFS_Dataset.masks[grid_name] = ~(
                            self.sample_netcdf_dataset[f'mask_{grid_name}'].values).astype(bool)
        else:
            raise _utilities.NoDataError(f'No WCOFS datasets found for {self.model_datetime} at the given hours.')

    def bounds(self, variable: str = 'psi') -> tuple:
        """
        Returns bounds of grid of given variable.

        :param variable: Variable name.
        :return: Tuple of (west, north, east, south)
        """

        grid_name = WCOFS_Dataset.variable_grids[variable]
        return WCOFS_Dataset.grid_bounds[grid_name]

    def data(self, variable: str, time_index: int) -> numpy.ndarray:
        """
        Get data of specified variable at specified hour.

        :param variable: Name of variable to retrieve.
        :param time_index: Time index to retrieve (days for avg, hours for others).
        :return: Array of data.
        """

        if self.source == 'avg' and time_index in self.time_indices:
            if time_index > 0:
                day_index = time_index - 1
                dataset_index = 1
            else:
                day_index = 0
                dataset_index = -1

            with self.dataset_locks[dataset_index]:
                # get surface layer; the last layer (of 40) at dimension 1
                # if variable in ['u', 'v']:
                #     raw_u = self.netcdf_datasets[dataset_index]['u'][day_index, -1, :-1, :].values
                #     raw_v = self.netcdf_datasets[dataset_index]['v'][day_index, -1, :, :-1].values
                #     theta = self.netcdf_datasets[dataset_index]['angle'][:-1, :-1].values
                #
                #     if variable == 'u':
                #         output_data = raw_u * numpy.cos(theta) - raw_v * numpy.sin(theta)
                #         extra_row = numpy.empty((1, output_data.shape[1]), dtype=output_data.dtype)
                #         extra_row[:] = numpy.nan
                #         output_data = numpy.concatenate((output_data, extra_row), axis=0)
                #     elif variable == 'v':
                #         output_data = raw_u * numpy.sin(theta) + raw_v * numpy.cos(theta)
                #         extra_column = numpy.empty((output_data.shape[0], 1), dtype=output_data.dtype)
                #         extra_column[:] = numpy.nan
                #         output_data = numpy.concatenate((output_data, extra_column), axis=1)
                # else:
                output_data = self.netcdf_datasets[dataset_index][variable][day_index, -1, :, :].values

        elif time_index in self.netcdf_datasets.keys():
            with self.dataset_locks[time_index]:
                output_data = self.netcdf_datasets[time_index][variable][0, :, :].values
        else:
            output_data = numpy.empty(WCOFS_Dataset.grid_shapes[WCOFS_Dataset.variable_grids[variable]])
            output_data[:] = numpy.nan

        return output_data

    def data_average(self, variable: str, time_indices: list = None) -> numpy.ndarray:
        """
        Writes interpolation of averaged data of given variable to output path.

        :param variable: Variable to use.
        :param time_indices: List of integers of time indices to use in average (days for avg, hours for others).
        :return: Array of data.
        """

        time_indices = time_indices if time_indices is not None else self.netcdf_datasets.keys()

        variable_data = []

        # concurrently populate array with data for every hour
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data, variable, hour): hour for hour in time_indices}

            for completed_future in concurrent.futures.as_completed(variable_futures):
                variable_data.append(completed_future.result())

            del variable_futures

        variable_data = numpy.mean(numpy.stack(variable_data), axis=2)

        return variable_data

    def write_rasters(self, output_dir: str, variables: list = None, hours: list = None, x_size: float = 0.04,
                      y_size: float = 0.04, drivers: list = ['GTiff']):
        """
        Writes interpolated rasters of given variables to output directory using concurrency.

        :param output_dir: Path to directory.
        :param variables: Variable names to use.
        :param hours: List of integers of hours to use in average.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        if variables is None:
            variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES

        # concurrently write rasters with data from each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            # get data average for each variable
            variable_mean_futures = {concurrency_pool.submit(self.data_average, variable_name, hours): variable_name for
                                     variable_name in variables}

            for completed_future in concurrent.futures.as_completed(variable_mean_futures):
                variable_name = variable_mean_futures[completed_future]

                data = completed_future.result()

                self.write_raster(os.path.join(output_dir, f'wcofs_{variable_name}.tiff'), variable_name,
                                  input_data=data, x_size=x_size, y_size=y_size, drivers=drivers)

            del variable_mean_futures

    def write_raster(self, output_filename: str, variable: str,
                     study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, hours: int = None,
                     input_data: numpy.ndarray = None, x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                     drivers: list = ['GTiff']):
        """
        Writes interpolated raster of given variable to output path.

        :param output_filename: Path of raster file to create.
        :param variable: Name of variable.
        :param study_area_polygon_filename: Path to vector file containing study area boundary.
        :param hours: List of hours from which to dataset data.
        :param input_data: Grid data to interpolate and write.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        study_area_polygon_filename, layer_name = study_area_polygon_filename.rsplit(':', 1)

        if layer_name == '':
            layer_name = None

        with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
            study_area_geojson = vector_layer.next()['geometry']

        grid_name = WCOFS_Dataset.variable_grids[variable]

        west = WCOFS_Dataset.grid_bounds[grid_name][0]
        north = WCOFS_Dataset.grid_bounds[grid_name][1]
        east = WCOFS_Dataset.grid_bounds[grid_name][2]
        south = WCOFS_Dataset.grid_bounds[grid_name][3]

        x_size = x_size if x_size is not None else self.x_size
        y_size = y_size if y_size is not None else self.y_size

        output_grid_lon = numpy.arange(west, east, x_size)
        output_grid_lat = numpy.arange(south, north, y_size)

        grid_transform = WCOFS_Dataset.grid_transforms[grid_name]

        if input_data is None:
            input_data = self.data_average(variable, hours)

        print(f'Starting {variable} interpolation')

        # interpolate data onto coordinate grid
        output_data = interpolate_grid(WCOFS_Dataset.data_coordinates[grid_name]['lon'],
                                       WCOFS_Dataset.data_coordinates[grid_name]['lat'], input_data, output_grid_lon,
                                       output_grid_lat)

        gdal_args = {
            'transform': grid_transform, 'height': output_data.shape[0], 'width': output_data.shape[1], 'count': 1,
            'dtype':     rasterio.float32, 'crs': RASTERIO_WGS84,
            'nodata':    numpy.array([fill_value]).astype(output_data.dtype).item()
        }

        with MemoryFile() as memory_file:
            with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                memory_raster.write(output_data, 1)

            with memory_file.open() as memory_raster:
                masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

        masked_data = masked_data[0, :, :]

        for driver in drivers:
            if driver == 'AAIGrid':
                file_extension = '.asc'
                gdal_args.update({'FORCE_CELLSIZE': 'YES'})
            elif driver == 'GTiff':
                file_extension = '.tiff'
            elif driver == 'GPKG':
                file_extension = '.gpkg'

            output_filename = f'{os.path.splitext(output_filename)[0]}{file_extension}'

            print(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(masked_data, 1)

    def write_vector(self, output_filename: str, layer_name: str = None, time_indices: list = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param time_indices: List of integers of hours to use in average.
        """

        variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES

        start_time = datetime.datetime.now()

        variable_means = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data_average, variable, time_indices): variable for
                                variable in variables}

            for completed_future in concurrent.futures.as_completed(variable_futures):
                variable = variable_futures[completed_future]
                variable_means[variable] = completed_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        schema = {
            'geometry': 'Point', 'properties': {
                'fid': 'int', 'row': 'int', 'col': 'int', 'rho_lon': 'float', 'rho_lat': 'float'
            }
        }

        # create layer
        fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84, layer=layer_name).close()

        start_time = datetime.datetime.now()

        print('Creating features...')

        # create features
        layer_features = []

        feature_index = 1

        grid_height, grid_width = WCOFS_Dataset.data_coordinates['psi']['lon'].shape

        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            feature_futures = []

            for col in range(grid_width):
                for row in range(grid_height):
                    if WCOFS_Dataset.masks['psi'][row, col] == 1:
                        feature_futures.append(
                                concurrency_pool.submit(self._create_qgis_feature, variable_means, row, col,
                                                        feature_index))
                        feature_index += 1

            for completed_future in concurrent.futures.as_completed(feature_futures):
                result = completed_future.result()

                if result is not None:
                    layer_features.append(result)

        print(f'creating features took {(datetime.datetime.now() - start_time).seconds} seconds')

        start_time = datetime.datetime.now()

        print(f'Writing {output_filename}:{layer_name}')

        # open layer in QGIS
        layer = QgsVectorLayer(f'{output_filename}|layername={layer_name}', layer_name, 'ogr')

        # put layer in editing mode
        layer.startEditing()

        # add features to layer
        layer.dataProvider().addFeatures(layer_features)

        # save changes to layer
        layer.commitChanges()

        print(f'writing features took {(datetime.datetime.now() - start_time).seconds} seconds')

    def write_vector_fiona(self, output_filename: str, layer_name: str = None, time_indices: list = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param time_indices: List of integers of hours to use in average.
        """

        variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES

        start_time = datetime.datetime.now()

        variable_means = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data_average, variable, time_indices): variable for
                                variable in variables}

            for completed_future in concurrent.futures.as_completed(variable_futures):
                variable = variable_futures[completed_future]
                variable_means[variable] = completed_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        schema = {
            'geometry': 'Point', 'properties': {
                'row': 'int', 'col': 'int', 'rho_lon': 'float', 'rho_lat': 'float'
            }
        }

        for variable in variables:
            schema['properties'][variable] = 'float'

        start_time = datetime.datetime.now()

        print('Creating records...')

        # create features
        layer_records = []

        grid_height, grid_width = WCOFS_Dataset.data_coordinates['psi']['lon'].shape

        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            feature_index = 1
            record_futures = []

            for col in range(grid_width):
                for row in range(grid_height):
                    if WCOFS_Dataset.masks['psi'][row, col] == 1:
                        # check if current record is unmasked
                        record_futures.append(
                                concurrency_pool.submit(self._create_fiona_record, variable_means, row, col,
                                                        feature_index))
                        feature_index += 1

            for completed_future in concurrent.futures.as_completed(record_futures):
                result = completed_future.result()

                if result is not None:
                    layer_records.append(result)

        print(f'creating records took {(datetime.datetime.now() - start_time).seconds} seconds')

        start_time = datetime.datetime.now()

        print(f'Writing {output_filename}:{layer_name}')

        # create layer
        with fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84,
                        layer=layer_name) as output_vector_file:
            output_vector_file.writerecords(layer_records)

        print(f'writing records took {(datetime.datetime.now() - start_time).seconds} seconds')

    def _create_qgis_feature(self, variable_means, row, col, feature_index):
        # get coordinates of cell center
        rho_lon = WCOFS_Dataset.data_coordinates['rho']['lon'][row, col]
        rho_lat = WCOFS_Dataset.data_coordinates['rho']['lat'][row, col]

        # create feature
        feature = QgsFeature()

        data = [feature_index, row, col, float(rho_lon), float(rho_lat)]
        for variable in variable_means.keys():
            data.append(float(variable_means[variable][row, col]))

        feature.setGeometry(QgsGeometry(QgsPoint(rho_lon, rho_lat)))
        feature.setAttributes(data)

        return feature

    def _create_fiona_record(self, variable_means, row, col, feature_index):
        # get coordinates of cell center
        rho_lon = WCOFS_Dataset.data_coordinates['rho']['lon'][row, col]
        rho_lat = WCOFS_Dataset.data_coordinates['rho']['lat'][row, col]

        record = {
            'geometry':      {
                'id': feature_index, 'type': 'Point', 'coordinates': (float(rho_lon), float(rho_lat))
            }, 'properties': {
                'row': row, 'col': col, 'rho_lon': float(rho_lon), 'rho_lat': float(rho_lat)
            }
        }

        for variable in variable_means.keys():
            record['properties'][variable] = float(variable_means[variable][row, col])

        return record

    def __repr__(self):
        used_params = [self.model_datetime.__repr__()]
        optional_params = [self.source, self.time_indices, self.x_size, self.y_size, self.uri]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class WCOFS_Range:
    """
    Range of WCOFS datasets.
    """

    grid_transforms = None
    grid_shapes = None
    grid_bounds = None
    data_coordinates = None
    variable_grids = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, source: str = None,
                 time_indices: list = None):
        """
        Create range of WCOFS datasets from the given time interval.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param source: One of 'stations', 'fields', 'avg', or '2ds'.
        :param time_indices: List of time indices (nowcast or forecast) to retrieve.
        :raises NoDataError: if data does not exist.
        """

        self.source = source
        self.time_indices = time_indices

        if self.source == 'avg':
            self.start_datetime = _utilities.round_to_day(start_datetime)
            self.end_datetime = _utilities.round_to_day(end_datetime)
        else:
            self.start_datetime = _utilities.round_to_hour(start_datetime)
            self.end_datetime = _utilities.round_to_hour(end_datetime)

        print(f'Collecting WCOFS stack between {self.start_datetime} and {self.end_datetime}...')

        # get all possible model dates that could overlap with the given time interval
        overlapping_models_start_datetime = self.start_datetime - datetime.timedelta(hours=WCOFS_MODEL_HOURS['f'])
        overlapping_models_end_datetime = self.end_datetime - datetime.timedelta(hours=WCOFS_MODEL_HOURS['n'])
        model_dates = _utilities.day_range(_utilities.round_to_day(overlapping_models_start_datetime, 'floor'),
                                           _utilities.round_to_day(overlapping_models_end_datetime, 'ceiling'))

        self.datasets = {}

        # concurrently populate dictionary with WCOFS dataset objects for every time in the given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            dataset_futures = {}

            for model_date in model_dates:
                if self.source == 'avg':
                    # construct start and end days from given time interval
                    start_duration = self.start_datetime - model_date
                    end_duration = self.end_datetime - model_date

                    start_day = round(start_duration.total_seconds() / (60 * 60 * 24))
                    end_day = round(end_duration.total_seconds() / (60 * 60 * 24))

                    if start_day <= WCOFS_MODEL_HOURS['n'] / 24:
                        start_day = WCOFS_MODEL_HOURS['n'] / 24
                    elif start_day >= WCOFS_MODEL_HOURS['f'] / 24:
                        start_day = WCOFS_MODEL_HOURS['f'] / 24
                    if end_day <= WCOFS_MODEL_HOURS['n'] / 24:
                        end_day = WCOFS_MODEL_HOURS['n'] / 24
                    elif end_day >= WCOFS_MODEL_HOURS['f'] / 24:
                        end_day = WCOFS_MODEL_HOURS['f'] / 24

                    overlapping_days = list(range(round(start_day) + 1, round(end_day) + 1))

                    for index in range(len(overlapping_days)):
                        if overlapping_days[index] <= 0:
                            overlapping_days[index] -= 1

                    if self.time_indices is not None:
                        time_indices = []

                        for day in self.time_indices:
                            if day >= 0:
                                if day - 1 in overlapping_days:
                                    time_indices.append(day)
                            else:
                                if day in overlapping_days:
                                    time_indices.append(day)
                    else:
                        time_indices = overlapping_days
                else:
                    model_datetime = model_date + datetime.timedelta(hours=3)

                    # construct start and end hours from given time interval
                    start_duration = self.start_datetime - model_datetime
                    end_duration = self.end_datetime - model_datetime

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

                    if self.time_indices is not None:
                        time_indices = []

                        for overlapping_hour in overlapping_hours:
                            if overlapping_hour in self.time_indices:
                                time_indices.append(overlapping_hour)
                    else:
                        time_indices = overlapping_hours

                # get dataset for the current hours (usually all hours)
                if time_indices is None or len(time_indices) > 0:
                    future = concurrency_pool.submit(WCOFS_Dataset, model_date, self.source, time_indices)

                    dataset_futures[future] = model_date

            for completed_future in concurrent.futures.as_completed(dataset_futures):
                model_date = dataset_futures[completed_future]

                if type(completed_future.exception()) is not _utilities.NoDataError:
                    result = completed_future.result()
                    self.datasets[model_date] = result

            del dataset_futures

        if len(self.datasets) > 0:
            self.sample_wcofs_dataset = next(iter(self.datasets.values()))

            with GLOBAL_LOCK:
                WCOFS_Range.grid_transforms = WCOFS_Dataset.grid_transforms
                WCOFS_Range.grid_shapes = WCOFS_Dataset.grid_shapes
                WCOFS_Range.grid_bounds = WCOFS_Dataset.grid_bounds
                WCOFS_Range.data_coordinates = WCOFS_Dataset.data_coordinates
                WCOFS_Range.variable_grids = WCOFS_Dataset.variable_grids
        else:
            raise _utilities.NoDataError(
                    f'No WCOFS datasets found between {self.start_datetime} and {self.end_datetime}.')

    def data(self, variable: str, model_datetime: datetime.datetime, time_index: int) -> numpy.ndarray:
        """
        Return data from given model run at given variable and hour.

        :param variable: Name of variable to use.
        :param model_datetime: Datetime of start of model run.
        :param time_index: Index of time to retrieve (days for avg, hours for others).
        :return: Matrix of data.
        """

        return self.datasets[model_datetime].data(variable, time_index)

    def data_stack(self, variable: str, input_datetime: datetime.datetime) -> dict:
        """
        Return dictionary of data for each model run within the given variable and datetime.

        :param variable: Name of variable to use.
        :param input_datetime: Datetime from which to retrieve data.
        :return: Dictionary of data for every model in the given datetime.
        """

        output_data = {}

        # concurrently populate dictionary with dictionaries for each model intersection with the given datetime
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {}

            for day, dataset in self.datasets.items():
                if self.source == 'avg':
                    # get current day index
                    timedelta = input_datetime - day
                    time_index = round(timedelta.total_seconds() / (60 * 60 * 24))
                    if time_index >= 0:
                        time_index += 1
                else:
                    # get current hour index
                    timedelta = input_datetime - day.replace(hour=3, minute=0, second=0)
                    time_index = round(timedelta.total_seconds() / (60 * 60))

                if time_index in dataset.time_indices:
                    future = concurrency_pool.submit(dataset.data, variable, time_index)
                    time_index_string = f'{"f" if time_index > 0 else "n"}{abs(time_index):03}'
                    data_futures[future] = f'{day.strftime("%Y%m%d")}_{time_index_string}'

            for completed_future in concurrent.futures.as_completed(data_futures):
                model_string = data_futures[completed_future]
                result = completed_future.result()

                if result is not None and len(result) > 0:
                    output_data[model_string] = result

            del data_futures
        return output_data

    def data_stacks(self, variable: str, start_datetime: datetime.datetime = None,
                    end_datetime: datetime.datetime = None) -> dict:
        """
        Return dictionary of data for each model run within the given variable and datetime.

        :param variable: Name of variable to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :return: Dictionary of data for every model for every time in the given time interval.
        """

        print(f'Aggregating {variable} data...')

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        if self.source == 'avg':
            time_range = _utilities.day_range(_utilities.round_to_day(start_datetime),
                                              _utilities.round_to_day(end_datetime))
        else:
            time_range = _utilities.hour_range(_utilities.round_to_hour(start_datetime),
                                               _utilities.round_to_hour(end_datetime))

        output_data = {}

        # concurrently populate dictionary with data stack for each time in given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {concurrency_pool.submit(self.data_stack, variable, data_datetime): data_datetime for
                            data_datetime in time_range}

            for completed_future in concurrent.futures.as_completed(data_futures):
                data_datetime = data_futures[completed_future]
                result = completed_future.result()

                if len(result) > 0:
                    output_data[data_datetime] = result

            del data_futures

        return output_data

    def data_averages(self, variable: str, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None) -> dict:
        """
        Collect averaged data for every time index in given time interval.

        :param variable: Name of variable to average.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :return: Dictionary of data for every model in the given datetime.
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        input_data = self.data_stacks(variable, start_datetime, end_datetime)
        stacked_arrays = {}

        for _, data_stack in input_data.items():
            for model_datetime, model_data in data_stack.items():
                if model_datetime in stacked_arrays.keys():
                    stacked_arrays[model_datetime] = numpy.dstack([stacked_arrays[model_datetime], model_data])
                else:
                    stacked_arrays[model_datetime] = model_data

        output_data = {}

        # concurrently populate dictionary with average of data stack for each time in given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {}

            for model_datetime, stacked_array in stacked_arrays.items():
                if len(stacked_array.shape) > 2:
                    future = concurrency_pool.submit(numpy.mean, stacked_array, axis=2)
                    data_futures[future] = model_datetime
                else:
                    output_data[model_datetime] = stacked_array

            for completed_future in concurrent.futures.as_completed(data_futures):
                model_datetime = data_futures[completed_future]
                output_data[model_datetime] = completed_future.result()

            del data_futures
        return output_data

    def write_rasters(self, output_dir: str, variables: list = None,
                      study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                      vector_components: bool = False, x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                      drivers: list = ['GTiff']):
        """
        Write raster data of given variables to given output directory, averaged over given time interval.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to use.
        :param study_area_polygon_filename: Path to vector file containing study area boundary.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param vector_components: Whether to write direction and magnitude rasters.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        if variables is None:
            variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES

        study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
            study_area_geojson = vector_layer.next()['geometry']

        # if cell sizes are not specified, get maximum coordinate differences between cell points on psi grid
        if x_size is None:
            x_size = numpy.max(numpy.diff(self.sample_wcofs_dataset.sample_netcdf_dataset['lon_psi'][:]))
        if y_size is None:
            y_size = numpy.max(numpy.diff(self.sample_wcofs_dataset.sample_netcdf_dataset['lat_psi'][:]))

        output_grid_coordinates = {}

        if vector_components:
            WCOFS_Range.variable_grids['dir'] = 'rho'
            WCOFS_Range.variable_grids['mag'] = 'rho'
            grid_variables = variables + ['dir', 'mag']
        else:
            grid_variables = variables

        for variable in grid_variables:
            output_grid_coordinates[variable] = {}

            grid_name = WCOFS_Range.variable_grids[variable]

            west = WCOFS_Range.grid_bounds[grid_name][0]
            north = WCOFS_Range.grid_bounds[grid_name][1]
            east = WCOFS_Range.grid_bounds[grid_name][2]
            south = WCOFS_Range.grid_bounds[grid_name][3]

            # if None not in (west, north, east, south):
            #     west_cell_offset = round((west - west) / x_size)
            #     north_cell_offset = round((north - north) / y_size)
            #     east_cell_offset = round((east - east) / x_size)
            #     south_cell_offset = round((south - south) / y_size)
            #
            #     west = west + (west_cell_offset * x_size)
            #     north = north + (north_cell_offset * y_size)
            #     east = east + (east_cell_offset * x_size)
            #     south = south + (south_cell_offset * y_size)

            # WCOFS grid starts at southwest corner
            output_grid_coordinates[variable]['lon'] = numpy.arange(west, east, x_size)
            output_grid_coordinates[variable]['lat'] = numpy.arange(south, north, y_size)

        start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_datetime, end_datetime): variable for
                variable in variables}

            for completed_future in concurrent.futures.as_completed(variable_futures):
                variable = variable_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        if vector_components:
            if self.source == '2ds':
                u_name = 'u_sur'
                v_name = 'v_sur'
            else:
                u_name = 'u'
                v_name = 'v'

            if u_name not in variable_data_stack_averages:
                variable_data_stack_averages[u_name] = self.data_averages(u_name, start_datetime, end_datetime)

            if v_name not in variable_data_stack_averages:
                variable_data_stack_averages[v_name] = self.data_averages(v_name, start_datetime, end_datetime)

        start_time = datetime.datetime.now()

        interpolated_data = {}

        # concurrently populate dictionary with interpolated data in given grid for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_interpolation_futures = {}

            for variable, variable_data_stack in variable_data_stack_averages.items():
                print(f'Starting {variable} interpolation...')

                grid_lon = output_grid_coordinates[variable]['lon']
                grid_lat = output_grid_coordinates[variable]['lat']

                grid_name = WCOFS_Range.variable_grids[variable]

                lon = WCOFS_Range.data_coordinates[grid_name]['lon']
                lat = WCOFS_Range.data_coordinates[grid_name]['lat']

                if len(grid_lon) > 0:
                    variable_interpolation_futures[variable] = {}

                    for model_string, model_data in variable_data_stack.items():
                        future = concurrency_pool.submit(interpolate_grid, lon, lat, model_data, grid_lon, grid_lat)

                        variable_interpolation_futures[variable][future] = model_string

            for variable, interpolation_futures in variable_interpolation_futures.items():
                interpolated_data[variable] = {}

                for completed_future in concurrent.futures.as_completed(interpolation_futures):
                    model_string = interpolation_futures[completed_future]
                    interpolated_data[variable][model_string] = completed_future.result()

            del variable_interpolation_futures

        print(f'parallel grid interpolation took {(datetime.datetime.now() - start_time).seconds} seconds')

        if vector_components:
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
                    grid_name = WCOFS_Range.variable_grids[variable]
                    grid_transform = WCOFS_Range.grid_transforms[grid_name]

                for model_string, data in variable_data_stack.items():
                    # flip the data to ensure northward y_size (see comment above)
                    raster_data = numpy.flip(data.astype(rasterio.float32), axis=0)

                    gdal_args = {
                        'width':  raster_data.shape[1], 'height': raster_data.shape[0], 'count': 1,
                        'crs':    RASTERIO_WGS84, 'transform': grid_transform, 'dtype': raster_data.dtype,
                        'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
                    }

                    with MemoryFile() as memory_file:
                        with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                            memory_raster.write(raster_data, 1)

                        with memory_file.open() as memory_raster:
                            masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

                    masked_data = masked_data[0, :, :]

                    for driver in drivers:
                        if driver == 'AAIGrid':
                            file_extension = 'asc'
                            gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                        elif driver == 'GTiff':
                            file_extension = 'tiff'
                        elif driver == 'GPKG':
                            file_extension = 'gpkg'

                        output_filename = os.path.join(output_dir, f'wcofs_{variable}_{model_string}.{file_extension}')

                        print(f'Writing to {output_filename}')
                        with rasterio.open(output_filename, mode='w', driver=driver, **gdal_args) as output_raster:
                            output_raster.write(masked_data, 1)

    def write_vector(self, output_filename: str, variables: list = None, start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: Path to output file.
        :param variables: List of variable names to write to vector file.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        """

        if variables is None:
            variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES

        start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_datetime, end_datetime): variable for
                variable in variables}

            for completed_future in concurrent.futures.as_completed(variable_futures):
                variable = variable_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        model_datetime_strings = [model_datetime for model_datetime in
                           next(iter(variable_data_stack_averages.values())).keys()]

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {'fid': 'int'}
        }

        for variable in variables:
            schema['properties'][variable] = 'float'

        # create layers
        for model_datetime_string in model_datetime_strings:
            fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84,
                       layer=model_datetime_string).close()

        print('Creating features...')

        # create features
        layer_features = {model_datetime: [] for model_datetime in model_datetime_strings}

        layer_feature_indices = {layer_name: 1 for layer_name in layer_features.keys()}

        grid_height, grid_width = WCOFS_Range.data_coordinates['psi']['lon'].shape

        for col in range(grid_width):
            for row in range(grid_height):
                # get coordinates of cell center
                rho_lon = WCOFS_Range.data_coordinates['rho']['lon'][row, col]
                rho_lat = WCOFS_Range.data_coordinates['rho']['lat'][row, col]

                for model_datetime_string in model_datetime_strings:
                    data = [variable_data_stack_averages[variable][model_datetime_string][row, col] for variable in variables]

                    if not numpy.isnan(data).all():
                        feature = QgsFeature()
                        feature.setGeometry(QgsGeometry(QgsPoint(rho_lon, rho_lat)))
                        feature.setAttributes(
                                [layer_feature_indices[model_datetime_string]] + [float(entry) for entry in data])

                        layer_features[model_datetime_string].append(feature)
                        layer_feature_indices[model_datetime_string] += 1

        # write queued features to layer
        for layer_name, layer_features in layer_features.items():
            print(f'Writing {output_filename}:{layer_name}')
            layer = QgsVectorLayer(f'{output_filename}|layername={layer_name}', layer_name, 'ogr')
            layer.startEditing()
            layer.dataProvider().addFeatures(layer_features)
            layer.commitChanges()

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.source, self.time_indices]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


def interpolate_grid(input_lon: numpy.ndarray, input_lat: numpy.ndarray, input_data: numpy.ndarray,
                     output_lon: numpy.ndarray, output_lat: numpy.ndarray) -> numpy.ndarray:
    """
    Interpolates the given data onto a coordinate grid.

    :param input_lon: Matrix of X coordinates in original grid.
    :param input_lat: Matrix of Y coordinates in original grid.
    :param input_data: Masked array of data values in original grid.
    :param output_lon: Longitude values of output grid.
    :param output_lat: Latitude values of output grid.
    :return: Interpolated values within WGS84 coordinate grid.
    """

    # get unmasked values only
    input_lon = input_lon[~numpy.isnan(input_data)]
    input_lat = input_lat[~numpy.isnan(input_data)]
    input_data = input_data[~numpy.isnan(input_data)]

    output_lon = output_lon[None, :]
    output_lat = output_lat[:, None]

    # get grid interpolation
    interpolated_grid = scipy.interpolate.griddata((input_lon, input_lat), input_data, (output_lon, output_lat),
                                                   method='nearest')

    return interpolated_grid


def write_convex_hull(netcdf_dataset: xarray.Dataset, output_filename: str, grid_name: str = 'psi'):
    """
    Extract the convex hull from the coordinate values of the given WCOFS NetCDF dataset, and write it to a file.

    :param netcdf_dataset: WCOFS format NetCDF dataset object.
    :param output_filename: Path to output file.
    :param grid_name: Name of grid. One of ('psi', 'rho', 'u', 'v').
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

    with fiona.open(output_filename, 'w', 'GPKG', schema=schema, crs=FIONA_WGS84, layer=layer_name) as vector_file:
        vector_file.write({'properties': {'name': layer_name}, 'geometry': shapely.geometry.mapping(polygon)})


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test\unrotated')

    start_datetime = datetime.datetime(2018, 8, 10)
    end_datetime = datetime.datetime(2018, 8, 11)

    wcofs_dataset = WCOFS_Dataset(uri=r'C:\Data\output\test\test2.nc')

    wcofs_range = WCOFS_Range(start_datetime, end_datetime, source='avg')
    # wcofs_range.write_rasters(output_dir, ['temp'])
    # wcofs_range.write_rasters(output_dir, ['u', 'v'], vector_components=True, drivers=['AAIGrid'])

    # from qgis.core import QgsApplication
    # qgis_application = QgsApplication([], True, None)
    # qgis_application.setPrefixPath(os.environ['QGIS_PREFIX_PATH'], True)
    # qgis_application.initQgis()
    # wcofs_range.write_vector(os.path.join(output_dir, r'wcofs.gpkg'))

    print('done')
