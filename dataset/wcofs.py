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
import netCDF4
import numpy
import pyproj
import rasterio.control
import rasterio.mask
import rasterio.warp
import scipy.interpolate
import scipy.ndimage
import shapely.geometry
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

        # netCDF4.num2date(time_var[:], time_var.units, '_'.join(list(reversed(time_var.calendar.split('_')))))

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
            self.netcdf_datasets[0] = netCDF4.Dataset(self.uri)
        elif self.source == 'avg':
            for current_day in self.time_indices:
                current_model_type = 'forecast' if current_day > 0 else 'nowcast'
                current_url = f'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/PyOFS/MODELS/{month_string}/nos.wcofs.avg.{current_model_type}.{date_string}.t{WCOFS_MODEL_RUN_HOUR:02}z.nc'

                try:
                    self.netcdf_datasets[current_day if current_day in [-1, 1] else 1] = netCDF4.Dataset(current_url)
                except OSError:
                    print(f'No PyOFS dataset found at {current_url}')
        else:
            # concurrently populate dictionary with NetCDF datasets for every hour in the given list of hours
            with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
                netcdf_futures = {}

                for current_hour in self.time_indices:
                    current_url = f'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/PyOFS/MODELS/{month_string}/nos.wcofs.{self.location}.{"n" if current_hour <= 0 else "f"}{abs(current_hour):03}.{date_string}.t{self.cycle:02}z.nc'

                    current_future = concurrency_pool.submit(netCDF4.Dataset, current_url)
                    netcdf_futures[current_future] = current_hour

                for current_future in concurrent.futures.as_completed(netcdf_futures):
                    try:
                        current_hour = netcdf_futures[current_future]
                        current_result = current_future.result()

                        if current_result is not None:
                            self.netcdf_datasets[current_hour] = current_result
                    except OSError:
                        # print(f'No PyOFS dataset found at {current_url}')
                        pass

                del netcdf_futures

        if len(self.netcdf_datasets) > 0:
            self.sample_netcdf_dataset = next(iter(self.netcdf_datasets.values()))

            self.dataset_locks = {hour: threading.Lock() for hour in self.netcdf_datasets.keys()}

            # get maximum pixel resolution that preserves data
            if self.x_size is None:
                self.x_size = numpy.max(numpy.diff(self.sample_netcdf_dataset[f'lon_psi'][:]))
            if self.y_size is None:
                self.y_size = numpy.max(numpy.diff(self.sample_netcdf_dataset[f'lat_psi'][:]))

            with GLOBAL_LOCK:
                if WCOFS_Dataset.grid_transforms is None:
                    WCOFS_Dataset.grid_transforms = {}
                    WCOFS_Dataset.grid_shapes = {}
                    WCOFS_Dataset.grid_bounds = {}
                    WCOFS_Dataset.data_coordinates = {}
                    WCOFS_Dataset.variable_grids = {}

                    for current_variable_name, current_variable in self.sample_netcdf_dataset.variables.items():
                        if 'location' in current_variable.ncattrs():
                            current_grid_name = GRID_LOCATIONS[current_variable.location]
                            WCOFS_Dataset.variable_grids[current_variable_name] = current_grid_name

                            # if current_grid_name not in WCOFS_Dataset.grid_names.keys():  #     WCOFS_Dataset.grid_names[current_grid_name] = []  #  # WCOFS_Dataset.grid_names[current_grid_name].append(current_variable_name)

                    for current_grid_name in GRID_LOCATIONS.values():
                        current_lon = self.sample_netcdf_dataset[f'lon_{current_grid_name}'][:]
                        current_lat = self.sample_netcdf_dataset[f'lat_{current_grid_name}'][:]

                        west = numpy.min(current_lon)
                        north = numpy.max(current_lat)
                        east = numpy.max(current_lon)
                        south = numpy.min(current_lat)

                        WCOFS_Dataset.grid_transforms[current_grid_name] = rasterio.transform.from_origin(west=west,
                                                                                                          north=south,
                                                                                                          xsize=self.x_size,
                                                                                                          ysize=-self.y_size)

                        WCOFS_Dataset.grid_bounds[current_grid_name] = (west, north, east, south)

                        WCOFS_Dataset.masks[current_grid_name] = self.sample_netcdf_dataset[
                                                                     f'mask_{current_grid_name}'][:]

                        WCOFS_Dataset.data_coordinates[current_grid_name] = {}
                        WCOFS_Dataset.data_coordinates[current_grid_name]['lon'] = current_lon
                        WCOFS_Dataset.data_coordinates[current_grid_name]['lat'] = current_lat
        else:
            raise _utilities.NoDataError(f'No PyOFS datasets found for {self.model_datetime} at the given hours.')

    def bounds(self, variable: str = 'psi') -> tuple:
        """
        Returns bounds of grid of given variable.

        :param variable: Variable name.
        :return: Tuple of (west, north, east, south)
        """

        grid_name = WCOFS_Dataset.variable_grids[variable]
        return WCOFS_Dataset.grid_bounds[grid_name]

    def data(self, variable: str, time_index: int) -> numpy.ma.MaskedArray:
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
                output_data = self.netcdf_datasets[dataset_index][variable][day_index, -1, :, :]
        elif time_index in self.netcdf_datasets.keys():
            with self.dataset_locks[time_index]:
                output_data = self.netcdf_datasets[time_index][variable][0, :, :]
        else:
            output_data = numpy.ma.MaskedArray([])

        return output_data

    def data_average(self, variable: str, time_indices: list = None) -> numpy.ma.MaskedArray:
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

            for current_future in concurrent.futures.as_completed(variable_futures):
                variable_data.append(current_future.result())

            del variable_futures

        variable_data = numpy.ma.mean(numpy.ma.dstack(variable_data), axis=2)

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
            variable_mean_futures = {
                concurrency_pool.submit(self.data_average, current_variable_name, hours): current_variable_name for
                current_variable_name in variables}

            for current_future in concurrent.futures.as_completed(variable_mean_futures):
                current_variable_name = variable_mean_futures[current_future]

                current_data = current_future.result()

                self.write_raster(os.path.join(output_dir, f'wcofs_{current_variable_name}.tiff'),
                                  current_variable_name, input_data=current_data, x_size=x_size, y_size=y_size,
                                  drivers=drivers)

            del variable_mean_futures

    def write_raster(self, output_filename: str, variable: str,
                     study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, hours: int = None,
                     input_data: numpy.ma.MaskedArray = None, x_size: float = 0.04, y_size: float = 0.04,
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
            'nodata':    output_data.fill_value.astype(rasterio.float32)
        }

        with MemoryFile() as memory_file:
            with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                memory_raster.write(output_data, 1)

            with memory_file.open() as memory_raster:
                masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

        masked_data = masked_data[0, :, :]

        for current_driver in drivers:
            if current_driver == 'AAIGrid':
                file_extension = '.asc'
                gdal_args.update({'FORCE_CELLSIZE': 'YES'})
            elif current_driver == 'GTiff':
                file_extension = '.tiff'
            elif current_driver == 'GPKG':
                file_extension = '.gpkg'

            current_output_filename = f'{os.path.splitext(output_filename)[0]}{file_extension}'

            print(f'Writing {current_output_filename}')
            with rasterio.open(current_output_filename, 'w', current_driver, **gdal_args) as output_raster:
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
            variable_futures = {
                concurrency_pool.submit(self.data_average, current_variable, time_indices): current_variable for
                current_variable in variables}

            for current_future in concurrent.futures.as_completed(variable_futures):
                current_variable = variable_futures[current_future]
                variable_means[current_variable] = current_future.result()

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

            for current_future in concurrent.futures.as_completed(feature_futures):
                current_result = current_future.result()

                if current_result is not None:
                    layer_features.append(current_result)

        print(f'creating features took {(datetime.datetime.now() - start_time).seconds} seconds')

        start_time = datetime.datetime.now()

        print(f'Writing {output_filename}:{layer_name}')

        # open layer in QGIS
        current_layer = QgsVectorLayer(f'{output_filename}|layername={layer_name}', layer_name, 'ogr')

        # put layer in editing mode
        current_layer.startEditing()

        # add features to layer
        current_layer.dataProvider().addFeatures(layer_features)

        # save changes to layer
        current_layer.commitChanges()

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
            variable_futures = {
                concurrency_pool.submit(self.data_average, current_variable, time_indices): current_variable for
                current_variable in variables}

            for current_future in concurrent.futures.as_completed(variable_futures):
                current_variable = variable_futures[current_future]
                variable_means[current_variable] = current_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        schema = {
            'geometry': 'Point', 'properties': {
                'row': 'int', 'col': 'int', 'rho_lon': 'float', 'rho_lat': 'float'
            }
        }

        for current_variable in variables:
            schema['properties'][current_variable] = 'float'

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

            for current_future in concurrent.futures.as_completed(record_futures):
                current_result = current_future.result()

                if current_result is not None:
                    layer_records.append(current_result)

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
        current_rho_lon = WCOFS_Dataset.data_coordinates['rho']['lon'][row, col]
        current_rho_lat = WCOFS_Dataset.data_coordinates['rho']['lat'][row, col]

        # create feature
        current_feature = QgsFeature()

        current_data = [feature_index, row, col, float(current_rho_lon), float(current_rho_lat)]
        for current_variable in variable_means.keys():
            current_data.append(float(variable_means[current_variable][row, col]))

        current_feature.setGeometry(QgsGeometry(QgsPoint(current_rho_lon, current_rho_lat)))
        current_feature.setAttributes(current_data)

        return current_feature

    def _create_fiona_record(self, variable_means, row, col, feature_index):
        # get coordinates of cell center
        current_rho_lon = WCOFS_Dataset.data_coordinates['rho']['lon'][row, col]
        current_rho_lat = WCOFS_Dataset.data_coordinates['rho']['lat'][row, col]

        current_record = {
            'geometry':      {
                'id': feature_index, 'type': 'Point', 'coordinates': (float(current_rho_lon), float(current_rho_lat))
            }, 'properties': {
                'row': row, 'col': col, 'rho_lon': float(current_rho_lon), 'rho_lat': float(current_rho_lat)
            }
        }

        for current_variable in variable_means.keys():
            current_record['properties'][current_variable] = float(variable_means[current_variable][row, col])

        return current_record

    def __repr__(self):
        used_params = [self.model_datetime.__repr__()]
        optional_params = [self.source, self.time_indices, self.x_size, self.y_size, self.uri]

        for current_param in optional_params:
            if current_param is not None:
                if 'str' in str(type(current_param)):
                    current_param = f'"{current_param}"'
                else:
                    current_param = str(current_param)

                used_params.append(current_param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class WCOFS_Range:
    """
    Range of PyOFS datasets.
    """

    grid_transforms = None
    grid_shapes = None
    grid_bounds = None
    data_coordinates = None
    variable_grids = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, source: str = None,
                 time_indices: list = None):
        """
        Create range of PyOFS datasets from the given time interval.

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

        print(f'Collecting PyOFS stack from {self.start_datetime} to {self.end_datetime}')

        # get all possible model dates that could overlap with the given time interval
        overlapping_models_start_datetime = self.start_datetime - datetime.timedelta(hours=WCOFS_MODEL_HOURS['f'])
        overlapping_models_end_datetime = self.end_datetime - datetime.timedelta(hours=WCOFS_MODEL_HOURS['n'])
        model_dates = _utilities.day_range(_utilities.round_to_day(overlapping_models_start_datetime, 'floor'),
                                           _utilities.round_to_day(overlapping_models_end_datetime, 'ceiling'))

        self.datasets = {}

        # concurrently populate dictionary with PyOFS dataset objects for every time in the given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            dataset_futures = {}

            for current_model_date in model_dates:
                if self.source == 'avg':
                    # construct start and end days from given time interval
                    start_duration = self.start_datetime - current_model_date
                    end_duration = self.end_datetime - current_model_date

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

                    overlapping_days = range(round(start_day), round(end_day))

                    if self.time_indices is not None:
                        current_time_indices = []

                        for current_day in self.time_indices:
                            if current_day >= 0:
                                if current_day - 1 in overlapping_days:
                                    current_time_indices.append(current_day)
                            else:
                                if current_day in overlapping_days:
                                    current_time_indices.append(current_day)
                    else:
                        current_time_indices = overlapping_days
                else:
                    current_model_datetime = current_model_date + datetime.timedelta(hours=3)

                    # construct start and end hours from given time interval
                    start_duration = self.start_datetime - current_model_datetime
                    end_duration = self.end_datetime - current_model_datetime

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

                    overlapping_hours = range(start_hour, end_hour)

                    if self.time_indices is not None:
                        current_time_indices = []

                        for current_overlapping_hour in overlapping_hours:
                            if current_overlapping_hour in self.time_indices:
                                current_time_indices.append(current_overlapping_hour)
                    else:
                        current_time_indices = overlapping_hours

                # get dataset for the current hours (usually all hours)
                if current_time_indices is None or len(current_time_indices) > 0:
                    current_future = concurrency_pool.submit(WCOFS_Dataset, current_model_date, self.source,
                                                             current_time_indices)

                    dataset_futures[current_future] = current_model_date

            for current_future in concurrent.futures.as_completed(dataset_futures):
                current_model_date = dataset_futures[current_future]

                if type(current_future.exception()) is not _utilities.NoDataError:
                    current_result = current_future.result()
                    self.datasets[current_model_date] = current_result

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
                    f'No PyOFS datasets found between {self.start_datetime} and {self.end_datetime}.')

    def data(self, variable: str, model_datetime: datetime.datetime, time_index: int) -> numpy.ma.MaskedArray:
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

            for current_day, current_dataset in self.datasets.items():
                if self.source == 'avg':
                    # get current day index
                    current_timedelta = input_datetime - current_day
                    current_time_index = round(current_timedelta.total_seconds() / (60 * 60 * 24))
                    if current_time_index >= 0:
                        current_time_index += 1
                else:
                    # get current hour index
                    current_timedelta = input_datetime - current_day.replace(hour=3, minute=0, second=0)
                    current_time_index = round(current_timedelta.total_seconds() / (60 * 60))

                if current_time_index in current_dataset.time_indices:
                    current_future = concurrency_pool.submit(current_dataset.data, variable, current_time_index)
                    current_time_index_string = f'{"f" if current_time_index > 0 else "n"}{abs(current_time_index):03}'
                    data_futures[current_future] = f'{current_day.strftime("%Y%m%d")}_{current_time_index_string}'

            for current_future in concurrent.futures.as_completed(data_futures):
                current_model_string = data_futures[current_future]
                current_result = current_future.result()

                if current_result is not None and len(current_result) > 0:
                    output_data[current_model_string] = current_result

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
            data_futures = {concurrency_pool.submit(self.data_stack, variable, current_datetime): current_datetime for
                            current_datetime in time_range}

            for current_future in concurrent.futures.as_completed(data_futures):
                current_datetime = data_futures[current_future]
                current_result = current_future.result()

                if len(current_result) > 0:
                    output_data[current_datetime] = current_result

            del data_futures

        # output_data = numpy.ma.mean(numpy.ma.vstack(output_data), axis=0)
        # output_data.mask[output_data > output_data.fill_value] = True

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

        for current_model_datetime, current_data_stack in input_data.items():
            for current_model_datetime, current_model_data in current_data_stack.items():
                if current_model_datetime in stacked_arrays.keys():
                    stacked_arrays[current_model_datetime] = numpy.dstack(
                            [stacked_arrays[current_model_datetime], current_model_data])
                else:
                    stacked_arrays[current_model_datetime] = current_model_data

        output_data = {}

        # concurrently populate dictionary with average of data stack for each time in given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            data_futures = {}

            for current_model_datetime, current_stacked_array in stacked_arrays.items():
                if len(current_stacked_array.shape) > 2:
                    current_future = concurrency_pool.submit(numpy.ma.mean, current_stacked_array, axis=2)
                    data_futures[current_future] = current_model_datetime
                else:
                    output_data[current_model_datetime] = current_stacked_array

            for current_future in concurrent.futures.as_completed(data_futures):
                current_model_datetime = data_futures[current_future]
                output_data[current_model_datetime] = current_future.result()

            del data_futures
        return output_data

    def write_rasters(self, output_dir: str, variables: list = None,
                      study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                      vector_components: bool = False, x_size: float = 0.04, y_size: float = 0.04,
                      fill_value: float = None, drivers: list = ['GTiff']):
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

        study_area_polygon_filename, layer_name = study_area_polygon_filename.rsplit(':', 1)

        if layer_name == '':
            layer_name = None

        with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
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

        for current_variable in grid_variables:
            output_grid_coordinates[current_variable] = {}

            grid_name = WCOFS_Range.variable_grids[current_variable]

            current_west = WCOFS_Range.grid_bounds[grid_name][0]
            current_north = WCOFS_Range.grid_bounds[grid_name][1]
            current_east = WCOFS_Range.grid_bounds[grid_name][2]
            current_south = WCOFS_Range.grid_bounds[grid_name][3]

            # if None not in (west, north, east, south):
            #     west_cell_offset = round((current_west - west) / x_size)
            #     north_cell_offset = round((current_north - north) / y_size)
            #     east_cell_offset = round((current_east - east) / x_size)
            #     south_cell_offset = round((current_south - south) / y_size)
            #
            #     current_west = west + (west_cell_offset * x_size)
            #     current_north = north + (north_cell_offset * y_size)
            #     current_east = east + (east_cell_offset * x_size)
            #     current_south = south + (south_cell_offset * y_size)

            # PyOFS grid starts at southwest corner
            output_grid_coordinates[current_variable]['lon'] = numpy.arange(current_west, current_east, x_size)
            output_grid_coordinates[current_variable]['lat'] = numpy.arange(current_south, current_north, y_size)

        start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data_averages, current_variable, start_datetime,
                                                        end_datetime): current_variable for current_variable in
                                variables}

            for current_future in concurrent.futures.as_completed(variable_futures):
                current_variable = variable_futures[current_future]
                variable_data_stack_averages[current_variable] = current_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        if vector_components:
            if self.source == '2ds':
                u_name = 'u_sur'
                v_name = 'v_sur'
            else:
                u_name = 'u'
                v_name = 'v'

            if u_name not in variables:
                variable_data_stack_averages[u_name] = self.data_averages(u_name, start_datetime, end_datetime)

            if v_name not in variables:
                variable_data_stack_averages[v_name] = self.data_averages(v_name, start_datetime, end_datetime)

        start_time = datetime.datetime.now()

        interpolated_data = {}

        # concurrently populate dictionary with interpolated data in given grid for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_interpolation_futures = {}

            for current_variable, current_variable_data_stack in variable_data_stack_averages.items():
                print(f'Starting {current_variable} interpolation...')

                current_grid_lon = output_grid_coordinates[current_variable]['lon']
                current_grid_lat = output_grid_coordinates[current_variable]['lat']

                grid_name = WCOFS_Range.variable_grids[current_variable]

                current_lon = WCOFS_Range.data_coordinates[grid_name]['lon'].filled()
                current_lat = WCOFS_Range.data_coordinates[grid_name]['lat'].filled()

                if len(current_grid_lon) > 0:
                    variable_interpolation_futures[current_variable] = {}

                    for current_model_string, current_model_data in current_variable_data_stack.items():
                        current_future = concurrency_pool.submit(interpolate_grid, current_lon, current_lat,
                                                                 current_model_data, current_grid_lon, current_grid_lat)

                        variable_interpolation_futures[current_variable][current_future] = current_model_string

            for current_variable, current_interpolation_futures in variable_interpolation_futures.items():
                interpolated_data[current_variable] = {}

                for current_future in concurrent.futures.as_completed(current_interpolation_futures):
                    current_model_string = current_interpolation_futures[current_future]
                    interpolated_data[current_variable][current_model_string] = current_future.result()

            del variable_interpolation_futures

        print(f'parallel grid interpolation took {(datetime.datetime.now() - start_time).seconds} seconds')

        if vector_components:
            interpolated_data['dir'] = {}
            interpolated_data['mag'] = {}

            u_data_stack = interpolated_data[u_name]
            v_data_stack = interpolated_data[v_name]

            for current_model_string in u_data_stack.keys():
                current_u_data = u_data_stack[current_model_string]
                current_v_data = v_data_stack[current_model_string]

                # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                interpolated_data['dir'][current_model_string] = (numpy.arctan2(current_u_data,
                                                                                current_v_data) + numpy.pi) * (
                                                                         180 / numpy.pi)
                interpolated_data['mag'][current_model_string] = numpy.sqrt(
                        numpy.square(current_u_data) + numpy.square(current_v_data))

            if u_name not in variables:
                del interpolated_data[u_name]

            if v_name not in variables:
                del interpolated_data[v_name]

        # write interpolated grids to raster files
        for current_variable, current_variable_data_stack in interpolated_data.items():
            if len(current_variable_data_stack) > 0:
                current_west = numpy.min(output_grid_coordinates[current_variable]['lon'])
                current_north = numpy.max(output_grid_coordinates[current_variable]['lat'])

                # PyOFS grid starts from southwest corner, but we'll be flipping the data in a moment so lets use the northwest point
                # TODO NOTE: You cannot use a negative (northward) y_size here, otherwise GDALSetGeoTransform will break at line 1253 of rasterio/_io.pyx
                if x_size is not None and y_size is not None:
                    grid_transform = rasterio.transform.from_origin(current_west, current_north, x_size, y_size)
                else:
                    grid_name = WCOFS_Range.variable_grids[current_variable]
                    grid_transform = WCOFS_Range.grid_transforms[grid_name]

                for current_model_string, current_data in current_variable_data_stack.items():
                    if fill_value is not None:
                        current_data.set_fill_value(float(fill_value))

                    # flip the data to ensure northward y_size (see comment above)
                    raster_data = numpy.flip(current_data.filled().astype(rasterio.float32), axis=0)

                    gdal_args = {
                        'width':  raster_data.shape[1], 'height': raster_data.shape[0], 'count': 1,
                        'crs':    RASTERIO_WGS84, 'transform': grid_transform, 'dtype': raster_data.dtype,
                        'nodata': current_data.fill_value.astype(raster_data.dtype)
                    }

                    with MemoryFile() as memory_file:
                        with memory_file.open(driver='GTiff', **gdal_args) as memory_raster:
                            memory_raster.write(raster_data, 1)

                        with memory_file.open() as memory_raster:
                            masked_data, masked_transform = rasterio.mask.mask(memory_raster, [study_area_geojson])

                    masked_data = masked_data[0, :, :]

                    for current_driver in drivers:
                        if current_driver == 'AAIGrid':
                            file_extension = 'asc'
                            gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                        elif current_driver == 'GTiff':
                            file_extension = 'tiff'
                        elif current_driver == 'GPKG':
                            file_extension = 'gpkg'

                        current_output_filename = os.path.join(output_dir,
                                                               f'wcofs_{current_variable}_{current_model_string}.{file_extension}')

                        print(f'Writing to {current_output_filename}')
                        with rasterio.open(current_output_filename, mode='w', driver=current_driver,
                                           **gdal_args) as output_raster:
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
            variable_futures = {concurrency_pool.submit(self.data_averages, current_variable, start_datetime,
                                                        end_datetime): current_variable for current_variable in
                                variables}

            for current_future in concurrent.futures.as_completed(variable_futures):
                current_variable = variable_futures[current_future]
                variable_data_stack_averages[current_variable] = current_future.result()

            del variable_futures

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).seconds} seconds')

        model_datetimes = [current_model_datetime for current_model_datetime in
                           next(iter(variable_data_stack_averages.values())).keys()]

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {'fid': 'int'}
        }

        for current_variable in variables:
            schema['properties'][current_variable] = 'float'

        # create layers
        for current_model_datetime in model_datetimes:
            fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84,
                       layer=current_model_datetime.strftime('%Y%m%d')).close()

        print('Creating features...')

        # create features
        layer_features = {current_model_datetime.strftime('%Y%m%d'): [] for current_model_datetime in model_datetimes}

        layer_feature_indices = {current_layer_name: 1 for current_layer_name in layer_features.keys()}

        grid_height, grid_width = WCOFS_Range.data_coordinates['psi']['lon'].shape

        for col in range(grid_width):
            for row in range(grid_height):
                # get coordinates of cell center
                current_rho_lon = WCOFS_Range.data_coordinates['rho']['lon'][row, col]
                current_rho_lat = WCOFS_Range.data_coordinates['rho']['lat'][row, col]

                for current_model_datetime in model_datetimes:
                    current_data = [variable_data_stack_averages[current_variable][current_model_datetime][row, col] for
                                    current_variable in variables]

                    if numpy.all([current_entry is not numpy.ma.masked for current_entry in current_data]):
                        current_feature = QgsFeature()
                        current_feature.setGeometry(QgsGeometry(QgsPoint(current_rho_lon, current_rho_lat)))
                        current_feature.setAttributes(
                                [layer_feature_indices[current_model_datetime.strftime('%Y%m%d')]] + [float(entry) for
                                                                                                      entry in
                                                                                                      current_data])

                        layer_features[current_model_datetime.strftime('%Y%m%d')].append(current_feature)
                        layer_feature_indices[current_model_datetime.strftime('%Y%m%d')] += 1
            print(f'processed column {col} of {grid_width}')

        # write queued features to layer
        for current_layer_name, current_layer_features in layer_features.items():
            print(f'Writing {output_filename}:{current_layer_name}')
            current_layer = QgsVectorLayer(f'{output_filename}|layername={current_layer_name}', current_layer_name,
                                           'ogr')
            current_layer.startEditing()
            current_layer.dataProvider().addFeatures(current_layer_features)
            current_layer.commitChanges()

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.source, self.time_indices]

        for current_param in optional_params:
            if current_param is not None:
                if 'str' in str(type(current_param)):
                    current_param = f'"{current_param}"'
                else:
                    current_param = str(current_param)

                used_params.append(current_param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


def interpolate_grid(input_lon: numpy.ndarray, input_lat: numpy.ndarray, input_data: numpy.ma.MaskedArray,
                     output_lon: numpy.ndarray, output_lat: numpy.ndarray) -> numpy.ma.MaskedArray:
    """
    Interpolates the given data onto a coordinate grid.

    :param input_lon: Matrix of X coordinates in original grid.
    :param input_lat: Matrix of Y coordinates in original grid.
    :param input_data: Masked array of data values in original grid.
    :param output_lon: Longitude values of output grid.
    :param output_lat: Latitude values of output grid.
    :return: Interpolated values within WGS84 coordinate grid.
    """

    fill_value = 1e+20

    # get unmasked values only
    input_lon = input_lon[~input_data.mask]
    input_lat = input_lat[~input_data.mask]
    input_data = input_data[~input_data.mask]

    output_lon = output_lon[None, :]
    output_lat = output_lat[:, None]

    # get grid interpolation
    interpolated_grid = scipy.interpolate.griddata((input_lon, input_lat), input_data, (output_lon, output_lat),
                                                   method='nearest', fill_value=fill_value)

    # input_grid = numpy.column_stack([input_lon.flatten(), input_lat.flatten(), input_data.flatten()])
    #
    # out_coords = numpy.stack([component.flatten() for component in numpy.meshgrid(output_lon, output_lat)])
    #
    # interpolated_grid = scipy.ndimage.interpolation.map_coordinates(input_grid, out_coords,
    #                                                                 mode='nearest', cval=fill_value)
    #
    # interpolated_grid = interpolated_grid.reshape(len(output_lon), len(output_lat))

    # mask out interpolated values on the edge of the fill
    # output_mask = numpy.logical_or(output_mask, interpolated_grid == fill_value, interpolated_grid > 1000)

    interpolated_grid = numpy.ma.MaskedArray(interpolated_grid, mask=interpolated_grid == fill_value,
                                             fill_value=fill_value)

    return interpolated_grid


def write_convex_hull(netcdf_dataset: netCDF4.Dataset, output_filename: str, grid_name: str = 'psi'):
    """
    Extract the convex hull from the coordinate values of the given PyOFS NetCDF dataset, and write it to a file.

    :param netcdf_dataset: PyOFS format NetCDF dataset object.
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
        points.append((lon_var[0, col].filled(), lat_var[0, col].filled()))

    # downwards over right col
    for row in range(height - 1):
        points.append((lon_var[row, width - 1].filled(), lat_var[row, width - 1].filled()))

    # leftward over bottom row
    for col in range(width - 1, -1, -1):
        points.append((lon_var[height - 1, col].filled(), lat_var[height - 1, col].filled()))

    # upwards over left col
    for row in range(height - 1, -1, -1):
        points.append((lon_var[row, 0].filled(), lat_var[row, 0].filled()))

    polygon = shapely.geometry.Polygon(points)

    schema = {'geometry': 'Polygon', 'properties': {'name': 'str'}}

    with fiona.open(output_filename, 'w', 'GPKG', schema=schema, crs=FIONA_WGS84, layer=layer_name) as vector_file:
        vector_file.write({'properties': {'name': layer_name}, 'geometry': shapely.geometry.mapping(polygon)})


if __name__ == '__main__':
    # start_datetime = datetime.datetime(2018, 8, 10)
    # end_datetime = datetime.datetime(2018, 8, 11)
    #
    # output_dir = r'C:\Data\output\test'
    #
    # wcofs_range = WCOFS_Range(start_datetime, end_datetime, source='avg')
    # wcofs_range.write_rasters(output_dir, ['temp'])

    wcofs_dataset = WCOFS_Dataset(datetime.datetime(2018, 8, 31), time_indices=[1])
    # mask_rho = wcofs_dataset.netcdf_datasets[1]['mask_rho'][:].filled().astype(rasterio.int16)
    #
    # # rasterio_wcofs_crs = rasterio.crs.CRS.from_wkt('+proj=ob_tran +o_proj=longlat +o_lat_p=37.4 +o_lon_p=-57.6')
    #
    # wcofs_origin = pyproj.transform(pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'),
    #                  pyproj.Proj('+proj=ob_tran +o_proj=longlat +o_lat_p=37.4 +o_lon_p=-57.6'), -123.00442595126076,
    #                  18.441782684328818, 0)
    #
    # rasterio_wcofs_transform = rasterio.transform.from_origin(wcofs_origin[0], wcofs_origin[1], 0.039, -0.039)
    #
    # with rasterio.open(os.path.join(DATA_DIR, r'output\test\mask_rho.tiff'), 'w', 'GTiff', height=mask_rho.shape[0],
    #                    width=mask_rho.shape[1], count=1, crs=RASTERIO_WGS84, transform=rasterio_wcofs_transform,
    #                    dtype=mask_rho.dtype) as output_raster:
    #     output_raster.write(mask_rho, 1)

    # from qgis.core import QgsApplication
    # qgis_application = QgsApplication([], True, None)
    # qgis_application.setPrefixPath(os.environ['QGIS_PREFIX_PATH'], True)
    # qgis_application.initQgis()
    #
    # wcofs_dataset.write_vector(os.path.join(DATA_DIR, r'output\test\wcofs_rho.gpkg'), layer_name='mask_qgis')
    # wcofs_dataset.write_vector_fiona(os.path.join(DATA_DIR, r'output\test\wcofs_rho.gpkg'), layer_name='mask_fiona')

    print('done')
