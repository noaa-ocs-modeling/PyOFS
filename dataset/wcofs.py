# coding=utf-8
"""
WCOFS model output data collection and transformation by interpolation onto Cartesian grid.

Created on Jun 25, 2018

@author: zachary.burnett
"""

from concurrent import futures
import datetime
import os
import threading

import fiona
import fiona.crs
import numpy
import pyproj
import rasterio.control
from rasterio.io import MemoryFile
import rasterio.mask
import rasterio.warp
from scipy import interpolate
import shapely.geometry
import xarray

from dataset import _utilities
from main import DATA_DIR

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})
FIONA_WGS84 = fiona.crs.from_epsg(4326)
WCOFS_PROJ4 = pyproj.Proj('+proj=ob_tran +o_proj=longlat +o_lat_p=37.4 +o_lon_p=-57.6')

GRID_LOCATIONS = {'face': 'rho', 'edge1': 'u', 'edge2': 'v', 'node': 'psi'}
COORDINATES = ['grid', 'ocean_time', 'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v', 'lon_psi', 'lat_psi',
               'angle', 'pm', 'pn']
STATIC_VARIABLES = ['h', 'f', 'mask_rho', 'mask_u', 'mask_v', 'mask_psi']
DATA_VARIABLES = {'other': ['u', 'v', 'w', 'temp', 'salt'], '2ds': ['u_sur', 'v_sur', 'temp_sur', 'salt_sur']}
WCOFS_MODEL_HOURS = {'n': -24, 'f': 72}
WCOFS_MODEL_RUN_HOUR = 3

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs.gpkg:study_area')
WCOFS_4KM_GRID_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs_4km_grid.nc')
WCOFS_2KM_GRID_FILENAME = os.path.join(DATA_DIR, 'reference', 'wcofs_2km_grid.nc')

GLOBAL_LOCK = threading.Lock()

SOURCE_URL = 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS'


class WCOFS_Dataset:
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

    def __init__(self, model_date: datetime.datetime, source: str = None, time_deltas: list = None,
                 x_size: float = None, y_size: float = None, grid_filename: str = None, source_url: str = None,
                 wcofs_string: str = 'wcofs'):
        """
        Creates new dataset object from datetime and given model parameters.

        :param model_date: Model run date.
        :param source: One of 'stations', 'fields', 'avg', or '2ds'.
        :param time_deltas: List of integers of times from model start for which to retrieve data (days for avg, hours for others).
        :param x_size: Size of cell in X direction.
        :param y_size: Size of cell in Y direction.
        :param grid_filename: Filename of NetCDF containing WCOFS grid coordinates.
        :param source_url: Directory containing NetCDF files.
        :param wcofs_string: WCOFS string in filename.
        :raises ValueError: if source is not valid.
        :raises NoDataError: if no datasets exist for the given model run.
        """

        valid_source_strings = ['stations', 'fields', 'avg', '2ds']

        if source is None:
            source = 'avg'
        elif source not in valid_source_strings:
            raise ValueError(f'Location must be one of {valid_source_strings}')

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
            source_url = SOURCE_URL
        
        # set start time to WCOFS model run time (0300 UTC)
        self.model_datetime = model_date.replace(hour=3, minute=0, second=0, microsecond=0)
        self.x_size = x_size
        self.y_size = y_size
        self.grid_filename = grid_filename
        self.source_url = source_url
        self.wcofs_string = wcofs_string

        month_string = self.model_datetime.strftime('%Y%m')
        date_string = self.model_datetime.strftime('%Y%m%d')

        self.netcdf_datasets = {}

        if self.source == 'avg':
            for day in self.time_deltas:
                if day > 0 and 1 in self.netcdf_datasets.keys():
                    continue

                model_type = 'nowcast' if day < 0 else 'forecast'
                url = f'{self.source_url}/{month_string}/nos.{self.wcofs_string}.avg.{model_type}.{date_string}.t{WCOFS_MODEL_RUN_HOUR:02}z.nc'

                try:
                    self.netcdf_datasets[-1 if day < 0 else 1] = xarray.open_dataset(url, decode_times=False)
                except OSError:
                    print(f'No WCOFS dataset found at {url}')
        else:
            # concurrently populate dictionary with NetCDF datasets for every hour in the given list of hours
            with futures.ThreadPoolExecutor() as concurrency_pool:
                running_futures = {}
                
                for hour in self.time_deltas:
                    url = f'{self.source_url}/{month_string}/nos.{self.wcofs_string}.{self.location}.{"n" if hour <= 0 else "f"}' + \
                          f'{abs(hour):03}.{date_string}.t{self.cycle:02}z.nc'
                    
                    future = concurrency_pool.submit(xarray.open_dataset, url, decode_times=False)
                    running_futures[future] = hour

                for completed_future in futures.as_completed(running_futures):
                    try:
                        hour = running_futures[completed_future]
                        result = completed_future.result()

                        if result is not None:
                            self.netcdf_datasets[hour] = result
                    except OSError:
                        # print(f'No WCOFS dataset found at {url}')
                        pass

                del running_futures
        
        if len(self.netcdf_datasets) > 0:
            self.sample_netcdf_dataset = next(iter(self.netcdf_datasets.values()))
            self.dataset_locks = {time_delta: threading.Lock() for time_delta in self.netcdf_datasets.keys()}

            with GLOBAL_LOCK:
                if WCOFS_Dataset.variable_grids is None:
                    WCOFS_Dataset.variable_grids = {}

                    for variable_name, variable in self.sample_netcdf_dataset.data_vars.items():
                        if 'location' in variable.attrs:
                            grid_name = GRID_LOCATIONS[variable.location]
                            WCOFS_Dataset.variable_grids[variable_name] = grid_name

            with GLOBAL_LOCK:
                if WCOFS_Dataset.data_coordinates is None:
                    WCOFS_Dataset.data_coordinates = {}
                    WCOFS_Dataset.masks = {}

                    wcofs_grid = xarray.open_dataset(self.grid_filename,
                                                     decode_times=False) if self.grid_filename is not None else self.sample_netcdf_dataset

                    for grid_name in GRID_LOCATIONS.values():
                        WCOFS_Dataset.masks[grid_name] = ~(wcofs_grid[f'mask_{grid_name}'].values).astype(bool)

                        lon = wcofs_grid[f'lon_{grid_name}'].values
                        lat = wcofs_grid[f'lat_{grid_name}'].values

                        WCOFS_Dataset.data_coordinates[grid_name] = {}
                        WCOFS_Dataset.data_coordinates[grid_name]['lon'] = lon
                        WCOFS_Dataset.data_coordinates[grid_name]['lat'] = lat

                    WCOFS_Dataset.angle = wcofs_grid['angle'].values

            with GLOBAL_LOCK:
                if WCOFS_Dataset.grid_shapes is None:
                    WCOFS_Dataset.grid_shapes = {}

                    for grid_name in GRID_LOCATIONS.values():
                        WCOFS_Dataset.grid_shapes[grid_name] = WCOFS_Dataset.data_coordinates[grid_name]['lon'].shape

            # set pixel resolution if not specified
            if self.x_size is None:
                self.x_size = numpy.max(numpy.diff(WCOFS_Dataset.data_coordinates['psi']['lon']))
            if self.y_size is None:
                self.y_size = numpy.max(numpy.diff(WCOFS_Dataset.data_coordinates['psi']['lat']))

            with GLOBAL_LOCK:
                if WCOFS_Dataset.grid_transforms is None:
                    WCOFS_Dataset.grid_transforms = {}

                    for grid_name in GRID_LOCATIONS.values():
                        lon = WCOFS_Dataset.data_coordinates[grid_name]['lon']
                        lat = WCOFS_Dataset.data_coordinates[grid_name]['lat']

                        west = numpy.min(lon)
                        south = numpy.min(lat)

                        WCOFS_Dataset.grid_transforms[grid_name] = rasterio.transform.from_origin(west=west,
                                                                                                  north=south,
                                                                                                  xsize=self.x_size,
                                                                                                  ysize=-self.y_size)

            with GLOBAL_LOCK:
                if WCOFS_Dataset.grid_bounds is None and 'wcofs_grid' in locals():
                    WCOFS_Dataset.grid_bounds = {}

                    for grid_name in GRID_LOCATIONS.values():
                        lon = WCOFS_Dataset.data_coordinates[grid_name]['lon']
                        lat = WCOFS_Dataset.data_coordinates[grid_name]['lat']

                        west = numpy.min(lon)
                        north = numpy.max(lat)
                        east = numpy.max(lon)
                        south = numpy.min(lat)

                        WCOFS_Dataset.grid_bounds[grid_name] = (west, north, east, south)

        else:
            raise _utilities.NoDataError(
                f'No WCOFS datasets found for {self.model_datetime} from {self.source_url} at the given time deltas ({self.time_deltas}).')
    
    def bounds(self, variable: str = 'psi') -> tuple:
        """
        Returns bounds of grid of given variable.

        :param variable: Variable name.
        :return: Tuple of (west, north, east, south)
        """

        grid_name = WCOFS_Dataset.variable_grids[variable]
        return WCOFS_Dataset.grid_bounds[grid_name]

    def data(self, variable: str, time_delta: int) -> numpy.ndarray:
        """
        Get data of specified variable at specified hour.

        :param variable: Name of variable to retrieve.
        :param time_delta: Time index to retrieve (days for avg, hours for others).
        :return: Array of data.
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
                        if variable in ['u', 'v']:
                            raw_u = self.netcdf_datasets[dataset_index]['u'][day_index, -1, :-1, :].values
                            raw_v = self.netcdf_datasets[dataset_index]['v'][day_index, -1, :, :-1].values
                            theta = WCOFS_Dataset.angle[:-1, :-1]
            
                            if variable == 'u':
                                output_data = raw_u * numpy.cos(theta) - raw_v * numpy.sin(theta)
                                extra_row = numpy.empty((1, output_data.shape[1]), dtype=output_data.dtype)
                                extra_row[:] = numpy.nan
                                output_data = numpy.concatenate((output_data, extra_row), axis=0)
                            elif variable == 'v':
                                output_data = raw_u * numpy.sin(theta) + raw_v * numpy.cos(theta)
                                extra_column = numpy.empty((output_data.shape[0], 1), dtype=output_data.dtype)
                                extra_column[:] = numpy.nan
                                output_data = numpy.concatenate((output_data, extra_column), axis=1)
                        else:
                            data_variable = self.netcdf_datasets[dataset_index][variable]
                            if len(data_variable.shape) == 3:
                                output_data = data_variable[day_index, :, :].values
                            if len(data_variable.shape) == 4:
                                output_data = data_variable[day_index, -1, :, :].values
            else:
                with self.dataset_locks[time_delta]:
                    output_data = self.netcdf_datasets[time_delta][variable][0, :, :].values

        return output_data

    def data_average(self, variable: str, time_deltas: list = None) -> numpy.ndarray:
        """
        Gets average of data from given time deltas.

        :param variable: Variable to use.
        :param time_deltas: List of integers of time indices to use in average (days for avg, hours for others).
        :return: Array of data.
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

    def write_rasters(self, output_dir: str, variables: list = None, filename_suffix: str = None,
                      time_deltas: list = None, study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      vector_components: bool = False, x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                      drivers: list = ['GTiff']):
        """
        Write averaged raster data of given variables to given output directory.

        :param output_dir: Path to directory.
        :param variables: Variable names to use.
        :param time_deltas: List of time indices to write.
        :param filename_suffix: Suffix for filenames.
        :param study_area_polygon_filename: Path to vector file containing study area boundary.
        :param vector_components: Whether to write direction and magnitude rasters.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        if variables is None:
            variables = DATA_VARIABLES['2ds'] if self.source == '2ds' else DATA_VARIABLES['other']

        study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
            study_area_geojson = next(iter(vector_layer))['geometry']

        # if cell sizes are not specified, get maximum coordinate differences between cell points on psi grid
        if x_size is None:
            x_size = numpy.max(numpy.diff(self.sample_netcdf_dataset['lon_psi'][:]))
        if y_size is None:
            y_size = numpy.max(numpy.diff(self.sample_netcdf_dataset['lat_psi'][:]))

        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        if vector_components:
            self.variable_grids['dir'] = 'rho'
            self.variable_grids['mag'] = 'rho'
            grid_variables = variables + ['dir', 'mag']
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

        start_time = datetime.datetime.now()

        variable_means = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}
            
            for variable in variables:
                wcofs_future = concurrency_pool.submit(self.data_average, variable, time_deltas)
                running_futures[wcofs_future] = variable

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                result = completed_future.result()

                if result is not None:
                    variable_means[variable] = result
            
            del running_futures
        
        if vector_components:
            if self.source == '2ds':
                u_name = 'u_sur'
                v_name = 'v_sur'
            else:
                u_name = 'u'
                v_name = 'v'

            if u_name not in variable_means:
                u_data = self.data_average(u_name, time_deltas)
    
                if u_data is not None:
                    variable_means[u_name] = u_data
            
            if v_name not in variable_means:
                v_data = self.data_average(v_name, time_deltas)
    
                if v_data is not None:
                    variable_means[v_name] = v_data
        
        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        start_time = datetime.datetime.now()

        interpolated_data = {}

        if len(variable_means) > 0:
            # concurrently populate dictionary with interpolated data in given grid for each variable
            with futures.ThreadPoolExecutor() as concurrency_pool:
                running_futures = {}
        
                for variable, variable_data in variable_means.items():
                    if variable_data is not None:
                        print(f'Starting {variable} interpolation...')
                
                        grid_lon = output_grid_coordinates[variable]['lon']
                        grid_lat = output_grid_coordinates[variable]['lat']
                
                        grid_name = self.variable_grids[variable]
                
                        lon = self.data_coordinates[grid_name]['lon']
                        lat = self.data_coordinates[grid_name]['lat']
                
                        if len(grid_lon) > 0:
                            running_future = concurrency_pool.submit(interpolate_grid, lon, lat, variable_data,
                                                                     grid_lon,
                                                                     grid_lat)
                            running_futures[running_future] = variable
        
                for completed_future in futures.as_completed(running_futures):
                    variable = running_futures[completed_future]
                    result = completed_future.result()
            
                    if result is not None:
                        interpolated_data[variable] = result
        
                del running_futures
    
            if vector_components:
                if 'u' in interpolated_data and 'v' in interpolated_data:
                    interpolated_data['dir'] = {}
                    interpolated_data['mag'] = {}
            
                    u_data = interpolated_data[u_name]
                    v_data = interpolated_data[v_name]
            
                    # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                    interpolated_data['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                    interpolated_data['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))
            
                    if u_name not in variables:
                        del interpolated_data[u_name]
            
                    if v_name not in variables:
                        del interpolated_data[v_name]
    
            print(f'parallel grid interpolation took ' + \
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
                'crs': RASTERIO_WGS84, 'transform': grid_transform, 'dtype': raster_data.dtype,
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

                output_filename = os.path.join(output_dir,
                                               f'wcofs_{variable}_{self.model_datetime.strftime("%Y%m%d")}' + \
                                               f'{filename_suffix}.{file_extension}')

            if os.path.isfile(output_filename):
                os.remove(output_filename)

            print(f'Writing to {output_filename}')
            with rasterio.open(output_filename, mode='w', driver=driver, **gdal_args) as output_raster:
                output_raster.write(masked_data, 1)


def write_vector(self, output_filename: str, layer_name: str = None, time_deltas: list = None):
    """
    Write average of surface velocity vector data for all hours in the given time interval to the provided output file.

    :param output_filename: Path to output file.
    :param layer_name: Name of layer to write.
    :param time_deltas: List of integers of hours to use in average.
    """
    
    variables = DATA_VARIABLES['2ds'] if self.source == '2ds' else DATA_VARIABLES['other']
    
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
    
    print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')
    
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
    
    with futures.ThreadPoolExecutor() as concurrency_pool:
        feature_index = 1
        running_futures = []
        
        for col in range(grid_width):
            for row in range(grid_height):
                if WCOFS_Dataset.masks['psi'][row, col] == 0:
                    # check if current record is unmasked
                    running_futures.append(
                        concurrency_pool.submit(self._create_fiona_record, variable_means, row, col,
                                                feature_index))
                    feature_index += 1
        
        for completed_future in futures.as_completed(running_futures):
            result = completed_future.result()
            
            if result is not None:
                layer_records.append(result)
    
    print(f'creating records took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')
    
    start_time = datetime.datetime.now()
    
    print(f'Writing {output_filename}:{layer_name}')
    
    # create layer
    with fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84,
                    layer=layer_name) as output_vector_file:
        output_vector_file.writerecords(layer_records)
    
    print(f'writing records took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')


def _create_fiona_record(self, variable_means, row, col, feature_index):
    # get coordinates of cell center
    rho_lon = WCOFS_Dataset.data_coordinates['rho']['lon'][row, col]
    rho_lat = WCOFS_Dataset.data_coordinates['rho']['lat'][row, col]
    
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


def __repr__(self):
    used_params = [self.model_datetime.__repr__()]
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


class WCOFS_Range:
    """
    Range of WCOFS datasets.
    """

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, source: str = '2ds',
                 time_deltas: list = None, x_size: float = None, y_size: float = None, grid_filename: str = None,
                 source_url: str = None, wcofs_string: str = 'wcofs'):
        """
        Create range of WCOFS datasets from the given time interval.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param source: One of 'stations', 'fields', 'avg', or '2ds'.
        :param time_deltas: List of time indices (nowcast or forecast) to retrieve.
        :param x_size: Size of cell in X direction.
        :param y_size: Size of cell in Y direction.
        :param grid_filename: Filename of NetCDF containing WCOFS grid coordinates.
        :param source_url: Directory containing NetCDF files.
        :param wcofs_string: WCOFS string in filename.
        :raises NoDataError: if data does not exist.
        """

        self.source = source
        self.time_deltas = time_deltas
        self.x_size = x_size
        self.y_size = y_size
        self.grid_filename = grid_filename
        self.source_url = source_url
        self.wcofs_string = wcofs_string

        if self.source == 'avg':
            self.start_datetime = _utilities.round_to_day(start_datetime)
            self.end_datetime = _utilities.round_to_day(end_datetime)
        else:
            self.start_datetime = _utilities.round_to_hour(start_datetime)
            self.end_datetime = _utilities.round_to_hour(end_datetime)

        print(f'Collecting WCOFS stack between {self.start_datetime} and {self.end_datetime}...')

        # get all possible model dates that could overlap with the given time interval
        overlapping_start_datetime = self.start_datetime - datetime.timedelta(hours=WCOFS_MODEL_HOURS['f'] - 24)
        overlapping_end_datetime = self.end_datetime + datetime.timedelta(hours=-WCOFS_MODEL_HOURS['n'] - 24)
        model_dates = _utilities.range_daily(_utilities.round_to_day(overlapping_start_datetime, 'floor'),
                                             _utilities.round_to_day(overlapping_end_datetime, 'ceiling'))

        self.datasets = {}

        # concurrently populate dictionary with WCOFS dataset objects for every time in the given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}
            
            for model_date in model_dates:
                if self.source == 'avg':
                    # construct start and end days from given time interval
                    start_duration = self.start_datetime - model_date
                    end_duration = self.end_datetime - model_date

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

                    if self.time_deltas is not None:
                        time_deltas = []

                        for overlapping_hour in overlapping_hours:
                            if overlapping_hour in self.time_deltas:
                                time_deltas.append(overlapping_hour)
                    else:
                        time_deltas = overlapping_hours

                # get dataset for the current hours (usually all hours)
                if time_deltas is None or len(time_deltas) > 0:
                    future = concurrency_pool.submit(WCOFS_Dataset, model_date=model_date, source=self.source,
                                                     time_deltas=self.time_deltas, x_size=self.x_size,
                                                     y_size=self.y_size, grid_filename=self.grid_filename,
                                                     source_url=self.source_url, wcofs_string=self.wcofs_string)

                    running_futures[future] = model_date

            for completed_future in futures.as_completed(running_futures):
                model_date = running_futures[completed_future]
                
                if type(completed_future.exception()) is not _utilities.NoDataError:
                    result = completed_future.result()
                    self.datasets[model_date] = result

            del running_futures
        
        if len(self.datasets) > 0:
            self.sample_wcofs_dataset = next(iter(self.datasets.values()))

            self.grid_transforms = WCOFS_Dataset.grid_transforms
            self.grid_shapes = WCOFS_Dataset.grid_shapes
            self.grid_bounds = WCOFS_Dataset.grid_bounds
            self.data_coordinates = WCOFS_Dataset.data_coordinates
            self.variable_grids = WCOFS_Dataset.variable_grids
        else:
            raise _utilities.NoDataError(
                f'No WCOFS datasets found between {self.start_datetime} and {self.end_datetime}.')

    def data(self, variable: str, model_datetime: datetime.datetime, time_delta: int) -> numpy.ndarray:
        """
        Return data from given model run at given variable and hour.

        :param variable: Name of variable to use.
        :param model_datetime: Datetime of start of model run.
        :param time_delta: Index of time to retrieve (days for avg, hours for others).
        :return: Matrix of data.
        """

        return self.datasets[model_datetime].data(variable, time_delta)

    def data_stack(self, variable: str, input_datetime: datetime.datetime) -> dict:
        """
        Return dictionary of data for each model run within the given variable and datetime.

        :param variable: Name of variable to use.
        :param input_datetime: Datetime from which to retrieve data.
        :return: Dictionary of data for every model in the given datetime.
        """

        output_data = {}

        # concurrently populate dictionary with dictionaries for each model intersection with the given datetime
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}
            
            for day, dataset in self.datasets.items():
                if self.source == 'avg':
                    # get current day index
                    time_difference = input_datetime - day
                    time_delta = round(time_difference.total_seconds() / (60 * 60 * 24))
                else:
                    # get current hour index
                    time_difference = input_datetime - day.replace(hour=3, minute=0, second=0)
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
            time_range = _utilities.range_daily(_utilities.round_to_day(start_datetime),
                                                _utilities.round_to_day(end_datetime))
        else:
            time_range = _utilities.range_hourly(_utilities.round_to_hour(start_datetime),
                                                 _utilities.round_to_hour(end_datetime))

        output_data = {}

        # concurrently populate dictionary with data stack for each time in given time interval
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {concurrency_pool.submit(self.data_stack, variable, data_datetime): data_datetime for
                               data_datetime in time_range}

            for completed_future in futures.as_completed(running_futures):
                data_datetime = running_futures[completed_future]
                result = completed_future.result()

                if len(result) > 0:
                    output_data[data_datetime] = result

            del running_futures
        
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
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}
            
            for model_datetime, stacked_array in stacked_arrays.items():
                if len(stacked_array.shape) > 2:
                    future = concurrency_pool.submit(numpy.mean, stacked_array, axis=2)
                    running_futures[future] = model_datetime
                else:
                    output_data[model_datetime] = stacked_array

            for completed_future in futures.as_completed(running_futures):
                model_datetime = running_futures[completed_future]
                output_data[model_datetime] = completed_future.result()

            del running_futures
        return output_data

    def write_rasters(self, output_dir: str, variables: list = None, filename_suffix: str = None,
                      study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                      start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                      vector_components: bool = False, x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                      drivers: list = ['GTiff']):
        """
        Write raster data of given variables to given output directory, averaged over given time interval.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to use.
        :param filename_suffix: Suffix for filenames.
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
            variables = DATA_VARIABLES['2ds'] if self.source == '2ds' else DATA_VARIABLES['other']

        study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)

        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None

        with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
            study_area_geojson = next(iter(vector_layer))['geometry']

        # if cell sizes are not specified, get maximum coordinate differences between cell points on psi grid
        if x_size is None:
            x_size = numpy.max(numpy.diff(self.sample_wcofs_dataset.sample_netcdf_dataset['lon_psi'][:]))
        if y_size is None:
            y_size = numpy.max(numpy.diff(self.sample_wcofs_dataset.sample_netcdf_dataset['lat_psi'][:]))

        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        if vector_components:
            self.variable_grids['dir'] = 'rho'
            self.variable_grids['mag'] = 'rho'
            grid_variables = variables + ['dir', 'mag']
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

        start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_datetime, end_datetime): variable for
                variable in variables}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del running_futures
        
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

        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        start_time = datetime.datetime.now()

        interpolated_data = {}

        # concurrently populate dictionary with interpolated data in given grid for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {}
            
            for variable, variable_data_stack in variable_data_stack_averages.items():
                print(f'Starting {variable} interpolation...')

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

        print(f'parallel grid interpolation took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

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
                        'crs': RASTERIO_WGS84, 'transform': grid_transform, 'dtype': raster_data.dtype,
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

                        output_filename = os.path.join(output_dir,
                                                       f'wcofs_{variable}_{model_string}{filename_suffix}.{file_extension}')

                        if os.path.isfile(output_filename):
                            os.remove(output_filename)

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
            variables = DATA_VARIABLES['2ds'] if self.source == '2ds' else DATA_VARIABLES['other']

        start_time = datetime.datetime.now()

        variable_data_stack_averages = {}

        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            running_futures = {
                concurrency_pool.submit(self.data_averages, variable, start_datetime, end_datetime): variable for
                variable in variables}

            for completed_future in futures.as_completed(running_futures):
                variable = running_futures[completed_future]
                variable_data_stack_averages[variable] = completed_future.result()

            del running_futures
        
        print(f'parallel data aggregation took {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

        model_datetime_strings = [model_datetime for model_datetime in
                                  next(iter(variable_data_stack_averages.values())).keys()]

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {'lon': 'float', 'lat': 'float'}
        }

        for variable in variables:
            schema['properties'][variable] = 'float'

        print('Creating records...')

        # create features
        layers = {model_datetime: [] for model_datetime in model_datetime_strings}

        layer_feature_indices = {layer_name: 1 for layer_name in layers.keys()}

        grid_height, grid_width = self.data_coordinates['psi']['lon'].shape

        for col in range(grid_width):
            for row in range(grid_height):
                # get coordinates of cell center
                rho_lon = self.data_coordinates['rho']['lon'][row, col]
                rho_lat = self.data_coordinates['rho']['lat'][row, col]

                for model_datetime_string in model_datetime_strings:
                    data = [float(variable_data_stack_averages[variable][model_datetime_string][row, col]) for variable
                            in variables]

                    if not numpy.isnan(data).all():
                        record = {
                            'id': layer_feature_indices[model_datetime_string],
                            'geometry': {'type': 'Point', 'coordinates': (rho_lon, rho_lat)}, 'properties': {
                                'lon': float(rho_lon), 'lat': float(rho_lat)
                            }
                        }

                        record['properties'].update(dict(zip(variables, data)))

                        layers[model_datetime_string].append(record)
                        layer_feature_indices[model_datetime_string] += 1

        # write queued features to layer
        for layer_name, layer_records in layers.items():
            print(f'Writing {output_filename}:{layer_name}')
            with fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84,
                            layer=layer_name) as layer:
                layer.writerecords(layer_records)

    def to_xarray(self, variables: list = None, mean: bool = True) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: List of variables to use.
        :param mean: Whether to average all time indices.
        :return: xarray Dataset of given variables.
        """

        data_arrays = {}

        if variables is None:
            variables = DATA_VARIABLES['2ds'] if self.source == '2ds' else DATA_VARIABLES['other']

        if mean:
            for variable in variables:
                grid = self.variable_grids[variable]

                data = self.data_averages(variable)
                data_stack = numpy.stack(
                    [time_delta_data for time_delta, time_delta_data in sorted(data.items(), reverse=True)], 0)

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
                    model_times = _utilities.range_daily(_utilities.round_to_day(self.start_datetime),
                                                         _utilities.round_to_day(self.end_datetime))
                else:
                    model_times = _utilities.range_hourly(_utilities.round_to_hour(self.start_datetime),
                                                          _utilities.round_to_hour(self.end_datetime))

                data = {}
                time_deltas = None

                for model_time in model_times:
                    model_datetime_data = self.data_stack(variable, model_time)
                    data[model_time] = numpy.stack(
                        [time_delta_data for time_delta, time_delta_data in
                         sorted(model_datetime_data.items(), reverse=True)], 0)

                    if time_deltas is None:
                        time_deltas = sorted(model_datetime_data.keys(), reverse=True)

                data_stack = numpy.stack([time_delta_data for time_delta, time_delta_data in sorted(data.items())], 0)

                data_array = xarray.DataArray(data_stack,
                                              coords={
                                                  'time': sorted(data.keys()),
                                                  'time_delta': time_deltas,
                                                  'lon': ((f'{grid}_eta', f'{grid}_xi'),
                                                          self.data_coordinates[grid]['lon']),
                                                  'lat': ((f'{grid}_eta', f'{grid}_xi'),
                                                          self.data_coordinates[grid]['lat'])
                                              },
                                              dims=('time', 'time_delta', f'{grid}_eta', f'{grid}_xi'))

                data_array.attrs['grid'] = grid
                data_arrays[variable] = data_array

        del data_stack

        output_dataset = xarray.Dataset(data_vars=data_arrays)

        del data_arrays

        return output_dataset

    def to_netcdf(self, output_file: str, variables: list = None, mean: bool = True):
        """
        Writes to NetCDF file.

        :param output_file: Output file to write.
        :param variables: List of variables to use.
        :param mean: Whether to average all time indices.
        """

        self.to_xarray(variables, mean).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
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

    :param input_lon: Matrix of X coordinates in original grid.
    :param input_lat: Matrix of Y coordinates in original grid.
    :param input_data: Masked array of data values in original grid.
    :param output_lon: Longitude values of output grid.
    :param output_lat: Latitude values of output grid.
    :param method: Interpolation method.
    :return: Interpolated values within output grid.
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
    WCOFS_Dataset.grid_transforms = None
    WCOFS_Dataset.grid_shapes = None
    WCOFS_Dataset.grid_bounds = None
    WCOFS_Dataset.data_coordinates = None
    WCOFS_Dataset.variable_grids = None
    WCOFS_Dataset.masks = None
    WCOFS_Dataset.angle = None


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
    output_dir = os.path.join(DATA_DIR, r'output\test')

    start_datetime = datetime.datetime(2018, 11, 8)
    end_datetime = start_datetime + datetime.timedelta(days=1)

    wcofs_range = WCOFS_Range(start_datetime, end_datetime, source='avg')
    wcofs_range.write_rasters(output_dir, variables=['salt', 'temp', 'u', 'v', 'zeta'])
    
    print('done')
