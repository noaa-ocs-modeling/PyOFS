# coding=utf-8
"""
RTOFS model output data collection and transformation by interpolation onto Cartesian grid.

Created on Jun 25, 2018

@author: zachary.burnett
"""

from concurrent import futures
import datetime
import os

import fiona
import fiona.crs
import numpy
import rasterio.control
from rasterio.io import MemoryFile
import rasterio.mask
import rasterio.warp
import xarray

from main import DATA_DIR

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})
FIONA_WGS84 = fiona.crs.from_epsg(4326)

COORDINATE_VARIABLES = ['time', 'lev', 'lat', 'lon']
DATA_VARIABLES = {
    'nowcast': {'salt': ['salinity'], 'temp': ['temperature'], 'uvel': ['u'], 'vvel': ['v']},
    'forecast': {'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'], 'diag': ['ssh', 'ice_coverage', 'ice_thickness']}
}

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RTOFS_NOMADS_URL = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs'


class RTOFS_Dataset:
    """
    Real-Time Ocean Forecasting System (RTOFS) NetCDF dataset.
    """
    
    def __init__(self, model_date: datetime.datetime, time_interval='daily'):
        """
        Creates new dataset object from datetime and given model parameters.

        :param model_date: Model run date.
        :param time_interval: Time interval of model output.
        """
        
        self.model_datetime = model_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        date_string = self.model_datetime.strftime('%Y%m%d')
        
        if time_interval == 'daily':
            for direction, variable_groups in DATA_VARIABLES.items():
                self.netcdf_datasets[direction] = {}
                
                for variable_group in variable_groups:
                    url = f'{RTOFS_NOMADS_URL}/rtofs_global{date_string}/rtofs_glo_3dz_{direction}_{time_interval}_{variable_group}'
                    
                    try:
                        self.netcdf_datasets[direction][variable_group] = xarray.open_dataset(url)
                    except OSError as error:
                        print(f'Error collecting RTOFS from {url}')
    
    def bounds(self, variable: str = 'psi') -> tuple:
        """
        Returns bounds of grid of given variable.

        :param variable: Variable name.
        :return: Tuple of (west, north, east, south)
        """
        
        grid_name = RTOFS_Dataset.variable_grids[variable]
        return RTOFS_Dataset.grid_bounds[grid_name]
    
    def data(self, variable: str, time_delta: int) -> numpy.ndarray:
        """
        Get data of specified variable at specified hour.

        :param variable: Name of variable to retrieve.
        :param time_delta: Time index to retrieve (days for avg, hours for others).
        :return: Array of data.
        """
        
        if time_delta in self.time_deltas:
            if self.source == 'avg':
                if time_delta >= 0:
                    dataset_index = 1
                    day_index = time_delta
                else:
                    dataset_index = -1
                    day_index = 0
                
                with self.dataset_locks[dataset_index]:
                    # get surface layer; the last layer (of 40) at dimension 1
                    if variable in ['u', 'v']:
                        raw_u = self.netcdf_datasets[dataset_index]['u'][day_index, -1, :-1, :].values
                        raw_v = self.netcdf_datasets[dataset_index]['v'][day_index, -1, :, :-1].values
                        theta = RTOFS_Dataset.angle[:-1, :-1]
                        
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
                        output_data = self.netcdf_datasets[dataset_index][variable][day_index, -1, :, :].values
            
            else:
                with self.dataset_locks[time_delta]:
                    output_data = self.netcdf_datasets[time_delta][variable][0, :, :].values
        else:
            output_data = numpy.empty(RTOFS_Dataset.grid_shapes[RTOFS_Dataset.variable_grids[variable]])
            output_data[:] = numpy.nan
        
        return output_data
    
    def data_average(self, variable: str, time_deltas: list = None) -> numpy.ndarray:
        """
        Writes interpolation of averaged data of given variable to output path.

        :param variable: Variable to use.
        :param time_deltas: List of integers of time indices to use in average (days for avg, hours for others).
        :return: Array of data.
        """
        
        time_deltas = time_deltas if time_deltas is not None else self.netcdf_datasets.keys()
        
        variable_data = []
        
        # concurrently populate array with data for every hour
        with futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data, variable, time_delta): time_delta for time_delta in
                                time_deltas}
            
            for completed_future in futures.as_completed(variable_futures):
                variable_data.append(completed_future.result())
            
            del variable_futures
        
        variable_data = numpy.mean(numpy.stack(variable_data), axis=0)
        
        return variable_data
    
    def write_rasters(self, output_dir: str, variables: list = None, time_deltas: list = None, x_size: float = 0.04,
                      y_size: float = 0.04, drivers: list = ['GTiff']):
        """
        Writes interpolated rasters of given variables to output directory using concurrency.

        :param output_dir: Path to directory.
        :param variables: Variable names to use.
        :param time_deltas: List of time indices to write.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """
        
        if variables is None:
            variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES
        
        # concurrently write rasters with data from each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            # get data average for each variable
            variable_mean_futures = {
                concurrency_pool.submit(self.data_average, variable_name, time_deltas): variable_name for variable_name
                in variables}
            
            for completed_future in futures.as_completed(variable_mean_futures):
                variable_name = variable_mean_futures[completed_future]
                
                data = completed_future.result()
                
                self.write_raster(os.path.join(output_dir, f'wcofs_{variable_name}.tiff'), variable_name,
                                  input_data=data, x_size=x_size, y_size=y_size, drivers=drivers)
            
            del variable_mean_futures
    
    def write_raster(self, output_filename: str, variable: str,
                     study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, time_deltas: int = None,
                     input_data: numpy.ndarray = None, x_size: float = 0.04, y_size: float = 0.04, fill_value=-9999,
                     drivers: list = ['GTiff']):
        """
        Writes interpolated raster of given variable to output path.

        :param output_filename: Path of raster file to create.
        :param variable: Name of variable.
        :param study_area_polygon_filename: Path to vector file containing study area boundary.
        :param time_deltas: List of time indices to write.
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
            study_area_geojson = next(iter(vector_layer))['geometry']
        
        grid_name = RTOFS_Dataset.variable_grids[variable]
        
        west = RTOFS_Dataset.grid_bounds[grid_name][0]
        north = RTOFS_Dataset.grid_bounds[grid_name][1]
        east = RTOFS_Dataset.grid_bounds[grid_name][2]
        south = RTOFS_Dataset.grid_bounds[grid_name][3]
        
        x_size = x_size if x_size is not None else self.x_size
        y_size = y_size if y_size is not None else self.y_size
        
        output_grid_lon = numpy.arange(west, east, x_size)
        output_grid_lat = numpy.arange(south, north, y_size)
        
        grid_transform = RTOFS_Dataset.grid_transforms[grid_name]
        
        if input_data is None:
            input_data = self.data_average(variable, time_deltas)
        
        print(f'Starting {variable} interpolation')
        
        # interpolate data onto coordinate grid
        output_data = interpolate_grid(RTOFS_Dataset.data_coordinates[grid_name]['lon'],
                                       RTOFS_Dataset.data_coordinates[grid_name]['lat'], input_data, output_grid_lon,
                                       output_grid_lat)
        
        gdal_args = {
            'transform': grid_transform, 'height': output_data.shape[0], 'width': output_data.shape[1], 'count': 1,
            'dtype': rasterio.float32, 'crs': RASTERIO_WGS84,
            'nodata': numpy.array([fill_value]).astype(output_data.dtype).item()
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
            
            if os.path.isfile(output_filename):
                os.remove(output_filename)
            
            print(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(masked_data, 1)
    
    def write_vector(self, output_filename: str, layer_name: str = None, time_deltas: list = None):
        """
        Write average of surface velocity vector data for all hours in the given time interval to the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param time_deltas: List of integers of hours to use in average.
        """
        
        variables = MEASUREMENT_VARIABLES_2DS if self.source == '2ds' else MEASUREMENT_VARIABLES
        
        start_time = datetime.datetime.now()
        
        variable_means = {}
        
        # concurrently populate dictionary with averaged data within given time interval for each variable
        with futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self.data_average, variable, time_deltas): variable for variable
                                in variables}
            
            for completed_future in futures.as_completed(variable_futures):
                variable = variable_futures[completed_future]
                variable_means[variable] = completed_future.result()
            
            del variable_futures
        
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
        
        grid_height, grid_width = RTOFS_Dataset.data_coordinates['psi']['lon'].shape
        
        with futures.ThreadPoolExecutor() as concurrency_pool:
            feature_index = 1
            record_futures = []
            
            for col in range(grid_width):
                for row in range(grid_height):
                    if RTOFS_Dataset.masks['psi'][row, col] == 0:
                        # check if current record is unmasked
                        record_futures.append(
                            concurrency_pool.submit(self._create_fiona_record, variable_means, row, col,
                                                    feature_index))
                        feature_index += 1
            
            for completed_future in futures.as_completed(record_futures):
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
        rho_lon = RTOFS_Dataset.data_coordinates['rho']['lon'][row, col]
        rho_lat = RTOFS_Dataset.data_coordinates['rho']['lat'][row, col]
        
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


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test')
    
    rtofs_dataset = RTOFS_Dataset(datetime.datetime.now())
    
    print('done')
