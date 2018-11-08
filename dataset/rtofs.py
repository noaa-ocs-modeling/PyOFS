# coding=utf-8
"""
RTOFS model output data collection and transformation by interpolation onto Cartesian grid.

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
import rasterio.control
import rasterio.features
from rasterio.io import MemoryFile
import rasterio.mask
import rasterio.warp
import xarray

from main import DATA_DIR

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})
FIONA_WGS84 = fiona.crs.from_epsg(4326)

COORDINATE_VARIABLES = ['time', 'lev', 'lat', 'lon']

DATASET_STRUCTURE = {
    '2ds': {
        'nowcast': {'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'],
                    'diag': ['ssh', 'ice_coverage', 'ice_thickness']},
        'forecast': {'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'],
                     'diag': ['ssh', 'ice_coverage', 'ice_thickness']}
    },
    '3dz': {
        'nowcast': {'salt': ['salinity'], 'temp': ['temperature'], 'uvel': ['u'], 'vvel': ['v']},
        'forecast': {'salt': ['salinity'], 'temp': ['temperature'], 'uvel': ['u'], 'vvel': ['v']}
    }
}

DATA_VARIABLES = {
    'salt': {'2ds': {'prog': 'sss'}, '3dz': {'salt': 'salinity'}},
    'temp': {'2ds': {'prog': 'sst'}, '3dz': {'temp': 'temperature'}},
    'u': {'2ds': {'prog': 'u_velocity'}, '3dz': {'uvel': 'u'}},
    'v': {'2ds': {'prog': 'v_velocity'}, '3dz': {'vvel': 'v'}},
    'ssh': {'2ds': {'diag': 'ssh'}},
    'ice_coverage': {'2ds': {'diag': 'ice_coverage'}},
    'ice_thickness': {'2ds': {'diag': 'ice_thickness'}}
}

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

SOURCE_URL = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs'

GLOBAL_LOCK = threading.Lock()


class RTOFS_Dataset:
    """
    Real-Time Ocean Forecasting System (RTOFS) NetCDF dataset.
    """
    
    def __init__(self, model_date: datetime.datetime, source: str = '2ds', time_interval: str = 'daily',
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME):
        """
        Creates new dataset object from datetime and given model parameters.

        :param model_date: Model run date.
        :param source: Either '2ds' or '3dz'.
        :param time_interval: Time interval of model output.
        :param study_area_polygon_filename: Filename of vector file containing study area boundary.
        """
        
        self.model_datetime = model_date.replace(hour=0, minute=0, second=0, microsecond=0)
        self.source = source
        self.time_interval = time_interval
        
        self.study_area_polygon_filename, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)
        
        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None
        
        self.netcdf_datasets = {}
        self.dataset_locks = {}
        
        date_string = self.model_datetime.strftime('%Y%m%d')
        
        if self.time_interval == 'daily':
            for forecast_direction, datasets in DATASET_STRUCTURE[self.source].items():
                self.netcdf_datasets[forecast_direction] = {}
                self.dataset_locks[forecast_direction] = {}
                
                date_dir = f'rtofs_global{date_string}'
                
                for dataset_name in datasets:
                    filename = f'rtofs_glo_{self.source}_{forecast_direction}_{self.time_interval}_{dataset_name}'
                    url = f'{SOURCE_URL}/{date_dir}/{filename}'
                    
                    try:
                        dataset = xarray.open_dataset(url)
                        
                        self.netcdf_datasets[forecast_direction][dataset_name] = dataset
                        self.dataset_locks[forecast_direction][dataset_name] = threading.Lock()
                        
                        self.lon = dataset['lon'].values
                        self.lat = dataset['lat'].values
                        
                        lon_pixel_size = dataset['lon'].resolution
                        lat_pixel_size = dataset['lat'].resolution
                        
                        # get first record in layer
                        with fiona.open(self.study_area_polygon_filename,
                                        layer=study_area_polygon_layer_name) as vector_layer:
                            study_area_geojson = next(iter(vector_layer))['geometry']
                        
                        global_west = numpy.min(self.lon)
                        global_north = numpy.max(self.lon)
                        
                        global_grid_transform = rasterio.transform.from_origin(global_west, global_north,
                                                                               lon_pixel_size, lat_pixel_size)
                        
                        # create array mask of data extent within study area, with the overall shape of the global VIIRS grid
                        raster_mask = rasterio.features.rasterize([study_area_geojson],
                                                                  out_shape=(self.lat.shape[0], self.lon.shape[0]),
                                                                  transform=global_grid_transform)
                        
                        # get indices of rows and columns within the data where the polygon mask exists
                        mask_row_indices, mask_col_indices = numpy.where(raster_mask == 1)
                        
                        west_index = numpy.min(mask_col_indices)
                        north_index = numpy.min(mask_row_indices)
                        east_index = numpy.max(mask_col_indices)
                        south_index = numpy.max(mask_row_indices)
                        
                        # create variable for the index bounds of the mask within the greater VIIRS global grid (west, north, east, south)
                        self.study_area_index_bounds = (west_index, north_index, east_index, south_index)
                        
                        study_area_west = self.netcdf_dataset['lon'][west_index]
                        study_area_north = self.netcdf_dataset['lat'][north_index]
                        
                        self.study_area_transform = rasterio.transform.from_origin(study_area_west, study_area_north,
                                                                                   lon_pixel_size, lat_pixel_size)
                    except OSError as error:
                        print(f'Error collecting RTOFS from {url}')
    
    def data(self, variable: str, forecast_direction: int) -> numpy.ndarray:
        """
        Get data of specified variable at specified hour.

        :param variable: Name of variable to retrieve.
        :param forecast_direction: Direction to retrieve.
        :return: Array of data.
        """
        
        if forecast_direction in DATASET_STRUCTURE[self.source]:
            if variable in DATA_VARIABLES:
                datasets = DATA_VARIABLES[variable][self.source]
                dataset_name, variable_name = next(iter(datasets.items()))
                
                with self.dataset_locks[forecast_direction][dataset_name]:
                    return self.netcdf_datasets[forecast_direction][dataset_name][variable_name][0, :, :].values
            else:
                raise ValueError(f'Variable must be one of {list(DATA_VARIABLES[self.source].keys())}.')
        else:
            raise ValueError(f'Forecast direction must be one of {list(DATASET_STRUCTURE[self.source].keys())}.')
    
    def data_average(self, variable: str, time_deltas: list = None) -> numpy.ndarray:
        """
        Gets average of data.

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
    
    def write_raster(self, output_filename: str, variable: str, forecast_direction: str,
                     study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                     fill_value=-9999, drivers: list = ['GTiff']):
        """
        Writes interpolated raster of given variable to output path.

        :param output_filename: Path of raster file to create.
        :param variable: Name of variable.
        :param forecast_direction: Direction to retrieve.
        :param study_area_polygon_filename: Path to vector file containing study area boundary.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """
        
        study_area_polygon_filename, layer_name = study_area_polygon_filename.rsplit(':', 1)
        
        if layer_name == '':
            layer_name = None
        
        with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
            study_area_geojson = next(iter(vector_layer))['geometry']
        
        output_data = self.data(variable, forecast_direction)
        grid_transform = rasterio.transform.from_bounds(self.lon.min(), self.lat.min(),
                                                        self.lon.max(), self.lat.max(),
                                                        self.lon.shape, self.lat.shape)
        
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
    
    def write_rasters(self, output_dir: str, variables: list = None, time_deltas: list = None, x_size: float = 0.04,
                      y_size: float = 0.04, drivers: list = ['GTiff']):
        """
        Writes rasters of given variables to output directory.

        :param output_dir: Path to directory.
        :param variables: Variable names to use.
        :param time_deltas: List of time indices to write.
        :param x_size: Cell size of output grid in X direction.
        :param y_size: Cell size of output grid in Y direction.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """
        
        if variables is None:
            variables = list(DATA_VARIABLES[self.source].keys())
        
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
    rtofs_dataset.write_raster(os.path.join(output_dir, 'test.tif'), 'temp', 'nowcast')
    
    print('done')
