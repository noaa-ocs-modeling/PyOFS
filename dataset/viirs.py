# coding=utf-8
"""
Sea surface temperature rasters from VIIRS (aboard Suomi NPP).

Created on Jun 13, 2018

@author: zachary.burnett
"""

from collections import OrderedDict
from concurrent import futures
import datetime
import ftplib
import math
import os
import threading

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

VIIRS_START_DATETIME = datetime.datetime.strptime('2012-03-01 00:10:00', '%Y-%m-%d %H:%M:%S')
VIIRS_PERIOD = datetime.timedelta(days=16)

PASS_TIMES_FILENAME = os.path.join(DATA_DIR, r"reference\viirs_pass_times.txt")
STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})

SOURCES = OrderedDict({'OpenDAP': OrderedDict({
    'NESDIS': 'https://www.star.nesdis.noaa.gov/thredds/dodsC',
    'JPL': 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3U',
    'NODC': 'https://data.nodc.noaa.gov/thredds/catalog/ghrsst/L3U'
}), 'FTP': OrderedDict({
    'NESDIS': 'ftp.star.nesdis.noaa.gov/pub/socd2/coastwatch/sst'
})
})


class VIIRS_Dataset:
    study_area_transform = None
    study_area_index_bounds = None
    study_area_geojson = None
    
    def __init__(self, granule_datetime: datetime.datetime,
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                 algorithm: str = 'OSPO', version: str = None, threading_lock: threading.Lock = None):
        """
        Retrieve VIIRS NetCDF dataset from NOAA with given datetime.

        :param granule_datetime: Dataset datetime.
        :param study_area_polygon_filename: Filename of vector file containing study area boundary.
        :param algorithm: Either 'STAR' or 'OSPO'.
        :param version: ACSPO algorithm version.
        :param threading_lock: Global lock in case of threaded dataset compilation.
        :raises NoDataError: if dataset does not exist.
        """
        
        # round minute to nearest 10 minutes (VIIRS data interval)
        self.granule_datetime = _utilities.round_to_ten_minutes(granule_datetime)
        
        self.study_area_polygon_filename, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)
        
        if study_area_polygon_layer_name == '':
            study_area_polygon_layer_name = None
        
        # use NRT flag if granule is less than 13 days old
        self.near_real_time = datetime.datetime.now() - granule_datetime <= datetime.timedelta(days=13)
        self.algorithm = algorithm
        
        if version is None:
            if granule_datetime >= datetime.datetime(2018, 11, 7, 15, 10):
                self.version = '2.60'
            elif granule_datetime >= datetime.datetime(2017, 9, 14, 12, 50):
                self.version = '2.41'
            else:
                self.version = '2.40'
        else:
            self.version = version
        
        self.url = None
        
        satellite = 'NPP'
        
        month_dir = f'{self.granule_datetime.year}/{self.granule_datetime.timetuple().tm_yday:03}'
        filename = f'{self.granule_datetime.strftime("%Y%m%d%H%M%S")}-{self.algorithm}-L3U_GHRSST-SSTsubskin-VIIRS_{satellite.upper()}-ACSPO_V{self.version}-v02.0-fv01.0.nc'
        
        for source_format, urls in SOURCES.items():
            for source, source_url in urls.items():
                if source_format == 'OpenDAP':
                    # TODO N20 does not have an OpenDAP archive
                    if satellite.upper() == 'N20':
                        continue
                    
                    if source == 'NESDIS':
                        url = f'{source_url}/gridSNPPVIIRS{"NRT" if self.near_real_time else "SCIENCE"}L3UWW00/{month_dir}/{filename}'
                    elif source == 'JPL':
                        url = f'{source_url}/VIIRS_{satellite}/{algorithm}/v{self.version}/{month_dir}/{filename}'
                    elif source in 'NODC':
                        url = f'{source_url}/VIIRS_{satellite}/{algorithm}/{month_dir}/{filename}'
                elif source_format == 'FTP':
                    host_url, input_dir = source_url.split('/', 1)
                    
                    if source == 'NESDIS':
                        if self.near_real_time:
                            ftp_path = f'/{input_dir}/nrt/viirs_acspo{self.version}/{satellite.lower()}/l3u/{month_dir}/{filename}'
                        else:
                            # TODO N20 does not have a reanalysis archive
                            if satellite.upper() == 'N20':
                                continue
                            
                            ftp_path = f'/{input_dir}/ran/viirs/{satellite.lower()}/l3u/{month_dir}/{filename}'
                        
                        url = f'{source_url}/{ftp_path.lstrip("/")}'
                try:
                    if source_format == 'OpenDAP':
                        self.netcdf_dataset = xarray.open_dataset(url)
                    elif source_format == 'FTP':
                        with ftplib.FTP(host_url) as ftp_connection:
                            ftp_connection.login()
                            temp_filename = os.path.join(DATA_DIR, 'tempfile.nc')
                            with open(temp_filename, 'wb') as temp_file:
                                ftp_connection.retrbinary(f'RETR {ftp_path}', temp_file.write)
                                self.netcdf_dataset = xarray.open_dataset(temp_filename)
                            
                            os.remove(temp_filename)
                    
                    self.url = url
                    break
                except Exception as error:
                    print(f'Error collecting dataset from {source} at {url}: {error}')
            
            if self.url is not None:
                break
        else:
            raise _utilities.NoDataError(f'{self.granule_datetime}: No VIIRS dataset found.')
        
        # construct rectangular polygon of granule extent
        if 'geospatial_bounds' in self.netcdf_dataset.attrs:
            self.data_extent = shapely.wkt.loads(self.netcdf_dataset.geospatial_bounds)
        elif 'geospatial_lon_min' in self.netcdf_dataset.attrs:
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
                    shapely.geometry.Polygon([(lon_min, lat_max), (180, lat_max), (180, lat_min), (lon_min, lat_min)]),
                    shapely.geometry.Polygon(
                        [(-180, lat_max), (lon_max, lat_max), (lon_max, lat_min), (-180, lat_min)])])
        else:
            print(f'{self.granule_datetime}: Dataset has no stored bounds...')
        
        lon_pixel_size = self.netcdf_dataset.geospatial_lon_resolution
        lat_pixel_size = self.netcdf_dataset.geospatial_lat_resolution
        
        if threading_lock is not None:
            threading_lock.acquire()
        
        if VIIRS_Dataset.study_area_index_bounds is None:
            # print(f'Calculating indices and transform from granule at {self.granule_datetime}...')
            
            # get first record in layer
            with fiona.open(self.study_area_polygon_filename, layer=study_area_polygon_layer_name) as vector_layer:
                VIIRS_Dataset.study_area_geojson = next(iter(vector_layer))['geometry']
            
            global_west = numpy.min(self.netcdf_dataset['lon'])
            global_north = numpy.max(self.netcdf_dataset['lat'])
            
            global_grid_transform = rasterio.transform.from_origin(global_west, global_north, lon_pixel_size,
                                                                   lat_pixel_size)
            
            # create array mask of data extent within study area, with the overall shape of the global VIIRS grid
            raster_mask = rasterio.features.rasterize([VIIRS_Dataset.study_area_geojson], out_shape=(
                self.netcdf_dataset['lat'].shape[0], self.netcdf_dataset['lon'].shape[0]),
                                                      transform=global_grid_transform)
            
            # get indices of rows and columns within the data where the polygon mask exists
            mask_row_indices, mask_col_indices = numpy.where(raster_mask == 1)
            
            west_index = numpy.min(mask_col_indices)
            north_index = numpy.min(mask_row_indices)
            east_index = numpy.max(mask_col_indices)
            south_index = numpy.max(mask_row_indices)
            
            # create variable for the index bounds of the mask within the greater VIIRS global grid (west, north, east, south)
            VIIRS_Dataset.study_area_index_bounds = (west_index, north_index, east_index, south_index)
            
            study_area_west = self.netcdf_dataset['lon'][west_index]
            study_area_north = self.netcdf_dataset['lat'][north_index]
            
            VIIRS_Dataset.study_area_transform = rasterio.transform.from_origin(study_area_west, study_area_north,
                                                                                lon_pixel_size, lat_pixel_size)
        
        if threading_lock is not None:
            threading_lock.release()
    
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
    
    def sst(self, sses_correction: bool = False) -> numpy.ndarray:
        """
        Return matrix of sea surface temperature.

        :return: Matrix of SST in Celsius.
        """
        
        west_index = VIIRS_Dataset.study_area_index_bounds[0]
        north_index = VIIRS_Dataset.study_area_index_bounds[1]
        east_index = VIIRS_Dataset.study_area_index_bounds[2]
        south_index = VIIRS_Dataset.study_area_index_bounds[3]
        
        # dataset SST data (masked array) using vertically reflected VIIRS grid
        output_sst_data = self.netcdf_dataset['sea_surface_temperature'][0, north_index:south_index,
                          west_index:east_index].values
        
        # check for unmasked data
        if not numpy.isnan(output_sst_data).all():
            if numpy.nanmax(output_sst_data) > 0:
                if numpy.nanmin(output_sst_data) <= 0:
                    output_sst_data[output_sst_data <= 0] = numpy.nan
                
                if sses_correction:
                    sses = self.sses()
                    
                    mismatched_records = len(numpy.where(numpy.isnan(output_sst_data) != (sses == 0))[0])
                    total_records = output_sst_data.shape[0] * output_sst_data.shape[1]
                    mismatch_percentage = mismatched_records / total_records * 100
                    
                    if mismatch_percentage > 0:
                        print(f'{self.granule_datetime}: SSES extent mismatch at {mismatch_percentage:.1f}%')
                    
                    output_sst_data -= sses
                
                # convert from Kelvin to Celsius (subtract 273.15)
                output_sst_data = output_sst_data - 273.15
            else:
                output_sst_data[:] = numpy.nan
        
        return output_sst_data
    
    def sses(self) -> numpy.ndarray:
        """
        Return matrix of sensor-specific error statistics.

        :return: Array of SSES bias in Celsius.
        """
        
        west_index = VIIRS_Dataset.study_area_index_bounds[0]
        north_index = VIIRS_Dataset.study_area_index_bounds[1]
        east_index = VIIRS_Dataset.study_area_index_bounds[2]
        south_index = VIIRS_Dataset.study_area_index_bounds[3]
        
        # dataset bias values using vertically reflected VIIRS grid
        sses_data = self.netcdf_dataset['sses_bias'][0, north_index:south_index, west_index:east_index].values
        
        # replace masked values with 0
        sses_data[numpy.isnan(sses_data)] = 0
        
        # offset by 2.048
        sses_data = sses_data - 2.048
        
        return sses_data
    
    def write_rasters(self, output_dir: str, variables: list = ['sst', 'sses'], filename_prefix: str = 'viirs',
                      fill_value: float = -9999.0, drivers: list = ['GTiff'], sses_correction: bool = False):
        """
        Write VIIRS rasters to file using data from given variables.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to write.
        :param filename_prefix: Prefix for output filenames.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from SST.
        """
        
        for variable in variables:
            if variable == 'sst':
                input_data = self.sst(sses_correction=sses_correction)
            elif variable == 'sses':
                input_data = self.sses()
                fill_value = 0
            
            if input_data is not None and not numpy.isnan(input_data).all():
                if fill_value is not -9999.0:
                    input_data[numpy.isnan(input_data)] = fill_value
                
                gdal_args = {
                    'height': input_data.shape[0], 'width': input_data.shape[1], 'count': 1, 'dtype': rasterio.float32,
                    'crs': RASTERIO_WGS84, 'transform': VIIRS_Dataset.study_area_transform, 'nodata': fill_value
                }
                
                for driver in drivers:
                    if driver == 'AAIGrid':
                        file_extension = 'asc'
                        gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                    elif driver == 'GTiff':
                        file_extension = 'tiff'
                    elif driver == 'GPKG':
                        file_extension = 'gpkg'
                    
                    output_filename = os.path.join(output_dir, f'{filename_prefix}_{variable}.{file_extension}')
                    
                    if os.path.isfile(output_filename):
                        os.remove(output_filename)
                    
                    # use rasterio to write to raster with GDAL args
                    print(f'Writing to {output_filename}')
                    with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                        output_raster.write(input_data, 1)
    
    def __repr__(self):
        used_params = [self.granule_datetime.__repr__()]
        optional_params = [self.study_area_polygon_filename, self.near_real_time, self.algorithm, self.version]
        
        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)
                
                used_params.append(param)
        
        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class VIIRS_Range:
    """
    Range of VIIRS dataset.
    """
    
    study_area_transform = None
    study_area_index_bounds = None
    
    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                 pass_times_filename: str = PASS_TIMES_FILENAME, algorithm: str = 'OSPO',
                 version: str = None):
        """
        Collect VIIRS datasets within time interval.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param study_area_polygon_filename: Filename of vector file of study area boundary.
        :param pass_times_filename: Path to text file with pass times.
        :param algorithm: Either 'STAR' or 'OSPO'.
        :param version: ACSPO algorithm version.
        :raises NoDataError: if data does not exist.
        """
        
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime if end_datetime < datetime.datetime.utcnow() else datetime.datetime.utcnow()
        self.study_area_polygon_filename = study_area_polygon_filename
        self.viirs_pass_times_filename = pass_times_filename
        self.algorithm = algorithm
        self.version = version
        
        self.pass_times = get_pass_times(self.start_datetime, self.end_datetime, self.viirs_pass_times_filename)
        
        if len(self.pass_times) > 0:
            print(
                f'Collecting VIIRS data from {len(self.pass_times)} passes between UTC {numpy.min(self.pass_times)} and UTC {numpy.max(self.pass_times)}...')
            
            # create dictionary to store scenes
            self.datasets = {}
            
            threading_lock = threading.Lock()
            
            # concurrently populate dictionary with VIIRS dataset object for each pass time in the given time interval
            with futures.ThreadPoolExecutor() as concurrency_pool:
                scene_futures = {
                    concurrency_pool.submit(VIIRS_Dataset, granule_datetime=pass_time,
                                            study_area_polygon_filename=self.study_area_polygon_filename,
                                            algorithm=self.algorithm, version=self.version,
                                            threading_lock=threading_lock): pass_time for pass_time in
                    self.pass_times}
                
                # yield results as threads complete
                for completed_future in futures.as_completed(scene_futures):
                    if type(completed_future.exception()) is not _utilities.NoDataError:
                        viirs_dataset = completed_future.result()
                        scene_datetime = scene_futures[completed_future]
                        
                        if not numpy.isnan(viirs_dataset.sst()).all():
                            self.datasets[scene_datetime] = viirs_dataset
                    else:
                        print(completed_future.exception())
            
            if len(self.datasets) > 0:
                VIIRS_Range.study_area_transform = VIIRS_Dataset.study_area_transform
                VIIRS_Range.study_area_index_bounds = VIIRS_Dataset.study_area_index_bounds
                
                self.sample_dataset = next(iter(self.datasets.values()))
                
                print(f'VIIRS data was found in {len(self.datasets)} passes.')
            else:
                raise _utilities.NoDataError(
                    f'No VIIRS datasets found between {self.start_datetime} and {self.end_datetime}.')
        else:
            raise _utilities.NoDataError(
                f'There are no VIIRS passes between {self.start_datetime} and {self.end_datetime}.')
    
    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: Tuple of cell sizes (x_size, y_size)
        """
        
        return (self.sample_dataset.netcdf_dataset.geospatial_lon_resolution,
                self.sample_dataset.netcdf_dataset.geospatial_lat_resolution)
    
    def write_rasters(self, output_dir: str, variables: list = ['sst', 'sses'], filename_prefix: str = 'viirs',
                      fill_value: float = None, drivers: list = ['GTiff'], sses_correction: bool = False):
        """
        Write individual VIIRS rasters to directory.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to write.
        :param filename_prefix: Prefix for output filenames.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        """
        
        # write a raster for each pass retrieved scene
        with futures.ThreadPoolExecutor() as concurrency_pool:
            for dataset_datetime, dataset in self.datasets.items():
                concurrency_pool.submit(dataset.write_rasters, output_dir, variables=variables,
                                        filename_prefix=f'{filename_prefix}_{dataset_datetime.strftime("%Y%m%d%H%M%S")}',
                                        fill_value=fill_value, drivers=drivers, sses_correction=sses_correction)
    
    def write_raster(self, output_dir: str, filename_prefix: str = None, filename_suffix: str = None,
                     start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                     average: bool = False, fill_value: float = -9999, drivers: list = ['GTiff'],
                     sses_correction: bool = False, variables: list = ['sst']):
        """
        Write VIIRS raster of SST data (either overlapped or averaged) from the given time interval.

        :param output_dir: Path to output directory.
        :param filename_prefix: Prefix for output filenames.
        :param filename_suffix: Suffix for output filenames.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param average: Whether to average rasters, otherwise overlap them.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param variables: List of variables to write (either 'sst' or 'sses').
        """
        
        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime
        
        dataset_datetimes = numpy.sort(list(self.datasets.keys()))
        
        # find first and last times within specified time interval
        start_index = numpy.searchsorted(dataset_datetimes, start_datetime)
        end_index = numpy.searchsorted(dataset_datetimes, end_datetime)
        
        pass_datetimes = dataset_datetimes[start_index:end_index]
        
        for variable in variables:
            output_data = None
            
            # check if user wants to average data
            if average:
                scenes_data = []
                
                # concurrently populate array with data from every VIIRS scene
                with futures.ThreadPoolExecutor() as concurrency_pool:
                    scenes_futures = []
                    
                    for pass_datetime in pass_datetimes:
                        dataset = self.datasets[pass_datetime]
                        
                        if variable == 'sst':
                            scenes_futures.append(concurrency_pool.submit(dataset.sst, sses_correction))
                        elif variable == 'sses':
                            scenes_futures.append(concurrency_pool.submit(dataset.sses))
                    
                    for completed_future in futures.as_completed(scenes_futures):
                        if completed_future._exception is None:
                            result = completed_future.result()
                            
                            if result is not None:
                                scenes_data.append(result)
                    
                    del scenes_futures
                
                if len(scenes_data) > 0:
                    output_data = numpy.nanmean(numpy.stack(scenes_data), axis=2)
            else:  # otherwise overlap based on datetime
                for pass_datetime in pass_datetimes:
                    if variable == 'sst':
                        scene_data = self.datasets[pass_datetime].sst(sses_correction=sses_correction)
                    elif variable == 'sses':
                        scene_data = self.datasets[pass_datetime].sses()
                        scene_data[scene_data == 0] = numpy.nan
                    
                    if output_data is None:
                        output_data = numpy.empty_like(scene_data)
                        output_data[:] = numpy.nan
                    
                    output_data[~numpy.isnan(scene_data)] = scene_data[~numpy.isnan(scene_data)]
            
            if output_data is not None:
                output_data[numpy.isnan(output_data)] = fill_value
                
                # define arguments to GDAL driver
                gdal_args = {
                    'height': output_data.shape[0], 'width': output_data.shape[1], 'count': 1, 'crs': RASTERIO_WGS84,
                    'transform': VIIRS_Range.study_area_transform
                }
                
                for driver in drivers:
                    if driver == 'AAIGrid':
                        file_extension = 'asc'
                        raster_data = output_data.astype(rasterio.float32)
                        gdal_args.update({
                            'dtype': raster_data.dtype,
                            'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item(),
                            'FORCE_CELLSIZE': 'YES'
                        })
                    elif driver == 'GTiff':
                        file_extension = 'tiff'
                        raster_data = output_data.astype(rasterio.float32)
                        gdal_args.update({
                            'dtype': raster_data.dtype,
                            'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
                        })
                    elif driver == 'GPKG':
                        file_extension = 'gpkg'
                        gpkg_dtype = rasterio.uint8
                        gpkg_fill_value = numpy.iinfo(gpkg_dtype).max
                        output_data[output_data == fill_value] = gpkg_fill_value
                        # scale data to within range of uint8
                        raster_data = ((gpkg_fill_value - 1) * (output_data - numpy.min(output_data)) / numpy.ptp(
                            output_data)).astype(gpkg_dtype)
                        gdal_args.update({
                            'dtype': gpkg_dtype, 'nodata': gpkg_fill_value
                        })  # , 'TILE_FORMAT': 'PNG8'})
                    else:
                        raster_data = numpy.empty_like(output_data)
                    
                    if filename_prefix is None:
                        current_filename_prefix = f'viirs_{variable}'
                    else:
                        current_filename_prefix = filename_prefix
                    
                    if filename_suffix is None:
                        start_datetime_string = start_datetime.strftime("%Y%m%d%H%M")
                        end_datetime_string = end_datetime.strftime("%Y%m%d%H%M")
                        
                        if '0000' in start_datetime_string and '0000' in end_datetime_string:
                            start_datetime_string = start_datetime_string.replace("0000", "")
                            end_datetime_string = end_datetime_string.replace("0000", "")
                        
                        current_filename_suffix = f'{start_datetime_string}_{end_datetime_string}'
                    else:
                        current_filename_suffix = filename_suffix
                    
                    output_filename = os.path.join(output_dir,
                                                   f'{current_filename_prefix}_{current_filename_suffix}.{file_extension}')
                    
                    if os.path.isfile(output_filename):
                        os.remove(output_filename)
                    
                    print(f'Writing {output_filename}')
                    with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                        output_raster.write(raster_data, 1)
            else:
                print(f'No VIIRS {variable} found between {start_datetime} and {end_datetime}.')
    
    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.study_area_polygon_filename, self.viirs_pass_times_filename,
                           self.algorithm, self.version]
        
        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)
                
                used_params.append(param)
        
        return f'{self.__class__.__name__}({", ".join(used_params)})'


def store_viirs_pass_times(study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                           start_datetime: datetime.datetime = VIIRS_START_DATETIME,
                           output_filename: str = PASS_TIMES_FILENAME, num_periods: int = 1,
                           data_source: str = 'STAR', subskin: bool = False,
                           acspo_version: str = '2.40'):
    """
    Compute VIIRS pass times from the given start date along the number of periods specified.

    :param study_area_polygon_filename: Path to vector file containing polygon of study area.
    :param start_datetime: Beginning of given VIIRS period.
    :param output_filename: Path to output file.
    :param num_periods: Number of periods to store.
    :param data_source: Either 'STAR' or 'OSPO'.
    :param subskin: Whether dataset should use subskin or not.
    :param acspo_version: ACSPO Version number (2.40 - 2.41)
    """
    
    start_datetime = _utilities.round_to_ten_minutes(start_datetime)
    end_datetime = _utilities.round_to_ten_minutes(start_datetime + (VIIRS_PERIOD * num_periods))
    
    print(
        f'Getting pass times between {start_datetime.strftime("%Y-%m-%d %H:%M:%S")} and {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
    
    datetime_range = _utilities.ten_minute_range(start_datetime, end_datetime)
    
    study_area_polygon_geopackage, study_area_polygon_layer_name = study_area_polygon_filename.rsplit(':', 1)
    
    if study_area_polygon_layer_name == '':
        study_area_polygon_layer_name = None
    
    # construct polygon from the first record in layer
    with fiona.open(study_area_polygon_geopackage, layer=study_area_polygon_layer_name) as vector_layer:
        study_area_polygon = shapely.geometry.Polygon(next(iter(vector_layer))['geometry']['coordinates'][0])
    
    lines = []
    
    for datetime_index in range(len(datetime_range)):
        current_datetime = datetime_range[datetime_index]
        
        # find number of cycles from the first orbit to the present day
        num_cycles = int((datetime.datetime.now() - start_datetime).days / 16)
        
        # iterate over each cycle
        for cycle_index in range(0, num_cycles):
            # get current datetime of interest
            cycle_offset = VIIRS_PERIOD * cycle_index
            cycle_datetime = current_datetime + cycle_offset
            
            try:
                # get dataset of new datetime
                dataset = VIIRS_Dataset(cycle_datetime, study_area_polygon_filename, data_source, subskin,
                                        acspo_version)
                
                # check if dataset falls within polygon extent
                if dataset.data_extent.is_valid and study_area_polygon.intersects(dataset.data_extent):
                    # get duration from current cycle start
                    cycle_duration = cycle_datetime - (start_datetime + cycle_offset)
                    
                    print(
                        f'{cycle_datetime.strftime("%Y%m%dT%H%M%S")} {cycle_duration.total_seconds()}: valid scene (checked {cycle_index + 1} cycle(s))')
                    lines.append(f'{cycle_datetime.strftime("%Y%m%dT%H%M%S")},{cycle_duration.total_seconds()}')
                
                # if we get to here, break and continue to the next datetime
                break
            except _utilities.NoDataError as error:
                print(error)
        else:
            print(f'{datetime.strftime("%Y%m%dT%H%M%S")}: missing dataset across all cycles')
    
    # write lines to file
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(lines))
    
    print('Wrote data to file')


def get_pass_times(start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                   pass_times_filename: str = PASS_TIMES_FILENAME):
    """
    Retreive array of datetimes of VIIRS passes within the given time interval, given initial period durations.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :param pass_times_filename: Filename of text file with durations of first VIIRS period.
    :return:
    """
    
    # get datetime of first pass in given file
    first_pass_row = numpy.genfromtxt(pass_times_filename, dtype=str, delimiter=',')[0, :]
    viirs_start_datetime = datetime.datetime.strptime(first_pass_row[0], '%Y%m%dT%H%M%S') - datetime.timedelta(
        seconds=float(first_pass_row[1]))
    
    # get starting datetime of the current VIIRS period
    period_start_datetime = viirs_start_datetime + datetime.timedelta(
        days=numpy.floor((start_datetime - viirs_start_datetime).days / 16) * 16)
    
    # get array of seconds since the start of the first 16-day VIIRS period
    pass_durations = numpy.genfromtxt(pass_times_filename, dtype=str, delimiter=',')[:, 1].T.astype(numpy.float32)
    pass_durations = numpy.asarray([datetime.timedelta(seconds=float(duration)) for duration in pass_durations])
    
    # add extra VIIRS periods to end of pass durations
    if end_datetime > (period_start_datetime + VIIRS_PERIOD):
        extra_periods = math.ceil((end_datetime - period_start_datetime) / VIIRS_PERIOD) - 1
        # print(f'Using {extra_periods} extra VIIRS periods.')
        for period in range(extra_periods):
            pass_durations = numpy.append(pass_durations, pass_durations[-360:] + pass_durations[-1])
    
    # get datetimes of VIIRS passes within the given time interval
    pass_times = period_start_datetime + pass_durations
    
    # find starting and ending times within the given time interval
    start_index = numpy.searchsorted(pass_times, start_datetime)
    end_index = numpy.searchsorted(pass_times, end_datetime)
    
    # ensure at least one datetime in range
    if start_index == end_index:
        end_index += 1
    
    # trim datetimes to within the given time interval
    pass_times = pass_times[start_index:end_index]
    
    return pass_times


if __name__ == '__main__':
    output_dir = os.path.join(DATA_DIR, r'output\test\viirs')
    
    viirs_dataset = VIIRS_Dataset(datetime.datetime(2018, 10, 29, 12))
    
    print('done')
