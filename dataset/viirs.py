"""
Sea surface temperature rasters from VIIRS (aboard Suomi NPP).

Created on Jun 13, 2018

@author: zachary.burnett
"""

import concurrent.futures
import datetime
import math
import os
import threading

import fiona
import netCDF4
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.wkt

from dataset import _utilities

VIIRS_START_DATETIME = datetime.datetime(2012, 3, 1, 0, 10)
VIIRS_PERIOD = datetime.timedelta(days=16)

DATA_DIR = os.environ['OFS_DATA']

VIIRS_PASS_TIMES_FILENAME = os.path.join(DATA_DIR, r"reference\viirs_pass_times.txt")
STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})


class VIIRS_Dataset:
    study_area_transform = None
    study_area_index_bounds = None
    study_area_geojson = None

    def __init__(self, granule_datetime: datetime.datetime,
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, near_real_time: bool = True,
                 data_source: str = 'OSPO', subskin: bool = True, acspo_version: str = '2.41',
                 threading_lock: threading.Lock = None):
        """
        Retrieve VIIRS NetCDF dataset from NOAA with given datetime.

        :param granule_datetime: Dataset datetime.
        :param study_area_polygon_filename: Filename of vector file containing study area boundary.
        :param near_real_time: Whether dataset should be Near Real Time.
        :param data_source: Either 'STAR' or 'OSPO'.
        :param subskin: Whether dataset should use subskin or not.
        :param acspo_version: ACSPO Version number (2.40 - 2.41).
        :param threading_lock: Global lock in case of threaded dataset compilation.
        :raises NoDataError: if dataset does not exist.
        """

        # round minute to nearest 10 minutes (VIIRS data interval)
        self.granule_datetime = _utilities.round_to_ten_minutes(granule_datetime)

        self.study_area_polygon_filename, self.layer_name = study_area_polygon_filename.rsplit(':', 1)

        if self.layer_name == '':
            self.layer_name = None

        self.near_real_time = near_real_time
        self.data_source = data_source
        self.subskin = subskin
        self.acspo_version = acspo_version

        self.url = f'https://www.star.nesdis.noaa.gov/thredds/dodsC/' \
                   f'gridSNPPVIIRS{"NRT" if self.near_real_time else "SCIENCE"}L3UWW00/' \
                   f'{self.granule_datetime.year}/{self.granule_datetime.timetuple().tm_yday:03}/' \
                   f'{self.granule_datetime.strftime("%Y%m%d%H%M%S")}-{self.data_source}-L3U_GHRSST-' \
                   f'SST{"sub" if self.subskin else ""}skin-VIIRS_NPP-ACSPO_V{self.acspo_version}-v02.0-fv01.0.nc'

        try:
            self.netcdf_dataset = netCDF4.Dataset(self.url)
        except OSError:
            # try flipped NRT setting
            try:
                self.near_real_time = not self.near_real_time
                self.url = f'https://www.star.nesdis.noaa.gov/thredds/dodsC/' \
                           f'gridSNPPVIIRS{"NRT" if self.near_real_time else "SCIENCE"}L3UWW00/' \
                           f'{self.granule_datetime.year}/{self.granule_datetime.timetuple().tm_yday:03}/' \
                           f'{self.granule_datetime.strftime("%Y%m%d%H%M%S")}-{self.data_source}-L3U_GHRSST-' \
                           f'SST{"sub" if self.subskin else ""}skin-VIIRS_NPP-ACSPO_V{self.acspo_version}-v02.0-fv01.0.nc'
                self.netcdf_dataset = netCDF4.Dataset(self.url)
            except OSError:
                raise _utilities.NoDataError(f'{self.granule_datetime}: No VIIRS dataset found at {self.url}')

        # construct rectangular polygon of granule extent
        if 'geospatial_bounds' in self.netcdf_dataset.ncattrs():
            self.data_extent = shapely.wkt.loads(self.netcdf_dataset.geospatial_bounds)
        else:
            lon_min = float(self.netcdf_dataset.geospatial_lon_min)
            lon_max = float(self.netcdf_dataset.geospatial_lon_max)
            lat_min = float(self.netcdf_dataset.geospatial_lat_min)
            lat_max = float(self.netcdf_dataset.geospatial_lat_max)

            if lon_min < lon_max:
                self.data_extent = shapely.geometry.Polygon(
                        [(lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min), (lon_min, lat_min)])
            else:
                # geospatial bounds cross the antimeridian, so we create a multipolygon
                polygons = []

                # portion in eastern hemisphere
                polygons.append(shapely.geometry.Polygon(
                        [(lon_min, lat_max), (180, lat_max), (180, lat_min), (lon_min, lat_min)]))

                # portion in western hemisphere
                polygons.append(shapely.geometry.Polygon(
                        [(-180, lat_max), (lon_max, lat_max), (lon_max, lat_min), (-180, lat_min)]))

                self.data_extent = shapely.geometry.MultiPolygon(polygons)

        # dataset data
        self.lon_var = self.netcdf_dataset['lon']
        self.lat_var = self.netcdf_dataset['lat']
        self.sst_var = self.netcdf_dataset['sea_surface_temperature']
        self.sses_bias_var = self.netcdf_dataset['sses_bias']

        # if self.sses_bias_var[:].mask.all():
        #     print(f'{self.datetime} has no bias')
        # else:
        #     print(f'{self.datetime} has bias')

        lon_pixel_size = self.netcdf_dataset.geospatial_lon_resolution
        lat_pixel_size = self.netcdf_dataset.geospatial_lat_resolution

        if threading_lock is not None:
            threading_lock.acquire()

        if VIIRS_Dataset.study_area_index_bounds is None:
            print(f'Calculating indices and transform from {self.granule_datetime}')

            # get first record in layer
            with fiona.open(self.study_area_polygon_filename, layer=self.layer_name) as vector_layer:
                VIIRS_Dataset.study_area_geojson = vector_layer.next()['geometry']

            global_west = numpy.min(self.netcdf_dataset['lon'][:])
            global_north = numpy.max(self.netcdf_dataset['lat'][:])

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

        return (self.netcdf_dataset.geospatial_lon_resolution, self.netcdf_dataset.geospatial_lat_resolution)

    def get_variable_data(self, variable: str = 'sst', correct_bias: bool = True) -> numpy.ma.MaskedArray:
        """
        Get data of specified variable within study area.

        :param variable: Name of variable, either 'sst' or 'sses_bias'.
        :param correct_bias: Whether to subtract SSES bias from SST.
        :return: Masked array of data for given variable.
        """

        if variable == 'sst':
            return self.get_sst(correct_bias)
        elif variable == 'sses_bias':
            return self.get_sses_bias()

    def get_sst(self, correct_bias: bool = True) -> numpy.ma.MaskedArray:
        """
        Get SST data, if it exists within the study area.

        :return: Matrix of SST in Celsius within index bounds.
        """

        west_index = VIIRS_Dataset.study_area_index_bounds[0]
        north_index = VIIRS_Dataset.study_area_index_bounds[1]
        east_index = VIIRS_Dataset.study_area_index_bounds[2]
        south_index = VIIRS_Dataset.study_area_index_bounds[3]

        # try:
        # dataset SST data (masked array) using vertically reflected VIIRS grid
        output_sst_data = self.sst_var[0, north_index:south_index, west_index:east_index]

        # check for unmasked data
        if not output_sst_data.mask.all():
            # mask negative values
            output_sst_data.mask[output_sst_data <= 0] = True
            # output_sst_data[output_sst_data.mask] = numpy.ma.masked

            # check mask again after modifying it on the previous line
            if not output_sst_data.mask.all():
                if correct_bias:
                    output_sst_data = output_sst_data - self.get_sses_bias()

                # convert from Kelvin to Celsius (subtract 273.15)
                output_sst_data = output_sst_data - self.sst_var.add_offset

                return output_sst_data

        # except Exception as error:
        #     print(self.netcdf_dataset['sea_surface_temperature'])
        #     print(f'Error collecting SST: {error}')

        return None

    def get_sses_bias(self) -> numpy.ndarray:
        """
        Return SSES bias.

        :return: Array of SSES bias.
        """

        west_index = VIIRS_Dataset.study_area_index_bounds[0]
        north_index = VIIRS_Dataset.study_area_index_bounds[1]
        east_index = VIIRS_Dataset.study_area_index_bounds[2]
        south_index = VIIRS_Dataset.study_area_index_bounds[3]

        # dataset bias values using vertically reflected VIIRS grid
        sses_bias_masked_data = self.sses_bias_var[0, north_index:south_index, west_index:east_index]

        # replace masked values with 0
        sses_bias_masked_data.set_fill_value(0)
        return sses_bias_masked_data.filled()

    def write_rasters(self, output_dir: str, variables: list = ['sst'], fill_value: float = None,
                      drivers: list = ['GTiff']):
        """
        Write VIIRS rasters to file using data from given variables.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to write.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        for current_variable in variables:
            input_data = self.get_variable_data(current_variable)

            if input_data is not None and not input_data.mask.all():
                if fill_value is not None:
                    input_data.set_fill_value(fill_value)

                gdal_args = {
                    'height': input_data.shape[0], 'width': input_data.shape[1], 'count': 1, 'dtype': rasterio.float32,
                    'crs':    RASTERIO_WGS84, 'transform': VIIRS_Dataset.study_area_transform,
                    'nodata': input_data.fill_value.astype(rasterio.float32)
                }

                for current_driver in drivers:
                    if current_driver == 'AAIGrid':
                        file_extension = 'asc'
                        gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                    elif current_driver == 'GTiff':
                        file_extension = 'tiff'
                    elif current_driver == 'GPKG':
                        file_extension = 'gpkg'

                    current_output_filename = os.path.join(output_dir, f'viirs_{current_variable}.{file_extension}')

                    # use rasterio to write to raster with GDAL args
                    print(f'Writing to {current_output_filename}')
                    with rasterio.open(current_output_filename, 'w', current_driver, **gdal_args) as output_raster:
                        output_raster.write(input_data, 1)

    def __repr__(self):
        used_params = [self.granule_datetime.__repr__()]
        optional_params = [self.study_area_polygon_filename, self.near_real_time, self.data_source, self.subskin,
                           self.acspo_version]

        for current_param in optional_params:
            if current_param is not None:
                if 'str' in str(type(current_param)):
                    current_param = f'"{current_param}"'
                else:
                    current_param = str(current_param)

                used_params.append(current_param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class VIIRS_Range:
    """
    Range of VIIRS dataset.
    """

    study_area_transform = None
    study_area_index_bounds = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                 viirs_pass_times_filename: str = VIIRS_PASS_TIMES_FILENAME, near_real_time: bool = True,
                 data_source: str = 'OSPO', subskin: bool = True, acspo_version: str = '2.41'):
        """
        Collect VIIRS datasets within time interval.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param study_area_polygon_filename: Filename of vector file of study area boundary.
        :param viirs_pass_times_filename: Path to text file with pass times.
        :param near_real_time: Whether dataset should be Near Real Time.
        :param data_source: Either 'STAR' or 'OSPO'.
        :param subskin: Whether dataset should use subskin or not.
        :param acspo_version: ACSPO Version number (2.40 - 2.41).
        :raises NoDataError: if data does not exist.
        """

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.study_area_polygon_filename = study_area_polygon_filename
        self.viirs_pass_times_filename = viirs_pass_times_filename
        self.near_real_time = near_real_time
        self.data_source = data_source
        self.subskin = subskin
        self.acspo_version = acspo_version

        self.pass_times = get_pass_times(self.start_datetime, self.end_datetime, self.viirs_pass_times_filename)

        print(f'Collecting VIIRS SST from {numpy.min(self.pass_times)} to {numpy.max(self.pass_times)}')

        # create dictionary to store scenes
        self.datasets = {}

        threading_lock = threading.Lock()

        # concurrently populate dictionary with VIIRS dataset object for each pass time in the given time interval
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            scene_futures = {concurrency_pool.submit(VIIRS_Dataset, current_datetime, self.study_area_polygon_filename,
                                                     self.near_real_time, self.data_source, self.subskin,
                                                     self.acspo_version, threading_lock): current_datetime for
                             current_datetime in self.pass_times}

            # yield results as threads complete
            for current_future in concurrent.futures.as_completed(scene_futures):
                if type(current_future.exception()) is not _utilities.NoDataError:
                    current_result = current_future.result()
                    current_datetime = scene_futures[current_future]
                    self.datasets[current_datetime] = current_result

        if len(self.datasets) > 0:
            VIIRS_Range.study_area_transform = VIIRS_Dataset.study_area_transform
            VIIRS_Range.study_area_index_bounds = VIIRS_Dataset.study_area_index_bounds

            self.sample_dataset = next(iter(self.datasets.values()))
        else:
            raise _utilities.NoDataError(
                    f'No VIIRS datasets found between {self.start_datetime} and {self.end_datetime}.')

    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: Tuple of cell sizes (x_size, y_size)
        """

        return (self.sample_dataset.netcdf_dataset.geospatial_lon_resolution,
                self.sample_dataset.netcdf_dataset.geospatial_lat_resolution)

    def write_rasters(self, output_dir: str, fill_value: float = None, drivers: list = ['GTiff']):
        """
        Write individual VIIRS rasters to directory.

        :param output_dir: Path to output directory.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        # write a raster for each pass retrieved scene
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            for current_datetime, current_dataset in self.datasets.items():
                current_output_filename = os.path.join(output_dir,
                                                       f'viirs_sst_{current_datetime.strftime("%Y%m%d%H%M%S")}.tiff')

                # write to raster file
                concurrency_pool.submit(current_dataset.write_rasters, current_output_filename, fill_value=fill_value,
                                        drivers=drivers)

    def write_raster(self, output_dir: str, start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None, correct_bias: bool = True, average: bool = True,
                     fill_value: float = None, drivers: list = ['GTiff']):
        """
        Write VIIRS raster of stacked SST data (averaged or overlapped).

        :param output_dir: Path to output directory.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param correct_bias: Whether to subtract SSES bias from SST.
        :param average: Whether to average rasters, otherwise overlap them.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        dataset_datetimes = numpy.sort(list(self.datasets.keys()))

        # find first time within specified time interval
        start_index = numpy.searchsorted(dataset_datetimes, start_datetime, side='left')

        # find last time within specified time interval
        end_index = numpy.searchsorted(dataset_datetimes, end_datetime, side='right')

        interval_datetimes = dataset_datetimes[start_index:end_index]

        # check if user wants to average data
        if average:
            scenes_data = []

            # concurrently populate array with data from every VIIRS scene
            with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
                scenes_futures = []

                for current_datetime in interval_datetimes:
                    current_dataset = self.datasets[current_datetime]
                    # scenes_data.append(current_dataset.get_sst(correct_bias))
                    scenes_futures.append(concurrency_pool.submit(current_dataset.get_sst, correct_bias))

                for current_future in concurrent.futures.as_completed(scenes_futures):
                    if current_future._exception is None:
                        current_result = current_future.result()

                        if current_result is not None:
                            scenes_data.append(current_result)

                del scenes_futures

            if len(scenes_data) > 0:
                output_sst_data = numpy.ma.mean(numpy.ma.dstack(scenes_data), axis=2)
            else:
                print(interval_datetimes)
                raise _utilities.NoDataError('No VIIRS data found in the given interval.')

            if fill_value is not None:
                output_sst_data.set_fill_value(fill_value)
        else:  # otherwise overlap based on datetime
            output_sst_data = None
            for current_datetime in interval_datetimes:
                current_sst_data = self.datasets[current_datetime].get_sst()

                if current_sst_data is not None:
                    if output_sst_data is None:
                        if fill_value is None:
                            fill_value = current_sst_data.fill_value

                        output_sst_data = numpy.ma.MaskedArray(
                                numpy.ones((current_sst_data.shape[0], current_sst_data.shape[1]),
                                           dtype=rasterio.float32) * fill_value, mask=True, fill_value=fill_value)
                    output_sst_data[~current_sst_data.mask] = current_sst_data[~current_sst_data.mask]

        # remove negative values
        output_sst_data.mask[output_sst_data <= 0] = True



        # mercator_pixel_x, mercator_pixel_y = pyproj.transform(pyproj.Proj({"init": "epsg:4326"}),
        #                                                       pyproj.Proj({"init": "epsg:3857"}),
        #                                                       VIIRS_Range.study_area_transform.a,
        #                                                       -VIIRS_Range.study_area_transform.e)
        #
        # mercator_origin_x, mercator_origin_y = pyproj.transform(pyproj.Proj({"init": "epsg:4326"}),
        #                                                         pyproj.Proj({"init": "epsg:3857"}),
        #                                                         VIIRS_Range.study_area_transform.c,
        #                                                         VIIRS_Range.study_area_transform.f)
        #
        # mercator_transform = rasterio.transform.from_origin(mercator_origin_x, mercator_origin_y, mercator_pixel_x,
        #                                                     mercator_pixel_y)

        # define arguments to GDAL driver
        gdal_args = {
            'height':    output_sst_data.shape[0], 'width': output_sst_data.shape[1], 'count': 1, 'crs': RASTERIO_WGS84,
            'transform': VIIRS_Range.study_area_transform
        }

        # if os.path.exists(output_filename):
        #     os.remove(output_filename)

        for current_driver in drivers:
            if current_driver == 'AAIGrid':
                file_extension = 'asc'
                raster_data = output_sst_data.filled().astype(numpy.float32)
                gdal_args.update({
                    'dtype':          raster_data.dtype, 'nodata': output_sst_data.fill_value.astype(raster_data.dtype),
                    'FORCE_CELLSIZE': 'YES'
                })
            elif current_driver == 'GTiff':
                file_extension = 'tiff'
                raster_data = output_sst_data.filled().astype(numpy.float32)
                gdal_args.update({
                    'dtype': raster_data.dtype, 'nodata': output_sst_data.fill_value.astype(raster_data.dtype)
                })
            elif current_driver == 'GPKG':
                file_extension = 'gpkg'
                gpkg_dtype = rasterio.uint8
                fill_value = numpy.iinfo(gpkg_dtype).max
                output_sst_data.set_fill_value(fill_value)
                # scale data to within range of uint8
                raster_data = ((fill_value - 1) * (output_sst_data - numpy.ma.min(output_sst_data)) / numpy.ma.ptp(
                        output_sst_data)).filled().astype(gpkg_dtype)
                gdal_args.update({
                    'dtype': raster_data.dtype, 'nodata': fill_value
                })  # , 'TILE_FORMAT': 'PNG8'})

            current_output_filename = os.path.join(output_dir, f'viirs_sst.{file_extension}')

            print(f'Writing {current_output_filename}')
            with rasterio.open(current_output_filename, 'w', current_driver, **gdal_args) as output_raster:
                output_raster.write(raster_data, 1)


def __repr__(self):
    used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
    optional_params = [self.study_area_polygon_filename, self.viirs_pass_times_filename, self.near_real_time,
                       self.data_source, self.subskin, self.acspo_version]

    for current_param in optional_params:
        if current_param is not None:
            if 'str' in str(type(current_param)):
                current_param = f'"{current_param}"'
            else:
                current_param = str(current_param)

            used_params.append(current_param)

    return f'{self.__class__.__name__}({", ".join(used_params)})'


def store_viirs_pass_times(study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                           start_datetime: datetime.datetime = VIIRS_START_DATETIME,
                           output_filename: str = VIIRS_PASS_TIMES_FILENAME, num_periods: int = 1,
                           near_real_time: bool = False, data_source: str = 'STAR', subskin: bool = False,
                           acspo_version: str = '2.40'):
    """
    Compute VIIRS pass times from the given start date along the number of periods specified.

    :param study_area_polygon_filename: Path to vector file containing polygon of study area.
    :param start_datetime: Beginning of given VIIRS period.
    :param output_filename: Path to output file.
    :param num_periods: Number of periods to store.
    :param near_real_time: Whether dataset should be Near Real Time.
    :param data_source: Either 'STAR' or 'OSPO'.
    :param subskin: Whether dataset should use subskin or not.
    :param acspo_version: ACSPO Version number (2.40 - 2.41)
    """

    start_datetime = _utilities.round_to_ten_minutes(start_datetime)
    end_datetime = _utilities.round_to_ten_minutes(start_datetime + (VIIRS_PERIOD * num_periods))

    print(
            f'Getting pass times between {start_datetime.strftime("%Y-%m-%d %H:%M:%S")} and {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}')

    datetime_range = _utilities.ten_minute_range(start_datetime, end_datetime)

    study_area_polygon_filename, layer_name = study_area_polygon_filename.rsplit(':', 1)

    if layer_name == '':
        layer_name = None

    # construct polygon from the first record in layer
    with fiona.open(study_area_polygon_filename, layer=layer_name) as vector_layer:
        study_area_polygon = shapely.geometry.Polygon(vector_layer.next()['geometry'])

    # create variable for the index bounds of the mask within the greater VIIRS global grid
    # index_bounds = None

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

            # get dataset of new datetime
            current_dataset = VIIRS_Dataset(cycle_datetime, study_area_polygon_filename, near_real_time, data_source,
                                            subskin, acspo_version)

            # if dataset is missing, try again at next cycle
            if current_dataset.netcdf_dataset is not None:
                # compute index bounds if not yet computed
                # if index_bounds is None:
                #     viirs_grid_transform = current_dataset.get_viirs_grid_transform()
                #     index_bounds = current_dataset.get_extent_indices()

                # check if dataset falls within polygon extent
                if current_dataset.data_extent.is_valid and study_area_polygon.intersects(current_dataset.data_extent):
                    # get duration from current cycle start
                    current_cycle_duration = cycle_datetime - (start_datetime + cycle_offset)

                    print(
                            f'{cycle_datetime.strftime("%Y%m%dT%H%M%S")} {current_cycle_duration.total_seconds()}: valid scene (checked {cycle_index + 1} cycle(s))')
                    lines.append('{0},{1}'.format(cycle_datetime.strftime('%Y%m%dT%H%M%S'),
                                                  current_cycle_duration.total_seconds()))

                # if we get to here, break and continue to the next datetime
                break
        else:
            print(f'{current_datetime.strftime("%Y%m%dT%H%M%S")}: missing dataset across all cycles')

    # write lines to file
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(lines))

    print('Wrote data to file')


def get_pass_times(start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                   viirs_pass_times_filename: str = VIIRS_PASS_TIMES_FILENAME):
    """
    Retreive array of datetimes of VIIRS passes within the given time interval, given initial period durations.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :param viirs_pass_times_filename: Filename of text file with durations of first VIIRS period.
    :return:
    """

    # get datetime of first pass in given file
    first_pass_row = numpy.genfromtxt(viirs_pass_times_filename, dtype=str, delimiter=',')[0, :]
    viirs_start_datetime = datetime.datetime.strptime(first_pass_row[0], '%Y%m%dT%H%M%S') - datetime.timedelta(
            seconds=float(first_pass_row[1]))

    # get starting datetime of the current VIIRS period
    current_period_start_datetime = viirs_start_datetime + datetime.timedelta(
            days=numpy.floor((start_datetime - viirs_start_datetime).days / 16) * 16)

    # get array of seconds since the start of the first 16-day VIIRS period
    pass_durations = numpy.genfromtxt(viirs_pass_times_filename, dtype=str, delimiter=',')[:, 1].T.astype(numpy.float32)
    pass_durations = numpy.asarray(
            [datetime.timedelta(seconds=float(current_duration)) for current_duration in pass_durations])

    # add extra VIIRS periods to end of pass durations
    if end_datetime > (current_period_start_datetime + VIIRS_PERIOD):
        extra_periods = math.ceil((end_datetime - current_period_start_datetime) / VIIRS_PERIOD) - 1
        # print(f'Using {extra_periods} extra VIIRS periods.')
        for period in range(extra_periods):
            pass_durations = numpy.append(pass_durations, pass_durations[-360:] + pass_durations[-1])

    # get datetimes of VIIRS passes within the given time interval
    pass_times = current_period_start_datetime + pass_durations

    # find starting and ending times within the given time interval
    start_index = numpy.searchsorted(pass_times, start_datetime, side='left')
    end_index = numpy.searchsorted(pass_times, end_datetime, side='right')

    # ensure at least one datetime in range
    if start_index == end_index:
        end_index += 1

    # trim datetimes to within the given time interval
    pass_times = pass_times[start_index:end_index]

    return pass_times


def check_mask(granule_datetime: datetime.datetime, study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
               threading_lock: threading.Lock = None):
    """
    Check whether SSES bias is totally masked in the study area for the given VIIRS dataset time.

    :param granule_datetime: Datetime of VIIRS image.
    :param study_area_polygon_filename: Filename of vector file containing study area.
    :param threading_lock: Lock for threaded execution.
    :return: Descriptive string of datetime and whether data is masked.
    """

    try:
        viirs_dataset = VIIRS_Dataset(granule_datetime, study_area_polygon_filename, threading_lock=threading_lock)
        masked = (viirs_dataset.get_sses_bias() == 0).all()
        if masked:
            return f'{granule_datetime}: masked'
        else:
            return f'{granule_datetime}: has unmasked values'
    except _utilities.NoDataError:
        return f'{granule_datetime}: no data'


def check_masks(start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                viirs_pass_times_filename: str = VIIRS_PASS_TIMES_FILENAME):
    """
    Check whether SSES bias is totally masked in the study area for the given VIIRS datasets within the given time interval.

    :param start_datetime: Beginning of time interval.
    :param end_datetime: End of time interval.
    :param study_area_polygon_filename: Filename of vector file containing study area.
    :param viirs_pass_times_filename: Filename to stored VIIRS pass durations.
    """

    pass_times = get_pass_times(start_datetime, end_datetime, viirs_pass_times_filename)

    print(f'Checking masks from {numpy.min(pass_times)} to {numpy.max(pass_times)}')

    threading_lock = threading.Lock()

    print(f'Starting {len(pass_times)} threads...')

    with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
        check_futures = [
            concurrency_pool.submit(check_mask, current_datetime, study_area_polygon_filename, threading_lock) for
            current_datetime in pass_times]

        for current_future in concurrent.futures.as_completed(check_futures):
            print(current_future.result())


if __name__ == '__main__':
    start_datetime = datetime.datetime(2018, 6, 10)
    end_datetime = datetime.datetime(2018, 6, 11)

    output_filename = r"C:\Data\viirs\avg_10_11.tiff"

    datasets = {}
    threading_lock = threading.Lock()
    pass_times = _utilities.ten_minute_range(start_datetime, end_datetime)

    # # write average of all scenes in specified time interval
    # viirs_range = VIIRS_Range(start_datetime, end_datetime)
    # viirs_range.write_raster(output_filename)

    print('done')
