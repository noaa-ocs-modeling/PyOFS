# coding=utf-8
"""
Sea surface temperature rasters from VIIRS (aboard Suomi NPP).

Created on Jun 13, 2018

@author: zachary.burnett
"""

import datetime
import ftplib
import math
import os
from collections import OrderedDict
from concurrent import futures

import fiona
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.wkt
import xarray

from dataset import CRS_EPSG, Logger
from dataset import _utilities
from main import DATA_DIR

VIIRS_START_DATETIME = datetime.datetime.strptime('2012-03-01 00:10:00', '%Y-%m-%d %H:%M:%S')
VIIRS_PERIOD = datetime.timedelta(days=16)

PASS_TIMES_FILENAME = os.path.join(DATA_DIR, r"reference\viirs_pass_times.txt")
STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})

NRT_DELAY = datetime.timedelta(hours=2)

SOURCE_URLS = OrderedDict({
    'OpenDAP': OrderedDict({
        'NESDIS': 'https://www.star.nesdis.noaa.gov/thredds/dodsC',
        'JPL': 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3U',
        'NODC': 'https://data.nodc.noaa.gov/thredds/catalog/ghrsst/L3U'
    }), 'FTP': OrderedDict({
        'NESDIS': 'ftp.star.nesdis.noaa.gov/pub/socd2/coastwatch/sst'
    })
})


class VIIRSDataset:
    study_area_transform = None
    study_area_extent = None
    study_area_bounds = None
    study_area_coordinates = None

    def __init__(self, granule_datetime: datetime.datetime, satellite: str = 'NPP',
                 study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME, algorithm: str = 'OSPO',
                 version: str = None, logger: Logger = None):
        """
        Retrieve VIIRS NetCDF dataset from NOAA with given datetime.

        :param granule_datetime: Dataset datetime.
        :param satellite: VIIRS platform.
        :param study_area_polygon_filename: Filename of vector file containing study area boundary.
        :param algorithm: Either 'STAR' or 'OSPO'.
        :param version: ACSPO algorithm version.
        :param logger: logbook logger
        :raises NoDataError: if dataset does not exist.
        """

        self.logger = logger

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

        self.satellite = satellite

        self.url = None

        month_dir = f'{self.granule_datetime.year}/{self.granule_datetime.timetuple().tm_yday:03}'
        filename = f'{self.granule_datetime.strftime("%Y%m%d%H%M%S")}-' + \
                   f'{self.algorithm}-L3U_GHRSST-SSTsubskin-VIIRS_{self.satellite.upper()}-ACSPO_V{self.version}-v02.0-fv01.0.nc'

        # TODO N20 does not yet have a reanalysis archive on NESDIS
        if self.satellite.upper() == 'N20' and not self.near_real_time:
            raise _utilities.NoDataError(f'{self.satellite.upper()} does not yet have a reanalysis archive')

        for source, source_url in SOURCE_URLS['OpenDAP'].items():
            if self.near_real_time:
                if source == 'NESDIS':
                    url = f'{source_url}/grid{self.satellite.upper()}VIIRSNRTL3UWW00/{month_dir}/{filename}'
                elif source == 'JPL':
                    url = f'{source_url}/VIIRS_{self.satellite.upper()}/{algorithm}/v{self.version}/{month_dir}/{filename}'
                elif source in 'NODC':
                    url = f'{source_url}/VIIRS_{self.satellite.upper()}/{algorithm}/{month_dir}/{filename}'
            else:
                if source == 'NESDIS':
                    url = f'{source_url}/grid{"" if self.near_real_time else "S"}{self.satellite.upper()}VIIRSSCIENCEL3UWW00/{month_dir}/{filename}'
                else:
                    if self.logger is not None:
                        self.logger.warn(f'{source} does not have a reanalysis archive')

            try:
                self.netcdf_dataset = xarray.open_dataset(url)
                self.url = url
                break
            except Exception as error:
                if self.logger is not None:
                    self.logger.error(f'Error collecting dataset from {source}: {error}')

            if self.url is not None:
                break

        if self.url is None:
            if self.logger is not None:
                self.logger.warn('Error collecting from OpenDAP; falling back to FTP...')

            for source, source_url in SOURCE_URLS['FTP'].items():
                host_url, ftp_input_dir = source_url.split('/', 1)

                if source == 'NESDIS':
                    if self.near_real_time:
                        ftp_path = f'/{ftp_input_dir}/nrt/viirs/{self.satellite.lower()}/l3u/{month_dir}/{filename}'
                    else:
                        ftp_path = f'/{ftp_input_dir}/ran/viirs/{"S" if self.satellite.upper() == "NPP" else ""}{self.satellite.lower()}/l3u/{month_dir}/{filename}'

                    url = f'{host_url}/{ftp_path.lstrip("/")}'

                try:
                    with ftplib.FTP(host_url) as ftp_connection:
                        ftp_connection.login()

                        output_dir = os.path.join(DATA_DIR, 'input', 'viirs')

                        if not os.path.exists(output_dir):
                            os.mkdir(output_dir)

                        output_filename = os.path.join(output_dir,
                                                       f'viirs_{self.granule_datetime.strftime("%Y%m%dT%H%M")}.nc')

                        if os.path.exists(output_filename):
                            os.remove(output_filename)

                        try:
                            with open(output_filename, 'wb') as output_file:
                                ftp_connection.retrbinary(f'RETR {ftp_path}', output_file.write)
                                self.netcdf_dataset = xarray.open_dataset(output_filename)
                        except Exception as error:
                            raise error
                        finally:
                            os.remove(output_filename)

                    self.url = url
                    break
                except Exception as error:
                    if self.logger is not None:
                        self.logger.error(f'Error collecting dataset from {source}: {error}')

                if self.url is not None:
                    break

        if self.url is None:
            raise _utilities.NoDataError(f'No VIIRS dataset found at {self.granule_datetime} UTC.')

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
                    shapely.geometry.Polygon(
                        [(lon_min, lat_max), (180, lat_max), (180, lat_min), (lon_min, lat_min)]),
                    shapely.geometry.Polygon(
                        [(-180, lat_max), (lon_max, lat_max), (lon_max, lat_min), (-180, lat_min)])])
        else:
            if self.logger is not None:
                self.logger.warn(f'{self.granule_datetime} UTC: Dataset has no stored bounds...')

        lon_pixel_size = self.netcdf_dataset.geospatial_lon_resolution
        lat_pixel_size = self.netcdf_dataset.geospatial_lat_resolution

        if VIIRSDataset.study_area_extent is None:
            if self.logger is not None:
                self.logger.debug(f'Calculating indices and transform from granule at {self.granule_datetime} UTC...')

            # get first record in layer
            with fiona.open(self.study_area_polygon_filename, layer=study_area_polygon_layer_name) as vector_layer:
                VIIRSDataset.study_area_extent = shapely.geometry.MultiPolygon(
                    [shapely.geometry.Polygon(polygon[0]) for polygon in
                     next(iter(vector_layer))['geometry']['coordinates']])

            VIIRSDataset.study_area_bounds = VIIRSDataset.study_area_extent.bounds
            VIIRSDataset.study_area_transform = rasterio.transform.from_origin(VIIRSDataset.study_area_bounds[0],
                                                                               VIIRSDataset.study_area_bounds[3],
                                                                               lon_pixel_size, lat_pixel_size)

        if VIIRSDataset.study_area_bounds is not None:
            self.netcdf_dataset = self.netcdf_dataset.isel(time=0).sel(lon=slice(VIIRSDataset.study_area_bounds[0],
                                                                                 VIIRSDataset.study_area_bounds[2]),
                                                                       lat=slice(VIIRSDataset.study_area_bounds[3],
                                                                                 VIIRSDataset.study_area_bounds[1]))

        if VIIRSDataset.study_area_coordinates is None:
            VIIRSDataset.study_area_coordinates = {
                'lon': self.netcdf_dataset['lon'], 'lat': self.netcdf_dataset['lat']
            }

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

    def data(self, variable: str = 'sst', sses_correction: bool = False) -> numpy.ndarray:
        if variable == 'sst':
            output_data = self._sst(sses_correction)
        elif variable == 'sses':
            output_data = self._sses()

        return output_data

    def _sst(self, sses_correction: bool = False) -> numpy.ndarray:
        """
        Return matrix of sea surface temperature.

        :return: Matrix of SST in Celsius.
        """

        # dataset SST data (masked array) using vertically reflected VIIRS grid
        output_sst_data = self.netcdf_dataset['sea_surface_temperature'].values

        # check for unmasked data
        if not numpy.isnan(output_sst_data).all():
            if numpy.nanmax(output_sst_data) > 0:
                if numpy.nanmin(output_sst_data) <= 0:
                    output_sst_data[output_sst_data <= 0] = numpy.nan

                if sses_correction:
                    sses = self._sses()

                    mismatched_records = len(numpy.where(numpy.isnan(output_sst_data) != (sses == 0))[0])
                    total_records = output_sst_data.shape[0] * output_sst_data.shape[1]
                    mismatch_percentage = mismatched_records / total_records * 100

                    if mismatch_percentage > 0:
                        if self.logger is not None:
                            self.logger.warn(
                                f'{self.granule_datetime} UTC: SSES extent mismatch at {mismatch_percentage:.1f}%')

                    output_sst_data -= sses

                # convert from Kelvin to Celsius (subtract 273.15)
                output_sst_data = output_sst_data - 273.15
            else:
                output_sst_data[:] = numpy.nan

        return output_sst_data

    def _sses(self) -> numpy.ndarray:
        """
        Return matrix of sensor-specific error statistics.

        :return: Array of SSES bias in Celsius.
        """

        # dataset bias values using vertically reflected VIIRS grid
        sses_data = self.netcdf_dataset['sses_bias'].values

        # replace masked values with 0
        sses_data[numpy.isnan(sses_data)] = 0

        # negative offset by 2.048
        sses_data -= 2.048

        return sses_data

    def write_rasters(self, output_dir: str, variables: list = ('sst', 'sses'), filename_prefix: str = 'viirs',
                      fill_value: float = -9999.0, driver: str = 'GTiff', sses_correction: bool = False):
        """
        Write VIIRS rasters to file using data from given variables.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to write.
        :param filename_prefix: Prefix for output filenames.
        :param fill_value: Desired fill value of output.
        :param driver: Strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from SST.
        """

        for variable in variables:
            input_data = self.data(variable, sses_correction)

            if variable == 'sses':
                fill_value = 0

            if input_data is not None and not numpy.isnan(input_data).all():
                if fill_value is not -9999.0:
                    input_data[numpy.isnan(input_data)] = fill_value

                gdal_args = {
                    'height': input_data.shape[0], 'width': input_data.shape[1], 'count': 1,
                    'dtype': rasterio.float32,
                    'crs': RASTERIO_CRS, 'transform': VIIRSDataset.study_area_transform, 'nodata': fill_value
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
                if self.logger is not None:
                    self.logger.notice(f'Writing to {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(input_data, 1)

    def __repr__(self):
        used_params = [self.granule_datetime.__repr__()]
        optional_params = [self.satellite, self.study_area_polygon_filename, self.near_real_time, self.algorithm,
                           self.version]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


class VIIRSRange:
    """
    Range of VIIRS dataset.
    """

    study_area_transform = None
    study_area_index_bounds = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                 satellites: list = ('NPP', 'N20'), study_area_polygon_filename: str = STUDY_AREA_POLYGON_FILENAME,
                 pass_times_filename: str = PASS_TIMES_FILENAME, algorithm: str = 'OSPO',
                 version: str = None, logger: Logger = None):
        """
        Collect VIIRS datasets within time interval.

        :param start_datetime: Beginning of time interval (in UTC).
        :param end_datetime: End of time interval (in UTC).
        :param satellites: List of VIIRS platforms.
        :param study_area_polygon_filename: Filename of vector file of study area boundary.
        :param pass_times_filename: Path to text file with pass times.
        :param algorithm: Either 'STAR' or 'OSPO'.
        :param version: ACSPO algorithm version.
        :param logger: logbook logger
        :raises NoDataError: if data does not exist.
        """

        self.logger = logger

        self.start_datetime = start_datetime
        if end_datetime > datetime.datetime.utcnow():
            # VIIRS near real time delay is 2 hours behind UTC
            self.end_datetime = datetime.datetime.utcnow() - NRT_DELAY
        else:
            self.end_datetime = end_datetime

        self.study_area_polygon_filename = study_area_polygon_filename
        self.viirs_pass_times_filename = pass_times_filename
        self.algorithm = algorithm
        self.version = version

        self.satellites = satellites

        self.pass_times = get_pass_times(self.start_datetime, self.end_datetime, self.viirs_pass_times_filename)

        if len(self.pass_times) > 0:
            if self.logger is not None:
                self.logger.info(f'Collecting VIIRS data from {len(self.pass_times)} passes between ' +
                                 f'{numpy.min(self.pass_times)} UTC and {numpy.max(self.pass_times)} UTC...')

            # create dictionary to store scenes
            self.datasets = {pass_time: {} for pass_time in self.pass_times}

            with futures.ThreadPoolExecutor() as concurrency_pool:
                for satellite in self.satellites:
                    running_futures = {}

                    for pass_time in self.pass_times:
                        running_future = concurrency_pool.submit(VIIRSDataset, granule_datetime=pass_time,
                                                                 study_area_polygon_filename=self.study_area_polygon_filename,
                                                                 algorithm=self.algorithm, version=self.version,
                                                                 satellite=satellite, logger=logger)
                        running_futures[running_future] = pass_time

                    for completed_future in futures.as_completed(running_futures):
                        if completed_future.exception() is None:
                            pass_time = running_futures[completed_future]
                            viirs_dataset = completed_future.result()
                            self.datasets[pass_time][satellite] = viirs_dataset
                        else:
                            if self.logger is not None:
                                self.logger.warning(f'Dataset creation error: {completed_future.exception()}')

                    del running_futures

            if len(self.datasets) > 0:
                VIIRSRange.study_area_transform = VIIRSDataset.study_area_transform
                VIIRSRange.study_area_extent = VIIRSDataset.study_area_extent
                VIIRSRange.study_area_bounds = VIIRSDataset.study_area_bounds

                if self.logger is not None:
                    self.logger.debug(f'VIIRS data was found in {len(self.datasets)} passes.')
            else:
                raise _utilities.NoDataError(
                    f'No VIIRS datasets found between {self.start_datetime} UTC and {self.end_datetime} UTC.')

        else:
            raise _utilities.NoDataError(
                f'There are no VIIRS passes between {self.start_datetime} UTC and {self.end_datetime} UTC.')

    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: Tuple of cell sizes (x_size, y_size)
        """

        sample_dataset = next(iter(self.datasets.values()))

        return (sample_dataset.netcdf_dataset.geospatial_lon_resolution,
                sample_dataset.netcdf_dataset.geospatial_lat_resolution)

    def data(self, start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
             average: bool = False, sses_correction: bool = False, variables: list = tuple(['sst']),
             satellite: str = None) -> dict:
        """
        Get VIIRS data (either overlapped or averaged) from the given time interval.

        :param start_datetime: Beginning of time interval (in UTC).
        :param end_datetime: End of time interval (in UTC).
        :param average: Whether to average rasters, otherwise overlap them.
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param variables: List of variables to write (either 'sst' or 'sses').
        :param satellite: VIIRS platform to retrieve. Default: per-granule averages of platform datasets.
        :return Dictionary of data per variable.
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        dataset_datetimes = numpy.sort(list(self.datasets.keys()))

        # find first and last times within specified time interval
        start_index = numpy.searchsorted(dataset_datetimes, start_datetime)
        end_index = numpy.searchsorted(dataset_datetimes, end_datetime)

        pass_datetimes = dataset_datetimes[start_index:end_index]

        if variables is None:
            variables = ['sst', 'sses']

        variables_data = {}

        for variable in variables:
            scenes_data = []

            for pass_datetime in pass_datetimes:
                if len(self.datasets[pass_datetime]) > 0:
                    if satellite is not None and satellite in self.datasets[pass_datetime]:
                        dataset = self.datasets[pass_datetime][satellite]
                        scene_data = dataset.data(variable, sses_correction)
                    else:
                        scene_data = numpy.nanmean(numpy.stack([dataset.data(variable, sses_correction) for dataset in
                                                                self.datasets[pass_datetime].values()], axis=0), axis=0)

                    if numpy.any(~numpy.isnan(scene_data)):
                        scenes_data.append(scene_data)

            variable_data = numpy.empty(
                (VIIRSDataset.study_area_coordinates['lat'].shape[0],
                 VIIRSDataset.study_area_coordinates['lon'].shape[0]))
            variable_data[:] = numpy.nan

            if len(scenes_data) > 0:
                # check if user wants to average data
                if average:
                    variable_data = numpy.nanmean(numpy.stack(scenes_data, axis=0), axis=0)
                else:  # otherwise overlap based on datetime
                    for scene_data in scenes_data:
                        if variable == 'sses':
                            scene_data[scene_data == 0] = numpy.nan

                        variable_data[~numpy.isnan(scene_data)] = scene_data[~numpy.isnan(scene_data)]

            variables_data[variable] = variable_data

        return variables_data

    def write_rasters(self, output_dir: str, variables: list = ('sst', 'sses'), filename_prefix: str = 'viirs',
                      fill_value: float = None, driver: str = 'GTiff', sses_correction: bool = False,
                      satellite: str = None):
        """
        Write individual VIIRS rasters to directory.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to write.
        :param filename_prefix: Prefix for output filenames.
        :param fill_value: Desired fill value of output.
        :param driver: String of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param satellite: VIIRS platform to retrieve. Default: per-granule averages of platform datasets.
        """

        # write a raster for each pass retrieved scene
        with futures.ThreadPoolExecutor() as concurrency_pool:
            for dataset_datetime, current_satellite in self.datasets.items():
                if current_satellite is None or current_satellite == satellite:
                    dataset = self.datasets[dataset_datetime][current_satellite]

                    concurrency_pool.submit(dataset.write_rasters, output_dir, variables=variables,
                                            filename_prefix=f'{filename_prefix}_' +
                                                            f'{dataset_datetime.strftime("%Y%m%d%H%M%S")}',
                                            fill_value=fill_value, drivers=driver, sses_correction=sses_correction)

    def write_raster(self, output_dir: str, filename_prefix: str = None, filename_suffix: str = None,
                     start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                     average: bool = False, fill_value: float = -9999, driver: str = 'GTiff',
                     sses_correction: bool = False, variables: list = tuple(['sst']), satellite: str = None):

        """
        Write VIIRS raster of SST data (either overlapped or averaged) from the given time interval.

        :param output_dir: Path to output directory.
        :param filename_prefix: Prefix for output filenames.
        :param filename_suffix: Suffix for output filenames.
        :param start_datetime: Beginning of time interval (in UTC).
        :param end_datetime: End of time interval (in UTC).
        :param average: Whether to average rasters, otherwise overlap them.
        :param fill_value: Desired fill value of output.
        :param driver: String of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param variables: List of variables to write (either 'sst' or 'sses').
        :param satellite: VIIRS platform to retrieve. Default: per-granule averages of platform datasets.
        """

        if start_datetime is None:
            start_datetime = self.start_datetime

        if end_datetime is None:
            end_datetime = self.end_datetime

        variable_data = self.data(start_datetime, end_datetime, average, sses_correction, variables, satellite)

        for variable, output_data in variable_data.items():
            if output_data is not None and numpy.any(~numpy.isnan(output_data)):
                output_data[numpy.isnan(output_data)] = fill_value

                raster_data = output_data.astype(rasterio.float32)

                # define arguments to GDAL driver
                gdal_args = {
                    'height': raster_data.shape[0], 'width': raster_data.shape[1], 'count': 1,
                    'crs': RASTERIO_CRS, 'dtype': raster_data.dtype,
                    'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item(),
                    'transform': VIIRSRange.study_area_transform
                }

                if driver == 'AAIGrid':
                    file_extension = 'asc'
                    gdal_args.update({
                        'FORCE_CELLSIZE': 'YES'
                    })
                elif driver == 'GPKG':
                    file_extension = 'gpkg'
                else:
                    file_extension = 'tiff'
                    gdal_args.update({
                        'TILED': 'YES'
                    })

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

                if self.logger is not None:
                    self.logger.info(f'Writing {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(raster_data, 1)
            else:
                if self.logger is not None:
                    self.logger.warning(
                        f'No {"VIIRS" if satellite is None else "VIIRS " + satellite} {variable} found between {start_datetime} and {end_datetime}.')

    def to_xarray(self, variables: list = ('sst', 'sses'), mean: bool = True, sses_correction: bool = False,
                  satellites: list = None) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: List of variables to use.
        :param mean: Whether to average all time indices.
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param satellites: VIIRS platforms to retrieve. Default: per-granule averages of platform datasets.
        :return: xarray Dataset of given variables.
        """

        data_arrays = {}

        coordinates = OrderedDict({
            'lat': VIIRSDataset.study_area_coordinates['lat'],
            'lon': VIIRSDataset.study_area_coordinates['lon']
        })

        if satellites is not None:
            coordinates['satellite'] = satellites

            satellites_data = [self.data(average=mean, sses_correction=sses_correction,
                                         variables=variables, satellite=satellite) for satellite in satellites]

            variables_data = {}

            for variable in variables:
                satellites_variable_data = [satellite_data[variable] for satellite_data in satellites_data if
                                            satellite_data[variable] is not None]
                variables_data[variable] = numpy.stack(satellites_variable_data, axis=2)
        else:
            variables_data = self.data(average=mean, sses_correction=sses_correction, variables=variables)

        for variable, variable_data in variables_data.items():
            data_arrays[variable] = xarray.DataArray(variable_data, coords=coordinates, dims=tuple(coordinates.keys()))

        output_dataset = xarray.Dataset(data_vars=data_arrays)

        del data_arrays

        return output_dataset

    def to_netcdf(self, output_file: str, variables: list = None, mean: bool = True, sses_correction: bool = False,
                  satellites: list = None):
        """
        Writes to NetCDF file.

        :param output_file: Output file to write.
        :param variables: List of variables to use.
        :param mean: Whether to average all time indices.
        :param sses_correction: Whether to subtract SSES bias from L3 sea surface temperature data.
        :param satellites: VIIRS platforms to retrieve. Default: per-granule averages of platform datasets.
        """

        self.to_xarray(variables, mean, sses_correction, satellites).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.satellites, self.study_area_polygon_filename, self.viirs_pass_times_filename,
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

    :param study_area_polygon_filename: path to vector file containing polygon of study area
    :param start_datetime: beginning of given VIIRS period (in UTC)
    :param output_filename: path to output file
    :param num_periods: number of periods to store
    :param data_source: either 'STAR' or 'OSPO'
    :param subskin: whether dataset should use subskin or not
    :param acspo_version: ACSPO Version number (2.40 - 2.41)
    """

    start_datetime = _utilities.round_to_ten_minutes(start_datetime)
    end_datetime = _utilities.round_to_ten_minutes(start_datetime + (VIIRS_PERIOD * num_periods))

    print(
        f'Getting pass times between {start_datetime.strftime("%Y-%m-%d %H:%M:%S")} and ' +
        f'{end_datetime.strftime("%Y-%m-%d %H:%M:%S")}')

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
                dataset = VIIRSDataset(cycle_datetime, study_area_polygon_filename, data_source, subskin,
                                       acspo_version)

                # check if dataset falls within polygon extent
                if dataset.data_extent.is_valid:
                    if study_area_polygon.intersects(dataset.data_extent):
                        # get duration from current cycle start
                        cycle_duration = cycle_datetime - (start_datetime + cycle_offset)

                        print(
                            f'{cycle_datetime.strftime("%Y%m%dT%H%M%S")} {cycle_duration.total_seconds()}: ' +
                            f'valid scene (checked {cycle_index + 1} cycle(s))')
                        lines.append(f'{cycle_datetime.strftime("%Y%m%dT%H%M%S")},{cycle_duration.total_seconds()}')

                # if we get to here, break and continue to the next datetime
                break
            except _utilities.NoDataError as error:
                print(error)
        else:
            print(f'{current_datetime.strftime("%Y%m%dT%H%M%S")}: missing dataset across all cycles')

        # write lines to file
        with open(output_filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

        print('Wrote data to file')


def get_pass_times(start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                   pass_times_filename: str = PASS_TIMES_FILENAME):
    """
    Retreive array of datetimes of VIIRS passes within the given time interval, given initial period durations.

    :param start_datetime: Beginning of time interval (in UTC).
    :param end_datetime: End of time interval (in UTC).
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
    output_dir = os.path.join(DATA_DIR, r'output\test')

    start_datetime = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    end_datetime = start_datetime + datetime.timedelta(days=1)

    viirs_range = VIIRSRange(start_datetime, end_datetime)
    viirs_range.write_raster(output_dir)

    print('done')
