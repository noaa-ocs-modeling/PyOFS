from collections import OrderedDict
from concurrent import futures
from datetime import datetime, timedelta
import ftplib
import math
import os
from os import PathLike
from pathlib import Path
from typing import Collection

import fiona.crs
import numpy
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.features
import shapely
import shapely.geometry
import shapely.wkt
import xarray

import PyOFS
from PyOFS import (
    CRS_EPSG,
    DATA_DIRECTORY,
    LEAFLET_NODATA_VALUE,
    TIFF_CREATION_OPTIONS,
    get_logger,
    utilities,
)

LOGGER = get_logger('PyOFS.VIIRS')

VIIRS_START_TIME = datetime.strptime('2012-03-01 00:10:00', '%Y-%m-%d %H:%M:%S')
VIIRS_PERIOD = timedelta(days=16)

PASS_TIMES_FILENAME = DATA_DIRECTORY / 'reference' / 'viirs_pass_times.txt'
STUDY_AREA_POLYGON_FILENAME = DATA_DIRECTORY / 'reference' / 'wcofs.gpkg:study_area'

OUTPUT_CRS = fiona.crs.from_epsg(CRS_EPSG)

NRT_DELAY = timedelta(hours=2)

SOURCE_URLS = OrderedDict(
    {
        'OpenDAP': OrderedDict(
            {
                'NESDIS': 'https://www.star.nesdis.noaa.gov/thredds/dodsC',
                'JPL': 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ghrsst/data/GDS2/L3U',
                'NODC': 'https://data.nodc.noaa.gov/thredds/catalog/ghrsst/L3U',
            }
        ),
        'FTP': OrderedDict({'NESDIS': 'ftp.star.nesdis.noaa.gov/pub/socd2/coastwatch/sst'}),
    }
)


class VIIRSDataset:
    """
    Visible Infrared Imaging Radiometer Suite (VIIRS) sea-surface temperature.
    """

    study_area_transform = None
    study_area_extent = None
    study_area_bounds = None
    study_area_coordinates = None

    def __init__(
        self,
        data_time: datetime = None,
        satellite: str = 'NPP',
        study_area_polygon_filename: PathLike = STUDY_AREA_POLYGON_FILENAME,
        algorithm: str = 'OSPO',
        version: str = None,
    ):
        """
        Retrieve VIIRS NetCDF observation from NOAA with given datetime.

        :param data_time: observation datetime
        :param satellite: VIIRS platform
        :param study_area_polygon_filename: filename of vector file containing study area boundary
        :param algorithm: either 'STAR' or 'OSPO'
        :param version: ACSPO algorithm version
        :raises NoDataError: if observation does not exist
        """

        if not isinstance(study_area_polygon_filename, Path):
            study_area_polygon_filename = Path(study_area_polygon_filename)

        if data_time is None:
            data_time = datetime.now()

        # round minute to nearest 10 minutes (VIIRS data interval)
        self.data_time = PyOFS.round_to_ten_minutes(data_time)

        self.satellite = satellite

        self.study_area_polygon_filename = study_area_polygon_filename

        # use NRT flag if granule is less than 13 days old
        self.near_real_time = datetime.now() - data_time <= timedelta(days=13)
        self.algorithm = algorithm

        if version is None:
            if data_time >= datetime(2019, 4, 23, 12, 50):
                self.version = '2.61'
            elif data_time >= datetime(2018, 11, 7, 15, 10):
                self.version = '2.60'
            elif data_time >= datetime(2017, 9, 14, 12, 50):
                self.version = '2.41'
            else:
                self.version = '2.40'
        else:
            self.version = version

        self.url = None

        day_dir = f'{self.data_time.year}/{self.data_time.timetuple().tm_yday:03}'
        filename = f'{self.data_time:%Y%m%d%H%M%S}-{self.algorithm}-L3U_GHRSST-SSTsubskin-VIIRS_{self.satellite.upper()}-ACSPO_V{self.version}-v02.0-fv01.0.nc'

        # TODO N20 does not yet have a reanalysis archive on NESDIS (as of March 8th, 2019)
        if self.satellite.upper() == 'N20' and not self.near_real_time:
            raise PyOFS.NoDataError(
                f'{self.satellite.upper()} does not yet have a reanalysis archive'
            )

        for source, source_url in SOURCE_URLS['OpenDAP'].items():
            url = source_url

            if self.near_real_time:
                if source == 'NESDIS':
                    url = f'{source_url}/grid{self.satellite.upper()}VIIRSNRTL3UWW00/{day_dir}/{filename}'
                elif source == 'JPL':
                    url = f'{source_url}/VIIRS_{self.satellite.upper()}/{algorithm}/v{self.version}/{day_dir}/{filename}'
                elif source in 'NODC':
                    url = f'{source_url}/VIIRS_{self.satellite.upper()}/{algorithm}/{day_dir}/{filename}'
            else:
                if source == 'NESDIS':
                    url = f'{source_url}/grid{"" if self.near_real_time else "S"}{self.satellite.upper()}VIIRSSCIENCEL3UWW00/{day_dir}/{filename}'
                else:
                    LOGGER.warning(f'{source} does not contain a reanalysis archive')

            try:
                self.dataset = xarray.open_dataset(url)
                self.url = url
                break
            except Exception as error:
                LOGGER.warning(f'{error.__class__.__name__}: {error}')

        if self.url is None:
            LOGGER.warning('Error collecting from OpenDAP; falling back to FTP...')

            for source, source_url in SOURCE_URLS['FTP'].items():
                host_url, ftp_input_dir = source_url.split('/', 1)
                ftp_path = ftp_input_dir
                url = host_url

                if source == 'NESDIS':
                    if self.near_real_time:
                        ftp_path = f'/{ftp_input_dir}/nrt/viirs/{self.satellite.lower()}/l3u/{day_dir}/{filename}'
                    else:
                        ftp_path = f'/{ftp_input_dir}/ran/viirs/{"S" if self.satellite.upper() == "NPP" else ""}{self.satellite.lower()}/l3u/{day_dir}/{filename}'

                    url = f'{host_url}/{ftp_path.lstrip("/")}'

                try:
                    with ftplib.FTP(host_url) as ftp_connection:
                        ftp_connection.login()

                        output_dir = DATA_DIRECTORY / 'input' / 'viirs'

                        if not output_dir.exists():
                            os.makedirs(output_dir, exist_ok=True)

                        output_filename = output_dir / f'viirs_{self.data_time:%Y%m%dT%H%M}.nc'

                        if output_filename.exists():
                            os.remove(output_filename)

                        try:
                            with open(output_filename, 'wb') as output_file:
                                ftp_connection.retrbinary(
                                    f'RETR {ftp_path}', output_file.write
                                )
                                self.dataset = xarray.open_dataset(output_filename)
                        except:
                            raise
                        finally:
                            os.remove(output_filename)

                    self.url = url
                    break
                except Exception as error:
                    LOGGER.warning(f'{error.__class__.__name__}: {error}')

                if self.url is not None:
                    break

        if self.url is None:
            raise PyOFS.NoDataError(f'No VIIRS observation found at {self.data_time} UTC.')

        # construct rectangular polygon of granule extent
        if 'geospatial_bounds' in self.dataset.attrs:
            self.data_extent = shapely.wkt.loads(self.dataset.geospatial_bounds)
        elif 'geospatial_lon_min' in self.dataset.attrs:
            lon_min = float(self.dataset.geospatial_lon_min)
            lon_max = float(self.dataset.geospatial_lon_max)
            lat_min = float(self.dataset.geospatial_lat_min)
            lat_max = float(self.dataset.geospatial_lat_max)

            if lon_min < lon_max:
                self.data_extent = shapely.geometry.Polygon(
                    [
                        (lon_min, lat_max),
                        (lon_max, lat_max),
                        (lon_max, lat_min),
                        (lon_min, lat_min),
                    ]
                )
            else:
                # geospatial bounds cross the antimeridian, so we create a multipolygon
                self.data_extent = shapely.geometry.MultiPolygon(
                    [
                        shapely.geometry.Polygon(
                            [
                                (lon_min, lat_max),
                                (180, lat_max),
                                (180, lat_min),
                                (lon_min, lat_min),
                            ]
                        ),
                        shapely.geometry.Polygon(
                            [
                                (-180, lat_max),
                                (lon_max, lat_max),
                                (lon_max, lat_min),
                                (-180, lat_min),
                            ]
                        ),
                    ]
                )
        else:
            LOGGER.warning(f'{self.data_time} UTC: Dataset has no stored bounds...')

        lon_pixel_size = self.dataset.geospatial_lon_resolution
        lat_pixel_size = self.dataset.geospatial_lat_resolution

        if VIIRSDataset.study_area_extent is None:
            LOGGER.debug(
                f'Calculating indices and transform from granule at {self.data_time} UTC...'
            )

            # get first record in layer
            VIIRSDataset.study_area_extent = shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.Polygon(polygon[0])
                    for polygon in utilities.get_first_record(
                    self.study_area_polygon_filename
                )['geometry']['coordinates']
                ]
            )

            VIIRSDataset.study_area_bounds = VIIRSDataset.study_area_extent.bounds
            VIIRSDataset.study_area_transform = rasterio.transform.from_origin(
                VIIRSDataset.study_area_bounds[0],
                VIIRSDataset.study_area_bounds[3],
                lon_pixel_size,
                lat_pixel_size,
            )

        if VIIRSDataset.study_area_bounds is not None:
            self.dataset = self.dataset.isel(time=0).sel(
                lon=slice(
                    VIIRSDataset.study_area_bounds[0], VIIRSDataset.study_area_bounds[2]
                ),
                lat=slice(
                    VIIRSDataset.study_area_bounds[3], VIIRSDataset.study_area_bounds[1]
                ),
            )

        if VIIRSDataset.study_area_coordinates is None:
            VIIRSDataset.study_area_coordinates = {
                'lon': self.dataset['lon'],
                'lat': self.dataset['lat'],
            }

    def bounds(self) -> tuple:
        """
        Get coordinate bounds of observation.

        :return: tuple of bounds (west, south, east, north)
        """

        return self.data_extent.bounds

    def cell_size(self) -> tuple:
        """
        Get cell sizes of observation.

        :return: tuple of cell sizes (x_size, y_size)
        """

        return self.dataset.geospatial_lon_resolution, self.dataset.geospatial_lat_resolution

    def data(self, variable: str = 'sst', correct_sses=True) -> numpy.array:
        """
        Retrieve data of given variable. Use 'sst_sses' to retrieve SST corrected with sensor-specific error statistic (SSES)

        :param variable: variable name (one of 'sst', 'sses', or 'sst_sses')
        :param correct_sses: whether to apply sensor bias
        :return: matrix of data in Celsius
        """

        if variable == 'sst':
            return self._sst(correct_sses)
        elif variable == 'sses':
            return self._sses()

    def _sst(self, correct_sses: bool = False) -> numpy.array:
        """
        Return matrix of sea surface temperature.

        :param correct_sses: whether to apply sensor bias
        :return: matrix of SST in Celsius
        """

        # observation SST data (masked array) using vertically reflected VIIRS grid
        output_sst_data = self.dataset['sea_surface_temperature'].values

        # check for unmasked data
        if not numpy.isnan(output_sst_data).all():
            if numpy.nanmax(output_sst_data) > 0:
                if numpy.nanmin(output_sst_data) <= 0:
                    output_sst_data[output_sst_data <= 0] = numpy.nan

                if correct_sses:
                    sses = self._sses()

                    mismatched_records = len(
                        numpy.where(numpy.isnan(output_sst_data) != (sses == 0))[0]
                    )
                    total_records = output_sst_data.shape[0] * output_sst_data.shape[1]
                    mismatch_percentage = mismatched_records / total_records * 100

                    if mismatch_percentage > 0:
                        LOGGER.warning(
                            f'{self.data_time} UTC: SSES extent mismatch at {mismatch_percentage:.1f}%'
                        )

                    output_sst_data -= sses

                # convert from Kelvin to Celsius (subtract 273.15)
                output_sst_data -= 273.15
            else:
                output_sst_data[:] = numpy.nan

        return output_sst_data

    def _sses(self) -> numpy.array:
        """
        Return matrix of sensor-specific error statistics.

        :return: array of SSES bias in Celsius
        """

        # observation bias values using vertically reflected VIIRS grid
        sses_data = self.dataset['sses_bias'].values

        # replace masked values with 0
        sses_data[numpy.isnan(sses_data)] = 0

        # negative offset by 2.048
        sses_data -= 2.048

        return sses_data

    def write_rasters(
        self,
        output_dir: PathLike,
        variables: Collection[str] = ('sst', 'sses'),
        filename_prefix: str = 'viirs',
        fill_value: float = LEAFLET_NODATA_VALUE,
        driver: str = 'GTiff',
        correct_sses: bool = False,
    ):
        """
        Write VIIRS rasters to file using data from given variables.

        :param output_dir: path to output directory
        :param variables: variable names to write
        :param filename_prefix: prefix for output filenames
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param correct_sses: whether to subtract SSES bias from SST
        """

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        for variable in variables:
            input_data = self.data(variable, correct_sses)

            if variable == 'sses':
                fill_value = 0

            if input_data is not None and not numpy.isnan(input_data).all():
                if fill_value is not None:
                    input_data[numpy.isnan(input_data)] = fill_value

                gdal_args = {
                    'height': input_data.shape[0],
                    'width': input_data.shape[1],
                    'count': 1,
                    'dtype': rasterio.float32,
                    'crs': CRS.from_dict(OUTPUT_CRS),
                    'transform': VIIRSDataset.study_area_transform,
                    'nodata': fill_value,
                }

                if driver == 'AAIGrid':
                    file_extension = 'asc'
                    gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                elif driver == 'GPKG':
                    file_extension = 'gpkg'
                else:
                    file_extension = 'tiff'
                    gdal_args.update(TIFF_CREATION_OPTIONS)

                output_filename = output_dir / f'{filename_prefix}_{variable}.{file_extension}'

                # use rasterio to write to raster with GDAL args
                LOGGER.info(f'Writing to {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(input_data, 1)
                    if driver == 'GTiff':
                        output_raster.build_overviews(
                            PyOFS.overview_levels(input_data.shape), Resampling['average']
                        )
                        output_raster.update_tags(ns='rio_overview', resampling='average')

    def __repr__(self):
        used_params = [self.data_time.__repr__()]
        optional_params = [
            self.satellite,
            self.study_area_polygon_filename,
            self.near_real_time,
            self.algorithm,
            self.version,
        ]

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
    Range of VIIRS observation.
    """

    study_area_transform = None
    study_area_index_bounds = None

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        satellites: list = ('NPP', 'N20'),
        study_area_polygon_filename: PathLike = STUDY_AREA_POLYGON_FILENAME,
        pass_times_filename: PathLike = PASS_TIMES_FILENAME,
        algorithm: str = 'OSPO',
        version: str = None,
    ):
        """
        Collect VIIRS datasets within time interval.

        :param start_time: beginning of time interval (in UTC)
        :param end_time: end of time interval (in UTC)
        :param satellites: VIIRS platforms
        :param study_area_polygon_filename: filename of vector file of study area boundary
        :param pass_times_filename: path to text file with pass times
        :param algorithm: either 'STAR' or 'OSPO'
        :param version: ACSPO algorithm version
        :raises NoDataError: if data does not exist
        """

        if not isinstance(study_area_polygon_filename, Path):
            study_area_polygon_filename = Path(study_area_polygon_filename)

        if not isinstance(pass_times_filename, Path):
            pass_times_filename = Path(pass_times_filename)

        self.start_time = start_time
        if end_time > datetime.utcnow():
            # VIIRS near real time delay is 2 hours behind UTC
            self.end_time = datetime.utcnow() - NRT_DELAY
        else:
            self.end_time = end_time

        self.satellites = satellites

        self.study_area_polygon_filename = study_area_polygon_filename
        self.viirs_pass_times_filename = pass_times_filename
        self.algorithm = algorithm
        self.version = version

        self.pass_times = get_pass_times(
            self.start_time, self.end_time, self.viirs_pass_times_filename
        )

        if len(self.pass_times) > 0:
            LOGGER.info(
                f'Collecting VIIRS data from {len(self.pass_times)} passes between {numpy.min(self.pass_times)} UTC and {numpy.max(self.pass_times)} UTC...'
            )

            # create dictionary to store scenes
            self.datasets = {pass_time: {} for pass_time in self.pass_times}

            with futures.ThreadPoolExecutor() as concurrency_pool:
                for satellite in self.satellites:
                    running_futures = {}

                    for pass_time in self.pass_times:
                        running_future = concurrency_pool.submit(
                            VIIRSDataset,
                            data_time=pass_time,
                            study_area_polygon_filename=self.study_area_polygon_filename,
                            algorithm=self.algorithm,
                            version=self.version,
                            satellite=satellite,
                        )
                        running_futures[running_future] = pass_time

                    for completed_future in futures.as_completed(running_futures):
                        if completed_future.exception() is None:
                            pass_time = running_futures[completed_future]
                            viirs_dataset = completed_future.result()
                            self.datasets[pass_time][satellite] = viirs_dataset
                        else:
                            LOGGER.warning(
                                f'Dataset creation error: {completed_future.exception()}'
                            )

                    del running_futures

            if len(self.datasets) > 0:
                VIIRSRange.study_area_transform = VIIRSDataset.study_area_transform
                VIIRSRange.study_area_extent = VIIRSDataset.study_area_extent
                VIIRSRange.study_area_bounds = VIIRSDataset.study_area_bounds

                LOGGER.debug(f'VIIRS data was found in {len(self.datasets)} passes.')
            else:
                raise PyOFS.NoDataError(
                    f'No VIIRS datasets found between {self.start_time} UTC and {self.end_time} UTC.'
                )

        else:
            raise PyOFS.NoDataError(
                f'There are no VIIRS passes between {self.start_time} UTC and {self.end_time} UTC.'
            )

    def cell_size(self) -> tuple:
        """
        Get cell sizes of observation.

        :return: tuple of cell sizes (x_size, y_size)
        """

        sample_dataset = next(iter(self.datasets.values()))

        return (
            sample_dataset.netcdf_dataset.geospatial_lon_resolution,
            sample_dataset.netcdf_dataset.geospatial_lat_resolution,
        )

    def data(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        average: bool = False,
        correct_sses: bool = False,
        variables: Collection[str] = tuple('sst'),
        satellite: str = None,
    ) -> dict:
        """
        Get VIIRS data (either overlapped or averaged) from the given time interval.

        :param start_time: beginning of time interval (in UTC)
        :param end_time: end of time interval (in UTC)
        :param average: whether to average rasters, otherwise overlap them
        :param correct_sses: whether to subtract SSES bias from L3 sea surface temperature data
        :param variables: variables to write (either 'sst' or 'sses')
        :param satellite: VIIRS platform to retrieve. Default: per-granule averages of platform datasets
        :return dictionary of data per variable
        """

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        dataset_times = numpy.sort(list(self.datasets.keys()))

        # find first and last times within specified time interval
        start_index = numpy.searchsorted(dataset_times, start_time)
        end_index = numpy.searchsorted(dataset_times, end_time)

        pass_times = dataset_times[start_index:end_index]

        if variables is None:
            variables = ['sst', 'sses']

        variables_data = {}

        for variable in variables:
            scenes_data = []

            for pass_time in pass_times:
                if len(self.datasets[pass_time]) > 0:
                    if satellite is not None and satellite in self.datasets[pass_time]:
                        dataset = self.datasets[pass_time][satellite]
                        scene_data = dataset.data(variable, correct_sses)
                    else:
                        scene_data = numpy.nanmean(
                            numpy.stack(
                                [
                                    dataset.data(variable, correct_sses)
                                    for dataset in self.datasets[pass_time].values()
                                ],
                                axis=0,
                            ),
                            axis=0,
                        )

                    if numpy.any(~numpy.isnan(scene_data)):
                        scenes_data.append(scene_data)

            variable_data = numpy.empty(
                (
                    VIIRSDataset.study_area_coordinates['lat'].shape[0],
                    VIIRSDataset.study_area_coordinates['lon'].shape[0],
                )
            )
            variable_data[:] = numpy.nan

            if len(scenes_data) > 0:
                # check if user wants to average data
                if average:
                    variable_data = numpy.nanmean(numpy.stack(scenes_data, axis=0), axis=0)
                else:  # otherwise overlap based on datetime
                    for scene_data in scenes_data:
                        if variable == 'sses':
                            scene_data[scene_data == 0] = numpy.nan

                        variable_data[~numpy.isnan(scene_data)] = scene_data[
                            ~numpy.isnan(scene_data)
                        ]

            variables_data[variable] = variable_data

        return variables_data

    def write_rasters(
        self,
        output_dir: PathLike,
        variables: Collection[str] = ('sst', 'sses'),
        filename_prefix: str = 'viirs',
        fill_value: float = None,
        driver: str = 'GTiff',
        correct_sses: bool = False,
        satellite: str = None,
    ):
        """
        Write individual VIIRS rasters to directory.

        :param output_dir: path to output directory
        :param variables: variable names to write
        :param filename_prefix: prefix for output filenames
        :param fill_value: desired fill value of output
        :param driver: string of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param correct_sses: whether to subtract SSES bias from L3 sea surface temperature data
        :param satellite: VIIRS platform to retrieve; if not specified, will average from both satellites
        """

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        # write a raster for each pass retrieved scene
        with futures.ThreadPoolExecutor() as concurrency_pool:
            for dataset_time, current_satellite in self.datasets.items():
                if current_satellite is None or current_satellite == satellite:
                    dataset = self.datasets[dataset_time][current_satellite]

                    concurrency_pool.submit(
                        dataset.write_rasters,
                        output_dir,
                        variables=variables,
                        filename_prefix=f'{filename_prefix}_{dataset_time:%Y%m%d%H%M%S}',
                        fill_value=fill_value,
                        drivers=driver,
                        correct_sses=correct_sses,
                    )

    def write_raster(
        self,
        output_dir: PathLike,
        filename_prefix: str = None,
        filename_suffix: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        average: bool = False,
        fill_value: float = LEAFLET_NODATA_VALUE,
        driver: str = 'GTiff',
        correct_sses: bool = False,
        variables: Collection[str] = tuple(['sst']),
        satellite: str = None,
    ):

        """
        Write VIIRS raster of SST data (either overlapped or averaged) from the given time interval.

        :param output_dir: path to output directory
        :param filename_prefix: prefix for output filenames
        :param filename_suffix: suffix for output filenames
        :param start_time: beginning of time interval (in UTC)
        :param end_time: end of time interval (in UTC)
        :param average: whether to average rasters, otherwise overlap them
        :param fill_value: desired fill value of output
        :param driver: string of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param correct_sses: whether to subtract SSES bias from L3 sea surface temperature data
        :param variables: variables to write (either 'sst' or 'sses')
        :param satellite: VIIRS platform to retrieve; if not specified, will average from both satellites
        """

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        variable_data = self.data(
            start_time, end_time, average, correct_sses, variables, satellite
        )

        for variable, output_data in variable_data.items():
            if output_data is not None and numpy.any(~numpy.isnan(output_data)):
                output_data[numpy.isnan(output_data)] = fill_value

                raster_data = output_data.astype(rasterio.float32)

                if fill_value is not None:
                    raster_data[numpy.isnan(raster_data)] = fill_value

                # define arguments to GDAL driver
                gdal_args = {
                    'height': raster_data.shape[0],
                    'width': raster_data.shape[1],
                    'count': 1,
                    'crs': OUTPUT_CRS,
                    'dtype': raster_data.dtype,
                    'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item(),
                    'transform': VIIRSRange.study_area_transform,
                }

                if driver == 'AAIGrid':
                    file_extension = 'asc'
                    gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                elif driver == 'GPKG':
                    file_extension = 'gpkg'
                else:
                    file_extension = 'tiff'
                    gdal_args.update(TIFF_CREATION_OPTIONS)

                if filename_prefix is None:
                    current_filename_prefix = f'viirs_{variable}'
                else:
                    current_filename_prefix = filename_prefix

                if filename_suffix is None:
                    start_time_string = f'{start_time:%Y%m%d%H%M}'
                    end_time_string = f'{end_time:%Y%m%d%H%M}'

                    if '0000' in start_time_string and '0000' in end_time_string:
                        start_time_string = start_time_string.replace('0000', '')
                        end_time_string = end_time_string.replace('0000', '')

                    current_filename_suffix = f'{start_time_string}_{end_time_string}'
                else:
                    current_filename_suffix = filename_suffix

                output_filename = (
                    output_dir
                    / f'{current_filename_prefix}_{current_filename_suffix}.{file_extension}'
                )

                LOGGER.info(f'Writing {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(raster_data, 1)
                    if driver == 'GTiff':
                        output_raster.build_overviews(
                            PyOFS.overview_levels(raster_data.shape), Resampling['average']
                        )
                        output_raster.update_tags(ns='rio_overview', resampling='average')
            else:
                LOGGER.warning(
                    f'No {"VIIRS" if satellite is None else "VIIRS " + satellite} {variable} found between {start_time} and {end_time}.'
                )

    def to_xarray(
        self,
        variables: Collection[str] = ('sst', 'sses'),
        mean: bool = True,
        correct_sses: bool = False,
        satellites: list = None,
    ) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: variables to use
        :param mean: whether to average all time indices
        :param correct_sses: whether to subtract SSES bias from L3 sea surface temperature data
        :param satellites: VIIRS platforms to retrieve; if not specified, will average from both satellites
        :return: xarray observation of given variables
        """

        output_dataset = xarray.Dataset()

        coordinates = OrderedDict(
            {
                'lat': VIIRSDataset.study_area_coordinates['lat'],
                'lon': VIIRSDataset.study_area_coordinates['lon'],
            }
        )

        if satellites is not None:
            coordinates['satellite'] = satellites

            satellites_data = [
                self.data(
                    average=mean,
                    correct_sses=correct_sses,
                    variables=variables,
                    satellite=satellite,
                )
                for satellite in satellites
            ]

            variables_data = {}

            for variable in variables:
                satellites_variable_data = [
                    satellite_data[variable]
                    for satellite_data in satellites_data
                    if satellite_data[variable] is not None
                ]
                variables_data[variable] = numpy.stack(satellites_variable_data, axis=2)
        else:
            variables_data = self.data(
                average=mean, correct_sses=correct_sses, variables=variables
            )

        for variable, variable_data in variables_data.items():
            output_dataset.update(
                {
                    variable: xarray.DataArray(
                        variable_data, coords=coordinates, dims=tuple(coordinates.keys())
                    )
                }
            )

        return output_dataset

    def to_netcdf(
        self,
        output_file: str,
        variables: Collection[str] = None,
        mean: bool = True,
        correct_sses: bool = False,
        satellites: list = None,
    ):
        """
        Writes to NetCDF file.

        :param output_file: output file to write
        :param variables: variables to use
        :param mean: whether to average all time indices
        :param correct_sses: whether to subtract SSES bias from L3 sea surface temperature data
        :param satellites: VIIRS platforms to retrieve; if not specified, will average from both satellites
        """

        self.to_xarray(variables, mean, correct_sses, satellites).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.start_time.__repr__(), self.end_time.__repr__()]
        optional_params = [
            self.satellites,
            self.study_area_polygon_filename,
            self.viirs_pass_times_filename,
            self.algorithm,
            self.version,
        ]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({", ".join(used_params)})'


def store_viirs_pass_times(
    satellite: str,
    study_area_polygon_filename: PathLike = STUDY_AREA_POLYGON_FILENAME,
    start_time: datetime = VIIRS_START_TIME,
    output_filename: str = PASS_TIMES_FILENAME,
    num_periods: int = 1,
    algorithm: str = 'STAR',
    version: str = '2.40',
):
    """
    Compute VIIRS pass times from the given start date along the number of periods specified.

    :param satellite: satellite for which to store pass times, either NPP or N20
    :param study_area_polygon_filename: path to vector file containing polygon of study area
    :param start_time: beginning of given VIIRS period (in UTC)
    :param output_filename: path to output file
    :param num_periods: number of periods to store
    :param algorithm: either 'STAR' or 'OSPO'
    :param version: ACSPO Version number (2.40 - 2.41)
    """

    if not isinstance(study_area_polygon_filename, Path):
        study_area_polygon_filename = Path(study_area_polygon_filename)

    start_time = PyOFS.round_to_ten_minutes(start_time)
    end_time = PyOFS.round_to_ten_minutes(start_time + (VIIRS_PERIOD * num_periods))

    LOGGER.info(
        f'Getting pass times between {start_time:%Y-%m-%d %H:%M:%S} and {end_time:%Y-%m-%d %H:%M:%S}'
    )

    datetime_range = PyOFS.ten_minute_range(start_time, end_time)

    # construct polygon from the first record in layer
    study_area_polygon = shapely.geometry.Polygon(
        utilities.get_first_record(study_area_polygon_filename)['geometry']['coordinates'][0]
    )

    lines = []

    for datetime_index in range(len(datetime_range)):
        current_time = datetime_range[datetime_index]

        # find number of cycles from the first orbit to the present day
        num_cycles = int((datetime.now() - start_time).days / 16)

        # iterate over each cycle
        for cycle_index in range(0, num_cycles):
            # get current datetime of interest
            cycle_offset = VIIRS_PERIOD * cycle_index
            cycle_time = current_time + cycle_offset

            try:
                # get observation of new datetime
                dataset = VIIRSDataset(
                    cycle_time, satellite, study_area_polygon_filename, algorithm, version
                )

                # check if observation falls within polygon extent
                if dataset.data_extent.is_valid:
                    if study_area_polygon.intersects(dataset.data_extent):
                        # get duration from current cycle start
                        cycle_duration = cycle_time - (start_time + cycle_offset)

                        LOGGER.info(
                            f'{cycle_time:%Y%m%dT%H%M%S} {cycle_duration / timedelta(seconds=1)}: valid scene (checked {cycle_index + 1} cycle(s))'
                        )
                        lines.append(
                            f'{cycle_time:%Y%m%dT%H%M%S},{cycle_duration / timedelta(seconds=1)}'
                        )

                # if we get to here, break and continue to the next datetime
                break
            except PyOFS.NoDataError as error:
                LOGGER.warning(f'{error.__class__.__name__}: {error}')
        else:
            LOGGER.warning(
                f'{current_time:%Y%m%dT%H%M%S}: missing observation across all cycles'
            )

        # write lines to file
        with open(output_filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

        LOGGER.info('Wrote data to file')


def get_pass_times(
    start_time: datetime,
    end_time: datetime,
    pass_times_filename: PathLike = PASS_TIMES_FILENAME,
):
    """
    Retreive array of datetimes of VIIRS passes within the given time interval, given initial period durations.

    :param start_time: beginning of time interval (in UTC)
    :param end_time: end of time interval (in UTC)
    :param pass_times_filename: filename of text file with durations of first VIIRS period
    :return:
    """

    if not isinstance(pass_times_filename, Path):
        pass_times_filename = Path(pass_times_filename)

    # get datetime of first pass in given file
    first_pass_row = numpy.genfromtxt(pass_times_filename, dtype=str, delimiter=',')[0, :]
    viirs_start_time = datetime.strptime(first_pass_row[0], '%Y%m%dT%H%M%S') - timedelta(
        seconds=float(first_pass_row[1])
    )

    # get starting datetime of the current VIIRS period
    period_start_time = viirs_start_time + timedelta(
        days=numpy.floor((start_time - viirs_start_time).days / 16) * 16
    )

    # get array of seconds since the start of the first 16-day VIIRS period
    pass_durations = numpy.genfromtxt(pass_times_filename, dtype=str, delimiter=',')[
                     :, 1
                     ].T.astype(numpy.float32)
    pass_durations = numpy.asarray(
        [timedelta(seconds=float(duration)) for duration in pass_durations]
    )

    # add extra VIIRS periods to end of pass durations
    if end_time > (period_start_time + VIIRS_PERIOD):
        extra_periods = math.ceil((end_time - period_start_time) / VIIRS_PERIOD) - 1
        for period in range(extra_periods):
            pass_durations = numpy.append(
                pass_durations, pass_durations[-360:] + pass_durations[-1]
            )

    # get datetimes of VIIRS passes within the given time interval
    pass_times = period_start_time + pass_durations

    # find starting and ending times within the given time interval
    start_index = numpy.searchsorted(pass_times, start_time)
    end_index = numpy.searchsorted(pass_times, end_time)

    # ensure at least one datetime in range
    if start_index == end_index:
        end_index += 1

    # trim datetimes to within the given time interval
    pass_times = pass_times[start_index:end_index]

    return pass_times


if __name__ == '__main__':
    output_dir = DATA_DIRECTORY / 'output' / 'test'

    start_time = datetime.utcnow() - timedelta(days=1)
    end_time = start_time + timedelta(days=1)

    viirs_range = VIIRSRange(start_time, end_time)
    viirs_range.write_raster(output_dir)

    print('done')
