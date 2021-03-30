from collections import OrderedDict
from datetime import date, datetime, timedelta
from os import PathLike
from pathlib import Path
import shutil
import tarfile
import threading
from typing import Collection

import fiona.crs
import numpy
import rasterio.control
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.features
import rasterio.mask
import rasterio.warp
from shapely import geometry
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

LOGGER = get_logger('PyOFS.RTOFS')

OUTPUT_CRS = fiona.crs.from_epsg(CRS_EPSG)

COORDINATE_VARIABLES = ['time', 'lev', 'lat', 'lon']

DATASET_STRUCTURE = {
    '2ds': {
        'nowcast': {
            'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'],
            'diag': ['ssh', 'ice_coverage', 'ice_thickness'],
        },
        'forecast': {
            'prog': ['sss', 'sst', 'u_velocity', 'v_velocity'],
            'diag': ['ssh', 'ice_coverage', 'ice_thickness'],
        },
    },
    '3dz': {
        'nowcast': {
            'salt': ['salinity'],
            'temp': ['temperature'],
            'uvel': ['u'],
            'vvel': ['v'],
        },
        'forecast': {
            'salt': ['salinity'],
            'temp': ['temperature'],
            'uvel': ['u'],
            'vvel': ['v'],
        },
    },
}

DATA_VARIABLES = {
    'sst': {'2ds': {'prog': 'sst'}, '3dz': {'temp': 'temperature'}},
    'sss': {'2ds': {'prog': 'sss'}, '3dz': {'salt': 'salinity'}},
    'ssu': {'2ds': {'prog': 'u_velocity'}, '3dz': {'uvel': 'u'}},
    'ssv': {'2ds': {'prog': 'v_velocity'}, '3dz': {'vvel': 'v'}},
    'ssh': {'2ds': {'diag': 'ssh'}},
}

TIME_DELTAS = {'daily': range(-3, 8 + 1)}

STUDY_AREA_POLYGON_FILENAME = DATA_DIRECTORY / r'reference\wcofs.gpkg:study_area'

SOURCE_URLS = {
    'NCEP': 'https://nomads.ncep.noaa.gov/dods/rtofs',
    'local': DATA_DIRECTORY / 'input' / 'rtofs',
}

GLOBAL_LOCK = threading.Lock()


class RTOFSDataset:
    """
    Real-Time Ocean Forecasting System (RTOFS) NetCDF observation.
    """

    def __init__(
        self,
        model_date: datetime = None,
        source: str = '2ds',
        time_interval: str = 'daily',
        study_area_polygon_filename: PathLike = STUDY_AREA_POLYGON_FILENAME,
        source_url: str = None,
        use_defaults: bool = True,
    ):
        """
        Creates new observation object from datetime and given model parameters.
        :param model_date: model run date
        :param source: either '2ds' or '3dz'
        :param time_interval: time interval of model output
        :param study_area_polygon_filename: filename of vector file containing study area boundary
        :param source_url: directory containing NetCDF files
        :param use_defaults: whether to fall back to default source URLs if the provided one does not exist
        """

        if not isinstance(study_area_polygon_filename, Path):
            study_area_polygon_filename = Path(study_area_polygon_filename)

        if model_date is None:
            model_date = datetime.now()

        if type(model_date) is date:
            self.model_time = datetime.combine(model_date, datetime.min.time())
        else:
            self.model_time = model_date.replace(hour=0, minute=0, second=0, microsecond=0)

        self.source = source
        self.time_interval = time_interval

        self.study_area_polygon_filename = study_area_polygon_filename
        self.study_area_geojson = utilities.get_first_record(self.study_area_polygon_filename)[
            'geometry'
        ]

        self.datasets = {}
        self.dataset_locks = {}

        date_string = f'{self.model_time:%Y%m%d}'
        date_dir = f'rtofs_global{date_string}'

        source_urls = SOURCE_URLS.copy()

        if source_url is not None:
            source_url = {'priority': source_url}
            if use_defaults:
                source_urls = {**source_url, **{source_urls}}

        self.source_names = []
        if self.time_interval == '3hrly' or self.time_interval == 'hrly' or self.time_interval == 'daily':
            # added due to the different hourly source for nowcast and forecast
            for self.time_interval in {'hrly', '3hrly'}:
                for source_name, source_url in source_urls.items():
                    for forecast_direction, datasets in DATASET_STRUCTURE[self.source].items():
                        if (forecast_direction == 'nowcast' and 'nowcast' in self.datasets and len(
                            self.datasets['nowcast']) > 0) or (
                            forecast_direction == 'forecast' and 'forecast' in self.datasets and len(
                            self.datasets['forecast']) > 0
                        ):
                            continue

                        self.datasets[forecast_direction] = {}
                        self.dataset_locks[forecast_direction] = {}

                        for dataset_name in datasets:
                            filename = f'rtofs_glo_{self.source}_{forecast_direction}_{self.time_interval}_{dataset_name}'
                            url = f'{source_url}/{date_dir}/{filename}'
                            if source_name == 'local':
                                url = f'{url}.nc'

                            try:
                                dataset = xarray.open_dataset(url)
                                self.datasets[forecast_direction][dataset_name] = dataset
                                self.dataset_locks[forecast_direction][dataset_name] = threading.Lock()
                                self.source_names.append(source_name)
                            except OSError as error:
                                LOGGER.warning(f'{error.__class__.__name__}: {error}')

        if (len(self.datasets['nowcast']) + len(self.datasets['forecast'])) > 0:
            if len(self.datasets['nowcast']) > 0:
                sample_dataset = next(iter(self.datasets['nowcast'].values()))
            else:
                sample_dataset = next(iter(self.datasets['forecast'].values()))

            self.lat = sample_dataset['lat'].values
            if not any(source_name == 'NCEP' for source_name in self.source_names):
                self.lon = sample_dataset['lon']
                self.raw_lon = self.lon
            else:
                # for some reason RTOFS from NCEP has longitude values shifted by 360
                self.raw_lon = sample_dataset['lon'].values
                self.lon = self.raw_lon - 180 - numpy.min(self.raw_lon)

            lat_pixel_size = numpy.mean(numpy.diff(sample_dataset['lat']))
            lon_pixel_size = numpy.mean(numpy.diff(sample_dataset['lon']))

            self.global_north = numpy.max(self.lat)
            self.global_west = numpy.min(self.lon)

            self.global_grid_transform = rasterio.transform.from_origin(
                self.global_west, self.global_north, lon_pixel_size, lat_pixel_size
            )

            (
                self.study_area_west,
                self.study_area_south,
                self.study_area_east,
                self.study_area_north,
            ) = geometry.shape(self.study_area_geojson).bounds

            self.study_area_transform = rasterio.transform.from_origin(
                self.study_area_west, self.study_area_north, lon_pixel_size, lat_pixel_size
            )
        else:
            raise PyOFS.NoDataError(f'No RTOFS datasets found for {self.model_time}.')

    def data(self, variable: str, time: datetime, crop: bool = True) -> xarray.DataArray:
        """
        Get data of specified variable at specified hour.
        :param variable: name of variable to retrieve
        :param time: time from which to retrieve data
        :param crop: whether to crop to study area extent
        :return: array of data
        """

        if time >= self.model_time:
            direction = 'forecast'
        else:
            direction = 'nowcast'

        if self.time_interval == 'daily':
            time = time.replace(hour=0, minute=0, second=0, microsecond=0)

        if direction in DATASET_STRUCTURE[self.source]:
            if len(self.datasets[direction]) > 0:
                if variable in DATA_VARIABLES:
                    datasets = DATA_VARIABLES[variable][self.source]
                    dataset_name, variable_name = next(iter(datasets.items()))

                    with self.dataset_locks[direction][dataset_name]:
                        data_variable = self.datasets[direction][dataset_name][
                            DATA_VARIABLES[variable][self.source][dataset_name]
                        ]
                        print(data_variable)
                        # TODO study areas that cross over longitude +74.16 may have problems here
                        if crop:
                            selection = data_variable.sel(
                                lon=slice(
                                    self.study_area_west + 360, self.study_area_east + 360
                                ),
                                lat=slice(self.study_area_south, self.study_area_north),
                            )
                        else:
                            western_selection = data_variable.sel(
                                lon=slice(180, numpy.max(self.raw_lon)),
                                lat=slice(numpy.min(self.lat), numpy.max(self.lat)),
                            )
                            eastern_selection = data_variable.sel(
                                lon=slice(numpy.min(self.raw_lon), 180),
                                lat=slice(numpy.min(self.lat), numpy.max(self.lat)),
                            )
                            selection = numpy.concatenate(
                                (western_selection, eastern_selection), axis=1
                            )

                        # to resample the 3 hr for forcast and 1hr for nowcast nc file to a daily
                        selections=selection.resample(time='D').mean()
                        selections = selections.sel(time=time, method='nearest')

                        # correction for the
                        if variable == 'ssh':
                            selections=selections + 0.25

                        selections = numpy.flip(selections.squeeze(), axis=0)

                        if selections.size > 0:
                            return selections
                        else:
                            raise PyOFS.NoDataError(
                                f'no RTOFS data for {time} within the cropped area ({self.study_area_west:.2f}, {self.study_area_south:.2f}), ({self.study_area_east:.2f}, {self.study_area_north:.2f})')
                else:
                    raise ValueError(
                        f'Variable must be one of {list(DATA_VARIABLES)}.'
                    )
            else:
                LOGGER.warning(
                    f'{direction} does not exist in RTOFS for {self.model_time:%Y%m%d}.'
                )
        else:
            raise ValueError(
                f'Direction must be one of {list(DATASET_STRUCTURE[self.source].keys())}.'
            )

    def write_rasters(
        self,
        output_dir: PathLike,
        variables: list,
        time: datetime,
        filename_prefix: str = None,
        filename_suffix: str = None,
        fill_value=LEAFLET_NODATA_VALUE,
        driver: str = 'GTiff',
        crop: bool = True,
    ):
        """
        Write averaged raster data of given variables to given output directory.
        :param output_dir: path to directory
        :param variables: variable names to use
        :param time: time from which to retrieve data
        :param filename_prefix: prefix for filenames
        :param filename_suffix: suffix for filenames
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param crop: whether to crop to study area extent
        """

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if variables is None:
            variables = DATA_VARIABLES[self.source]

        if filename_prefix is None:
            filename_prefix = 'rtofs'
        filename_suffix = f'_{filename_suffix}' if filename_suffix is not None else ''

        if self.time_interval == 'daily':
            time = time.replace(hour=0, minute=0, second=0, microsecond=0)

        time_delta = int((time - self.model_time) / timedelta(days=1))
        direction = 'forecast' if time_delta >= 0 else 'nowcast'
        time_delta_string = f'{direction[0]}{abs(time_delta) + 1 if direction == "forecast" else abs(time_delta):03}'

        variable_means = {}
        for variable in variables:
            if variable not in ['dir', 'mag']:
                try:
                    variable_means[variable] = self.data(variable, time, crop)
                except KeyError:
                    LOGGER.warning(f'variable "{variable}" not found in RTOFS dataset')
                except Exception as error:
                    LOGGER.warning(error)

        variable_means = {
            variable: variable_mean.values
            for variable, variable_mean in variable_means.items()
            if variable_mean is not None
        }

        if 'dir' in variables or 'mag' in variables:
            u_name = 'ssu'
            v_name = 'ssv'

            if u_name not in variable_means:
                u_data = self.data(u_name, time, crop)
                u_data = u_data.values if u_data is not None else None
            else:
                u_data = variable_means[u_name]

            if v_name not in variable_means:
                v_data = self.data(v_name, time, crop)
                v_data = v_data.values if v_data is not None else None
            else:
                v_data = variable_means[v_name]

            if u_data is not None and v_data is not None:
                variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (
                    180 / numpy.pi
                )
                variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

        # write interpolated grids to raster files
        for variable, variable_mean in variable_means.items():
            if variable_mean is not None and variable_mean.size > 0:
                if crop:
                    transform = self.study_area_transform
                else:
                    transform = self.global_grid_transform

                if fill_value is not None:
                    variable_mean[numpy.isnan(variable_mean)] = fill_value

                gdal_args = {
                    'transform': transform,
                    'height': variable_mean.shape[0],
                    'width': variable_mean.shape[1],
                    'count': 1,
                    'dtype': rasterio.float32,
                    'crs': CRS.from_dict(OUTPUT_CRS),
                    'nodata': numpy.array([fill_value]).astype(variable_mean.dtype).item(),
                }

                if driver == 'AAIGrid':
                    file_extension = 'asc'
                    gdal_args.update({'FORCE_CELLSIZE': 'YES'})
                elif driver == 'GPKG':
                    file_extension = 'gpkg'
                else:
                    file_extension = 'tiff'
                    gdal_args.update(TIFF_CREATION_OPTIONS)

                output_filename = f'{filename_prefix}_{variable}_{self.model_time:%Y%m%d}_{time_delta_string}{filename_suffix}.{file_extension}'
                output_filename = output_dir / output_filename

                LOGGER.info(f'Writing {output_filename}')
                with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                    output_raster.write(variable_mean, 1)
                    if driver == 'GTiff':
                        output_raster.build_overviews(
                            PyOFS.overview_levels(variable_mean.shape), Resampling['average']
                        )
                        output_raster.update_tags(ns='rio_overview', resampling='average')

    def write_raster(
        self,
        output_filename: PathLike,
        variable: str,
        time: datetime,
        fill_value=LEAFLET_NODATA_VALUE,
        driver: str = 'GTiff',
        crop: bool = True,
    ):
        """
        Writes interpolated raster of given variable to output path.
        :param output_filename: path of raster file to create
        :param variable: name of variable
        :param time: time from which to retrieve data
        :param fill_value: desired fill value of output
        :param driver: strings of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param crop: whether to crop to study area extent
        """

        if not isinstance(output_filename, Path):
            output_filename = Path(output_filename)

        output_data = self.data(variable, time, crop).values

        if output_data is not None:
            if crop:
                transform = self.study_area_transform
            else:
                transform = self.global_grid_transform

            gdal_args = {
                'transform': transform,
                'height': output_data.shape[0],
                'width': output_data.shape[1],
                'count': 1,
                'dtype': rasterio.float32,
                'crs': CRS.from_dict(OUTPUT_CRS),
                'nodata': numpy.array([fill_value]).astype(output_data.dtype).item(),
            }

            if driver == 'AAIGrid':
                file_extension = 'asc'
                gdal_args.update({'FORCE_CELLSIZE': 'YES'})
            elif driver == 'GPKG':
                file_extension = 'gpkg'
            else:
                file_extension = 'tiff'
                gdal_args.update(TIFF_CREATION_OPTIONS)

            output_filename = f'{output_filename.stem}.{file_extension}'

            LOGGER.info(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(output_data, 1)
                if driver == 'GTiff':
                    output_raster.build_overviews(
                        PyOFS.overview_levels(output_data.shape), Resampling['average']
                    )
                    output_raster.update_tags(ns='rio_overview', resampling='average')

    def to_xarray(
        self, variables: Collection[str] = None, mean: bool = True
    ) -> xarray.Dataset:
        """
        Converts to xarray Dataset.
        :param variables: variables to use
        :param mean: whether to average all time indices
        :return: xarray observation of given variables
        """

        if variables is None:
            variables = ('sst', 'sss', 'ssu', 'ssv', 'ssh')

        output_dataset = xarray.Dataset()

        if self.time_interval == 'daily':
            times = [
                self.model_time + timedelta(days=day_delta)
                for day_delta in TIME_DELTAS['daily']
            ]
        else:
            times = None

        coordinates = None
        variables_data = {}

        for variable in variables:
            variable_data = [self.data(variable=variable, time=time) for time in times]

            if mean:
                coordinates = OrderedDict(
                    {
                        'lat': variable_data[0]['lat'].values,
                        'lon': variable_data[0]['lon'].values,
                    }
                )
            else:
                coordinates = OrderedDict(
                    {
                        'time': times,
                        'lat': variable_data[0]['lat'].values,
                        'lon': variable_data[0]['lon'].values,
                    }
                )

            variable_data = numpy.stack(variable_data, axis=0)
            variable_data = numpy.squeeze(variable_data)

            if mean:
                variable_data = numpy.nanmean(variable_data, axis=0)

            variables_data[variable] = variable_data

        for variable, variable_data in variables_data.items():
            output_dataset.update(
                {
                    variable: xarray.DataArray(
                        variable_data, coords=coordinates, dims=tuple(coordinates.keys())
                    )
                },
            )

        return output_dataset

    def to_netcdf(
        self, output_file: str, variables: Collection[str] = None, mean: bool = True
    ):
        """
        Writes to NetCDF file.
        :param output_file: output file to write
        :param mean: whether to average all time indices
        :param variables: variables to use
        """

        self.to_xarray(variables, mean).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.model_time.__repr__()]
        optional_params = [self.source, self.time_interval, self.study_area_polygon_filename]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


def extract_rtofs_tar(tar_filename: PathLike, output_directory: PathLike, overwrite: bool = False):
    if not isinstance(tar_filename, Path):
        tar_filename = Path(tar_filename)
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)
    temporary_directory = output_directory / 'extracted'
    output_filename = output_directory / f'rtofs_glo_2ds_forecast_daily_prog.nc'

    if not output_filename.exists() or overwrite:
        with tarfile.open(tar_filename) as tar_file:
            try:

                tar_file.extractall(temporary_directory)

            except Exception as error:
                LOGGER.warning(f'{error.__class__.__name__} - {error}')
            extracted_directory = temporary_directory / tar_filename.stem
            for filename in extracted_directory.iterdir():
                new_filename = temporary_directory / filename.name
                if not new_filename.exists():
                    filename.rename(temporary_directory / filename.name)
            shutil.rmtree(extracted_directory)

        datasets_6hr = {

            int(filename.stem.split('_')[3][1:]): xarray.open_dataset(filename)
            for filename in temporary_directory.iterdir()

        }

        first_dataset = list(datasets_6hr.values())[0]

        # date_string, day_percentage = str(first_dataset['Date'].values[0]).split('.')
        # start_time = datetime.strptime(date_string, '%Y%m%d') + (float(day_percentage) / 100) * timedelta(days=1)

        coordinates = OrderedDict(
            {
                'time': numpy.array([dataset['MT'].values[0] for dataset in datasets_6hr.values()]),
                'lat': first_dataset['Latitude'][:, 0].values,
                'lon': first_dataset['Longitude'][0, :].values,
            }
        )

        data_arrays = {}
        data_variable_names = {'temperature': 'sst', 'u': 'u_velocity', 'v': 'v_velocity', 'salinity': 'sss', 'ssh': 'ssh'}
        for input_name, output_name in data_variable_names.items():
            if input_name in first_dataset:
                variable_data = numpy.stack([dataset[input_name][0, 0, :, :] for dataset in datasets_6hr.values()], 0)

                data_arrays[output_name] = xarray.DataArray(variable_data, coords=coordinates, dims=coordinates.keys(),
                                                            name=output_name)

        for dataset in datasets_6hr.values():
            dataset.close()

        output_dataset = xarray.Dataset()
        for variable_name, data_array in data_arrays.items():
            output_dataset.update({variable_name: data_array})

        output_dataset=output_dataset.resample(time='D').mean()
        output_dataset.to_netcdf(output_filename)

        shutil.rmtree(temporary_directory)


if __name__ == '__main__':
    input_dir = DATA_DIRECTORY / 'input'/'rtofs'

    tar_filenames = [tar_filename for tar_filename in input_dir.iterdir() if tar_filename.suffix == '.tar']
    for tar_filename in tar_filenames:
        try:
            extract_rtofs_tar(tar_filename, input_dir / f'rtofs_global{tar_filename.stem.split(".")[1]}')
        except Exception as error:
            LOGGER.error(f'{error.__class__.__name__} - {error}')

    print('done')
