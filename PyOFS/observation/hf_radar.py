# coding=utf-8
"""
High Frequency Radar stations measuring water surface speed on the West Coast.

Created on Jun 13, 2018

@author: zachary.burnett
"""

import datetime
import logging
import os
from typing import Collection

import fiona
import fiona.crs
import numpy
import rasterio
import scipy.interpolate
import xarray

from PyOFS import CRS_EPSG, utilities, LEAFLET_NODATA_VALUE

DATA_VARIABLES = {'ssu': 'u', 'ssv': 'v', 'dopx': 'DOPx', 'dopy': 'DOPy'}

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})
FIONA_CRS = fiona.crs.from_epsg(CRS_EPSG)

NRT_DELAY = datetime.timedelta(hours=1)

# either UCSD (University of California San Diego) or NDBC (National Data Buoy Center); NDBC has larger extent but only for the past 4 days
SOURCE_URLS = {
    'NDBC': 'https://dods.ndbc.noaa.gov/thredds/dodsC',
    'UCSD': 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC'
}


class HFRadarRange:
    """
    High Frequency (HF) Radar NetCDF observation of surface current velocities.
    """

    grid_transform = None

    def __init__(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None, resolution: int = 6):
        """
        Creates new observation object from source.

        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param resolution: desired observation resolution in kilometers
        :raises NoDataError: if observation does not exist.
        """

        if start_time is None:
            start_time = datetime.datetime.now()

        self.start_time = start_time

        if end_time is None:
            end_time = self.start_time + datetime.timedelta(days=1)

        if end_time > datetime.datetime.utcnow():
            # HFR near real time delay is 1 hour behind UTC
            self.end_time = datetime.datetime.utcnow() - NRT_DELAY
        else:
            self.end_time = end_time

        self.resolution = resolution

        # NDBC only keeps observations within the past 4 days
        for source, source_url in SOURCE_URLS.items():
            # get URL
            if source == 'NDBC':
                url = f'{source_url}/hfradar_uswc_{self.resolution}km'
            elif source == 'UCSD':
                url = f'{source_url}/{self.resolution}km/hourly/RTV/HFRADAR_US_West_Coast_{self.resolution}km_Resolution_Hourly_RTV_best.ncd'
            else:
                url = source

            try:
                self.dataset = xarray.open_dataset(url)
                self.url = url
                break
            except OSError:
                logging.exception(f'error reading dataset from {url}')
        else:
            raise utilities.NoDataError(f'No HFR observations found between {self.start_time} and {self.end_time}')

        raw_times = self.dataset['time']

        self.dataset['time'] = xarray.DataArray(numpy.array(raw_times.values, dtype='datetime64[h]'),
                                                coords=raw_times.coords,
                                                dims=raw_times.dims, attrs=raw_times.attrs)

        self.dataset = self.dataset.sel(time=slice(self.start_time, self.end_time))

        logging.info(f'Collecting HFR velocity between {str(self.dataset["time"].min().values)[:19]} and ' + \
                     f'{str(self.dataset["time"].max().values)[:19]}...')

        if HFRadarRange.grid_transform is None:
            lon = self.dataset['lon'].values
            lat = self.dataset['lat'].values

            # define image properties
            west = numpy.min(lon)
            north = numpy.max(lat)

            self.mean_x_size = numpy.mean(numpy.diff(lon))
            self.mean_y_size = numpy.mean(numpy.diff(lat))

            # get rasterio geotransform of HFR observation (flipped latitude)
            self.grid_transform = rasterio.transform.from_origin(west, north, self.mean_x_size, self.mean_y_size)

    def data(self, variable: str, time: datetime.datetime, dop_threshold: float = None) -> numpy.array:
        """
        Get data for the specified variable at a single time.

        :param variable: variable name
        :param time: time to retrieve
        :param dop_threshold: threshold for Dilution of Precision (DOP) above which data should be discarded
        :return: array of data.
        """

        output_data = self.dataset[DATA_VARIABLES[variable]].sel(time)

        if dop_threshold is not None:
            output_data[~self.dop_mask(dop_threshold)] = numpy.nan

        return output_data.values

    def data_average(self, variable: str, start_time: datetime.datetime = None, end_time: datetime.datetime = None,
                     dop_threshold: float = None, include_incomplete: bool = False) -> numpy.array:
        """
        Get data for the specified variable at a single time.

        :param variable: variable name
        :param start_time: start of time interval
        :param end_time: end of time interval
        :param dop_threshold: threshold for Dilution of Precision (DOP) above which data should be discarded
        :param include_incomplete: whether to keep incomplete time series
        :return: array of data
        """

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        data_array = self.dataset[DATA_VARIABLES[variable]].sel(time=slice(start_time, end_time))

        if dop_threshold is not None:
            data_array.values[~self.dop_mask(dop_threshold)] = numpy.nan

        output_data = numpy.nanmean(data_array, axis=0)

        if not include_incomplete:
            output_data[numpy.isnan(data_array).any(axis=0)] = numpy.nan

        return output_data

    def bounds(self) -> tuple:
        """
        Get coordinate bounds of observation.

        :return: tuple of bounds (west, north, east, south)
        """

        return (
            self.dataset.geospatial_lon_min, self.dataset.geospatial_lat_max, self.dataset.geospatial_lon_max,
            self.dataset.geospatial_lat_min)

    def cell_size(self) -> tuple:
        """
        Get cell sizes of observation.

        :return: tuple of cell sizes (x, y)
        """

        return abs(self.mean_x_size), abs(self.mean_y_size)

    def write_sites(self, output_filename: str, layer_name: str):
        """
        Writes HFR radar facility locations to specified file and layer.

        :param output_filename: path to output file
        :param layer_name: name of layer to write
        """

        layer_records = []

        for site_index in range(self.dataset['nSites']):
            site_code = self.dataset['site_code'][site_index].tobytes().decode().strip('\x00').strip()
            site_network_code = self.dataset['site_netCode'][site_index].tobytes().decode().strip('\x00').strip()
            lon = float(self.dataset['site_lon'][site_index])
            lat = float(self.dataset['site_lat'][site_index])

            record = {
                'id': site_index + 1,
                'geometry': {
                    'type': 'Point',
                    'coordinates': (lon, lat)},
                'properties': {
                    'code': site_code,
                    'net_code': site_network_code,
                    'lon': float(lon),
                    'lat': float(lat)
                }
            }

            layer_records.append(record)

        schema = {
            'geometry': 'Point',
            'properties': {
                'code': 'str',
                'net_code': 'str',
                'lon': 'float',
                'lat': 'float'
            }
        }

        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
            layer.writerecords(layer_records)

    def write_vectors(self, output_filename: str, variables: Collection[str] = None,
                      start_time: datetime.datetime = None,
                      end_time: datetime.datetime = None, dop_threshold: float = 0.5):
        """
        Write HFR data to a layer of the provided output file for every hour in the given time interval.

        :param output_filename: path to output file
        :param variables: variable names to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param dop_threshold: threshold for Dilution of Precision (DOP) above which data should be discarded
        """

        if variables is None:
            variables = DATA_VARIABLES

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        time_interval_selection = self.dataset.sel(time=slice(start_time, end_time))

        if dop_threshold is not None:
            dop_mask = ((self.dataset['DOPx'].sel(time=slice(start_time, end_time)) <= dop_threshold) & (
                    self.dataset['DOPy'].sel(time=slice(start_time, end_time)) <= dop_threshold)).values
            time_interval_selection[~dop_mask] = numpy.nan

        # create dict to store features
        layers = {}

        # create layer using OGR, then add features using QGIS
        for hfr_time in time_interval_selection['time']:
            hfr_time = datetime.datetime.utcfromtimestamp(
                (hfr_time.values - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's'))
            layer_name = f'{hfr_time:%Y%m%dT%H%M%S}'

            hfr_data = time_interval_selection.sel(time=hfr_time)

            # create features
            layer_records = []

            feature_index = 1

            for col in range(len(self.dataset['lon'])):
                for row in range(len(self.dataset['lat'])):
                    data = [float(hfr_data[variable_name][row, col].values) for variable, variable_name in
                            variables.items()]

                    # stop if record has masked values
                    if not (numpy.isnan(data)).all():
                        lon = self.dataset['lon'][col]
                        lat = self.dataset['lat'][row]

                        record = {
                            'id': feature_index,
                            'geometry': {
                                'type': 'Point',
                                'coordinates': (lon, lat)
                            },
                            'properties': {
                                'lon': float(lon),
                                'lat': float(lat)
                            }
                        }

                        record['properties'].update(dict(zip(list(variables.keys()), data)))

                        layer_records.append(record)
                        feature_index += 1

            layers[layer_name] = layer_records

        # write queued features to their respective layers
        schema = {
            'geometry': 'Point', 'properties': {
                'u': 'float',
                'v': 'float',
                'lat': 'float',
                'lon': 'float',
                'dop_lat': 'float',
                'dop_lon': 'float'
            }
        }

        for layer_name, layer_records in layers.items():
            with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
                layer.writerecords(layer_records)

    def write_vector(self, output_filename: str, layer_name: str = 'ssuv', variables: Collection[str] = None,
                     start_time: datetime.datetime = None, end_time: datetime.datetime = None,
                     dop_threshold: float = 0.5):
        """
        Write average of HFR data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: path to output file
        :param layer_name: name of layer to write
        :param variables: variable names to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param dop_threshold: threshold for Dilution of Precision (DOP) above which data should be discarded
        """

        if variables is None:
            variables = DATA_VARIABLES

        variable_means = {variable: self.data_average(variable, start_time, end_time, dop_threshold) for
                          variable in variables}

        # define layer schema
        schema = {
            'geometry': 'Point',
            'properties': {
                'lon': 'float',
                'lat': 'float'
            }
        }

        schema['properties'].update({variable: 'float' for variable in variables})

        # create features
        layer_records = []

        feature_index = 1

        for col in range(len(self.dataset['lon'])):
            for row in range(len(self.dataset['lat'])):
                data = [float(variable_means[variable][row, col]) for variable in variables]

                # stop if record has masked values
                if not (numpy.isnan(data)).all():
                    lon = self.dataset['lon'][col]
                    lat = self.dataset['lat'][row]

                    record = {
                        'id': feature_index,
                        'geometry': {
                            'type': 'Point',
                            'coordinates': (lon, lat)
                        },
                        'properties': {
                            'lon': float(lon),
                            'lat': float(lat)
                        }
                    }

                    record['properties'].update(dict(zip(variables, data)))

                    layer_records.append(record)
                    feature_index += 1

        # write queued features to layer
        logging.info(f'Writing {output_filename}')
        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
            layer.writerecords(layer_records)

    def write_rasters(self, output_dir: str, filename_prefix: str = 'hfr', filename_suffix: str = '',
                      variables: Collection[str] = None,
                      start_time: datetime.datetime = None, end_time: datetime.datetime = None,
                      fill_value: float = LEAFLET_NODATA_VALUE,
                      driver: str = 'GTiff', dop_threshold: float = None):
        """
        Write average of HFR data for all hours in the given time interval to rasters.

        :param output_dir: path to output directory
        :param filename_prefix: prefix for output filenames
        :param filename_suffix: suffix for output filenames
        :param variables: variable names to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param fill_value: desired fill value of output
        :param driver: string of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid')
        :param dop_threshold: threshold for dilution of precision above which data is not useable
        """

        if variables is None:
            variables = DATA_VARIABLES

        if filename_suffix is not '':
            filename_suffix = f'_{filename_suffix}'

        variable_means = {variable: self.data_average(variable, start_time, end_time, dop_threshold) for
                          variable in variables if variable not in ['dir', 'mag']}

        if 'dir' in variables or 'mag' in variables:
            if 'ssu' in variables:
                u_data = variable_means['ssu']
            else:
                u_data = self.data_average('ssu', start_time, end_time, dop_threshold)

            if 'ssv' in variables:
                v_data = variable_means['ssv']
            else:
                v_data = self.data_average('ssv', start_time, end_time, dop_threshold)

            # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
            variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
            variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

        for variable, variable_data in variable_means.items():
            raster_data = variable_data.astype(rasterio.float32)

            gdal_args = {
                'height': raster_data.shape[0],
                'width': raster_data.shape[1],
                'count': 1,
                'dtype': raster_data.dtype,
                'crs': RASTERIO_CRS,
                'transform': self.grid_transform,
                'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
            }

            if driver == 'AAIGrid':
                file_extension = 'asc'

                # interpolate to regular grid in case of ASCII grid
                mean_cell_length = numpy.min(self.cell_size())
                west, north, east, south = self.bounds()

                input_lon, input_lat = numpy.meshgrid(self.dataset['lon'], self.dataset['lat'])
                output_lon = numpy.arange(west, east, mean_cell_length)[None, :]
                output_lat = numpy.arange(south, north, mean_cell_length)[:, None]

                raster_data = scipy.interpolate.griddata((input_lon.flatten(), input_lat.flatten()),
                                                         raster_data.flatten(),
                                                         (output_lon, output_lat), method='nearest',
                                                         fill_value=fill_value).astype(
                    raster_data.dtype)

                gdal_args.update({
                    'height': raster_data.shape[0],
                    'width': raster_data.shape[1],
                    'FORCE_CELLSIZE': 'YES',
                    'transform': rasterio.transform.from_origin(numpy.min(output_lon),
                                                                numpy.max(output_lat),
                                                                numpy.max(numpy.diff(output_lon)),
                                                                numpy.max(numpy.diff(output_lon)))
                })
            elif driver == 'GPKG':
                file_extension = 'gpkg'
            else:
                file_extension = 'tiff'
                gdal_args.update({
                    'TILED': 'YES'
                })

            if fill_value is not None:
                raster_data[numpy.isnan(raster_data)] = fill_value

            output_filename = os.path.join(output_dir,
                                           f'{filename_prefix}_{variable}{filename_suffix}.{file_extension}')

            logging.info(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(numpy.flipud(raster_data), 1)

    def dop_mask(self, threshold: float, start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        """
        Get mask of time series with a dilution of precision (DOP) below the given threshold.

        :param threshold: DOP threshold
        :param start_time: start time
        :param end_time: end time
        :return: boolean mask
        """

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        dop_x = self.dataset['DOPx'].sel(time=slice(start_time, end_time))
        dop_y = self.dataset['DOPy'].sel(time=slice(start_time, end_time))
        return ((dop_x <= threshold) & (dop_y <= threshold)).values

    def to_xarray(self, variables: Collection[str] = None, start_time: datetime.datetime = None,
                  end_time: datetime.datetime = None,
                  mean: bool = True, dop_threshold: float = 0.5) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: variables to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param mean: whether to average all time indices
        :param dop_threshold: threshold for Dilution of Precision (DOP) above which data should be discarded
        :return: xarray observation of given variables
        """

        output_dataset = xarray.Dataset()

        if variables is None:
            variables = DATA_VARIABLES

        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        if mean:
            for variable in variables:
                output_data = self.data_average(variable, start_time=start_time, end_time=end_time,
                                                dop_threshold=dop_threshold)

                output_dataset.update({variable: xarray.DataArray(output_data,
                                                                  coords={'lat': self.dataset['lat'],
                                                                          'lon': self.dataset['lon']},
                                                                  dims=('lat', 'lon'))})
        else:
            for variable in variables:
                output_data = self.dataset[DATA_VARIABLES[variable]].sel(time=slice(start_time, end_time))

                if dop_threshold is not None:
                    output_data.values[~self.dop_mask(dop_threshold)] = numpy.nan

                output_dataset.update({variable: output_data})

        return output_dataset

    def to_netcdf(self, output_file: str, variables: Collection[str] = None, start_time: datetime.datetime = None,
                  end_time: datetime.datetime = None, mean: bool = True):
        """
        Writes to NetCDF file.

        :param output_file: output file to write
        :param variables: variables to use
        :param start_time: beginning of time interval
        :param end_time: end of time interval
        :param mean: whether to average all time indices
        """

        self.to_xarray(variables, start_time, end_time, mean).to_netcdf(output_file)

    def __repr__(self):
        used_params = [self.start_time.__repr__(), self.end_time.__repr__()]
        optional_params = [self.resolution]

        for param in optional_params:
            if param is not None:
                if 'str' in str(type(param)):
                    param = f'"{param}"'
                else:
                    param = str(param)

                used_params.append(param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


def discard_incomplete_time_series(data: xarray.DataArray):
    assert 'time' in data.coords


if __name__ == '__main__':
    from matplotlib import pyplot
    from pandas.plotting import register_matplotlib_converters

    from PyOFS import DATA_DIRECTORY

    register_matplotlib_converters()

    output_dir = os.path.join(DATA_DIRECTORY, r'output\test')

    start_time = datetime.datetime(2019, 2, 6)
    end_time = start_time + datetime.timedelta(days=1)

    hfr_range = HFRadarRange(start_time, end_time, source='UCSD')

    cell = hfr_range.dataset.isel(lon=67, lat=270)

    _, axes = pyplot.subplots(nrows=2)

    axes[0].plot(cell['time'], cell['u'])
    axes[0].plot(cell['time'], cell['v'])
    axes[1].plot(cell['time'], cell['DOPx'])
    axes[1].plot(cell['time'], cell['DOPy'])

    for axis in axes:
        axis.legend()

    pyplot.show()

    # hfr_range.write_rasters(output_dir, filename_suffix='hfr_no_DOP')
    # hfr_range.write_rasters(output_dir, filename_suffix='hfr_0.5_DOP', dop_threshold=0.5)
    # hfr_range.write_rasters(output_dir, filename_suffix='hfr_0.1_DOP', dop_threshold=0.1)

    # hfr_range.write_vector(os.path.join(output_dir, 'hfr_no_DOP.gpkg'))
    # hfr_range.write_vector(os.path.join(output_dir, 'hfr_0.5_DOP.gpkg'), dop_threshold=0.5)
    # hfr_range.write_vector(os.path.join(output_dir, 'hfr_0.1_DOP.gpkg'), dop_threshold=0.1)

    print('done')
