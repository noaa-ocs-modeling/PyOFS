# coding=utf-8
"""
High Frequency Radar stations measuring water surface speed on the West Coast.

Created on Jun 13, 2018

@author: zachary.burnett
"""

import datetime
import os

import fiona
import fiona.crs
import numpy
import rasterio
import scipy.interpolate
import xarray

from dataset import CRS_EPSG, Logger
from dataset import _utilities
from main import DATA_DIR

DATA_VARIABLES = {'ssu': 'u', 'ssv': 'v', 'dopx': 'DOPx', 'dopy': 'DOPy'}

RASTERIO_CRS = rasterio.crs.CRS({'init': f'epsg:{CRS_EPSG}'})
FIONA_CRS = fiona.crs.from_epsg(CRS_EPSG)

NRT_DELAY = datetime.timedelta(hours=1)

SOURCE_URLS = {
    'NDBC': 'https://dods.ndbc.noaa.gov/thredds/dodsC',
    'UCSD': 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC'
}


class HFRRange:
    """
    High Frequency Radar (HFR) NetCDF dataset of surface current velocities.
    """

    grid_transform = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, resolution: int = 6,
                 source: str = None, logger: Logger = None):
        """
        Creates new dataset object from source.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param resolution: Desired dataset resolution in kilometers.
        :param source: Either UCSD (University of California San Diego) or NDBC (National Data Buoy Center). NDBC has a larger extent but only for the past 4 days.
        :param logger: logbook logger
        :raises NoDataError: if dataset does not exist.
        """

        self.logger = logger

        self.start_datetime = start_datetime
        if end_datetime > datetime.datetime.utcnow():
            # HFR near real time delay is 1 hour behind UTC
            self.end_datetime = datetime.datetime.utcnow() - NRT_DELAY
        else:
            self.end_datetime = end_datetime

        self.resolution = resolution

        # get NDBC dataset if input time is within 4 days, otherwise get UCSD dataset
        if source is not None:
            self.source = source
        elif (datetime.datetime.now() - self.start_datetime) < datetime.timedelta(days=4):
            self.source = 'NDBC'
        else:
            self.source = 'UCSD'

        # get URL
        if self.source == 'NDBC':
            self.url = f'{SOURCE_URLS["NDBC"]}/hfradar_uswc_{self.resolution}km'
        elif self.source == 'UCSD':
            self.url = f'{SOURCE_URLS["UCSD"]}/{self.resolution}km/hourly/RTV/' + \
                       f'HFRADAR_US_West_Coast_{self.resolution}km_Resolution_Hourly_RTV_best.ncd'

        try:
            self.netcdf_dataset = xarray.open_dataset(self.url)
        except OSError:
            raise _utilities.NoDataError(f'No HFR dataset found at {self.url}')

        raw_times = self.netcdf_dataset['time']

        self.netcdf_dataset['time'] = xarray.DataArray(numpy.array(raw_times.values, dtype='datetime64[h]'),
                                                       coords=raw_times.coords, dims=raw_times.dims,
                                                       attrs=raw_times.attrs)

        self.netcdf_dataset = self.netcdf_dataset.sel(time=slice(self.start_datetime, self.end_datetime))

        if self.logger is not None:
            self.logger.info(f'Collecting HFR velocity from {self.source} between ' +
                             f'{str(self.netcdf_dataset["time"].min().values)[:19]}' +
                             f' and {str(self.netcdf_dataset["time"].max().values)[:19]}...')

        if HFRRange.grid_transform is None:
            lon = self.netcdf_dataset['lon'].values
            lat = self.netcdf_dataset['lat'].values

            # define image properties
            west = numpy.min(lon)
            north = numpy.max(lat)

            self.mean_x_size = numpy.mean(numpy.diff(lon))
            self.mean_y_size = numpy.mean(numpy.diff(lat))

            # get rasterio geotransform of HFR dataset (flipped latitude)
            self.grid_transform = rasterio.transform.from_origin(west, north, self.mean_x_size, self.mean_y_size)

    def data(self, variable: str, time: datetime.datetime, dop_threshold: float = None) -> numpy.ndarray:
        """
        Get data for the specified variable at a single time.

        :param variable: Variable name.
        :param time: Time to retrieve.
        :param dop_threshold: Threshold for Dilution of Precision (DOP) above which data should be discarded.
        :return: Array of data.
        """

        output_data = self.netcdf_dataset[DATA_VARIABLES[variable]].sel(time).values

        if dop_threshold is not None:
            dop_mask = ((self.netcdf_dataset['DOPx'].sel(time=time) <= dop_threshold) | (
                    self.netcdf_dataset['DOPy'].sel(time=time) <= dop_threshold))
            output_data[~dop_mask] = numpy.nan

        return output_data

    def data_average(self, variable: str, start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None, dop_threshold: float = None) -> numpy.ndarray:
        """
        Get data for the specified variable at a single time.

        :param variable: Variable name.
        :param start_datetime: Start of time interval.
        :param end_datetime: End of time interval.
        :param dop_threshold: Threshold for Dilution of Precision (DOP) above which data should be discarded.
        :return: Array of data.
        """

        if start_datetime is None:
            start_datetime = self.start_datetime

        if end_datetime is None:
            end_datetime = self.end_datetime

        output_data = self.netcdf_dataset[DATA_VARIABLES[variable]].sel(time=slice(start_datetime, end_datetime)).values

        if dop_threshold is not None:
            dop_mask = ((self.netcdf_dataset['DOPx'].sel(time=slice(start_datetime, end_datetime)) <= dop_threshold) | (
                    self.netcdf_dataset['DOPy'].sel(time=slice(start_datetime, end_datetime)) <= dop_threshold)).values
            output_data[~dop_mask] = numpy.nan

        output_data = numpy.nanmean(output_data, axis=0)

        return output_data

    def bounds(self) -> tuple:
        """
        Get coordinate bounds of dataset.

        :return: Tuple of bounds (west, north, east, south)
        """

        return (self.netcdf_dataset.geospatial_lon_min, self.netcdf_dataset.geospatial_lat_max,
                self.netcdf_dataset.geospatial_lon_max, self.netcdf_dataset.geospatial_lat_min)

    def cell_size(self) -> tuple:
        """
        Get cell sizes of dataset.

        :return: Tuple of cell sizes (x, y)
        """

        return abs(self.mean_x_size), abs(self.mean_y_size)

    def write_sites(self, output_filename: str, layer_name: str):
        """
        Writes HFR radar facility locations to specified file and layer.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        """

        layer_records = []

        for site_index in range(self.netcdf_dataset['nSites']):
            site_code = self.netcdf_dataset['site_code'][site_index].tobytes().decode().strip('\x00').strip()
            site_network_code = self.netcdf_dataset['site_netCode'][site_index].tobytes().decode().strip('\x00').strip()
            lon = float(self.netcdf_dataset['site_lon'][site_index])
            lat = float(self.netcdf_dataset['site_lat'][site_index])

            record = {
                'id': site_index + 1, 'geometry': {'type': 'Point', 'coordinates': (lon, lat)}, 'properties': {
                    'code': site_code, 'net_code': site_network_code, 'lon': float(lon), 'lat': float(lat)
                }
            }

            layer_records.append(record)

        schema = {
            'geometry': 'Point', 'properties': {
                'code': 'str', 'net_code': 'str', 'lon': 'float', 'lat': 'float'
            }
        }

        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
            layer.writerecords(layer_records)

    def write_vectors(self, output_filename: str, variables: list = None, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None, dop_threshold: float = 0.5):
        """
        Write HFR data to a layer of the provided output file for every hour in the given time interval.

        :param output_filename: Path to output file.
        :param variables: List of variable names to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param dop_threshold: Threshold for Dilution of Precision (DOP) above which data should be discarded.
        """

        if variables is None:
            variables = DATA_VARIABLES

        if start_datetime is None:
            start_datetime = self.start_datetime

        if end_datetime is None:
            end_datetime = self.end_datetime

        time_interval_selection = self.netcdf_dataset.sel(time=slice(start_datetime, end_datetime))

        if dop_threshold is not None:
            dop_mask = ((self.netcdf_dataset['DOPx'].sel(time=slice(start_datetime, end_datetime)) <= dop_threshold) | (
                    self.netcdf_dataset['DOPy'].sel(time=slice(start_datetime, end_datetime)) <= dop_threshold)).values
            time_interval_selection[~dop_mask] = numpy.nan

        # create dict to store features
        layers = {}

        # create layer using OGR, then add features using QGIS
        for hfr_time in time_interval_selection['time']:
            hfr_datetime = datetime.datetime.utcfromtimestamp(
                (hfr_time.values - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's'))
            layer_name = f'{hfr_datetime.strftime("%Y%m%dT%H%M%S")}'

            hfr_data = time_interval_selection.sel(time=hfr_time)

            # create features
            layer_records = []

            feature_index = 1

            for col in range(len(self.netcdf_dataset['lon'])):
                for row in range(len(self.netcdf_dataset['lat'])):
                    data = [float(hfr_data[variable_name][row, col].values) for variable, variable_name in
                            variables.items()]

                    # stop if record has masked values
                    if not (numpy.isnan(data)).all():
                        lon = self.netcdf_dataset['lon'][col]
                        lat = self.netcdf_dataset['lat'][row]

                        record = {
                            'id': feature_index, 'geometry': {'type': 'Point', 'coordinates': (lon, lat)},
                            'properties': {'lon': float(lon), 'lat': float(lat)}
                        }

                        record['properties'].update(dict(zip(list(variables.keys()), data)))

                        layer_records.append(record)
                        feature_index += 1

            layers[layer_name] = layer_records

        # write queued features to their respective layers
        schema = {
            'geometry': 'Point', 'properties': {
                'u': 'float', 'v': 'float', 'lat': 'float', 'lon': 'float', 'dop_lat': 'float',
                'dop_lon': 'float'
            }
        }

        for layer_name, layer_records in layers.items():
            with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
                layer.writerecords(layer_records)

    def write_vector(self, output_filename: str, layer_name: str = 'ssuv', variables: list = None,
                     start_datetime: datetime.datetime = None, end_datetime: datetime.datetime = None,
                     dop_threshold: float = 0.5):
        """
        Write average of HFR data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param variables: List of variable names to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param dop_threshold: Threshold for Dilution of Precision (DOP) above which data should be discarded.
        """

        if variables is None:
            variables = DATA_VARIABLES

        variable_means = {variable: self.data_average(variable, start_datetime, end_datetime, dop_threshold) for
                          variable in variables}

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {
                'lon': 'float', 'lat': 'float'
            }
        }

        schema['properties'].update({variable: 'float' for variable in variables})

        # create features
        layer_records = []

        feature_index = 1

        for col in range(len(self.netcdf_dataset['lon'])):
            for row in range(len(self.netcdf_dataset['lat'])):
                data = [float(variable_means[variable][row, col]) for variable in variables]

                # stop if record has masked values
                if not (numpy.isnan(data)).all():
                    lon = self.netcdf_dataset['lon'][col]
                    lat = self.netcdf_dataset['lat'][row]

                    record = {
                        'id': feature_index, 'geometry': {'type': 'Point', 'coordinates': (lon, lat)},
                        'properties': {'lon': float(lon), 'lat': float(lat)}
                    }

                    record['properties'].update(dict(zip(variables, data)))

                    layer_records.append(record)
                    feature_index += 1

        # write queued features to layer
        if self.logger is not None:
            self.logger.info(f'Writing {output_filename}')
        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_CRS) as layer:
            layer.writerecords(layer_records)

    def write_rasters(self, output_dir: str, filename_prefix: str = 'hfr', filename_suffix: str = '',
                      variables: list = None, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None, fill_value: float = -9999, driver: str = 'GTiff',
                      dop_threshold: float = 0.5):
        """
        Write average of HFR data for all hours in the given time interval to rasters.

        :param output_dir: Path to output directory.
        :param filename_prefix: Prefix for output filenames.
        :param filename_suffix: Suffix for output filenames.
        :param variables: List of variable names to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param fill_value: Desired fill value of output.
        :param driver: String of valid GDAL driver (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param dop_threshold: Threshold for dilution of precision above which data is not useable.
        """

        if variables is None:
            variables = DATA_VARIABLES

        if filename_suffix is not '':
            filename_suffix = f'_{filename_suffix}'

        variable_means = {variable: self.data_average(variable, start_datetime, end_datetime, dop_threshold) for
                          variable in variables if variable not in ['dir', 'mag']}

        if 'dir' in variables or 'mag' in variables:
            if 'ssu' in variables:
                u_data = variable_means['ssu']
            else:
                u_data = self.data_average('ssu', start_datetime, end_datetime, dop_threshold)

            if 'ssv' in variables:
                v_data = variable_means['ssv']
            else:
                v_data = self.data_average('ssv', start_datetime, end_datetime, dop_threshold)

            # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
            variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
            variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

        for variable, variable_data in variable_means.items():
            raster_data = variable_data.astype(rasterio.float32)

            gdal_args = {
                'height': raster_data.shape[0], 'width': raster_data.shape[1], 'count': 1,
                'dtype': raster_data.dtype, 'crs': RASTERIO_CRS, 'transform': self.grid_transform,
                'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
            }

            if driver == 'AAIGrid':
                file_extension = 'asc'

                # interpolate to regular grid in case of ASCII grid
                mean_cell_length = numpy.min(self.cell_size())
                west, north, east, south = self.bounds()

                input_lon, input_lat = numpy.meshgrid(self.netcdf_dataset['lon'], self.netcdf_dataset['lat'])
                output_lon = numpy.arange(west, east, mean_cell_length)[None, :]
                output_lat = numpy.arange(south, north, mean_cell_length)[:, None]

                raster_data = scipy.interpolate.griddata((input_lon.flatten(), input_lat.flatten()),
                                                         raster_data.flatten(), (output_lon, output_lat),
                                                         method='nearest', fill_value=fill_value).astype(
                    raster_data.dtype)

                gdal_args.update({
                    'height': raster_data.shape[0], 'width': raster_data.shape[1], 'FORCE_CELLSIZE': 'YES',
                    'transform': rasterio.transform.from_origin(numpy.min(output_lon), numpy.max(output_lat),
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

            output_filename = os.path.join(output_dir,
                                           f'{filename_prefix}_{variable}{filename_suffix}.{file_extension}')

            if self.logger is not None:
                self.logger.info(f'Writing {output_filename}')
            with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                output_raster.write(numpy.flipud(raster_data), 1)

    def to_xarray(self, variables: list = None, mean: bool = True) -> xarray.Dataset:
        """
        Converts to xarray Dataset.

        :param variables: List of variables to use.
        :param mean: Whether to average all time indices.
        :return: xarray Dataset of given variables.
        """

        data_arrays = {}

        if variables is None:
            variables = DATA_VARIABLES

        if mean:
            for variable in variables:
                data_arrays[variable] = xarray.DataArray(numpy.mean(self.data_average(variable), axis=0),
                                                         coords={'lat': self.netcdf_dataset['lat'],
                                                                 'lon': self.netcdf_dataset['lon']},
                                                         dims=('lat', 'lon'))
        else:
            for variable in variables:
                data_arrays[variable] = xarray.DataArray(self.data_average(variable),
                                                         coords={'lat': self.netcdf_dataset['lat'],
                                                                 'lon': self.netcdf_dataset['lon']},
                                                         dims=('time', 'lat', 'lon'))

        return xarray.Dataset(data_vars=data_arrays)

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
        optional_params = [self.resolution]

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

    start_datetime = datetime.datetime(2018, 11, 15)
    end_datetime = start_datetime + datetime.timedelta(days=1)

    hfr_range = HFRRange(start_datetime, end_datetime)
    hfr_range.write_rasters(output_dir)

    print('done')
