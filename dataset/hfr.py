"""
High Frequency Radar stations measuring water surface speed on the West Coast.

Created on Jun 13, 2018

@author: zachary.burnett
"""

import concurrent.futures
import datetime
import os

import fiona
import fiona.crs
import netCDF4
import numpy
import rasterio
import scipy.interpolate
from qgis.core import QgsFeature, QgsGeometry, QgsPoint, QgsVectorLayer

from dataset import _utilities

MEASUREMENT_VARIABLES = ['u', 'v', 'DOPx', 'DOPy']

FIONA_WGS84 = fiona.crs.from_epsg(4326)
RASTERIO_WGS84 = rasterio.crs.CRS({"init": "epsg:4326"})


class HFR_Range:
    """
    High Frequency Radar (HFR) NetCDF dataset of surface current velocities.
    """

    grid_transform = None

    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, resolution: int = 6,
                 source: str = None):
        """
        Creates new dataset object from source.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param resolution: Desired dataset resolution in kilometers.
        :param source: Either UCSD (University of California San Diego) or NDBC (National Data Buoy Center). NDBC has a larger extent but only for the past 4 days.
        :raises NoDataError: if dataset does not exist.
        """

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.resolution = resolution

        # get NDBC dataset if input time is within 4 days, otherwise get UCSD dataset
        if source is not None:
            self.source = source
        elif (datetime.datetime.now() - self.start_datetime) < datetime.timedelta(days=4, seconds=57600):
            self.source = 'NDBC'
        else:
            self.source = 'UCSD'

        # print(f'Using {self.source} data...')

        # get URL
        if self.source == 'UCSD':
            self.url = f'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/{self.resolution}km/hourly/RTV/HFRADAR_US_West_Coast_{self.resolution}km_Resolution_Hourly_RTV_best.ncd'
        elif self.source == 'NDBC':
            # url = r"C:\Data\hfr\hfradar_uswc_6km.nc"
            self.url = f'https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{self.resolution}km'

        try:
            self.netcdf_dataset = netCDF4.Dataset(self.url)
        except OSError:
            raise _utilities.NoDataError(f'No HFR dataset found at {self.url}')

        time_var = self.netcdf_dataset['time']

        # get datetimes
        if self.source == 'UCSD':
            # parse datetime objects from "hours since 2011-10-01 00:00:00.000
            # UTC" using netCDF4 date parser
            start_hour, end_hour = netCDF4.date2num([self.start_datetime, self.end_datetime], time_var.units,
                                                    time_var.calendar)

            self.start_index = numpy.searchsorted(time_var[:], start_hour)
            self.end_index = numpy.searchsorted(time_var[:], end_hour)

            self.datetimes = netCDF4.num2date(time_var[self.start_index:self.end_index], units=time_var.units,
                                              calendar=time_var.calendar)
        elif self.source == 'NDBC':
            # decode strings from datetime character byte arrays
            datetime_strings = time_var[:].tobytes().decode().split('\x00' * 44)[:-1]

            self.start_index = numpy.searchsorted(datetime_strings, self.start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
            self.end_index = numpy.searchsorted(datetime_strings, self.end_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))

            datetime_strings_subset = datetime_strings[self.start_index:self.end_index]

            if len(datetime_strings_subset) > 0:
                # parse datetime objects from strings using vectorized version of datetime.strptime()
                vector_strptime = numpy.vectorize(datetime.datetime.strptime)
                self.datetimes = vector_strptime(datetime_strings_subset, '%Y-%m-%dT%H:%M:%SZ')
            else:
                self.datetimes = [datetime_strings[self.start_index]]

        print(f'Collecting HFR velocity from {numpy.min(self.datetimes)} to {numpy.max(self.datetimes)}')

        self.data = {'lon': self.netcdf_dataset['lon'][:], 'lat': self.netcdf_dataset['lat'][:]}

        # concurrently populate dictionary with data for each variable
        with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
            variable_futures = {concurrency_pool.submit(self._threaded_collect, current_variable): current_variable for
                                current_variable in MEASUREMENT_VARIABLES}

            for current_future in concurrent.futures.as_completed(variable_futures):
                if current_future._exception is None:
                    current_variable = variable_futures[current_future]
                    self.data[current_variable] = current_future.result()

            del variable_futures

        if HFR_Range.grid_transform is None:
            # define image properties
            west = numpy.min(self.data['lon'])
            north = numpy.max(self.data['lat'])

            self.x_size = numpy.mean(numpy.diff(self.data['lon']))
            self.y_size = numpy.mean(numpy.diff(self.data['lat']))

            # get rasterio geotransform of HFR dataset (flipped latitude)
            self.grid_transform = rasterio.transform.from_origin(west, north, self.x_size, self.y_size)

    def _threaded_collect(self, variable_name: str) -> numpy.ma.MaskedArray:
        """
        Hacky method that creates a new NetCDF dataset to retrieve data.
        This function is necessary to perform multithreaded reading on the same dataset.

        :param variable_name: Name of variable.
        :return: Array of data for variable.
        """

        return netCDF4.Dataset(self.url)[variable_name][self.start_index:self.end_index, :, :]

    def get_datetime_indices(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime) -> numpy.ndarray:
        """
        Returns indices where datetimes in the current dataset exist within the hourly range between the given datetimes.

        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :return: Array of indices.
        """

        # get range of times spaced by hour intervals between specified
        # endpoints, rounded to the hour
        hourly_range = _utilities.hour_range(_utilities.round_to_hour(start_datetime),
                                             _utilities.round_to_hour(end_datetime))

        try:
            datetime_indices = numpy.where(numpy.in1d(self.datetimes, hourly_range, assume_unique=True))[0]
        except:
            datetime_indices = None

        if len(datetime_indices) == 0 or datetime_indices is None:
            print('Specified time interval does not exist within dataset.')
            datetime_indices = None

        return datetime_indices

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

        return (abs(self.x_size), abs(self.y_size))

    def write_sites(self, output_filename: str, layer_name: str):
        """
        Writes HFR radar facility locations to specified file and layer.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        """

        radar_sites_code = self.netcdf_dataset['site_code'][:]
        radar_sites_network_code = self.netcdf_dataset['site_netCode'][:]
        radar_sites_lon = self.netcdf_dataset['site_lon'][:]
        radar_sites_lat = self.netcdf_dataset['site_lat'][:]

        current_layer_records = []

        schema = {
            'geometry': 'Point', 'properties': {
                'code': 'str', 'net_code': 'str', 'lon': 'float', 'lat': 'float'
            }
        }

        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema,
                        crs=FIONA_WGS84) as current_layer:
            for site_index in range(len(radar_sites_code)):
                current_site_code = radar_sites_code[site_index, :].tobytes().decode().strip('\x00').strip()
                current_site_network_code = radar_sites_network_code[site_index, :].tobytes().decode().strip(
                        '\x00').strip()
                current_lon = float(radar_sites_lon[site_index])
                current_lat = float(radar_sites_lat[site_index])

                current_record = {
                    'id':         site_index + 1,
                    'geometry':   {'type': 'Point', 'coordinates': (current_lon, current_lat)}, 'properties': {
                        'code': current_site_code, 'net_code': current_site_network_code, 'lon': float(current_lon),
                        'lat':  float(current_lat)
                    }
                }

                current_layer_records.append(current_record)

            current_layer.writerecords(current_layer_records)

    def write_vectors(self, output_filename: str, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None):
        """
        Write HFR data to a layer of the provided output file for every hour in the given time interval.

        :param output_filename: Path to output file.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)

        # dataset data
        hfr_u = self.data['u'][datetime_indices, :, :]
        hfr_v = self.data['v'][datetime_indices, :, :]
        hfr_lon = self.data['lon'][:]
        hfr_lat = self.data['lat'][:]
        hfr_dop_lon = self.data['dop_lon'][datetime_indices, :, :]
        hfr_dop_lat = self.data['dop_lat'][datetime_indices, :, :]

        # define layer schema
        schema = {
            'geometry': 'Point', 'properties': {
                'u': 'float', 'v': 'float', 'lat': 'float', 'lon': 'float', 'dop_lat': 'float', 'dop_lon': 'float'
            }
        }

        # create dict to store features
        layer_features = {}

        # create layer using OGR, then add features using QGIS
        for datetime_index in range(len(datetime_indices)):
            current_datetime = self.datetimes[datetime_indices[datetime_index]]

            current_layer_name = f'{current_datetime.strftime("%Y%m%dT%H%M%S")}'

            # create QGIS features
            current_layer_features = []

            feature_index = 1

            for lon_index in range(len(hfr_lon)):
                for lat_index in range(len(hfr_lat)):
                    current_u = hfr_u[datetime_index, lat_index, lon_index]

                    # check if record has values
                    if current_u is not numpy.ma.masked:
                        current_v = hfr_v[datetime_index, lat_index, lon_index]
                        current_dop_lon = hfr_dop_lon[datetime_index, lat_index, lon_index]
                        current_dop_lat = hfr_dop_lat[datetime_index, lat_index, lon_index]
                        current_lon = hfr_lon[lon_index]
                        current_lat = hfr_lat[lat_index]

                        current_point = QgsGeometry(QgsPoint(current_lon, current_lat))

                        current_feature = QgsFeature()

                        current_feature.setAttributes(
                                [feature_index, float(current_u), float(current_v), float(current_lon),
                                 float(current_lat), float(current_dop_lon), float(current_dop_lat)])

                        current_feature.setGeometry(current_point)

                        current_layer_features.append(current_feature)

                        feature_index += 1

            layer_features[current_layer_name] = current_layer_features

        # write queued features to their respective layers
        for current_layer_name, current_layer_features in layer_features.items():
            current_layer = QgsVectorLayer(f'{output_filename}|layername={current_layer_name}', current_layer_name,
                                           'ogr')

            # open layer for editing
            current_layer.startEditing()

            # add features to layer
            current_layer.dataProvider().addFeatures(current_layer_features)

            # write changes to layer
            current_layer.commitChanges()

    def write_vector(self, output_filename: str, layer_name: str = 'uv', start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None):
        """
        Write average of HFR data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)

        if datetime_indices is not None and len(datetime_indices) > 0:
            measurement_variables = ['u', 'v', 'DOPx', 'DOPy']

            variable_means = {}

            # concurrently populate dictionary with averaged data for each
            # variable
            with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
                variable_futures = {
                    concurrency_pool.submit(numpy.ma.mean, self.data[current_variable][datetime_indices, :, :],
                                            axis=0): current_variable for current_variable in measurement_variables}

                for current_future in concurrent.futures.as_completed(variable_futures):
                    current_variable = variable_futures[current_future]
                    variable_means[current_variable] = current_future.result()

                del variable_futures

            # define layer schema
            schema = {
                'geometry': 'Point', 'properties': {
                    'lon': 'float', 'lat': 'float'
                }
            }

            schema['properties'].update({current_variable: 'float' for current_variable in measurement_variables})

            # create layer
            fiona.open(output_filename, 'w', driver='GPKG', schema=schema, crs=FIONA_WGS84, layer=layer_name).close()

            # dataset data
            hfr_lon = self.data['lon']
            hfr_lat = self.data['lat']

            # create features
            layer_features = []

            feature_index = 1

            for lon_index in range(len(hfr_lon)):
                for lat_index in range(len(hfr_lat)):
                    current_data = [variable_means[current_variable][lat_index, lon_index] for current_variable in
                                    measurement_variables]

                    # stop if record has masked values
                    if numpy.all(~numpy.isnan(current_data)):
                        current_lon = hfr_lon[lon_index]
                        current_lat = hfr_lat[lat_index]

                        current_feature = QgsFeature()
                        current_feature.setGeometry(QgsGeometry(QgsPoint(current_lon, current_lat)))
                        current_feature.setAttributes(
                                [feature_index, float(current_lon), float(current_lat)] + [float(entry) for entry in
                                                                                           current_data])

                        layer_features.append(current_feature)
                        feature_index += 1

            # write queued features to layer
            print(f'Writing {output_filename}')
            current_layer = QgsVectorLayer('{0}|layername={1}'.format(output_filename, layer_name), layer_name, 'ogr')
            current_layer.startEditing()
            current_layer.dataProvider().addFeatures(layer_features)
            current_layer.commitChanges()

    def write_rasters(self, output_dir: str, variables: list = None, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None, vector_components: bool = False, fill_value: float = None,
                      drivers: list = ['GTiff']):
        """
        Write average of HFR data for all hours in the given time interval to rasters.

        :param output_dir: Path to output directory.
        :param variables: List of variable names to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param vector_components: Whether to write direction and magnitude rasters.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        """

        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime

        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)

        if datetime_indices is not None:
            variables = variables if variables is not None else ['u', 'v', 'DOPx', 'DOPy']

            variable_means = {}

            # concurrently populate dictionary with averaged data for each variable
            with concurrent.futures.ThreadPoolExecutor() as concurrency_pool:
                variable_futures = {
                    concurrency_pool.submit(numpy.ma.mean, self.data[current_variable][datetime_indices, :, :],
                                            axis=0): current_variable for current_variable in variables}

                for current_future in concurrent.futures.as_completed(variable_futures):
                    current_variable = variable_futures[current_future]
                    variable_means[current_variable] = current_future.result()

                del variable_futures

            if vector_components:
                if 'u' in variables:
                    u_data = variable_means['u']
                else:
                    u_data = numpy.ma.mean(self.data['u'][datetime_indices, :, :], axis=0)

                if 'v' in variables:
                    v_data = variable_means['v']
                else:
                    v_data = numpy.ma.mean(self.data['v'][datetime_indices, :, :], axis=0)

                # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))

            for current_variable, current_data in variable_means.items():
                if fill_value is not None:
                    current_data.set_fill_value(fill_value)

                raster_data = current_data.filled().astype(rasterio.float32)

                gdal_args = {
                    'height': raster_data.shape[0], 'width': raster_data.shape[1], 'count': 1,
                    'dtype':  raster_data.dtype, 'crs': RASTERIO_WGS84, 'transform': self.grid_transform,
                    'nodata': current_data.fill_value.astype(raster_data.dtype)
                }

                for current_driver in drivers:
                    if current_driver == 'AAIGrid':
                        file_extension = 'asc'

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
                            'height':    raster_data.shape[0], 'width': raster_data.shape[1], 'FORCE_CELLSIZE': 'YES',
                            'transform': rasterio.transform.from_origin(numpy.min(output_lon), numpy.max(output_lat),
                                                                        numpy.max(numpy.diff(output_lon)),
                                                                        numpy.max(numpy.diff(output_lon)))
                        })
                    elif current_driver == 'GTiff':
                        file_extension = 'tiff'
                    elif current_driver == 'GPKG':
                        file_extension = 'gpkg'

                    current_output_filename = os.path.join(output_dir, f'hfr_{current_variable}.{file_extension}')

                    print(f'Writing {current_output_filename}')
                    with rasterio.open(current_output_filename, 'w', current_driver, **gdal_args) as output_raster:
                        output_raster.write(numpy.flipud(raster_data), 1)

    def __repr__(self):
        used_params = [self.start_datetime.__repr__(), self.end_datetime.__repr__()]
        optional_params = [self.resolution]

        for current_param in optional_params:
            if current_param is not None:
                if 'str' in str(type(current_param)):
                    current_param = f'"{current_param}"'
                else:
                    current_param = str(current_param)

                used_params.append(current_param)

        return f'{self.__class__.__name__}({str(", ".join(used_params))})'


if __name__ == '__main__':
    start_datetime = datetime.datetime(2018, 7, 14)
    end_datetime = datetime.datetime.now()

    output_dir = r'C:\Data\hfr\compare'
    resolution = 6

    # get dataset from source
    print('UCSD')
    hfr_dataset_UCSD = HFR_Range(start_datetime, end_datetime, resolution, source='UCSD')

    print('NDBC')
    hfr_dataset_NDBC = HFR_Range(start_datetime, end_datetime, resolution, source='NDBC')

    date_interval_string = f'{start_datetime.strftime("%m%d%H")}_{end_datetime.strftime("%m%d%H")}'

    # write HFR rasters
    hfr_dataset_UCSD.write_rasters(output_dir, f'UCSD_{date_interval_string}')
    hfr_dataset_NDBC.write_rasters(output_dir, f'NDBC_{date_interval_string}')

    print('done')
