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

from main import DATA_DIR
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
        
        # get URL
        if self.source == 'UCSD':
            self.url = f'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/{self.resolution}km/hourly/RTV/HFRADAR_US_West_Coast_{self.resolution}km_Resolution_Hourly_RTV_best.ncd'
        elif self.source == 'NDBC':
            self.url = f'https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{self.resolution}km'
        
        try:
            self.netcdf_dataset = xarray.open_dataset(self.url)
        except OSError:
            raise _utilities.NoDataError(f'No HFR dataset found at {self.url}')
        
        self.datetimes = numpy.array(self.netcdf_dataset['time'].values, dtype='datetime64[h]')
        
        self.start_index = numpy.searchsorted(self.datetimes, numpy.datetime64(self.start_datetime))
        self.end_index = numpy.searchsorted(self.datetimes, numpy.datetime64(self.end_datetime))
        
        self.datetimes = self.datetimes[self.start_index:self.end_index]
        
        print(
            f'Collecting HFR velocity from {self.source} between {numpy.min(self.datetimes)} and {numpy.max(self.datetimes)}...')
        
        self.data = {
            'lon': self.netcdf_dataset['lon'].values, 'lat': self.netcdf_dataset['lat'].values
        }
        
        for variable in MEASUREMENT_VARIABLES:
            self.data[variable] = self.netcdf_dataset[variable][self.start_index:self.end_index, :, :].values
        
        if HFR_Range.grid_transform is None:
            # define image properties
            west = numpy.min(self.data['lon'])
            north = numpy.max(self.data['lat'])
            
            self.x_size = numpy.mean(numpy.diff(self.data['lon']))
            self.y_size = numpy.mean(numpy.diff(self.data['lat']))
            
            # get rasterio geotransform of HFR dataset (flipped latitude)
            self.grid_transform = rasterio.transform.from_origin(west, north, self.x_size, self.y_size)
    
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
            datetime_indices = numpy.where(
                numpy.in1d(self.datetimes, numpy.array(hourly_range).astype(numpy.datetime64), assume_unique=True))[
                0]
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
        
        return abs(self.x_size), abs(self.y_size)
    
    def write_sites(self, output_filename: str, layer_name: str):
        """
        Writes HFR radar facility locations to specified file and layer.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        """
        
        radar_sites_code = self.netcdf_dataset['site_code'].values
        radar_sites_network_code = self.netcdf_dataset['site_netCode'].values
        radar_sites_lon = self.netcdf_dataset['site_lon'].values
        radar_sites_lat = self.netcdf_dataset['site_lat'].values
        
        layer_records = []
        
        for site_index in range(len(radar_sites_code)):
            site_code = radar_sites_code[site_index, :].tobytes().decode().strip('\x00').strip()
            site_network_code = radar_sites_network_code[site_index, :].tobytes().decode().strip('\x00').strip()
            lon = float(radar_sites_lon[site_index])
            lat = float(radar_sites_lat[site_index])
            
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
        
        with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_WGS84) as layer:
            layer.writerecords(layer_records)
    
    def write_vectors(self, output_filename: str, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None, dop_threshold: float = 0.5):
        """
        Write HFR data to a layer of the provided output file for every hour in the given time interval.

        :param output_filename: Path to output file.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param dop_threshold: Threshold for dilution of precision above which data is not useable.
        """
        
        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime
        
        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)
        
        numpy.warnings.filterwarnings('ignore')
        
        dop_mask = ((self.data['DOPx'][datetime_indices, :, :] <= dop_threshold) | (
                self.data['DOPy'][datetime_indices, :, :] <= dop_threshold))
        
        numpy.warnings.filterwarnings('default')
        
        # dataset data
        hfr_u = self.data['u'][datetime_indices, :, :]
        hfr_v = self.data['v'][datetime_indices, :, :]
        hfr_lon = self.data['lon'][:]
        hfr_lat = self.data['lat'][:]
        hfr_dop_lon = self.data['DOPx'][datetime_indices, :, :]
        hfr_dop_lat = self.data['DOPy'][datetime_indices, :, :]
        
        hfr_u[~dop_mask] = numpy.nan
        hfr_v[~dop_mask] = numpy.nan
        hfr_dop_lon[~dop_mask] = numpy.nan
        hfr_dop_lat[~dop_mask] = numpy.nan
        
        # create dict to store features
        layers = {}
        
        # create layer using OGR, then add features using QGIS
        for datetime_index in range(len(datetime_indices)):
            vector_datetime = self.datetimes[datetime_indices[datetime_index]]
            
            layer_name = f'{vector_datetime.astype(datetime.datetime).strftime("%Y%m%dT%H%M%S")}'
            
            # create QGIS features
            layer_records = []
            
            feature_index = 1
            
            for lon_index in range(len(hfr_lon)):
                for lat_index in range(len(hfr_lat)):
                    u = hfr_u[datetime_index, lat_index, lon_index]
                    
                    # check if record has values
                    if not numpy.isnan(u):
                        v = hfr_v[datetime_index, lat_index, lon_index]
                        dop_lon = hfr_dop_lon[datetime_index, lat_index, lon_index]
                        dop_lat = hfr_dop_lat[datetime_index, lat_index, lon_index]
                        lon = hfr_lon[lon_index]
                        lat = hfr_lat[lat_index]
                        
                        record = {
                            'id': feature_index, 'geometry': {'type': 'Point', 'coordinates': (lon, lat)},
                            'properties': {
                                'lon': float(lon), 'lat': float(lat), 'u': float(u), 'v': float(v),
                                'DOPx': float(dop_lon), 'DOPy': float(dop_lat)
                            }
                        }
                        
                        layer_records.append(record)
                        feature_index += 1
            
            layers[layer_name] = layer_records
        
        # write queued features to their respective layers
        schema = {
            'geometry': 'Point', 'properties': {
                'u': 'float', 'v': 'float', 'lat': 'float', 'lon': 'float', 'dop_lat': 'float', 'dop_lon': 'float'
            }
        }
        
        for layer_name, layer_records in layers.items():
            with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_WGS84) as layer:
                layer.writerecords(layer_records)
    
    def write_vector(self, output_filename: str, layer_name: str = 'uv', start_datetime: datetime.datetime = None,
                     end_datetime: datetime.datetime = None, dop_threshold: float = 0.5):
        """
        Write average of HFR data for all hours in the given time interval to a single layer of the provided output file.

        :param output_filename: Path to output file.
        :param layer_name: Name of layer to write.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param dop_threshold: Threshold for dilution of precision above which data is not useable.
        """
        
        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime
        
        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)
        
        if datetime_indices is not None and len(datetime_indices) > 0:
            measurement_variables = ['u', 'v', 'DOPx', 'DOPy']
            
            numpy.warnings.filterwarnings('ignore')
            
            dop_mask = ((self.data['DOPx'][datetime_indices, :, :] <= dop_threshold) | (
                    self.data['DOPy'][datetime_indices, :, :] <= dop_threshold))
            
            numpy.warnings.filterwarnings('default')
            
            variable_means = {}
            
            for variable in measurement_variables:
                variable_data = self.data[variable][datetime_indices, :, :]
                variable_data[~dop_mask] = numpy.nan
                
                variable_mean = numpy.mean(variable_data, axis=0)
                variable_means[variable] = variable_mean
            
            # define layer schema
            schema = {
                'geometry': 'Point', 'properties': {
                    'lon': 'float', 'lat': 'float'
                }
            }
            
            schema['properties'].update({variable: 'float' for variable in measurement_variables})
            
            # dataset data
            hfr_lon = self.data['lon']
            hfr_lat = self.data['lat']
            
            # create features
            layer_records = []
            
            feature_index = 1
            
            for lon_index in range(len(hfr_lon)):
                for lat_index in range(len(hfr_lat)):
                    data = [float(variable_means[variable][lat_index, lon_index]) for variable in measurement_variables]
                    
                    # stop if record has masked values
                    if numpy.all(~numpy.isnan(data)):
                        lon = hfr_lon[lon_index]
                        lat = hfr_lat[lat_index]
                        
                        record = {
                            'id': feature_index, 'geometry': {'type': 'Point', 'coordinates': (lon, lat)},
                            'properties': {
                                'lon': float(lon), 'lat': float(lat)
                            }
                        }
                        
                        record['properties'].update(dict(zip(measurement_variables, data)))
                        
                        layer_records.append(record)
                        feature_index += 1
            
            # write queued features to layer
            print(f'Writing {output_filename}')
            with fiona.open(output_filename, 'w', 'GPKG', layer=layer_name, schema=schema, crs=FIONA_WGS84) as layer:
                layer.writerecords(layer_records)
    
    def write_rasters(self, output_dir: str, filename_prefix: str = 'hfr', filename_suffix: str = '',
                      variables: list = None, start_datetime: datetime.datetime = None,
                      end_datetime: datetime.datetime = None, vector_components: bool = False,
                      fill_value: float = -9999, drivers: list = ['GTiff'], dop_threshold: float = 0.5):
        """
        Write average of HFR data for all hours in the given time interval to rasters.

        :param output_dir: Path to output directory.
        :param filename_prefix: Prefix for output filenames.
        :param filename_suffix: Suffix for output filenames.
        :param variables: List of variable names to use.
        :param start_datetime: Beginning of time interval.
        :param end_datetime: End of time interval.
        :param vector_components: Whether to write direction and magnitude rasters.
        :param fill_value: Desired fill value of output.
        :param drivers: List of strings of valid GDAL drivers (currently one of 'GTiff', 'GPKG', or 'AAIGrid').
        :param dop_threshold: Threshold for dilution of precision above which data is not useable.
        """
        
        start_datetime = start_datetime if start_datetime is not None else self.start_datetime
        end_datetime = end_datetime if end_datetime is not None else self.end_datetime
        
        # get indices of selected datetimes
        datetime_indices = self.get_datetime_indices(start_datetime, end_datetime)
        
        if filename_suffix is not '':
            filename_suffix = f'_{filename_suffix}'
        
        if datetime_indices is not None:
            variables = variables if variables is not None else ['u', 'v', 'DOPx', 'DOPy']
            
            numpy.warnings.filterwarnings('ignore')
            
            dop_mask = ((self.data['DOPx'][datetime_indices, :, :] <= dop_threshold) | (
                    self.data['DOPy'][datetime_indices, :, :] <= dop_threshold))
            
            numpy.warnings.filterwarnings('default')
            
            variable_means = {}
            
            for variable in variables:
                variable_data = self.data[variable][datetime_indices, :, :]
                variable_data[~dop_mask] = numpy.nan
                
                variable_mean = numpy.mean(variable_data, axis=0)
                variable_means[variable] = variable_mean
            
            if vector_components:
                if 'u' in variables:
                    u_data = variable_means['u']
                else:
                    u_data = numpy.mean(self.data['u'][datetime_indices, :, :], axis=0)
                
                if 'v' in variables:
                    v_data = variable_means['v']
                else:
                    v_data = numpy.mean(self.data['v'][datetime_indices, :, :], axis=0)
                
                # calculate direction and magnitude of vector in degrees (0-360) and in metres per second
                variable_means['dir'] = (numpy.arctan2(u_data, v_data) + numpy.pi) * (180 / numpy.pi)
                variable_means['mag'] = numpy.sqrt(numpy.square(u_data) + numpy.square(v_data))
            
            for variable, variable_data in variable_means.items():
                raster_data = variable_data.astype(rasterio.float32)
                
                gdal_args = {
                    'height': raster_data.shape[0], 'width': raster_data.shape[1], 'count': 1,
                    'dtype': raster_data.dtype, 'crs': RASTERIO_WGS84, 'transform': self.grid_transform,
                    'nodata': numpy.array([fill_value]).astype(raster_data.dtype).item()
                }
                
                for driver in drivers:
                    if driver == 'AAIGrid':
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
                            'height': raster_data.shape[0], 'width': raster_data.shape[1], 'FORCE_CELLSIZE': 'YES',
                            'transform': rasterio.transform.from_origin(numpy.min(output_lon), numpy.max(output_lat),
                                                                        numpy.max(numpy.diff(output_lon)),
                                                                        numpy.max(numpy.diff(output_lon)))
                        })
                    elif driver == 'GTiff':
                        file_extension = 'tiff'
                    elif driver == 'GPKG':
                        file_extension = 'gpkg'
                    
                    output_filename = os.path.join(output_dir,
                                                   f'{filename_prefix}_{variable}{filename_suffix}.{file_extension}')
                    
                    if os.path.isfile(output_filename):
                        os.remove(output_filename)
                    
                    print(f'Writing {output_filename}')
                    with rasterio.open(output_filename, 'w', driver, **gdal_args) as output_raster:
                        output_raster.write(numpy.flipud(raster_data), 1)
    
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
    
    start_datetime = datetime.datetime(2018, 10, 15)
    end_datetime = datetime.datetime(2018, 10, 15) + datetime.timedelta(days=1)
    
    # get dataset from source
    hfr_dataset = HFR_Range(start_datetime, end_datetime)
    
    # write HFR raster
    hfr_dataset.write_vector(os.path.join(output_dir, 'hfr.gpkg'))
    
    print('done')
