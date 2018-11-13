import datetime
import os

import xarray

from main import DATA_DIR

WORKSPACE_DIR = os.path.join(DATA_DIR, 'validation')

if __name__ == '__main__':
    # setup start and ending times
    start_datetime = datetime.datetime(2018, 11, 8)
    end_datetime = start_datetime + datetime.timedelta(days=1)
    day_dir = os.path.join(WORKSPACE_DIR, start_datetime.strftime('%Y%m%d'))
    os.mkdir(day_dir)

    # define filenames for NetCDF
    nc_filenames = {'hfr': os.path.join(day_dir, 'hfr.nc'),
                    'viirs': os.path.join(day_dir, 'viirs.nc'),
                    'viirs_corrected': os.path.join(day_dir, 'viirs_corrected.nc'),
                    'wcofs_u': os.path.join(day_dir, 'wcofs_u.nc'),
                    'wcofs_v': os.path.join(day_dir, 'wcofs_v.nc'),
                    'wcofs_sst': os.path.join(day_dir, 'wcofs_sst.nc')}

    # write HFR NetCDF files if they do not exist
    if not os.path.exists(nc_filenames['hfr']):
        from dataset import hfr

        hfr_range = hfr.HFR_Range(start_datetime, end_datetime)
        hfr_range.to_netcdf(nc_filenames['hfr'])

    # write VIIRS NetCDF files if they do not exist
    if not os.path.exists(nc_filenames['viirs']) or not os.path.exists(nc_filenames['viirs_corrected']):
        from dataset import viirs

        viirs_range = viirs.VIIRS_Range(start_datetime, end_datetime)
        viirs_range.to_netcdf(os.path.join(day_dir, 'viirs.nc'), variables=['sst'])
        viirs_range.to_netcdf(os.path.join(day_dir, 'viirs_corrected.nc'), variables=['sst'], sses_correction=True)

    # write WCOFS NetCDF files if they do not exist
    if not os.path.exists(nc_filenames['wcofs_u']) or not os.path.exists(nc_filenames['wcofs_v']) or not os.path.exists(
            nc_filenames['wcofs_sst']):
        from dataset import wcofs

        wcofs_range = wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg')

        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range.to_netcdf(nc_filenames['wcofs_u'], variables=['u'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_v'], variables=['v'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_sst'], variables=['temp'])

    # load datasets from local NetCDF files
    datasets = {'hfr': xarray.open_dataset(nc_filenames['hfr']),
                'viirs': xarray.open_dataset(nc_filenames['viirs']),
                'viirs_corrected': xarray.open_dataset(nc_filenames['viirs_corrected']),
                'wcofs_u': xarray.open_dataset(nc_filenames['wcofs_u']),
                'wcofs_v': xarray.open_dataset(nc_filenames['wcofs_v']),
                'wcofs_sst': xarray.open_dataset(nc_filenames['wcofs_sst'])}

    # create dict to store differences
    nearest_neighbor_differences = {'sst', 'ssc'}

    # get dimensions of WCOFS dataset (time delta, eta, rho)
    wcofs_dimensions = {'sst': list(datasets['wcofs_sst']['temp'].coords),
                        'u': list(datasets['wcofs_u']['temp'].coords), 'v': list(datasets['wcofs_v']['temp'].coords)}

    # compare sea-surface temperature
    wcofs_sst = datasets['wcofs_sst']
    viirs_sst = datasets['viirs']
    viirs_sst_corrected = datasets['viirs_corrected']

    # iterate over every WCOFS cell in rho grid
    for time_delta in wcofs_dimensions['sst'][0]:
        for row in wcofs_dimensions['sst'][1]:
            for col in wcofs_dimensions['sst'][2]:
                pass

    # compare sea-surface currents
    hfr = datasets['hfr']
    wcofs_u = datasets['wcofs_u']
    wcofs_v = datasets['wcofs_v']

    # iterate over every WCOFS cell in u grid
    for time_delta in wcofs_dimensions['u'][0]:
        for row in wcofs_dimensions['u'][1]:
            for col in wcofs_dimensions['u'][2]:
                pass

    # iterate over every WCOFS cell in v grid
    for time_delta in wcofs_dimensions['v'][0]:
        for row in wcofs_dimensions['v'][1]:
            for col in wcofs_dimensions['v'][2]:
                pass
