import datetime
import os
import sys
from concurrent import futures

import numpy
import xarray

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from main import DATA_DIR
from dataset.wcofs import interpolate_grid

WORKSPACE_DIR = os.path.join(DATA_DIR, 'validation')

if __name__ == '__main__':
    # setup start and ending times
    start_datetime = datetime.datetime(2018, 11, 8)
    end_datetime = start_datetime + datetime.timedelta(days=1)
    day_dir = os.path.join(WORKSPACE_DIR, start_datetime.strftime('%Y%m%d'))
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    # define filenames for NetCDF
    nc_filenames = {'hfr': os.path.join(day_dir, 'data/hfr.nc'),
                    'viirs': os.path.join(day_dir, 'data/viirs.nc'),
                    'wcofs_u': os.path.join(day_dir, 'data/wcofs_u.nc'),
                    'wcofs_v': os.path.join(day_dir, 'data/wcofs_v.nc'),
                    'wcofs_sst': os.path.join(day_dir, 'data/wcofs_sst.nc')}

    # write HFR NetCDF file if it does not exist
    if not os.path.exists(nc_filenames['hfr']):
        from dataset import hfr

        hfr_range = hfr.HFR_Range(start_datetime, end_datetime)
        hfr_range.to_netcdf(nc_filenames['hfr'])

    # write VIIRS NetCDF file if it does not exist
    if not os.path.exists(nc_filenames['viirs']):
        from dataset import viirs

        viirs_range = viirs.VIIRS_Range(start_datetime, end_datetime)
        viirs_range.to_netcdf(os.path.join(day_dir, 'viirs.nc'), variables=['sst'])

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
    datasets = {'hfr': None, 'viirs': None, 'wcofs_u': None, 'wcofs_v': None, 'wcofs_sst': None}
    for dataset in datasets:
        datasets[dataset] = xarray.open_dataset(nc_filenames[dataset])

    # get dimensions of WCOFS dataset (time delta, eta, rho)
    wcofs_dimensions = {'sst': list(datasets['wcofs_sst']['temp'].coords),
                        'u': list(datasets['wcofs_u']['u'].coords), 'v': list(datasets['wcofs_v']['v'].coords)}

    # interpolate nearest-neighbor observational data onto WCOFS grid
    data = {'obser': {'sst': datasets['viirs']['sst'].values, 'u': datasets['hfr']['u'].values,
                      'v': datasets['hfr']['v'].values},
            'model': {'sst': {}, 'u': {}, 'v': {}}}

    time_deltas = dict(zip(range(len(datasets['wcofs_sst'][wcofs_dimensions['sst'][0]].values)),
                           sorted(datasets['wcofs_sst'][wcofs_dimensions['sst'][0]].values, reverse=True)))

    if len(os.listdir(os.path.join(day_dir, 'residuals'))) == 0:
        print('interpolating WCOFS data onto observational grids...')
        with futures.ThreadPoolExecutor() as concurrency_pool:
            sst_futures = {}
            v_futures = {}
            u_futures = {}

            for time_delta_index, time_delta in time_deltas.items():
                sst_futures[concurrency_pool.submit(interpolate_grid,
                                                    datasets['wcofs_sst']['lon'].values,
                                                    datasets['wcofs_sst']['lat'].values,
                                                    datasets['wcofs_sst']['temp'][time_delta_index, :, :].values,
                                                    datasets['viirs']['lon'].values,
                                                    datasets['viirs']['lat'].values,
                                                    'linear')] = time_delta

                u_futures[concurrency_pool.submit(interpolate_grid,
                                                  datasets['wcofs_u']['lon'].values,
                                                  datasets['wcofs_u']['lat'].values,
                                                  datasets['wcofs_u']['u'][time_delta_index, :, :].values,
                                                  datasets['hfr']['lon'].values,
                                                  datasets['hfr']['lat'].values,
                                                  'linear')] = time_delta

                v_futures[concurrency_pool.submit(interpolate_grid,
                                                  datasets['wcofs_v']['lon'].values,
                                                  datasets['wcofs_v']['lat'].values,
                                                  datasets['wcofs_v']['v'][time_delta_index, :, :].values,
                                                  datasets['hfr']['lon'].values,
                                                  datasets['hfr']['lat'].values,
                                                  'linear')] = time_delta

            for completed_future in futures.as_completed(sst_futures):
                time_delta = sst_futures[completed_future]
                data['model']['sst'][time_delta] = completed_future.result()

            for completed_future in futures.as_completed(u_futures):
                time_delta = u_futures[completed_future]
                data['model']['u'][time_delta] = completed_future.result()

            for completed_future in futures.as_completed(v_futures):
                time_delta = v_futures[completed_future]
                data['model']['v'][time_delta] = completed_future.result()

        for time_delta in time_deltas.values():
            sst_residuals = data['obser']['sst'] - data['model']['sst'][time_delta]
            u_residuals = data['obser']['u'] - data['model']['u'][time_delta]
            v_residuals = data['obser']['v'] - data['model']['v'][time_delta]

            sst_dataarray = xarray.DataArray(sst_residuals, coords=datasets['viirs']['sst'].coords)
            u_dataarray = xarray.DataArray(u_residuals, coords=datasets['hfr']['u'].coords)
            v_dataarray = xarray.DataArray(v_residuals, coords=datasets['hfr']['v'].coords)

            dataset_path = os.path.join(day_dir, f'residuals/residuals_{time_delta}.nc')

            residuals_dataset = xarray.Dataset({'sst': sst_dataarray, 'u': u_dataarray, 'v': v_dataarray})
            residuals_dataset.to_netcdf(dataset_path)

    # get total sum of squares (variance proportional)
    sst_tot_sum_sq = numpy.nansum(numpy.square(data['obser']['sst'] - numpy.nanmean(data['obser']['sst'])))
    u_tot_sum_sq = numpy.nansum(numpy.square(data['obser']['u'] - numpy.nanmean(data['obser']['u'])))
    v_tot_sum_sq = numpy.nansum(numpy.square(data['obser']['v'] - numpy.nanmean(data['obser']['v'])))

    for time_delta in time_deltas.values():
        dataset_path = os.path.join(day_dir, f'residuals/residuals_{time_delta}.nc')

        residuals_dataset = xarray.open_dataset(dataset_path)

        sst_residuals = residuals_dataset['sst'].values
        u_residuals = residuals_dataset['u'].values
        v_residuals = residuals_dataset['v'].values

        # get squared residuals
        sst_res_sq = numpy.square(sst_residuals)
        u_res_sq = numpy.square(u_residuals)
        v_res_sq = numpy.square(v_residuals)

        # get root mean squared error
        sst_rmse = numpy.sqrt(numpy.nanmean(sst_res_sq))
        u_rmse = numpy.sqrt(numpy.nanmean(u_res_sq))
        v_rmse = numpy.sqrt(numpy.nanmean(v_res_sq))

        print(f'{time_delta} RMSE SST: {sst_rmse}')
        print(f'{time_delta} RMSE U: {u_rmse}')
        print(f'{time_delta} RMSE V: {v_rmse}')

        # get sum of residuals
        sst_res_sum_sq = numpy.nansum(sst_res_sq)
        u_res_sum_sq = numpy.nansum(u_res_sq)
        v_res_sum_sq = numpy.nansum(v_res_sq)

        # get coefficient of determination (R^2)
        sst_r_sq = 1 - (sst_res_sum_sq / sst_tot_sum_sq)
        u_r_sq = 1 - (u_res_sum_sq / u_tot_sum_sq)
        v_r_sq = 1 - (v_res_sum_sq / v_tot_sum_sq)

        print(f'{time_delta} R^2 SST: {sst_r_sq}')
        print(f'{time_delta} R^2 U: {u_r_sq}')
        print(f'{time_delta} R^2 V: {v_r_sq}')

    print('done')
