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

# UTC offset of study area
UTC_OFFSET = 8

if __name__ == '__main__':
    # setup start and ending times
    start_datetime = datetime.datetime(2018, 11, 14)
    end_datetime = start_datetime + datetime.timedelta(days=1)

    day_dir = os.path.join(WORKSPACE_DIR, start_datetime.strftime('%Y%m%d'))
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    data_dir = os.path.join(day_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    residuals_dir = os.path.join(day_dir, 'residuals')
    if not os.path.exists(residuals_dir):
        os.mkdir(residuals_dir)

    # define filenames for NetCDF
    nc_filenames = {'hfr': os.path.join(data_dir, 'hfr.nc'),
                    'viirs': os.path.join(data_dir, 'viirs.nc'),
                    'wcofs_sst_DA': os.path.join(data_dir, 'wcofs_sst_DA.nc'),
                    'wcofs_u_DA': os.path.join(data_dir, 'wcofs_u_DA.nc'),
                    'wcofs_v_DA': os.path.join(data_dir, 'wcofs_v_DA.nc'),
                    'wcofs_sst_noDA': os.path.join(data_dir, 'wcofs_sst_noDA.nc'),
                    'wcofs_u_noDA': os.path.join(data_dir, 'wcofs_u_noDA.nc'),
                    'wcofs_v_noDA': os.path.join(data_dir, 'wcofs_v_noDA.nc')}

    # write HFR NetCDF file if it does not exist
    if not os.path.exists(nc_filenames['hfr']):
        from dataset import hfr

        hfr_range = hfr.HFR_Range(start_datetime, end_datetime)
        hfr_range.to_netcdf(nc_filenames['hfr'])

    # write VIIRS NetCDF file if it does not exist
    if not os.path.exists(nc_filenames['viirs']):
        from dataset import viirs

        utc_start_datetime = start_datetime + datetime.timedelta(hours=UTC_OFFSET)
        utc_end_datetime = end_datetime + datetime.timedelta(hours=UTC_OFFSET)

        viirs_range = viirs.VIIRS_Range(utc_start_datetime, utc_end_datetime)
        viirs_range.to_netcdf(nc_filenames['viirs'], variables=['sst'])

    # write WCOFS NetCDF files if they do not exist
    if not os.path.exists(nc_filenames['wcofs_sst_DA']) or not os.path.exists(
            nc_filenames['wcofs_u_DA']) or not os.path.exists(nc_filenames['wcofs_v_DA']):
        from dataset import wcofs

        wcofs_range = wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg')

        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range.to_netcdf(nc_filenames['wcofs_sst_DA'], variables=['temp'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_u_DA'], variables=['u'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_v_DA'], variables=['v'])

    if not os.path.exists(nc_filenames['wcofs_u_noDA']) or not os.path.exists(
            nc_filenames['wcofs_v_noDA']) or not os.path.exists(nc_filenames['wcofs_sst_noDA']):
        from dataset import wcofs

        wcofs_range_noDA = wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg',
                                             grid_filename=wcofs.WCOFS_4KM_GRID_FILENAME,
                                             source_url=os.path.join(DATA_DIR, 'input', 'wcofs', 'avg'),
                                             wcofs_string='wcofs4')

        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_sst_noDA'], variables=['temp'])
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_u_noDA'], variables=['u'])
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_v_noDA'], variables=['v'])

    # load datasets from local NetCDF files
    datasets = {'hfr': None, 'viirs': None, 'wcofs_sst_DA': None, 'wcofs_u_DA': None, 'wcofs_v_DA': None,
                'wcofs_sst_noDA': None, 'wcofs_u_noDA': None, 'wcofs_v_noDA': None}
    for dataset in datasets:
        datasets[dataset] = xarray.open_dataset(nc_filenames[dataset])

    # get dimensions of WCOFS dataset (time delta, eta, rho)
    wcofs_dimensions = {'sst': list(datasets['wcofs_sst_DA']['temp'].coords),
                        'u': list(datasets['wcofs_u_DA']['u'].coords), 'v': list(datasets['wcofs_v_DA']['v'].coords)}

    # interpolate nearest-neighbor observational data onto WCOFS grid
    data = {'obser': {'sst': datasets['viirs']['sst'].values, 'u': datasets['hfr']['u'].values,
                      'v': datasets['hfr']['v'].values},
            'DA_model': {'sst': {}, 'u': {}, 'v': {}},
            'noDA_model': {'sst': {}, 'u': {}, 'v': {}}}

    time_deltas = dict(zip(range(len(datasets['wcofs_sst_DA'][wcofs_dimensions['sst'][0]].values)),
                           sorted(datasets['wcofs_sst_DA'][wcofs_dimensions['sst'][0]].values, reverse=True)))

    if len(os.listdir(os.path.join(day_dir, 'residuals'))) == 0:
        print('interpolating WCOFS data onto observational grids...')
        with futures.ThreadPoolExecutor() as concurrency_pool:
            sst_futures = {'DA_model': {}, 'noDA_model': {}}
            u_futures = {'DA_model': {}, 'noDA_model': {}}
            v_futures = {'DA_model': {}, 'noDA_model': {}}

            for time_delta_index, time_delta in time_deltas.items():
                try:
                    sst_DA_future = concurrency_pool.submit(interpolate_grid,
                                                            datasets['wcofs_sst_DA']['lon'].values,
                                                            datasets['wcofs_sst_DA']['lat'].values,
                                                            datasets['wcofs_sst_DA']['temp'][time_delta_index, :,
                                                            :].values,
                                                            datasets['viirs']['lon'].values,
                                                            datasets['viirs']['lat'].values,
                                                            'linear')

                    sst_noDA_future = concurrency_pool.submit(interpolate_grid,
                                                              datasets['wcofs_sst_noDA']['lon'].values,
                                                              datasets['wcofs_sst_noDA']['lat'].values,
                                                              datasets['wcofs_sst_noDA']['temp'][time_delta_index, :,
                                                              :].values,
                                                              datasets['viirs']['lon'].values,
                                                              datasets['viirs']['lat'].values,
                                                              'linear')

                    u_DA_future = concurrency_pool.submit(interpolate_grid,
                                                          datasets['wcofs_u_DA']['lon'].values,
                                                          datasets['wcofs_u_DA']['lat'].values,
                                                          datasets['wcofs_u_DA']['u'][time_delta_index, :, :].values,
                                                          datasets['hfr']['lon'].values,
                                                          datasets['hfr']['lat'].values,
                                                          'linear')

                    u_noDA_future = concurrency_pool.submit(interpolate_grid,
                                                            datasets['wcofs_u_noDA']['lon'].values,
                                                            datasets['wcofs_u_noDA']['lat'].values,
                                                            datasets['wcofs_u_noDA']['u'][time_delta_index, :,
                                                            :].values,
                                                            datasets['hfr']['lon'].values,
                                                            datasets['hfr']['lat'].values,
                                                            'linear')

                    v_DA_future = concurrency_pool.submit(interpolate_grid,
                                                          datasets['wcofs_v_DA']['lon'].values,
                                                          datasets['wcofs_v_DA']['lat'].values,
                                                          datasets['wcofs_v_DA']['v'][time_delta_index, :, :].values,
                                                          datasets['hfr']['lon'].values,
                                                          datasets['hfr']['lat'].values,
                                                          'linear')

                    v_noDA_future = concurrency_pool.submit(interpolate_grid,
                                                            datasets['wcofs_v_noDA']['lon'].values,
                                                            datasets['wcofs_v_noDA']['lat'].values,
                                                            datasets['wcofs_v_noDA']['v'][time_delta_index, :,
                                                            :].values,
                                                            datasets['hfr']['lon'].values,
                                                            datasets['hfr']['lat'].values,
                                                            'linear')

                    sst_futures['DA_model'][sst_DA_future] = time_delta
                    sst_futures['noDA_model'][sst_noDA_future] = time_delta
                    u_futures['DA_model'][u_DA_future] = time_delta
                    u_futures['noDA_model'][u_noDA_future] = time_delta
                    v_futures['DA_model'][v_DA_future] = time_delta
                    v_futures['noDA_model'][v_noDA_future] = time_delta
                except IndexError as error:
                    print(f'{time_delta} IndexError: {error}')
                    continue

            for completed_future in futures.as_completed(sst_futures['DA_model']):
                time_delta = sst_futures['DA_model'][completed_future]
                data['DA_model']['sst'][time_delta] = completed_future.result()

            for completed_future in futures.as_completed(sst_futures['noDA_model']):
                time_delta = sst_futures['noDA_model'][completed_future]
                data['noDA_model']['sst'][time_delta] = completed_future.result()

            del sst_futures

            for completed_future in futures.as_completed(u_futures['DA_model']):
                time_delta = u_futures['DA_model'][completed_future]
                data['DA_model']['u'][time_delta] = completed_future.result()

            for completed_future in futures.as_completed(u_futures['noDA_model']):
                time_delta = u_futures['noDA_model'][completed_future]
                data['noDA_model']['u'][time_delta] = completed_future.result()

            del u_futures

            for completed_future in futures.as_completed(v_futures['DA_model']):
                time_delta = v_futures['DA_model'][completed_future]
                data['DA_model']['v'][time_delta] = completed_future.result()

            for completed_future in futures.as_completed(v_futures['noDA_model']):
                time_delta = v_futures['noDA_model'][completed_future]
                data['noDA_model']['v'][time_delta] = completed_future.result()

            del v_futures

        # write datasets to NetCDF files
        for time_delta in time_deltas.values():
            try:
                sst_DA_residuals = data['obser']['sst'] - data['DA_model']['sst'][time_delta]
                sst_noDA_residuals = data['obser']['sst'] - data['noDA_model']['sst'][time_delta]
                u_DA_residuals = data['obser']['u'] - data['DA_model']['u'][time_delta]
                u_noDA_residuals = data['obser']['u'] - data['noDA_model']['u'][time_delta]
                v_DA_residuals = data['obser']['v'] - data['DA_model']['v'][time_delta]
                v_noDA_residuals = data['obser']['v'] - data['noDA_model']['v'][time_delta]

                sst_DA_dataarray = xarray.DataArray(sst_DA_residuals, coords=datasets['viirs']['sst'].coords)
                sst_noDA_dataarray = xarray.DataArray(sst_noDA_residuals, coords=datasets['viirs']['sst'].coords)
                u_DA_dataarray = xarray.DataArray(u_DA_residuals, coords=datasets['hfr']['u'].coords)
                u_noDA_dataarray = xarray.DataArray(u_noDA_residuals, coords=datasets['hfr']['u'].coords)
                v_DA_dataarray = xarray.DataArray(v_DA_residuals, coords=datasets['hfr']['v'].coords)
                v_noDA_dataarray = xarray.DataArray(v_noDA_residuals, coords=datasets['hfr']['v'].coords)

                dataset_path = os.path.join(residuals_dir, f'residuals_{time_delta}.nc')

                residuals_dataset = xarray.Dataset(
                    {'sst_DA': sst_DA_dataarray, 'u_DA': u_DA_dataarray, 'v_DA': v_DA_dataarray,
                     'sst_noDA': sst_noDA_dataarray, 'u_noDA': u_noDA_dataarray, 'v_noDA': v_noDA_dataarray})
                residuals_dataset.to_netcdf(dataset_path)
            except KeyError as error:
                print(f'{time_delta} KeyError: {error}')

    metrics = {}

    # get total sum of squared variance
    sst_tot_sumsq = numpy.nansum(numpy.square(data['obser']['sst'] - numpy.nanmean(data['obser']['sst'])))
    u_tot_sumsq = numpy.nansum(numpy.square(data['obser']['u'] - numpy.nanmean(data['obser']['u'])))
    v_tot_sumsq = numpy.nansum(numpy.square(data['obser']['v'] - numpy.nanmean(data['obser']['v'])))

    for time_delta in time_deltas.values():
        time_delta_metrics = {}

        dataset_path = os.path.join(residuals_dir, f'residuals_{time_delta}.nc')

        try:
            residuals_dataset = xarray.open_dataset(dataset_path)
        except FileNotFoundError as error:
            print(f'{time_delta} FileNotFoundError: {error}')
            continue

        sst_DA_residuals = residuals_dataset['sst_DA'].values
        sst_noDA_residuals = residuals_dataset['sst_noDA'].values
        u_DA_residuals = residuals_dataset['u_DA'].values
        u_noDA_residuals = residuals_dataset['u_noDA'].values
        v_DA_residuals = residuals_dataset['v_DA'].values
        v_noDA_residuals = residuals_dataset['v_noDA'].values

        # get squared residuals
        sst_DA_res_sq = numpy.square(sst_DA_residuals)
        sst_noDA_res_sq = numpy.square(sst_DA_residuals)
        u_DA_res_sq = numpy.square(u_DA_residuals)
        u_noDA_res_sq = numpy.square(u_noDA_residuals)
        v_DA_res_sq = numpy.square(v_DA_residuals)
        v_noDA_res_sq = numpy.square(v_noDA_residuals)

        # get root mean squared error
        sst_DA_rmse = numpy.sqrt(numpy.nanmean(sst_DA_res_sq))
        sst_noDA_rmse = numpy.sqrt(numpy.nanmean(sst_noDA_res_sq))
        u_DA_rmse = numpy.sqrt(numpy.nanmean(u_DA_res_sq))
        u_noDA_rmse = numpy.sqrt(numpy.nanmean(u_noDA_res_sq))
        v_DA_rmse = numpy.sqrt(numpy.nanmean(v_DA_res_sq))
        v_noDA_rmse = numpy.sqrt(numpy.nanmean(v_noDA_res_sq))

        # get sum of residuals
        sst_DA_res_sumsq = numpy.nansum(sst_DA_res_sq)
        sst_noDA_res_sumsq = numpy.nansum(sst_noDA_res_sq)
        u_DA_res_sumsq = numpy.nansum(u_DA_res_sq)
        u_noDA_res_sumsq = numpy.nansum(u_noDA_res_sq)
        v_DA_res_sumsq = numpy.nansum(v_DA_res_sq)
        v_noDA_res_sumsq = numpy.nansum(v_noDA_res_sq)

        # get coefficient of determination (R^2)
        sst_DA_rsq = 1 - (sst_DA_res_sumsq / sst_tot_sumsq)
        sst_noDA_rsq = 1 - (sst_noDA_res_sumsq / sst_tot_sumsq)
        u_DA_rsq = 1 - (u_DA_res_sumsq / u_tot_sumsq)
        u_noDA_rsq = 1 - (u_noDA_res_sumsq / u_tot_sumsq)
        v_DA_rsq = 1 - (v_DA_res_sumsq / v_tot_sumsq)
        v_noDA_rsq = 1 - (v_noDA_res_sumsq / v_tot_sumsq)

        time_delta_metrics['sst_DA_rmse'] = sst_DA_rmse
        time_delta_metrics['sst_noDA_rmse'] = sst_noDA_rmse
        time_delta_metrics['u_DA_rmse'] = u_DA_rmse
        time_delta_metrics['u_noDA_rmse'] = u_noDA_rmse
        time_delta_metrics['v_DA_rmse'] = v_DA_rmse
        time_delta_metrics['v_noDA_rmse'] = v_noDA_rmse

        time_delta_metrics['sst_DA_rsq'] = sst_DA_rsq
        time_delta_metrics['sst_noDA_rsq'] = sst_noDA_rsq
        time_delta_metrics['u_DA_rsq'] = u_DA_rsq
        time_delta_metrics['u_noDA_rsq'] = u_noDA_rsq
        time_delta_metrics['v_DA_rsq'] = v_DA_rsq
        time_delta_metrics['v_noDA_rsq'] = v_noDA_rsq

        metrics[time_delta] = time_delta_metrics

    print('done')
