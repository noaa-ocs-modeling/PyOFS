from concurrent import futures
import datetime
import os
import sys

import numpy
import xarray

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from main import DATA_DIR
from dataset.wcofs import interpolate_grid

WORKSPACE_DIR = os.path.join(DATA_DIR, 'validation')

# UTC offset of study area
UTC_OFFSET = 8


def to_netcdf(start_datetime: datetime.datetime, end_datetime: datetime.datetime, output_dir: str):
    """
    Writes HFR, VIIRS, and WCOFS data to NetCDF files at the given filenames.

    :param start_datetime: Start of time interval.
    :param end_datetime: End of time interval.
    :param output_dir: Output directory.
    """
    
    # define filenames for NetCDF
    nc_filenames = {
        'hfr': os.path.join(output_dir, 'hfr.nc'),
        'viirs': os.path.join(output_dir, 'viirs.nc'),
        'wcofs_sst_noDA': os.path.join(output_dir, 'wcofs_sst_noDA.nc'),
        'wcofs_sst_DA': os.path.join(output_dir, 'wcofs_sst_DA.nc'),
        'wcofs_u_noDA': os.path.join(output_dir, 'wcofs_u_noDA.nc'),
        'wcofs_u_DA': os.path.join(output_dir, 'wcofs_u_DA.nc'),
        'wcofs_v_noDA': os.path.join(output_dir, 'wcofs_v_noDA.nc'),
        'wcofs_v_DA': os.path.join(output_dir, 'wcofs_v_DA.nc')
    }
    
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
    
    if not os.path.exists(nc_filenames['wcofs_sst_DA']) or not os.path.exists(
            nc_filenames['wcofs_u_DA']) or not os.path.exists(nc_filenames['wcofs_v_DA']):
        from dataset import wcofs
        
        wcofs_range = wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg')
        
        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range.to_netcdf(nc_filenames['wcofs_sst_DA'], variables=['temp'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_u_DA'], variables=['u'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_v_DA'], variables=['v'])


def from_netcdf(input_dir: str) -> dict:
    """
    Read NetCDF files from directory.
    
    :param input_dir: Directory with NetCDF files.
    :return: Dictionary of xarray Dataset objects.
    """
    
    # define filenames for NetCDF
    nc_filenames = {
        'hfr': os.path.join(input_dir, 'hfr.nc'),
        'viirs': os.path.join(input_dir, 'viirs.nc'),
        'wcofs_sst_noDA': os.path.join(input_dir, 'wcofs_sst_noDA.nc'),
        'wcofs_sst_DA': os.path.join(input_dir, 'wcofs_sst_DA.nc'),
        'wcofs_u_noDA': os.path.join(input_dir, 'wcofs_u_noDA.nc'),
        'wcofs_u_DA': os.path.join(input_dir, 'wcofs_u_DA.nc'),
        'wcofs_v_noDA': os.path.join(input_dir, 'wcofs_v_noDA.nc'),
        'wcofs_v_DA': os.path.join(input_dir, 'wcofs_v_DA.nc')
    }
    
    # load datasets from local NetCDF files
    return {dataset: xarray.open_dataset(nc_filenames[dataset]) for dataset in nc_filenames}


def interpolate_grids(datasets: dict) -> dict:
    """
    Interpolate model grids onto observational grids.
    
    :param datasets: Dictionary of xarray Dataset objects.
    :return: Dictioanry of interpolated values.
    """
    
    data = {
        'noDA_model': {'sst': {}, 'u': {}, 'v': {}},
        'DA_model': {'sst': {}, 'u': {}, 'v': {}}
    }
    
    # get dimensions of WCOFS dataset (time delta, eta, rho)
    wcofs_dimensions = {
        'sst': list(datasets['wcofs_sst_noDA']['temp'].coords),
        'u': list(datasets['wcofs_u_noDA']['u'].coords),
        'v': list(datasets['wcofs_v_DA']['v'].coords)
    }
    
    with futures.ThreadPoolExecutor() as concurrency_pool:
        sst_futures = {'noDA_model': {}, 'DA_model': {}}
        u_futures = {'noDA_model': {}, 'DA_model': {}}
        v_futures = {'noDA_model': {}, 'DA_model': {}}
        
        for time_delta_index, time_delta in dict(
                zip(range(len(datasets['wcofs_sst_noDA'][wcofs_dimensions['sst'][0]].values)),
                    sorted(datasets['wcofs_sst_noDA'][wcofs_dimensions['sst'][0]].values, reverse=True))).items():
            try:
                
                sst_noDA_future = concurrency_pool.submit(interpolate_grid,
                                                          datasets['wcofs_sst_noDA']['lon'].values,
                                                          datasets['wcofs_sst_noDA']['lat'].values,
                                                          datasets['wcofs_sst_noDA']['temp'][time_delta_index, :,
                                                          :].values,
                                                          datasets['viirs']['lon'].values,
                                                          datasets['viirs']['lat'].values,
                                                          'linear')
                
                sst_DA_future = concurrency_pool.submit(interpolate_grid,
                                                        datasets['wcofs_sst_DA']['lon'].values,
                                                        datasets['wcofs_sst_DA']['lat'].values,
                                                        datasets['wcofs_sst_DA']['temp'][time_delta_index, :,
                                                        :].values,
                                                        datasets['viirs']['lon'].values,
                                                        datasets['viirs']['lat'].values,
                                                        'linear')
                
                u_noDA_future = concurrency_pool.submit(interpolate_grid,
                                                        datasets['wcofs_u_noDA']['lon'].values,
                                                        datasets['wcofs_u_noDA']['lat'].values,
                                                        datasets['wcofs_u_noDA']['u'][time_delta_index, :,
                                                        :].values,
                                                        datasets['hfr']['lon'].values,
                                                        datasets['hfr']['lat'].values,
                                                        'linear')
                
                u_DA_future = concurrency_pool.submit(interpolate_grid,
                                                      datasets['wcofs_u_DA']['lon'].values,
                                                      datasets['wcofs_u_DA']['lat'].values,
                                                      datasets['wcofs_u_DA']['u'][time_delta_index, :, :].values,
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
                
                v_DA_future = concurrency_pool.submit(interpolate_grid,
                                                      datasets['wcofs_v_DA']['lon'].values,
                                                      datasets['wcofs_v_DA']['lat'].values,
                                                      datasets['wcofs_v_DA']['v'][time_delta_index, :, :].values,
                                                      datasets['hfr']['lon'].values,
                                                      datasets['hfr']['lat'].values,
                                                      'linear')
                
                sst_futures['noDA_model'][sst_noDA_future] = time_delta
                sst_futures['DA_model'][sst_DA_future] = time_delta
                u_futures['noDA_model'][u_noDA_future] = time_delta
                u_futures['DA_model'][u_DA_future] = time_delta
                v_futures['noDA_model'][v_noDA_future] = time_delta
                v_futures['DA_model'][v_DA_future] = time_delta
            except IndexError as error:
                print(f'{time_delta} IndexError: {error}')
                continue
        
        for completed_future in futures.as_completed(sst_futures['noDA_model']):
            time_delta = sst_futures['noDA_model'][completed_future]
            data['noDA_model']['sst'][time_delta] = completed_future.result()
        
        for completed_future in futures.as_completed(sst_futures['DA_model']):
            time_delta = sst_futures['DA_model'][completed_future]
            data['DA_model']['sst'][time_delta] = completed_future.result()
        
        del sst_futures
        
        for completed_future in futures.as_completed(u_futures['noDA_model']):
            time_delta = u_futures['noDA_model'][completed_future]
            data['noDA_model']['u'][time_delta] = completed_future.result()
        
        for completed_future in futures.as_completed(u_futures['DA_model']):
            time_delta = u_futures['DA_model'][completed_future]
            data['DA_model']['u'][time_delta] = completed_future.result()
        
        del u_futures
        
        for completed_future in futures.as_completed(v_futures['noDA_model']):
            time_delta = v_futures['noDA_model'][completed_future]
            data['noDA_model']['v'][time_delta] = completed_future.result()
        
        for completed_future in futures.as_completed(v_futures['DA_model']):
            time_delta = v_futures['DA_model'][completed_future]
            data['DA_model']['v'][time_delta] = completed_future.result()
        
        del v_futures
    
    return data


def rmse(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    Calculate root-mean-square error (RMSE) given observational data and model output.
    
    :param x: Array of observational data.
    :param y: Array of model output (in same grid as x).
    :return: Root-mean-square error.
    """
    
    squared_residuals = numpy.square(x - y)
    return numpy.sqrt(numpy.nanmean(squared_residuals))


def r_squ(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    Calculate determination coefficient (R^2) given observational data and model output.
    R^2 is a value from 0 to 1 of the amount of observational variance explained by the model.

    :param x: Array of observational data.
    :param y: Array of model output (in same grid as x).
    :return: Determination coefficient.
    """
    
    sum_of_squares_of_residuals = numpy.nansum(numpy.square(x - y))
    sum_of_squares_of_variance = numpy.nansum(numpy.square(x - numpy.nanmean(x)))
    return 1 - (sum_of_squares_of_residuals / sum_of_squares_of_variance)


if __name__ == '__main__':
    # setup start and ending times
    start_datetime = datetime.datetime(2018, 11, 11)
    end_datetime = start_datetime + datetime.timedelta(days=1)
    
    day_dir = os.path.join(WORKSPACE_DIR, start_datetime.strftime('%Y%m%d'))
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    to_netcdf(start_datetime, end_datetime, day_dir)

    datasets = from_netcdf(day_dir)
    
    # interpolate nearest-neighbor observational data onto WCOFS grid
    data = {'obser': {
        'sst': datasets['viirs']['sst'].values,
        'u': datasets['hfr']['u'].values,
        'v': datasets['hfr']['v'].values}
    }
    
    print('interpolating WCOFS data onto observational grids...')
    data.update(interpolate_grids(datasets))

    time_deltas = data['DA_model']['sst'].keys()
    
    metrics = {}
    
    for time_delta in time_deltas:
        metrics[time_delta] = {
            'rmse': {
                'sst': {
                    'noDA': rmse(data['obser']['sst'], data['noDA_model']['sst'][time_delta]),
                    'DA': rmse(data['obser']['sst'], data['DA_model']['sst'][time_delta])
                },
                'u': {
                    'noDA': rmse(data['obser']['u'], data['noDA_model']['u'][time_delta]),
                    'DA': rmse(data['obser']['u'], data['DA_model']['u'][time_delta])
                },
                'v': {
                    'noDA': rmse(data['obser']['v'], data['noDA_model']['v'][time_delta]),
                    'DA': rmse(data['obser']['v'], data['DA_model']['v'][time_delta])
                }
            },
            'r_squ': {
                'sst': {
                    'noDA': r_squ(data['obser']['sst'], data['noDA_model']['sst'][time_delta]),
                    'DA': r_squ(data['obser']['sst'], data['DA_model']['sst'][time_delta])
                },
                'u': {
                    'noDA': r_squ(data['obser']['u'], data['noDA_model']['u'][time_delta]),
                    'DA': r_squ(data['obser']['u'], data['DA_model']['u'][time_delta])
                },
                'v': {
                    'noDA': r_squ(data['obser']['v'], data['noDA_model']['v'][time_delta]),
                    'DA': r_squ(data['obser']['v'], data['DA_model']['v'][time_delta])
                }
            }
        }
    
    for time_delta, methods in metrics.items():
        print(f'{time_delta}:')
        for metric, variables in methods.items():
            print(f'{metric}:')
            for variable, assimilations in variables.items():
                no_da_metric = assimilations["noDA"]
                da_metric = assimilations["DA"]
                
                print(f'{no_da_metric:5.2f} -> {da_metric:5.2f}: {da_metric - no_da_metric: 5.2f}')
    
    print('done')
