from concurrent import futures
from datetime import datetime, timedelta
import os
from os import PathLike
from pathlib import Path

import numpy
import xarray

from PyOFS import DATA_DIRECTORY, get_logger
from PyOFS.model import wcofs
from PyOFS.observation import hf_radar, viirs

LOGGER = get_logger('PyOFS.valid')

WORKSPACE_DIR = DATA_DIRECTORY / 'validation'

# UTC offset of study area
UTC_OFFSET = 8


def to_netcdf(start_time: datetime, end_time: datetime, output_dir: PathLike):
    """
    Writes HFR, VIIRS, and WCOFS data to NetCDF files at the given filenames.

    :param start_time: Start of time interval.
    :param end_time: End of time interval.
    :param output_dir: Output directory.
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # define filenames for NetCDF
    nc_filenames = {
        'hfr': output_dir / 'hfr.nc',
        'viirs': output_dir / 'viirs.nc',
        'wcofs_sst_noDA': output_dir / 'wcofs_sst_noDA.nc',
        'wcofs_sst_DA': output_dir / 'wcofs_sst_DA.nc',
        'wcofs_u_noDA': output_dir / 'wcofs_u_noDA.nc',
        'wcofs_u_DA': output_dir / 'wcofs_u_DA.nc',
        'wcofs_v_noDA': output_dir / 'wcofs_v_noDA.nc',
        'wcofs_v_DA': output_dir / 'wcofs_v_DA.nc',
    }

    # write HFR NetCDF file if it does not exist
    if not nc_filenames['hfr'].exists():
        hfr_range = hf_radar.HFRadarRange(start_time, end_time)
        hfr_range.to_netcdf(nc_filenames['hfr'])

    # write VIIRS NetCDF file if it does not exist
    if not nc_filenames['viirs'].exists():
        utc_start_time = start_time + timedelta(hours=UTC_OFFSET)
        utc_end_time = end_time + timedelta(hours=UTC_OFFSET)

        viirs_range = viirs.VIIRSRange(utc_start_time, utc_end_time)
        viirs_range.to_netcdf(nc_filenames['viirs'], variables=['sst'])

    # write WCOFS NetCDF files if they do not exist
    if (
            not nc_filenames['wcofs_u_noDA'].exists()
            or not nc_filenames['wcofs_v_noDA'].exists()
            or not nc_filenames['wcofs_sst_noDA'].exists()
    ):
        wcofs_range_noDA = wcofs.WCOFSRange(
            start_time,
            end_time,
            source='avg',
            grid_filename=wcofs.WCOFS_4KM_GRID_FILENAME,
            source_url=DATA_DIRECTORY / 'input' / 'wcofs' / 'avg',
            wcofs_string='wcofs4',
        )

        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_sst_noDA'], variables=['temp'])
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_u_noDA'], variables=['u'])
        wcofs_range_noDA.to_netcdf(nc_filenames['wcofs_v_noDA'], variables=['v'])

    if (
            not nc_filenames['wcofs_sst_DA'].exists()
            or not nc_filenames['wcofs_u_DA'].exists()
            or not nc_filenames['wcofs_v_DA'].exists()
    ):
        wcofs_range = wcofs.WCOFSRange(start_time, end_time, source='avg')

        # TODO find a way to combine WCOFS variables without raising MemoryError
        wcofs_range.to_netcdf(nc_filenames['wcofs_sst_DA'], variables=['temp'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_u_DA'], variables=['u'])
        wcofs_range.to_netcdf(nc_filenames['wcofs_v_DA'], variables=['v'])


def from_netcdf(input_dir: PathLike) -> dict:
    """
    Read NetCDF files from directory.
    
    :param input_dir: Directory with NetCDF files.
    :return: Dictionary of xarray Dataset objects.
    """

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    # define filenames for NetCDF
    nc_filenames = {
        'hfr': input_dir / 'hfr.nc',
        'viirs': input_dir / 'viirs.nc',
        'wcofs_sst_noDA': input_dir / 'wcofs_sst_noDA.nc',
        'wcofs_sst_DA': input_dir / 'wcofs_sst_DA.nc',
        'wcofs_u_noDA': input_dir / 'wcofs_u_noDA.nc',
        'wcofs_u_DA': input_dir / 'wcofs_u_DA.nc',
        'wcofs_v_noDA': input_dir / 'wcofs_v_noDA.nc',
        'wcofs_v_DA': input_dir / 'wcofs_v_DA.nc',
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
        'DA_model': {'sst': {}, 'u': {}, 'v': {}},
    }

    # get dimensions of WCOFS observation (time delta, eta, rho)
    wcofs_dimensions = {
        'sst': list(datasets['wcofs_sst_noDA']['temp'].coords),
        'u': list(datasets['wcofs_u_noDA']['u'].coords),
        'v': list(datasets['wcofs_v_DA']['v'].coords),
    }

    with futures.ThreadPoolExecutor() as concurrency_pool:
        sst_futures = {'noDA_model': {}, 'DA_model': {}}
        u_futures = {'noDA_model': {}, 'DA_model': {}}
        v_futures = {'noDA_model': {}, 'DA_model': {}}

        for time_delta_index, time_delta in enumerate(
                sorted(datasets['wcofs_sst_noDA'][wcofs_dimensions['sst'][0]].values, reverse=True)
        ):
            try:

                sst_noDA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_sst_noDA']['lon'].values,
                    datasets['wcofs_sst_noDA']['lat'].values,
                    datasets['wcofs_sst_noDA']['temp'][time_delta_index, :, :].values,
                    datasets['viirs']['lon'].values,
                    datasets['viirs']['lat'].values,
                    'linear',
                )

                sst_DA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_sst_DA']['lon'].values,
                    datasets['wcofs_sst_DA']['lat'].values,
                    datasets['wcofs_sst_DA']['temp'][time_delta_index, :, :].values,
                    datasets['viirs']['lon'].values,
                    datasets['viirs']['lat'].values,
                    'linear',
                )

                u_noDA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_u_noDA']['lon'].values,
                    datasets['wcofs_u_noDA']['lat'].values,
                    datasets['wcofs_u_noDA']['u'][time_delta_index, :, :].values,
                    datasets['hfr']['lon'].values,
                    datasets['hfr']['lat'].values,
                    'linear',
                )

                u_DA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_u_DA']['lon'].values,
                    datasets['wcofs_u_DA']['lat'].values,
                    datasets['wcofs_u_DA']['u'][time_delta_index, :, :].values,
                    datasets['hfr']['lon'].values,
                    datasets['hfr']['lat'].values,
                    'linear',
                )

                v_noDA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_v_noDA']['lon'].values,
                    datasets['wcofs_v_noDA']['lat'].values,
                    datasets['wcofs_v_noDA']['v'][time_delta_index, :, :].values,
                    datasets['hfr']['lon'].values,
                    datasets['hfr']['lat'].values,
                    'linear',
                )

                v_DA_future = concurrency_pool.submit(
                    wcofs.interpolate_grid,
                    datasets['wcofs_v_DA']['lon'].values,
                    datasets['wcofs_v_DA']['lat'].values,
                    datasets['wcofs_v_DA']['v'][time_delta_index, :, :].values,
                    datasets['hfr']['lon'].values,
                    datasets['hfr']['lat'].values,
                    'linear',
                )

                sst_futures['noDA_model'][sst_noDA_future] = time_delta
                sst_futures['DA_model'][sst_DA_future] = time_delta
                u_futures['noDA_model'][u_noDA_future] = time_delta
                u_futures['DA_model'][u_DA_future] = time_delta
                v_futures['noDA_model'][v_noDA_future] = time_delta
                v_futures['DA_model'][v_DA_future] = time_delta
            except IndexError as error:
                LOGGER.warning(f'{error.__class__.__name__}: {error}')
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


def rmse(x: numpy.array, y: numpy.array) -> float:
    """
    Calculate root-mean-square error (RMSE) given observational data and model output.
    
    :param x: Array of observational data.
    :param y: Array of model output (in same grid as x).
    :return: Root-mean-square error.
    """

    squared_residuals = numpy.square(x - y)
    return numpy.sqrt(numpy.nanmean(squared_residuals))


def r_squ(x: numpy.array, y: numpy.array) -> float:
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
    start_time = datetime(2018, 11, 11)
    end_time = start_time + timedelta(days=1)

    day_dir = WORKSPACE_DIR / f'{start_time:%Y%m%d}'
    if not day_dir.exists():
        os.makedirs(day_dir, exist_ok=True)

    to_netcdf(start_time, end_time, day_dir)

    datasets = from_netcdf(day_dir)

    # interpolate nearest-neighbor observational data onto WCOFS grid
    data = {
        'obser': {
            'sst': datasets['viirs']['sst'].values,
            'u': datasets['hfr']['u'].values,
            'v': datasets['hfr']['v'].values,
        }
    }

    LOGGER.info('interpolating WCOFS data onto observational grids...')
    data.update(interpolate_grids(datasets))

    time_deltas = data['DA_model']['sst'].keys()

    metrics = {}

    for time_delta in time_deltas:
        metrics[time_delta] = {
            'rmse': {
                'sst': {
                    'noDA': rmse(data['obser']['sst'], data['noDA_model']['sst'][time_delta]),
                    'DA': rmse(data['obser']['sst'], data['DA_model']['sst'][time_delta]),
                },
                'u': {
                    'noDA': rmse(data['obser']['u'], data['noDA_model']['u'][time_delta]),
                    'DA': rmse(data['obser']['u'], data['DA_model']['u'][time_delta]),
                },
                'v': {
                    'noDA': rmse(data['obser']['v'], data['noDA_model']['v'][time_delta]),
                    'DA': rmse(data['obser']['v'], data['DA_model']['v'][time_delta]),
                },
            },
            'r_squ': {
                'sst': {
                    'noDA': r_squ(data['obser']['sst'], data['noDA_model']['sst'][time_delta]),
                    'DA': r_squ(data['obser']['sst'], data['DA_model']['sst'][time_delta]),
                },
                'u': {
                    'noDA': r_squ(data['obser']['u'], data['noDA_model']['u'][time_delta]),
                    'DA': r_squ(data['obser']['u'], data['DA_model']['u'][time_delta]),
                },
                'v': {
                    'noDA': r_squ(data['obser']['v'], data['noDA_model']['v'][time_delta]),
                    'DA': r_squ(data['obser']['v'], data['DA_model']['v'][time_delta]),
                },
            },
        }

    for time_delta, methods in metrics.items():
        LOGGER.info(f'{time_delta}:')
        for metric, variables in methods.items():
            LOGGER.info(f'{metric}:')
            for variable, assimilations in variables.items():
                no_da_metric = assimilations['noDA']
                da_metric = assimilations['DA']

                LOGGER.info(
                    f'{no_da_metric:5.2f} -> {da_metric:5.2f}: {da_metric - no_da_metric: 5.2f}'
                )

    print('done')
