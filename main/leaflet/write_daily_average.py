# coding=utf-8
"""
Daily average WCOFS source data.

Created on Aug 21, 2018

@author: zachary.burnett
"""

import datetime
import logging
import os
import sys
from typing import Collection, Union

import pytz
from leaflet import azure, write_json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from PyOFS import DATA_DIR, LEAFLET_NODATA_VALUE
from PyOFS.observation import hf_radar, viirs, smap, data_buoy
from PyOFS.model import wcofs, rtofs

LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference')

# offset from study area to UTC
STUDY_AREA_TIMEZONE = 'US/Pacific'
STUDY_AREA_TO_UTC = datetime.timedelta(
    hours=-datetime.datetime.now(pytz.timezone(STUDY_AREA_TIMEZONE)).utcoffset().total_seconds() / 3600)

# range of day deltas that models reach
MODEL_DAY_DELTAS = {'WCOFS': range(-1, 3), 'RTOFS': range(-3, 9)}


def write_observation(output_dir: str, observation_date: Union[datetime.datetime, datetime.date],
                      observation: str):
    """
    Writes daily average of observational data on given date.

    :param output_dir: output directory to write files
    :param observation_date: fate of observation
    :param observation: observation to write
    :raise _utilities.NoDataError: if no data found
    """

    if type(observation_date) is datetime.date:
        day_start = datetime.datetime.combine(observation_date, datetime.time.min)
    elif type(observation_date) is datetime.datetime:
        day_start = observation_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        day_start = observation_date

    day_noon = day_start + datetime.timedelta(hours=12)
    day_end = day_start + datetime.timedelta(days=1)

    if observation is 'smap':
        monthly_dir = os.path.join(output_dir, 'monthly_averages')

        if not os.path.isdir(monthly_dir):
            os.mkdir(monthly_dir)

        observation_dir = os.path.join(monthly_dir, observation_date.strftime('%Y%m'))
    else:
        daily_dir = os.path.join(output_dir, 'daily_averages')

        if not os.path.isdir(daily_dir):
            os.mkdir(daily_dir)

        observation_dir = os.path.join(daily_dir, observation_date.strftime('%Y%m%d'))

    if not os.path.isdir(observation_dir):
        os.mkdir(observation_dir)

    day_start_ndbc = day_start + datetime.timedelta(hours=2)
    day_end_ndbc = day_end + datetime.timedelta(hours=2)

    day_start_utc = day_start + STUDY_AREA_TO_UTC
    day_noon_utc = day_noon + STUDY_AREA_TO_UTC
    day_end_utc = day_end + STUDY_AREA_TO_UTC

    try:
        if observation == 'hf_radar':
            hfr_range = hf_radar.HFRadarRange(day_start_ndbc, day_end_ndbc)
            hfr_range.write_rasters(observation_dir, filename_suffix=f'{observation_date.strftime("%Y%m%d")}',
                                    variables=['dir', 'mag'], driver='AAIGrid', fill_value=LEAFLET_NODATA_VALUE,
                                    dop_threshold=0.5)
            del hfr_range
        elif observation == 'viirs':
            viirs_range = viirs.VIIRSRange(day_start, day_end_utc)
            viirs_range.write_raster(observation_dir,
                                     filename_suffix=day_start.strftime('%Y%m%d%H%M') + '_' + day_noon.strftime(
                                         '%Y%m%d%H%M'),
                                     start_time=day_start_utc, end_time=day_noon_utc,
                                     fill_value=LEAFLET_NODATA_VALUE, driver='GTiff', correct_sses=False,
                                     variables=['sst'])
            viirs_range.write_raster(observation_dir,
                                     filename_suffix=day_noon.strftime('%Y%m%d%H%M') + '_' + day_end.strftime(
                                         '%Y%m%d%H%M'),
                                     start_time=day_noon_utc, end_time=day_end_utc,
                                     fill_value=LEAFLET_NODATA_VALUE, driver='GTiff', correct_sses=False,
                                     variables=['sst'])
            del viirs_range
        elif observation == 'smap':
            smap_dataset = smap.SMAPDataset()
            smap_dataset.write_rasters(observation_dir, data_time=day_start_utc, fill_value=LEAFLET_NODATA_VALUE,
                                       driver='GTiff', variables=['sss'])
            del smap_dataset
        elif observation == 'data_buoy':
            data_buoy_range = data_buoy.DataBuoyRange(data_buoy.WCOFS_NDBC_STATIONS_FILENAME)
            output_filename = os.path.join(observation_dir,
                                           f'ndbc_data_buoys_{observation_date.strftime("%Y%m%d")}.gpkg')
            data_buoy_range.write_vector(output_filename, day_start_ndbc, day_end_ndbc)
            del data_buoy_range
    except Exception as error:
        error_type, exc_obj, error_traceback = sys.exc_info()
        filename = os.path.split(error_traceback.tb_frame.f_code.co_filename)[1]
        logging.warning(f'{error} ({filename}:{error_traceback.tb_lineno})')


def write_rtofs(output_dir: str, model_run_date: Union[datetime.datetime, datetime.date],
                day_deltas: range = MODEL_DAY_DELTAS['RTOFS'],
                scalar_variables: Collection[str] = ('sst', 'sss', 'ssh'),
                vector_variables: Collection[str] = ('dir', 'mag')):
    """
    Writes daily average of RTOFS output on given date.

    :param output_dir: output directory to write files
    :param model_run_date: date of model run
    :param day_deltas: time deltas for which to write model output
    :param scalar_variables: list of scalar variables to use
    :param vector_variables: list of vector variables to use
    :raise _utilities.NoDataError: if no data found
    """

    if type(model_run_date) is datetime.date:
        model_run_date = datetime.datetime.combine(model_run_date, datetime.datetime.min.time())

    daily_dir = os.path.join(output_dir, 'daily_averages')

    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: os.path.join(daily_dir, (model_run_date + datetime.timedelta(days=day_delta)).strftime("%Y%m%d")) for
        day_delta in day_deltas
    }

    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.mkdir(daily_average_dir)

    try:
        rtofs_dataset = None

        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['RTOFS']:
                time_delta_string = f'{"f" if day_delta >= 0 else "n"}{abs(day_delta) + 1 if day_delta >= 0 else abs(day_delta):03}'

                day_of_forecast = model_run_date + datetime.timedelta(days=day_delta)

                existing_files = os.listdir(daily_average_dir)
                existing_files = [filename for filename in existing_files if
                                  'rtofs' in filename and time_delta_string in filename]

                if rtofs_dataset is None and not all(
                        any(variable in filename for filename in existing_files) for variable in
                        list(scalar_variables) + list(vector_variables)):
                    rtofs_dataset = rtofs.RTOFSDataset(model_run_date, source='2ds', time_interval='daily')

                scalar_variables_to_write = [variable for variable in scalar_variables if
                                             not any(variable in filename for filename in existing_files)]

                if len(scalar_variables_to_write) > 0:
                    rtofs_dataset.write_rasters(daily_average_dir, variables=scalar_variables_to_write,
                                                time=day_of_forecast,
                                                driver='GTiff')
                else:
                    logging.info(f'Skipping RTOFS day {day_delta} scalar variables')

                if not all(any(vector_variable in filename for filename in existing_files) for vector_variable in
                           vector_variables):
                    rtofs_dataset.write_rasters(daily_average_dir, variables=vector_variables, time=day_of_forecast,
                                                driver='AAIGrid')
                else:
                    logging.info(f'Skipping RTOFS day {day_delta} uv')
        del rtofs_dataset
    except Exception as error:
        error_type, exc_obj, error_traceback = sys.exc_info()
        filename = os.path.split(error_traceback.tb_frame.f_code.co_filename)[1]
        logging.warning(f'{error} ({filename}:{error_traceback.tb_lineno})')


def write_wcofs(output_dir: str, model_run_date: Union[datetime.datetime, datetime.date, int, float],
                day_deltas: range = MODEL_DAY_DELTAS['WCOFS'],
                scalar_variables: Collection[str] = ('sst', 'sss', 'ssh'),
                vector_variables: Collection[str] = ('dir', 'mag'), data_assimilation: bool = True,
                grid_size_km: int = 4):
    """
    Writes daily average of model output on given date.

    :param output_dir: output directory to write files
    :param model_run_date: date of model run
    :param day_deltas: time deltas for which to write model output
    :param scalar_variables: list of scalar variables to use
    :param vector_variables: list of vector variables to use
    :param data_assimilation: whether to retrieve model with data assimilation
    :param grid_size_km: cell size in km
    :raise _utilities.NoDataError: if no data found
    """

    if type(model_run_date) is datetime.date:
        model_run_date = datetime.datetime.combine(model_run_date, datetime.datetime.min.time())

    daily_dir = os.path.join(output_dir, 'daily_averages')

    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: os.path.join(daily_dir, (model_run_date + datetime.timedelta(days=day_delta)).strftime("%Y%m%d")) for
        day_delta in day_deltas
    }

    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.mkdir(daily_average_dir)

    if scalar_variables is None:
        scalar_variables = []

    if vector_variables is None:
        vector_variables = []

    if grid_size_km == 4:
        grid_filename = wcofs.WCOFS_4KM_GRID_FILENAME
        if data_assimilation:
            wcofs_string = 'wcofs'
        else:
            wcofs_string = 'wcofs4'
    else:
        grid_filename = wcofs.WCOFS_2KM_GRID_FILENAME
        wcofs_string = 'wcofs2'
        wcofs.reset_dataset_grid()

    try:
        wcofs_dataset = None

        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['WCOFS']:
                wcofs_direction = 'forecast' if day_delta >= 0 else 'nowcast'
                time_delta_string = f'{wcofs_direction[0]}{abs(day_delta) + 1 if wcofs_direction == "forecast" else abs(day_delta):03}'

                wcofs_filename_suffix = time_delta_string

                if not data_assimilation:
                    wcofs_filename_suffix = f'{wcofs_filename_suffix}_noDA'

                if not data_assimilation or grid_size_km == 2:
                    wcofs_filename_suffix = f'{wcofs_filename_suffix}_{grid_size_km}km'

                existing_files = os.listdir(daily_average_dir)

                if data_assimilation:
                    if grid_size_km == 4:
                        existing_files = [filename for filename in existing_files if
                                          'wcofs' in filename and time_delta_string in filename and 'noDA' not in filename]
                    else:
                        existing_files = [filename for filename in existing_files if
                                          'wcofs' in filename and time_delta_string in filename and 'noDA' not in filename and f'{grid_size_km}km' in filename]
                else:
                    if grid_size_km == 4:
                        existing_files = [filename for filename in existing_files if
                                          'wcofs' in filename and time_delta_string in filename and 'noDA' in filename]
                    else:
                        existing_files = [filename for filename in existing_files if
                                          'wcofs' in filename and time_delta_string in filename and 'noDA' in filename and f'{grid_size_km}km' in filename]

                if wcofs_dataset is None and not all(
                        any(variable in filename for filename in existing_files) for variable in
                        list(scalar_variables) + list(vector_variables)):
                    if grid_size_km == 4:
                        wcofs_dataset = wcofs.WCOFSDataset(model_run_date, source='avg', wcofs_string=wcofs_string)
                    else:
                        wcofs_dataset = wcofs.WCOFSDataset(model_run_date, source='avg',
                                                           grid_filename=grid_filename,
                                                           source_url=os.path.join(DATA_DIR, 'input/wcofs/avg'),
                                                           wcofs_string=wcofs_string)

                scalar_variables_to_write = [variable for variable in scalar_variables if
                                             not any(variable in filename for filename in existing_files)]

                if len(scalar_variables_to_write) > 0:
                    wcofs_dataset.write_rasters(daily_average_dir, scalar_variables_to_write,
                                                filename_suffix=wcofs_filename_suffix,
                                                time_deltas=[day_delta], fill_value=LEAFLET_NODATA_VALUE,
                                                driver='GTiff')
                else:
                    logging.info(f'Skipping WCOFS day {day_delta} scalar variables')

                if not all(any(vector_variable in filename for filename in existing_files) for vector_variable in
                           vector_variables):
                    wcofs_dataset.write_rasters(daily_average_dir, vector_variables,
                                                filename_suffix=wcofs_filename_suffix,
                                                time_deltas=[day_delta], fill_value=LEAFLET_NODATA_VALUE,
                                                driver='AAIGrid')
                else:
                    logging.info(f'Skipping WCOFS day {day_delta} uv')
        del wcofs_dataset

        if grid_size_km == 2:
            wcofs.reset_dataset_grid()
    except Exception as error:
        error_type, exc_obj, error_traceback = sys.exc_info()
        filename = os.path.split(error_traceback.tb_frame.f_code.co_filename)[1]
        logging.warning(f'{error} ({filename}:{error_traceback.tb_lineno})')


def write_daily_average(output_dir: str, output_date: Union[datetime.datetime, datetime.date, int, float],
                        day_deltas: range = MODEL_DAY_DELTAS['WCOFS']):
    """
    Writes daily average of observational data and model output on given date.

    :param output_dir: output directory to write files
    :param output_date: date of data run
    :param day_deltas: time deltas for which to write model output
    """

    # write initial message
    logging.info(f'Starting file conversion for {output_date}')

    logging.info('Processing HFR SSUV...')
    write_observation(output_dir, output_date, 'hf_radar')
    logging.info('Processing VIIRS SST...')
    write_observation(output_dir, output_date, 'viirs')
    logging.info('Processing SMAP SSS...')
    write_observation(output_dir, output_date, 'smap')
    # logging.info('Processing NDBC data...')
    # write_observation(output_dir, output_date, 'data_buoy')

    logging.info(f'Wrote observations to {output_dir}')

    logging.info('Processing RTOFS...')  # RTOFS forecast is uploaded at 1700 UTC
    write_rtofs(output_dir, output_date, day_deltas)
    logging.info('Processing WCOFS...')
    write_wcofs(output_dir, output_date, day_deltas)
    logging.info('Processing WCOFS noDA...')
    write_wcofs(output_dir, output_date, day_deltas, data_assimilation=False)
    logging.info(f'Wrote models to {output_dir}')

    logging.info(f'Finished writing files. Total time: ' +
                 f'{(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')


if __name__ == '__main__':
    # create folders if they do not exist
    for dir_path in [OUTPUT_DIR, LOG_DIR]:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    start_time = datetime.datetime.now()

    log_path = os.path.join(LOG_DIR, f'{start_time.strftime("%Y%m%d")}_conversion.log')
    if log_path is None:
        logging.basicConfig(level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S',
                            format='[%(asctime)s] %(levelname)s: %(message)s')
    else:
        logging.basicConfig(filename=log_path, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                            format='[%(asctime)s] %(levelname)s: %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)

    # disable complaints from Fiona environment within conda
    logging.root.manager.loggerDict['fiona._env'].setLevel(logging.CRITICAL)

    # define dates over which to collect data (dates after today are for WCOFS forecast)
    day_deltas = MODEL_DAY_DELTAS['WCOFS']

    # from PyOFS import utilities
    #
    # model_run_dates = utilities.range_daily(datetime.datetime.now(), datetime.datetime(2019, 6, 14))
    # for model_run_date in model_run_dates:
    #     write_daily_average(OUTPUT_DIR, model_run_date, day_deltas)

    model_run_date = datetime.date.today()
    write_daily_average(OUTPUT_DIR, model_run_date, day_deltas)

    files_json_filename = os.path.join(REFERENCE_DIR, 'files.json')

    write_json.dir_structure_to_json(OUTPUT_DIR, files_json_filename)

    azure_credential_files = [os.path.join(DATA_DIR, azure_credentials_filename) for azure_credentials_filename in
                              ['azure_credentials.txt', 'azure_credentials_old.txt']]

    for azure_credential_file in azure_credential_files:
        with open(azure_credential_file) as credentials_file:
            azure_blob_url, credentials = (line.strip('\n') for line in credentials_file.readlines())

        azure.upload_to_azure(files_json_filename, f'{azure_blob_url}/reference/files.json', credentials,
                              overwrite=True)

        for day_delta in day_deltas:
            day = model_run_date + datetime.timedelta(days=day_delta)
            daily_averages_dir = os.path.join(OUTPUT_DIR, 'daily_averages', day.strftime('%Y%m%d'))
            azure.upload_to_azure(daily_averages_dir, f'{azure_blob_url}/output/daily_averages/', credentials,
                                  overwrite=True)

    print('done')
