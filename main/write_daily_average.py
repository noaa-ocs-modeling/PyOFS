# coding=utf-8
"""
Daily average WCOFS source data.

Created on Aug 21, 2018

@author: zachary.burnett
"""
import datetime
import os
import sys

import pytz

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from dataset import _utilities, hfr, viirs, wcofs, rtofs

from main import json_dir_structure, DATA_DIR

JSON_PATH = os.path.join(DATA_DIR, r'reference\model_dates.json')
LOG_DIR = os.path.join(DATA_DIR, 'log')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
DAILY_AVERAGES_DIR = os.path.join(OUTPUT_DIR, 'daily_averages')

# offset from study area to UTC
STUDY_AREA_TIMEZONE = 'US/Pacific'
STUDY_AREA_TO_UTC = datetime.timedelta(
    hours=-datetime.datetime.now(pytz.timezone(STUDY_AREA_TIMEZONE)).utcoffset().total_seconds() / 3600)

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999

# range of day deltas that models reach
MODEL_DAY_DELTAS = {'WCOFS': range(-1, 2 + 1), 'RTOFS': range(-3, 8 + 1)}


def write_observational_data(output_dir: str, observation_date: datetime.datetime, log_path: str):
    if 'datetime.date' in str(type(observation_date)):
        start_datetime = datetime.datetime.combine(observation_date, datetime.datetime.min.time())
    elif 'datetime.datetime' in str(type(observation_date)):
        start_datetime = observation_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_datetime = observation_date

    end_datetime = start_datetime + datetime.timedelta(days=1)

    if end_datetime > datetime.datetime.now():
        end_datetime = datetime.datetime.now()
    
    # write HFR rasters
    try:
        hfr_start_datetime = start_datetime + datetime.timedelta(hours=2)
        hfr_end_datetime = end_datetime + datetime.timedelta(hours=2)

        hfr_range = hfr.HFR_Range(hfr_start_datetime, hfr_end_datetime)
        hfr_range.write_rasters(output_dir, filename_suffix=f'{observation_date.strftime("%Y%m%d")}',
                                variables=['u', 'v'], vector_components=True, drivers=['AAIGrid'],
                                fill_value=LEAFLET_NODATA_VALUE)
        del hfr_range
    except _utilities.NoDataError as error:
        print(error)
        with open(log_path, 'a') as log_file:
            log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) HFR: {error}\n')
    
    # write VIIRS rasters
    try:
        utc_start_datetime = start_datetime + STUDY_AREA_TO_UTC
        utc_morning_datetime = start_datetime + datetime.timedelta(hours=6) + STUDY_AREA_TO_UTC
        utc_evening_datetime = start_datetime + datetime.timedelta(hours=18) + STUDY_AREA_TO_UTC
        utc_end_datetime = start_datetime + datetime.timedelta(hours=24) + STUDY_AREA_TO_UTC
        
        viirs_range = viirs.VIIRS_Range(utc_start_datetime, utc_end_datetime)
        
        viirs_range.write_raster(output_dir,
                                 filename_suffix=f'{start_datetime.strftime("%Y%m%d")}_morning',
                                 start_datetime=utc_start_datetime, end_datetime=utc_morning_datetime,
                                 fill_value=LEAFLET_NODATA_VALUE, drivers=['GTiff'], sses_correction=False,
                                 variables=['sst'])
        viirs_range.write_raster(output_dir,
                                 filename_suffix=f'{start_datetime.strftime("%Y%m%d")}_daytime',
                                 start_datetime=utc_morning_datetime, end_datetime=utc_evening_datetime,
                                 fill_value=LEAFLET_NODATA_VALUE, drivers=['GTiff'], sses_correction=False,
                                 variables=['sst'])
        viirs_range.write_raster(output_dir,
                                 filename_suffix=f'{start_datetime.strftime("%Y%m%d")}_evening',
                                 start_datetime=utc_evening_datetime, end_datetime=utc_end_datetime,
                                 fill_value=LEAFLET_NODATA_VALUE, drivers=['GTiff'], sses_correction=False,
                                 variables=['sst'])
        del viirs_range
    except _utilities.NoDataError as error:
        print(error)
        with open(log_path, 'a') as log_file:
            log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) VIIRS: {error}\n')


def write_model_output(output_dir: str, model_run_date: datetime.datetime, day_deltas: list, log_path: str):
    if 'datetime.date' in str(type(model_run_date)):
        model_run_date = datetime.datetime.combine(model_run_date, datetime.datetime.min.time())
    
    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: os.path.join(output_dir, (model_run_date + datetime.timedelta(days=day_delta)).strftime("%Y%m%d"))
        for day_delta in day_deltas}
    
    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.mkdir(daily_average_dir)
    
    # write RTOFS rasters
    try:
        rtofs_dataset = rtofs.RTOFS_Dataset(model_run_date, source='2ds', time_interval='daily')
        
        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['RTOFS']:
                start_datetime = model_run_date + datetime.timedelta(days=day_delta)
                rtofs_filename_prefix = f'rtofs'
                rtofs_direction = 'forecast' if day_delta >= 0 else 'nowcast'
                time_delta_string = f'{rtofs_direction[0]}{abs(day_delta) + 1 if rtofs_direction == "forecast" else abs(day_delta):03}'
                rtofs_filename_suffix = f'{start_datetime.strftime("%Y%m%d")}_{time_delta_string}'
                
                rtofs_dataset.write_raster(
                    os.path.join(daily_average_dir, f'{rtofs_filename_prefix}_sst_{rtofs_filename_suffix}'),
                    variable='temp', time=start_datetime, direction=rtofs_direction)

                # RTOFS has a same-day nowcast
                if day_delta == 0:
                    rtofs_direction = 'nowcast'
                    time_delta_string = f'{rtofs_direction[0]}{abs(day_delta) + 1 if rtofs_direction == "forecast" else abs(day_delta):03}'
                    rtofs_filename_suffix = f'{start_datetime.strftime("%Y%m%d")}_{time_delta_string}'
    
                    rtofs_dataset.write_raster(
                        os.path.join(daily_average_dir, f'{rtofs_filename_prefix}_sst_{rtofs_filename_suffix}'),
                        variable='temp', time=start_datetime, direction=rtofs_direction)
    except _utilities.NoDataError as error:
        print(error)
        with open(log_path, 'a') as log_file:
            log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) RTOFS: {error}\n')
    
    # write WCOFS rasters
    try:
        wcofs_dataset = wcofs.WCOFS_Dataset(model_run_date, source='avg')
    
        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['WCOFS']:
                wcofs_direction = 'forecast' if day_delta >= 0 else 'nowcast'
                time_delta_string = f'{wcofs_direction[0]}{abs(day_delta) + 1 if wcofs_direction == "forecast" else abs(day_delta):03}'
                wcofs_filename_suffix = f'{time_delta_string}'
            
                wcofs_dataset.write_rasters(daily_average_dir, ['temp'], filename_suffix=wcofs_filename_suffix,
                                            time_deltas=[day_delta], fill_value=LEAFLET_NODATA_VALUE, drivers=['GTiff'])
                wcofs_dataset.write_rasters(daily_average_dir, ['u', 'v'], filename_suffix=wcofs_filename_suffix,
                                            time_deltas=[day_delta], vector_components=True,
                                            fill_value=LEAFLET_NODATA_VALUE, drivers=['AAIGrid'])
    except _utilities.NoDataError as error:
        print(error)
        with open(log_path, 'a') as log_file:
            log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) WCOFS: {error}\n')

    try:
        wcofs_4km_dataset = wcofs.WCOFS_Dataset(model_run_date, source='avg',
                                                grid_filename=wcofs.WCOFS_4KM_GRID_FILENAME,
                                                source_url=os.path.join(DATA_DIR, 'input/wcofs/avg'),
                                                wcofs_string='wcofs4')
    
        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['WCOFS']:
                wcofs_direction = 'forecast' if day_delta >= 0 else 'nowcast'
                time_delta_string = f'{wcofs_direction[0]}{abs(day_delta) + 1 if wcofs_direction == "forecast" else abs(day_delta):03}'
                wcofs_filename_suffix = f'{time_delta_string}_noDA_4km'
            
                wcofs_4km_dataset.write_rasters(daily_average_dir, ['temp'], filename_suffix=wcofs_filename_suffix,
                                                time_deltas=[day_delta], fill_value=LEAFLET_NODATA_VALUE,
                                                drivers=['GTiff'])
                wcofs_4km_dataset.write_rasters(daily_average_dir, ['u', 'v'], filename_suffix=wcofs_filename_suffix,
                                                time_deltas=[day_delta], vector_components=True,
                                                fill_value=LEAFLET_NODATA_VALUE, drivers=['AAIGrid'])
    except _utilities.NoDataError as error:
        print(error)
        with open(log_path, 'a') as log_file:
            log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) WCOFS: {error}\n')

    # wcofs.reset_dataset_grid()
    #
    # try:
    #     wcofs_2km_dataset = wcofs.WCOFS_Dataset(model_run_date, source='avg',
    #                                             grid_filename=wcofs.WCOFS_2KM_GRID_FILENAME,
    #                                             source_url=os.path.join(DATA_DIR, 'input/wcofs/avg'),
    #                                             wcofs_string='wcofs2')
    #
    #     for day_delta, daily_average_dir in output_dirs.items():
    #         if day_delta in MODEL_DAY_DELTAS['WCOFS']:
    #             wcofs_direction = 'forecast' if day_delta >= 0 else 'nowcast'
    #             time_delta_string = f'{wcofs_direction[0]}{abs(day_delta) + 1 if wcofs_direction == "forecast" else abs(day_delta):03}'
    #             wcofs_filename_suffix = f'{time_delta_string}_noDA_2km'
    #
    #             wcofs_2km_dataset.write_rasters(daily_average_dir, ['temp'], filename_suffix=wcofs_filename_suffix,
    #                                             time_deltas=[day_delta], x_size=0.02, y_size=0.02,
    #                                             fill_value=LEAFLET_NODATA_VALUE, drivers=['GTiff'])
    #             wcofs_2km_dataset.write_rasters(daily_average_dir, ['u', 'v'], filename_suffix=wcofs_filename_suffix,
    #                                             time_deltas=[day_delta], vector_components=True, x_size=0.02,
    #                                             y_size=0.02, fill_value=LEAFLET_NODATA_VALUE, drivers=['AAIGrid'])
    # except _utilities.NoDataError as error:
    #     print(error)
    #     with open(log_path, 'a') as log_file:
    #         log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s) WCOFS: {error}\n')


def write_daily_average(output_dir: str, output_date: datetime.datetime, day_deltas: list, log_path: str):
    # get time for log
    start_time = datetime.datetime.now()
    
    # write observational data to directory for specified date
    observation_dir = os.path.join(output_dir, output_date.strftime("%Y%m%d"))
    if not os.path.isdir(observation_dir):
        os.mkdir(observation_dir)
    write_observational_data(observation_dir, output_date, log_path)
    
    # write to log
    message = f'Wrote observational data to {observation_dir}'
    with open(log_path, 'a') as log_file:
        log_file.write(
            f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
    print(message)
    
    # write models to directories
    write_model_output(output_dir, output_date, day_deltas, log_path)
    
    # write to log
    message = f'Wrote model output.'
    with open(log_path, 'a') as log_file:
        log_file.write(
            f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
    print(message)
    
    # get time for log
    start_time = datetime.datetime.now()
    
    # populate JSON file with new directory structure so that JavaScript application can see it
    json_dir_structure.populate_json(OUTPUT_DIR, JSON_PATH)
    
    # write to log
    message = f'Updated directory structure at {JSON_PATH}'
    with open(log_path, 'a') as log_file:
        log_file.write(
            f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
    print(message)


if __name__ == '__main__':
    # create folders if they do not exist
    for daily_average_dir in [OUTPUT_DIR, DAILY_AVERAGES_DIR, LOG_DIR]:
        if not os.path.isdir(daily_average_dir):
            os.mkdir(daily_average_dir)
    
    start_time = datetime.datetime.now()
    
    # define log filename
    log_path = os.path.join(LOG_DIR, f'{start_time.strftime("%Y%m%d")}_daily_average.log')
    
    # write initial message
    message = f'Starting file conversion...'
    with open(log_path, 'a') as log_file:
        log_file.write(
            f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
    print(message)
    
    # define dates over which to collect data (dates after today are for WCOFS forecast)
    day_deltas = [-1, 0, 1, 2]
    
    # model_run_dates = dataset._utilities.day_range(datetime.datetime(2018, 10, 2), datetime.datetime(2018, 9, 1))
    # for model_run_date in model_run_dates:
    #     write_daily_average(os.path.join(DATA_DIR, DAILY_AVERAGES_DIR), model_run_date, day_deltas, log_path)
    
    model_run_date = datetime.date.today()
    write_daily_average(os.path.join(DATA_DIR, DAILY_AVERAGES_DIR), model_run_date, day_deltas, log_path)
    
    message = f'Finished writing files. Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds'
    with open(log_path, 'a') as log_file:
        log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}(0.00s): {message}\n')
    print(message)
    
    print('done')
