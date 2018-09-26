# coding=utf-8
"""
Daily average WCOFS source data.

Created on Aug 21, 2018

@author: zachary.burnett
"""

import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import dataset._utilities, dataset.hfr, dataset.viirs, dataset.wcofs

from main import json_dir_structure

DATA_DIR = os.environ['OFS_DATA']
LOG_DIR = os.path.join(DATA_DIR, 'log')
JSON_PATH = os.path.join(DATA_DIR, r'reference\model_dates.json')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
DAILY_AVERAGES_DIR = os.path.join(OUTPUT_DIR, 'daily_averages')

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999


def write_daily_average(output_dir, model_run_date: datetime.datetime, day_deltas, log_path):
    if type(model_run_date) == 'datetime.Date':
        model_run_date = datetime.datetime.combine(model_run_date, datetime.datetime.min.time())

    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: os.path.join(output_dir, (model_run_date + datetime.timedelta(days=day_delta)).strftime("%Y%m%d"))
        for day_delta in day_deltas}

    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.mkdir(daily_average_dir)

        start_time = datetime.datetime.now()

        start_datetime = model_run_date + datetime.timedelta(days=day_delta)
        end_datetime = start_datetime + datetime.timedelta(days=1)

        if day_delta <= 0:
            try:
                # print(f'Processing HFR for {start_datetime}')
                hfr_range = dataset.hfr.HFR_Range(start_datetime, end_datetime)
                hfr_range.write_rasters(daily_average_dir, filename_suffix=f'{start_datetime.strftime("%Y%m%d")}',
                                        variables=['u', 'v'], vector_components=True, drivers=['AAIGrid'],
                                        fill_value=LEAFLET_NODATA_VALUE)
                del hfr_range
            except dataset._utilities.NoDataError as error:
                with open(log_path, 'a') as log_file:
                    log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                    print(error)

            try:
                # print(f'Processing VIIRS for {start_datetime}')
                viirs_range = dataset.viirs.VIIRS_Range(start_datetime, end_datetime)

                morning_datetime = start_datetime + datetime.timedelta(hours=6)
                evening_datetime = start_datetime + datetime.timedelta(hours=18)

                viirs_range.write_raster(daily_average_dir,
                                         filename_prefix=f'viirs_sst_{start_datetime.strftime("%Y%m%d")}_morning',
                                         start_datetime=start_datetime, end_datetime=morning_datetime,
                                         drivers=['GTiff'], fill_value=LEAFLET_NODATA_VALUE)
                viirs_range.write_raster(daily_average_dir,
                                         filename_prefix=f'viirs_sst_{start_datetime.strftime("%Y%m%d")}_daytime',
                                         start_datetime=morning_datetime, end_datetime=evening_datetime,
                                         drivers=['GTiff'], fill_value=LEAFLET_NODATA_VALUE)
                viirs_range.write_raster(daily_average_dir,
                                         filename_prefix=f'viirs_sst_{start_datetime.strftime("%Y%m%d")}_evening',
                                         start_datetime=evening_datetime, end_datetime=end_datetime, drivers=['GTiff'],
                                         fill_value=LEAFLET_NODATA_VALUE)
                del viirs_range
            except dataset._utilities.NoDataError as error:
                with open(log_path, 'a') as log_file:
                    log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                    print(error)

        # only retrieve forecasts that have not already been written
        if day_delta == -1:
            wcofs_time_indices = [-1]
        else:
            wcofs_time_indices = [day_delta + 1]

        try:
            # print(f'Processing WCOFS for {date}')
            wcofs_range = dataset.wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg',
                                                    time_indices=wcofs_time_indices)
            wcofs_range.write_rasters(daily_average_dir, ['temp'], drivers=['GTiff'], fill_value=LEAFLET_NODATA_VALUE)
            wcofs_range.write_rasters(daily_average_dir, ['u', 'v'], vector_components=True, drivers=['AAIGrid'],
                                      fill_value=LEAFLET_NODATA_VALUE)
            del wcofs_range
        except dataset._utilities.NoDataError as error:
            with open(log_path, 'a') as log_file:
                log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                print(error)

        with open(log_path, 'a') as log_file:
            message = f'Wrote files to {daily_average_dir}'
            log_file.write(
                    f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
            print(message)

        start_time = datetime.datetime.now()

        # populate JSON file with new directory structure so that JavaScript application can see it
        json_dir_structure.populate_json(OUTPUT_DIR, JSON_PATH)

        with open(log_path, 'a') as log_file:
            message = f'Updated directory structure at {JSON_PATH}'
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
    with open(log_path, 'a') as log_file:
        message = f'Starting file conversion...'
        log_file.write(
                f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
        print(message)

    # get current date
    model_run_dates = [start_time.replace(hour=0, minute=0, second=0, microsecond=0)]
    # model_run_dates = dataset._utilities.day_range(datetime.datetime(2018, 9, 16),
    #                                                datetime.datetime.now() + datetime.timedelta(days=1))

    # define dates over which to collect data (dates after today are for WCOFS forecast)
    day_deltas = [-1, 0, 1, 2]

    for model_run_date in model_run_dates:
        write_daily_average(os.path.join(DATA_DIR, DAILY_AVERAGES_DIR), model_run_date, day_deltas, log_path)

    with open(log_path, 'a') as log_file:
        message = f'Finished writing files. Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds'
        log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}(0.00s): {message}\n')
        print(message)

    print('done')
