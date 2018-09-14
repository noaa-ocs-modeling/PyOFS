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
DAILY_AVERAGE_DIR = os.path.join(OUTPUT_DIR, 'daily_averages')

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999


def write_daily_average(output_dir, current_model_run_date, day_deltas, log_path):
    # define directories to which output rasters will be written
    output_dirs = {current_day_delta: os.path.join(output_dir, (
            current_model_run_date + datetime.timedelta(days=current_day_delta)).strftime("%Y%m%d")) for
                   current_day_delta in day_deltas}

    for current_day_delta, current_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

        current_start_time = datetime.datetime.now()

        current_date = current_model_run_date + datetime.timedelta(days=current_day_delta)
        next_date = current_date + datetime.timedelta(days=1)

        if current_day_delta <= 0:
            try:
                # print(f'Processing HFR for {current_date}')
                current_hfr = dataset.hfr.HFR_Range(current_date, next_date)
                current_hfr.write_rasters(current_dir, ['u', 'v'], vector_components=True, drivers=['AAIGrid'],
                                          fill_value=LEAFLET_NODATA_VALUE)
            except dataset._utilities.NoDataError as error:
                with open(log_path, 'a') as log_file:
                    log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                    print(error)

            try:
                # print(f'Processing VIIRS for {current_date}')
                current_viirs = dataset.viirs.VIIRS_Range(current_date, next_date)
                current_viirs.write_raster(current_dir, drivers=['GTiff'], average=False,
                                           fill_value=LEAFLET_NODATA_VALUE)
            except dataset._utilities.NoDataError as error:
                with open(log_path, 'a') as log_file:
                    log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                    print(error)

        # only retrieve forecasts that have not already been written
        current_wcofs_time_indices = None
        if current_day_delta == -1:
            current_wcofs_time_indices = [-1]
        else:
            current_wcofs_time_indices = [current_day_delta + 1]

        try:
            # print(f'Processing WCOFS for {current_date}')
            current_wcofs = dataset.wcofs.WCOFS_Range(current_date, next_date, source='avg',
                                                      time_indices=current_wcofs_time_indices)
            current_wcofs.write_rasters(current_dir, ['temp'], drivers=['GTiff'], fill_value=LEAFLET_NODATA_VALUE)
            current_wcofs.write_rasters(current_dir, ['u', 'v'], vector_components=True, drivers=['AAIGrid'],
                                        fill_value=LEAFLET_NODATA_VALUE)
        except dataset._utilities.NoDataError as error:
            with open(log_path, 'a') as log_file:
                log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): {error}\n')
                print(error)

        with open(log_path, 'a') as log_file:
            message = f'Wrote files to {current_dir}'
            log_file.write(
                    f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s): {message}\n')
            print(message)

            current_start_time = datetime.datetime.now()

            # populate JSON file with new directory structure so that JavaScript application can see it
            json_dir_structure.populate_json(OUTPUT_DIR, JSON_PATH)

            message = f'Updated directory structure at {JSON_PATH}'
            log_file.write(
                    f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s): {message}\n')
            print(message)


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # create folders if they do not exist
    for current_dir in [OUTPUT_DIR, DAILY_AVERAGE_DIR, LOG_DIR]:
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

    # define log filename
    log_path = os.path.join(LOG_DIR, f'{datetime.datetime.now().strftime("%Y%m%d")}_daily_average.log')

    # write initial message
    with open(log_path, 'a') as log_file:
        message = f'Starting file conversion...'
        log_file.write(
                f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ({(datetime.datetime.now() - start_time).total_seconds():.2f}s): {message}\n')
        print(message)

    # # initialize QGIS application
    # qgis_application = QgsApplication([], True, None)
    # qgis_application.setPrefixPath(os.environ['QGIS_PREFIX_PATH'], True)
    # qgis_application.initQgis()

    # get current date
    model_run_dates = [start_time.replace(hour=0, minute=0, second=0, microsecond=0)]
    # model_run_dates = _utilities.day_range(datetime.datetime(2018, 8, 27), datetime.datetime(2018, 9, 6))

    # define dates over which to collect data (dates after today are for WCOFS forecast)
    day_deltas = [-1, 0, 1, 2]

    for current_model_run_date in model_run_dates:
        write_daily_average(os.path.join(DATA_DIR, DAILY_AVERAGE_DIR), current_model_run_date, day_deltas, log_path)

    with open(log_path, 'a') as log_file:
        message = f'Finished writing files. Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds'
        log_file.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}(0.00s): {message}\n')
        print(message)

    print('done')
