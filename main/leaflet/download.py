# coding=utf-8
"""
FTP pulldown of custom WCOFS data slices (2 and 4 kilometer non-DA).

Created on Aug 9, 2018

@author: zachary.burnett
"""

import datetime
import ftplib
import logging
import os
import sys

from PyOFS.logging import create_logger

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from PyOFS import DATA_DIRECTORY

TIDEPOOL_URL = 'tidepool.nos.noaa.gov'
INPUT_DIR = '/pub/outgoing/CSDL'
WCOFS_OPTION_DIR = '/pub/outgoing/CSDL/testssh'

OUTPUT_DIR = os.path.join(DATA_DIRECTORY, 'input')
LOG_DIR = os.path.join(DATA_DIRECTORY, 'log')

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    num_downloads = 0

    wcofs_dir = os.path.join(OUTPUT_DIR, 'wcofs')
    rtofs_dir = os.path.join(OUTPUT_DIR, 'rtofs')

    avg_dir = os.path.join(wcofs_dir, 'avg')
    fwd_dir = os.path.join(wcofs_dir, 'fwd')
    obs_dir = os.path.join(wcofs_dir, 'obs')
    mod_dir = os.path.join(wcofs_dir, 'mod')
    option_dir = os.path.join(wcofs_dir, 'option', f'{datetime.datetime.now():%Y%m}')

    month_dir = os.path.join(avg_dir, f'{datetime.datetime.now():%Y%m}')

    # create folders if they do not exist
    for directory in [OUTPUT_DIR, LOG_DIR, wcofs_dir, rtofs_dir, avg_dir, fwd_dir, obs_dir, mod_dir, month_dir,
                      option_dir]:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    # define log filename
    log_path = os.path.join(LOG_DIR, f'{datetime.datetime.now():%Y%m%d}_download.log')

    # check whether logfile exists
    log_exists = os.path.exists(log_path)

    create_logger('', log_path, file_level=logging.INFO, console_level=logging.DEBUG,
                  log_format='[%(asctime)s] %(levelname)-8s: %(message)s')

    # write initial message
    logging.info('Starting FTP transfer...')

    # instantiate FTP connection
    with ftplib.FTP(TIDEPOOL_URL) as ftp_connection:
        ftp_connection.login()

        path_map = {}
        for input_path in ftp_connection.nlst(INPUT_DIR):
            filename = os.path.basename(input_path)

            if 'rtofs' in filename:
                output_path = os.path.join(rtofs_dir, filename)
            elif 'wcofs' in filename:
                if filename[-4:] == '.sur':
                    filename = filename[:-4]

                if 'fwd' in filename:
                    output_path = os.path.join(fwd_dir, filename)
                elif 'obs' in filename:
                    output_path = os.path.join(obs_dir, filename)
                elif 'mod' in filename:
                    output_path = os.path.join(mod_dir, filename)
                else:
                    output_path = os.path.join(month_dir, filename)
            else:
                output_path = os.path.join(OUTPUT_DIR, filename)

            path_map[input_path] = output_path

        for input_path in ftp_connection.nlst(WCOFS_OPTION_DIR):
            filename = os.path.basename(input_path)

            if 'wcofs' in filename:
                if filename[-4:] == '.sur':
                    filename = filename[:-4]

                output_path = os.path.join(option_dir, filename)
            else:
                logging.warning(f'no options set up for {input_path}')

            path_map[input_path] = output_path

        for input_path, output_path in path_map.items():
            filename = os.path.basename(input_path)

            # filter for NetCDF and TAR archives
            if '.nc' in filename or '.tar' in filename:
                current_start_time = datetime.datetime.now()

                # download file (copy via binary connection) to local destination if it does not already exist
                if not (os.path.exists(output_path) and os.stat(output_path).st_size > 232000):
                    with open(output_path, 'wb') as output_file:
                        try:
                            ftp_connection.retrbinary(f'RETR {input_path}', output_file.write)
                            logging.info(f'Copied "{input_path}" to "{output_path}" ' +
                                         f'({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s, {os.stat(output_path).st_size / 1000} KB)')
                            num_downloads += 1
                        except Exception as error:
                            logging.info(f'error with "{input_path}": {error.__class__.__name__} - {error}')
                else:
                    # only write 'file exists' message on the first run of the day
                    logging.log(logging.DEBUG if log_exists else logging.INFO,
                                'Destination file already exists: ' + \
                                f'"{output_path}", {os.stat(output_path).st_size / 1000} KB')

    logging.info(f'Downloaded {num_downloads} files. ' +
                 f'Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

    print('done')

    if num_downloads == 0:
        exit(1)
