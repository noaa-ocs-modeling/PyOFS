# coding=utf-8
"""
FTP pulldown of custom WCOFS data slices (2 and 4 kilometer non-DA).

Created on Aug 9, 2018

@author: zachary.burnett
"""

import sys

import datetime
import ftplib
import logging
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from config import DATA_DIR

TIDEPOOL_URL = 'tidepool.nos.noaa.gov'
INPUT_DIR = '/pub/outgoing/CSDL'

OUTPUT_DIR = os.path.join(DATA_DIR, 'input')
LOG_DIR = os.path.join(DATA_DIR, 'log')

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    num_downloads = 0

    wcofs_dir = os.path.join(OUTPUT_DIR, 'wcofs')
    rtofs_dir = os.path.join(OUTPUT_DIR, 'rtofs')

    avg_dir = os.path.join(wcofs_dir, 'avg')
    fwd_dir = os.path.join(wcofs_dir, 'fwd')
    obs_dir = os.path.join(wcofs_dir, 'obs')
    mod_dir = os.path.join(wcofs_dir, 'mod')

    month_dir = os.path.join(avg_dir, datetime.datetime.now().strftime('%Y%m'))

    # create folders if they do not exist
    for directory in [OUTPUT_DIR, LOG_DIR, wcofs_dir, rtofs_dir, avg_dir, fwd_dir, obs_dir, mod_dir, month_dir]:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    # define log filename
    log_path = os.path.join(LOG_DIR, f'{datetime.datetime.now().strftime("%Y%m%d")}_download.log')

    # check whether logfile exists
    log_exists = os.path.exists(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # write initial message
    logging.info('Starting FTP transfer...')

    # instantiate FTP connection
    with ftplib.FTP(TIDEPOOL_URL) as ftp_connection:
        ftp_connection.login()
        input_paths = ftp_connection.nlst(INPUT_DIR)

        for input_path in input_paths:
            extension = os.path.splitext(input_path)[-1]
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

            # filter for NetCDF and TAR archives
            if '.nc' in filename or '.tar' in filename:
                current_start_time = datetime.datetime.now()

                # download file (copy via binary connection) to local destination if it does not already exist
                if not (os.path.exists(output_path) and os.stat(output_path).st_size > 232000):
                    with open(output_path, 'wb') as output_file:
                        ftp_connection.retrbinary(f'RETR {input_path}', output_file.write)
                        logging.info(f'Copied "{input_path}" ' +
                                     f'({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s) ' +
                                     f' to "{output_path}", {os.stat(output_path).st_size / 1000} KB')
                        num_downloads += 1
                else:
                    # only write 'file exists' message on the first run of the day
                    if not log_exists:
                        logging.info('Destination file already exists: ' +
                                     f'"{output_path}", {os.stat(output_path).st_size / 1000} KB')

    logging.info(f'Downloaded {num_downloads} files. ' +
                 f'Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds')

    if num_downloads == 0:
        exit(1)

    print('done')
