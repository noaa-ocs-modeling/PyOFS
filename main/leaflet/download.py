# coding=utf-8
"""
FTP pulldown of custom WCOFS data slices (2 and 4 kilometer non-DA).

Created on Aug 9, 2018

@author: zachary.burnett
"""

from datetime import date, datetime, timedelta
import ftplib
import logging
import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.parent.parent)

from PyOFS import DATA_DIRECTORY, get_logger

TIDEPOOL_URL = '137.75.111.166'
INPUT_DIRECTORY = '/pub/outgoing/CSDL'
# WCOFS_EXPERIMENTAL_DIRECTORY = '/pub/outgoing/CSDL/testssh'

OUTPUT_DIRECTORY = DATA_DIRECTORY / 'input'
LOG_DIRECTORY = DATA_DIRECTORY / 'log'


def previous_months(to_months: int) -> [date]:
    months = [date.today().replace(day=1)]
    for _ in range(to_months):
        months.append((months[-1] - timedelta(days=1)).replace(day=1))
    return months


if __name__ == '__main__':
    start_time = datetime.now()

    num_downloads = 0

    wcofs_dir = OUTPUT_DIRECTORY / 'wcofs'
    rtofs_dir = OUTPUT_DIRECTORY / 'rtofs'

    avg_dir = wcofs_dir / 'avg'
    fwd_dir = wcofs_dir / 'fwd'
    obs_dir = wcofs_dir / 'obs'
    mod_dir = wcofs_dir / 'mod'
    # experimental_dir = wcofs_dir / 'exp' / f'{datetime.now():%Y%m}'

    month_directories = {
        month_string: avg_dir / month_string
        for month_string in (f'{month:%Y%m}' for month in previous_months(6))
    }

    # create folders if they do not exist
    for directory in [
                         OUTPUT_DIRECTORY,
                         LOG_DIRECTORY,
                         wcofs_dir,
                         rtofs_dir,
                         avg_dir,
                         fwd_dir,
                         obs_dir,
                         mod_dir,
                     ] + list(
        month_directories.values()
    ):  # experimental_dir]:
        if not directory.exists():
            os.makedirs(directory, exist_ok=True)

    # define log filename
    log_path = LOG_DIRECTORY / f'{datetime.now():%Y%m%d}_download.log'

    # check whether logfile exists
    log_exists = log_path.exists()

    logger = get_logger(
        'download', log_path, file_level=logging.INFO, console_level=logging.DEBUG
    )

    # write initial message
    logger.info('Starting FTP transfer...')

    # instantiate FTP connection
    with ftplib.FTP(TIDEPOOL_URL) as ftp_connection:
        ftp_connection.login()

        path_map = {}
        for input_path in ftp_connection.nlst(INPUT_DIRECTORY):
            filename = os.path.basename(input_path)

            if 'rtofs' in filename:
                output_path = rtofs_dir / filename
            elif 'wcofs' in filename:
                if filename[-4:] == '.sur':
                    filename = filename[:-4]

                if 'fwd' in filename:
                    output_path = fwd_dir / filename
                elif 'obs' in filename:
                    output_path = obs_dir / filename
                elif 'mod' in filename:
                    output_path = mod_dir / filename
                else:
                    for month_string, month_directory in month_directories.items():
                        if month_string in filename:
                            output_path = month_directory / filename
                            break
                    else:
                        output_path = avg_dir / filename
            else:
                output_path = OUTPUT_DIRECTORY / filename

            path_map[input_path] = output_path

        sizes = {}
        for input_path in path_map:
            try:
                size = ftp_connection.size(input_path)
                sizes[size] = (input_path, path_map[input_path])
            except ftplib.error_perm:
                pass
        if len(sizes) > 0:
            old_path_map = path_map.copy()
            path_map = {sizes[size][0]: sizes[size][1] for size in sorted(sizes)}
            path_map.update(
                {
                    input_path: output_path
                    for input_path, output_path in old_path_map.items()
                    if input_path not in path_map
                }
            )
            del sizes

        # for input_path in ftp_connection.nlst(WCOFS_EXPERIMENTAL_DIRECTORY):
        #     filename = os.path.basename(input_path)
        #
        #     if 'wcofs' in filename:
        #         if filename[-4:] == '.sur':
        #             filename = filename[:-4]
        #
        #         output_path = experimental_dir / filename
        #     else:
        #         logger.warning(f'no options set up for {input_path}')
        #
        #     path_map[input_path] = output_path

        logger.info(f'found {len(path_map)} files at FTP remote')
        for input_path, output_path in path_map.items():
            filename = os.path.basename(input_path)

            # filter for NetCDF and TAR archives
            if '.nc' in filename or '.tar' in filename:
                current_start_time = datetime.now()

                # download file (copy via binary connection) to local destination if it does not already exist
                if not (output_path.exists() and os.stat(output_path).st_size > 232000):
                    with open(output_path, 'wb') as output_file:
                        try:
                            ftp_connection.retrbinary(f'RETR {input_path}', output_file.write)
                            logger.info(
                                f'Copied "{input_path}" to "{output_path}" ({(datetime.now() - current_start_time) / timedelta(seconds=1):.2f}s, {os.stat(output_path).st_size / 1000} KB)'
                            )
                            num_downloads += 1
                        except Exception as error:
                            logger.exception(
                                f'input path: {input_path}, {output_path}: {output_path}'
                            )
                else:
                    # only write 'file exists' message on the first run of the day
                    logger.log(
                        logging.DEBUG if log_exists else logging.INFO,
                        f'Destination file already exists: "{output_path}", {os.stat(output_path).st_size / 1000} KB',
                    )

    logger.info(
        f'Downloaded {num_downloads} files. Total time: {(datetime.now() - start_time) / timedelta(seconds=1):.2f} seconds'
    )

    print('done')

    if num_downloads == 0:
        exit(1)
