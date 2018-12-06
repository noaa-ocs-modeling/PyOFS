# coding=utf-8
"""
FTP pulldown of custom WCOFS data slices (2 and 4 kilometer non-DA).

Created on Aug 9, 2018

@author: zachary.burnett
"""

import datetime
import ftplib
import os

DATA_DIR = os.environ['OFS_DATA']

FTP_URI = 'tidepool.nos.noaa.gov'
INPUT_DIR = '/pub/outgoing/CSDL'

OUTPUT_DIR = os.path.join(DATA_DIR, 'input')
LOG_DIR = os.path.join(DATA_DIR, 'log')

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # create boolean flag to determine if script found any new files
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
    log_path = os.path.join(LOG_DIR, f'{datetime.datetime.now().strftime("%Y%m%d")}_ftp_transfer.log')

    # check whether logfile exists
    log_exists = os.path.exists(log_path)

    # write initial message
    with open(log_path, 'a') as log_file:
        message = f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ' + \
                  f'({(datetime.datetime.now() - start_time).total_seconds():.2f}s): Starting FTP transfer...' + '\n'
        log_file.write(message)

    # instantiate FTP connection
    with ftplib.FTP(FTP_URI) as ftp_connection:
        ftp_connection.login()
        input_paths = ftp_connection.nlst(INPUT_DIR)

        for input_path in input_paths:
            with open(log_path, 'a') as log_file:
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
                            message = f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ' + \
                                      f'({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s): ' + \
                                      f'Copied "{input_path}" to ' + \
                                      f'"{output_path}", {os.stat(output_path).st_size / 1000} KB'
                            log_file.write(message + '\n')
                            print(message)
                            num_downloads += 1
                    else:
                        message = f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} ' + \
                                  f'({(datetime.datetime.now() - current_start_time).total_seconds():.2f}s): ' + \
                                  'Destination file already exists: ' + \
                                  f'"{output_path}", {os.stat(output_path).st_size / 1000} KB'
                        
                        # only write 'file exists' message on the first run of the day
                        if not log_exists:
                            log_file.write(message + '\n')
                        print(message)

    with open(log_path, 'a') as log_file:
        message = f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")} (0.00s): Downloaded {num_downloads} files. ' + \
                  f'Total time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds' + '\n'
        log_file.write(message)
        print(message)
    
        if num_downloads == 0:
            exit(1)

    print('done')
