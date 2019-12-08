# coding=utf-8
"""
Check existence of files.

Created on Apr 10, 2019

@author: zachary.burnett
"""
import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from main.leaflet import write_json

observations = {'hfr': ['dir', 'mag'], 'viirs': ['sst']}
models = {'wcofs': ['dir', 'mag', 'sst', 'ssh', 'sss'], 'rtofs': ['dir', 'mag', 'sst', 'ssh', 'sss']}
time_deltas = ['n001', 'f001', 'f002', 'f003']


def check_files(input_dir: str) -> dict:
    missing_files = {}

    structure = write_json.get_directory_structure(input_dir)

    for day, filenames in structure['output']['daily_averages'].items():
        for observation, variables in observations.items():
            for variable in variables:
                if variable in ['dir', 'mag']:
                    extension = 'asc'
                else:
                    extension = 'tiff'

                filename = f'{observation}_{variable}_{day}.{extension}'

                if filename in filenames:
                    if day not in missing_files:
                        missing_files[day] = []

                    missing_files[day].append(filename)

        for model, variables in models.items():
            for time_delta in time_deltas:
                if time_delta[0] == 'n':
                    day_delta = int(time_delta[1:])
                else:
                    day_delta = (int(time_delta[1:]) - 1) * -1

                date = f'{datetime.datetime.strptime(day, "%Y%m%d") + datetime.timedelta(days=day_delta):%Y%m%d}'

                for variable in variables:
                    if variable in ['dir', 'mag']:
                        extension = 'asc'
                    else:
                        extension = 'tiff'

                    filename = f'{model}_{variable}_{date}_{time_delta}.{extension}'

                    if filename not in filenames:
                        if day not in missing_files:
                            missing_files[day] = []

                        missing_files[day].append(filename)

    return missing_files


if __name__ == '__main__':
    from PyOFS import DATA_DIRECTORY

    missing_files = check_files(os.path.join(DATA_DIRECTORY, 'output'))

    print(missing_files)
