from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path

from PyOFS import DATA_DIRECTORY, get_logger
from main.leaflet import write_json

LOGGER = get_logger('PyOFS.check')

observations = {'hfr': ['dir', 'mag'], 'viirs': ['sst']}
models = {
    'wcofs': ['dir', 'mag', 'sst', 'ssh', 'sss'],
    'rtofs': ['dir', 'mag', 'sst', 'ssh', 'sss'],
}
time_deltas = ['n001', 'f001', 'f002', 'f003']


def check_files(input_dir: PathLike) -> dict:
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

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

                date = f'{datetime.strptime(day, "%Y%m%d") + timedelta(days=day_delta):%Y%m%d}'

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
    missing_files = check_files(DATA_DIRECTORY / 'output')

    for missing_file_date, current_missing_files in missing_files.items():
        print(
            f'{missing_file_date} - {len(current_missing_files)} files missing: ({current_missing_files})'
        )

    print('done')
