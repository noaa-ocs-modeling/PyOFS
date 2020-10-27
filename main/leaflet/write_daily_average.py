from datetime import date, datetime, time, timedelta
import logging
import os
from os import PathLike
from pathlib import Path
import sys
from typing import Collection, Union

import pytz

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from main.leaflet.write_azure import upload_to_azure, sync_with_azure
from main.leaflet import write_json
from PyOFS import DATA_DIRECTORY, LEAFLET_NODATA_VALUE, NoDataError, get_logger
from PyOFS.observation import hf_radar, viirs, smap, data_buoy
from PyOFS.model import wcofs, rtofs

# disable complaints from Fiona environment within conda
logging.root.manager.loggerDict['fiona._env'].setLevel(logging.CRITICAL)

LOG_DIRECTORY = Path(DATA_DIRECTORY) / 'log'
LOG_FILENAME = Path(LOG_DIRECTORY) / f'{datetime.now():%Y%m%d}_conversion.log'
OUTPUT_DIRECTORY = Path(DATA_DIRECTORY) / 'output'
REFERENCE_DIRECTORY = Path(DATA_DIRECTORY) / 'reference'

# offset from study area to UTC
STUDY_AREA_TIMEZONE = 'US/Pacific'
STUDY_AREA_TO_UTC = timedelta(
    hours=-datetime.now(pytz.timezone(STUDY_AREA_TIMEZONE)).utcoffset() / timedelta(hours=1)
)

# range of day deltas that models reach
MODEL_DAY_DELTAS = {'WCOFS': range(-1, 3), 'RTOFS': range(-3, 9)}

LOGGER = get_logger('PyOFS', LOG_FILENAME, file_level=logging.INFO, console_level=logging.INFO)


def write_observation(
        output_dir: PathLike, observation_date: Union[datetime, date], observation: str
):
    """
    Writes daily average of observational data on given date.

    :param output_dir: output directory to write files
    :param observation_date: fate of observation
    :param observation: observation to write
    :raise _utilities.NoDataError: if no data found
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if type(observation_date) is date:
        day_start = datetime.combine(observation_date, time.min)
    elif type(observation_date) is datetime:
        day_start = observation_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        day_start = observation_date

    day_noon = day_start + timedelta(hours=12)
    day_end = day_start + timedelta(days=1)

    if observation is 'smap':
        monthly_dir = output_dir / 'monthly_averages'

        if not os.path.isdir(monthly_dir):
            os.makedirs(monthly_dir, exist_ok=True)

        observation_dir = monthly_dir / f'{observation_date:%Y%m}'
    else:
        daily_dir = output_dir / 'daily_averages'

        if not os.path.isdir(daily_dir):
            os.makedirs(daily_dir, exist_ok=True)

        observation_dir = daily_dir / f'{observation_date:%Y%m%d}'

    if not os.path.isdir(observation_dir):
        os.makedirs(observation_dir, exist_ok=True)

    day_start_ndbc = day_start + timedelta(hours=2)
    day_end_ndbc = day_end + timedelta(hours=2)

    day_start_utc = day_start + STUDY_AREA_TO_UTC
    day_noon_utc = day_noon + STUDY_AREA_TO_UTC
    day_end_utc = day_end + STUDY_AREA_TO_UTC

    try:
        if observation == 'hf_radar':
            hfr_range = hf_radar.HFRadarRange(day_start_ndbc, day_end_ndbc)
            hfr_range.write_rasters(
                observation_dir,
                filename_suffix=f'{observation_date:%Y%m%d}',
                variables=['dir', 'mag'],
                driver='AAIGrid',
                fill_value=LEAFLET_NODATA_VALUE,
                dop_threshold=0.5,
            )
            del hfr_range
        elif observation == 'viirs':
            viirs_range = viirs.VIIRSRange(day_start, day_end_utc)
            viirs_range.write_raster(
                observation_dir,
                filename_suffix=f'{day_start:%Y%m%d%H%M}_{day_noon:%Y%m%d%H%M}',
                start_time=day_start_utc,
                end_time=day_noon_utc,
                fill_value=LEAFLET_NODATA_VALUE,
                driver='GTiff',
                correct_sses=False,
                variables=['sst'],
            )
            viirs_range.write_raster(
                observation_dir,
                filename_suffix=f'{day_noon:%Y%m%d%H%M}_{day_end:%Y%m%d%H%M}',
                start_time=day_noon_utc,
                end_time=day_end_utc,
                fill_value=LEAFLET_NODATA_VALUE,
                driver='GTiff',
                correct_sses=False,
                variables=['sst'],
            )
            del viirs_range
        elif observation == 'smap':
            smap_dataset = smap.SMAPDataset()
            smap_dataset.write_rasters(
                observation_dir,
                data_time=day_start_utc,
                fill_value=LEAFLET_NODATA_VALUE,
                driver='GTiff',
                variables=['sss'],
            )
            del smap_dataset
        elif observation == 'data_buoy':
            data_buoy_range = data_buoy.DataBuoyRange(data_buoy.WCOFS_NDBC_STATIONS_FILENAME)
            output_filename = (
                    observation_dir / f'ndbc_data_buoys_{observation_date:%Y%m%d}.gpkg'
            )
            data_buoy_range.write_vector(output_filename, day_start_ndbc, day_end_ndbc)
            del data_buoy_range
    except NoDataError as error:
        LOGGER.warning(f'{error.__class__.__name__}: {error}')
    except:
        LOGGER.exception(f'observation: {observation}')


def write_rtofs(
        output_dir: PathLike,
        model_run_date: Union[datetime, date],
        day_deltas: range = MODEL_DAY_DELTAS['RTOFS'],
        scalar_variables: Collection[str] = ('sst', 'sss', 'ssh'),
        vector_variables: Collection[str] = ('dir', 'mag'),
        overwrite: bool = False,
):
    """
    Writes daily average of RTOFS output on given date.

    :param output_dir: output directory to write files
    :param model_run_date: date of model run
    :param day_deltas: time deltas for which to write model output
    :param scalar_variables: list of scalar variables to use
    :param vector_variables: list of vector variables to use
    :param overwrite: whether to overwrite existing files
    :raise _utilities.NoDataError: if no data found
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if type(model_run_date) is date:
        model_run_date = datetime.combine(model_run_date, datetime.min.time())

    daily_dir = output_dir / 'daily_averages'

    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: daily_dir / f'{model_run_date + timedelta(days=day_delta):%Y%m%d}'
        for day_delta in day_deltas
    }

    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.makedirs(daily_average_dir, exist_ok=True)

    try:
        rtofs_dataset = None

        for day_delta, daily_average_dir in output_dirs.items():
            if day_delta in MODEL_DAY_DELTAS['RTOFS']:
                time_delta_string = f'{"f" if day_delta >= 0 else "n"}{abs(day_delta) + 1 if day_delta >= 0 else abs(day_delta):03}'

                day_of_forecast = model_run_date + timedelta(days=day_delta)

                if overwrite:
                    existing_files = []
                else:
                    existing_files = os.listdir(daily_average_dir)
                    existing_files = [
                        filename
                        for filename in existing_files
                        if 'rtofs' in filename and time_delta_string in filename
                    ]

                    if rtofs_dataset is None and not all(
                            any(variable in filename for filename in existing_files)
                            for variable in list(scalar_variables) + list(vector_variables)
                    ):
                        rtofs_dataset = rtofs.RTOFSDataset(
                            model_run_date, source='2ds', time_interval='daily'
                        )

                scalar_variables_to_write = [
                    variable
                    for variable in scalar_variables
                    if not any(variable in filename for filename in existing_files)
                ]

                if rtofs_dataset is not None:
                    if len(scalar_variables_to_write) > 0:
                        rtofs_dataset.write_rasters(
                            daily_average_dir,
                            variables=scalar_variables_to_write,
                            time=day_of_forecast,
                            driver='GTiff',
                        )
                    else:
                        LOGGER.debug(f'Skipping RTOFS day {day_delta} scalar variables')

                    if not all(
                            any(vector_variable in filename for filename in existing_files)
                            for vector_variable in vector_variables
                    ):
                        rtofs_dataset.write_rasters(
                            daily_average_dir,
                            variables=vector_variables,
                            time=day_of_forecast,
                            driver='AAIGrid',
                        )
                    else:
                        LOGGER.debug(f'Skipping RTOFS day {day_delta} uv')
        del rtofs_dataset
    except NoDataError as error:
        LOGGER.warning(f'{error.__class__.__name__}: {error}')
    except:
        LOGGER.exception(f'model run date: {model_run_date}, day deltas: {day_deltas}')


def write_wcofs(
        output_dir: PathLike,
        model_run_date: Union[datetime, date, int, float],
        day_deltas: range = MODEL_DAY_DELTAS['WCOFS'],
        scalar_variables: Collection[str] = ('sst', 'sss', 'ssh'),
        vector_variables: Collection[str] = ('dir', 'mag'),
        data_assimilation: bool = True,
        grid_size_km: int = 4,
        source_url: str = None,
        use_defaults: bool = True,
        suffix: str = None,
        overwrite: bool = False,
):
    """
    Writes daily average of model output on given date.

    :param output_dir: output directory to write files
    :param model_run_date: date of model run
    :param day_deltas: time deltas for which to write model output
    :param scalar_variables: list of scalar variables to use
    :param vector_variables: list of vector variables to use
    :param data_assimilation: whether to retrieve model with data assimilation
    :param grid_size_km: cell size in km
    :param source_url: URL of source
    :param use_defaults: whether to fall back to default source URLs if the provided one does not exist
    :param suffix: suffix to append to output filename
    :param overwrite: whether to overwrite existing files
    :raise _utilities.NoDataError: if no data found
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if type(model_run_date) is date:
        model_run_date = datetime.combine(model_run_date, datetime.min.time())

    daily_dir = output_dir / 'daily_averages'

    # define directories to which output rasters will be written
    output_dirs = {
        day_delta: daily_dir / f'{model_run_date + timedelta(days=day_delta):%Y%m%d}'
        for day_delta in day_deltas
    }

    for day_delta, daily_average_dir in output_dirs.items():
        # ensure output directory exists
        if not os.path.isdir(daily_average_dir):
            os.makedirs(daily_average_dir, exist_ok=True)

    if scalar_variables is None:
        scalar_variables = []

    if vector_variables is None:
        vector_variables = []

    if grid_size_km == 4:
        grid_filename = wcofs.WCOFS_4KM_GRID_FILENAME
        if data_assimilation:
            wcofs_string = 'wcofs'
        else:
            wcofs_string = 'wcofs-free'
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

                if suffix is not None:
                    wcofs_filename_suffix = f'{wcofs_filename_suffix}_{suffix}'

                if overwrite:
                    existing_files = []
                else:
                    existing_files = os.listdir(daily_average_dir)

                    if data_assimilation:
                        if grid_size_km == 4:
                            existing_files = [
                                filename
                                for filename in existing_files
                                if 'wcofs' in filename
                                   and time_delta_string in filename
                                   and 'noDA' not in filename
                                   and (suffix in filename if suffix is not None else True)
                            ]
                        else:
                            existing_files = [
                                filename
                                for filename in existing_files
                                if 'wcofs' in filename
                                   and time_delta_string in filename
                                   and 'noDA' not in filename
                                   and (suffix in filename if suffix is not None else True)
                                   and f'{grid_size_km}km' in filename
                            ]
                    else:
                        if grid_size_km == 4:
                            existing_files = [
                                filename
                                for filename in existing_files
                                if 'wcofs' in filename
                                   and time_delta_string in filename
                                   and 'noDA' in filename
                                   and (suffix in filename if suffix is not None else True)
                            ]
                        else:
                            existing_files = [
                                filename
                                for filename in existing_files
                                if 'wcofs' in filename
                                   and time_delta_string in filename
                                   and 'noDA' in filename
                                   and (suffix in filename if suffix is not None else True)
                                   and f'{grid_size_km}km' in filename
                            ]

                if wcofs_dataset is None and not all(
                        any(variable in filename for filename in existing_files)
                        for variable in list(scalar_variables) + list(vector_variables)
                ):
                    if grid_size_km == 4:
                        wcofs_dataset = wcofs.WCOFSDataset(
                            model_run_date,
                            source='avg',
                            wcofs_string=wcofs_string,
                            source_url=source_url,
                            use_defaults=use_defaults,
                        )
                    else:
                        wcofs_dataset = wcofs.WCOFSDataset(
                            model_run_date,
                            source='avg',
                            grid_filename=grid_filename,
                            source_url=source_url,
                            use_defaults=use_defaults,
                            wcofs_string=wcofs_string,
                        )
                if wcofs_dataset is not None:
                    scalar_variables_to_write = [
                        variable
                        for variable in scalar_variables
                        if not any(variable in filename for filename in existing_files)
                    ]

                    if len(scalar_variables_to_write) > 0:
                        wcofs_dataset.write_rasters(
                            daily_average_dir,
                            scalar_variables_to_write,
                            filename_suffix=wcofs_filename_suffix,
                            time_deltas=[day_delta],
                            fill_value=LEAFLET_NODATA_VALUE,
                            driver='GTiff',
                        )
                    else:
                        LOGGER.debug(f'Skipping WCOFS day {day_delta} scalar variables')

                    if not all(
                            any(vector_variable in filename for filename in existing_files)
                            for vector_variable in vector_variables
                    ):
                        wcofs_dataset.write_rasters(
                            daily_average_dir,
                            vector_variables,
                            filename_suffix=wcofs_filename_suffix,
                            time_deltas=[day_delta],
                            fill_value=LEAFLET_NODATA_VALUE,
                            driver='AAIGrid',
                        )
                    else:
                        LOGGER.debug(f'Skipping WCOFS day {day_delta} uv')
        del wcofs_dataset

        if grid_size_km == 2:
            wcofs.reset_dataset_grid()
    except NoDataError as error:
        LOGGER.warning(f'{error.__class__.__name__}: {error}')
    except:
        LOGGER.exception(f'model run date: {model_run_date}, day deltas: {day_deltas}')


def write_observations(
        output_dir: PathLike,
        output_date: Union[datetime, date, int, float],
        day_deltas: range = MODEL_DAY_DELTAS['WCOFS'],
):
    """
    Writes daily average of observational data on given date.

    :param output_dir: output directory to write files
    :param output_date: date of data run
    :param day_deltas: time deltas for which to write model output
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    LOGGER.info(f'Starting file conversion for {output_date}')

    LOGGER.info('Processing HFR SSUV...')
    write_observation(output_dir, output_date, 'hf_radar')
    LOGGER.info('Processing VIIRS SST...')
    write_observation(output_dir, output_date, 'viirs')
    LOGGER.info('Processing SMAP SSS...')
    write_observation(output_dir, output_date, 'smap')
    # LOGGER.info('Processing NDBC data...')
    # write_observation(output_dir, output_date, 'data_buoy')
    LOGGER.info(f'Wrote observations to {output_dir}')


def write_models(
        output_dir: PathLike,
        output_date: Union[datetime, date, int, float],
        day_deltas: range = MODEL_DAY_DELTAS['WCOFS'],
):
    """
    Writes daily average of model output on given date.

    :param output_dir: output directory to write files
    :param output_date: date of data run
    :param day_deltas: time deltas for which to write model output
    """

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    LOGGER.info('Processing RTOFS...')  # RTOFS forecast is uploaded at 1700 UTC
    write_rtofs(output_dir, output_date, day_deltas)
    LOGGER.info('Processing WCOFS DA...')
    write_wcofs(output_dir, output_date, day_deltas)
    # LOGGER.info('Processing WCOFS experimental DA...')
    # write_wcofs(output_dir, output_date, day_deltas, source_url=DATA_DIRECTORY / 'input/wcofs/option',
    #             use_defaults=False, suffix='exp')
    LOGGER.info('Processing WCOFS noDA...')
    write_wcofs(output_dir, output_date, day_deltas, data_assimilation=False)
    LOGGER.info(f'Wrote models to {output_dir}')


if __name__ == '__main__':
    # create folders if they do not exist
    for dir_path in [OUTPUT_DIRECTORY, LOG_DIRECTORY]:
        if not dir_path.is_dir():
            os.makedirs(dir_path, exist_ok=True)

    # define dates over which to collect data (dates after today are for WCOFS forecast)
    day_deltas = MODEL_DAY_DELTAS['WCOFS']

    start_time = datetime.now()

    # from PyOFS import range_daily
    #
    # model_run_dates = range_daily(datetime.today(), datetime(2020, 6, 19))
    # for model_run_date in model_run_dates:
    #     LOGGER.info(f'Starting file conversion for {model_run_date:%Y%m%d}')
    #     write_observations(OUTPUT_DIRECTORY, model_run_date, day_deltas)
    #     write_models(OUTPUT_DIRECTORY, model_run_date, day_deltas)

    model_run_date = date.today()
    LOGGER.info(f'Starting file conversion for {model_run_date:%Y%m%d}')
    write_observations(OUTPUT_DIRECTORY, model_run_date, day_deltas)
    write_models(OUTPUT_DIRECTORY, model_run_date, day_deltas)

    LOGGER.info(
        f'Finished writing files. Total time: {(datetime.now() - start_time) / timedelta(seconds=1):.2f} seconds'
    )

    start_time = datetime.now()

    files_json_filename = REFERENCE_DIRECTORY / 'files.json'
    write_json.dir_structure_to_json(OUTPUT_DIRECTORY, files_json_filename)

    with open(DATA_DIRECTORY / 'azure_credentials.txt') as credentials_file:
        azure_blob_url, credentials = (
            line.strip('\n') for line in credentials_file.readlines()
        )

    azcopy_path = r'C:\Working\azcopy.exe'

    upload_to_azure(
        files_json_filename,
        f'{azure_blob_url}/reference/files.json',
        credentials,
        overwrite=True,
        azcopy_path=azcopy_path,
    )
    sync_with_azure(
        OUTPUT_DIRECTORY, f'{azure_blob_url}/output', credentials, azcopy_path=azcopy_path
    )

    LOGGER.info(
        f'Finished uploading files. Total time: {(datetime.now() - start_time) / timedelta(seconds=1):.2f} seconds'
    )

    print('done')
