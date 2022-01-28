import os
from os import PathLike
from pathlib import Path
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import NoCredentialsError

from PyOFS import get_logger

LOGGER = get_logger('PyOFS.azure')


def upload_to_azure(
    local_path: PathLike,
    remote_path: PathLike,
    credentials: str,
    overwrite: bool = False,
    azcopy_path: PathLike = None,
    **kwargs,
):
    if not isinstance(azcopy_path, Path):
        azcopy_path = Path(azcopy_path)

    LOGGER.info(f'Uploading {local_path} to {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    if azcopy_path is not None:
        azcopy_dir = azcopy_path.parent
        azcopy_filename = azcopy_path.name
        os.chdir(azcopy_dir)
    else:
        azcopy_filename = 'azcopy.exe'

    kwargs_string = ' '.join(f'--{key}={value}' for key, value in kwargs.items())
    os.system(
        f'{azcopy_filename} copy "{local_path}" "{remote_path}?{credentials}" '
        f'--overwrite={str(overwrite).lower()} --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5 {kwargs_string}'
    )


def sync_with_azure(
    local_path: PathLike,
    remote_path: PathLike,
    credentials: str,
    azcopy_path: PathLike = None,
    **kwargs,
):
    if not isinstance(azcopy_path, Path):
        azcopy_path = Path(azcopy_path)

    LOGGER.info(f'Synchronizing {local_path} with {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    if azcopy_path is not None:
        azcopy_dir = azcopy_path.parent
        azcopy_filename = azcopy_path.name
        os.chdir(azcopy_dir)
    else:
        azcopy_filename = 'azcopy.exe'

    kwargs_string = ' '.join(f'--{key}={value}' for key, value in kwargs.items())
    os.system(
        f'{azcopy_filename} sync "{local_path}" "{remote_path}?{credentials}" {kwargs_string}'
    )


def upload_to_aws(local_file, bucket_name, s3_file, ACCESS_KEY, SECRET_KEY):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket_name, s3_file)
        print("Upload Successful for files.json")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def sync_with_aws(local_folder, bucket_name, ACCESS_KEY, SECRET_KEY):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    upload_start_time = datetime.now() - timedelta(days=10)

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            if os.path.basename(root) > f'{upload_start_time:%Y%m%d}':
                s3_filename = 'WCOFS/viewer/data/output/daily_averages/' + os.path.basename(root) + '/' + file
                file_exist = key_existing_size__list(s3, bucket_name, s3_filename)  # check if the file have size
                if file_exist is None:
                    try:
                        s3.upload_file(os.path.join(root, file), bucket_name, s3_filename)

                    except ClientError as exc:
                        if exc.response['Error']['Code'] != '404':
                            raise

    print("Upload Successful for daily average data")


def key_existing_size__list(client, bucket_name, file_name):
    """return the key's size if it exist, else None"""
    response = client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=file_name,
    )
    for obj in response.get('Contents', []):
        if obj['Key'] == file_name:
            return obj['Size']


if __name__ == '__main__':
    local_data_path = Path(r'D:\data\OFS')
    azcopy_path = Path(r'C:\Working\azcopy.exe')
    azure_credentials_filename = Path(r'D:\data\azure_credentials.txt')

    with open(azure_credentials_filename) as credentials_file:
        azure_blob_url, credentials = (
            line.strip('\n') for line in credentials_file.readlines()
        )

    upload_to_azure(
        local_data_path / 'reference' / 'files.json',
        f'{azure_blob_url}/data/reference/files.json',
        credentials,
        overwrite=True,
        azcopy_path=azcopy_path,
    )
    sync_with_azure(
        local_data_path / 'output',
        f'{azure_blob_url}/data/output',
        credentials,
        azcopy_path=azcopy_path,
    )

    print('done')
