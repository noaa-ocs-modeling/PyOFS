import os

from PyOFS import get_logger

LOGGER = get_logger('PyOFS.azure')


def upload_to_azure(local_path: str, remote_path: str, credentials: str, overwrite: bool = False, azcopy_path: str = None, **kwargs):
    LOGGER.info(f'Uploading {local_path} to {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    if azcopy_path is not None:
        azcopy_dir, azcopy_filename = os.path.split(azcopy_path)
        os.chdir(azcopy_dir)
    else:
        azcopy_filename = 'azcopy.exe'

    kwargs_string = ' '.join(f'--{key}={value}' for key, value in kwargs.items())
    os.system(f'{azcopy_filename} copy "{local_path}" "{remote_path}?{credentials}" '
              f'--overwrite={str(overwrite).lower()} --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5 {kwargs_string}')


def sync_with_azure(local_path: str, remote_path: str, credentials: str, azcopy_path: str = None, **kwargs):
    LOGGER.info(f'Synchronizing {local_path} with {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    if azcopy_path is not None:
        azcopy_dir, azcopy_filename = os.path.split(azcopy_path)
        os.chdir(azcopy_dir)
    else:
        azcopy_filename = 'azcopy.exe'

    kwargs_string = ' '.join(f'--{key}={value}' for key, value in kwargs.items())
    os.system(f'{azcopy_filename} sync "{local_path}" "{remote_path}?{credentials}" {kwargs_string}')


if __name__ == '__main__':
    local_data_path = r'D:\data'
    azcopy_path = r"C:\Working\azcopy.exe"

    with open(r"D:\data\azure_credentials.txt") as credentials_file:
        azure_blob_url, credentials = (line.strip('\n') for line in credentials_file.readlines())

    upload_to_azure(local_data_path / 'reference' / 'files.json', f'{azure_blob_url}/reference/files.json', credentials, overwrite=True, azcopy_path=azcopy_path)
    sync_with_azure(local_data_path / 'output', f'{azure_blob_url}/output', credentials, azcopy_path=azcopy_path)

    print('done')
