import os

AZCOPY_PATH = r"C:\Working\azcopy.exe"


def upload_to_azure(local_path: str, remote_path: str, credentials: str, overwrite: bool = False):
    print(f'Uploading {local_path} to {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    azcopy_dir, azcopy_filename = os.path.split(AZCOPY_PATH)
    os.chdir(azcopy_dir)
    os.system(f'{azcopy_filename} copy "{local_path}" "{remote_path}?{credentials}" ' +
              f'--overwrite={str(overwrite).lower()} --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5')


def sync_with_azure(local_path: str, remote_path: str, credentials: str):
    print(f'Synchronizing {local_path} with {remote_path}')

    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    azcopy_dir, azcopy_filename = os.path.split(AZCOPY_PATH)
    os.chdir(azcopy_dir)
    os.system(f'{azcopy_filename} sync "{local_path}" "{remote_path}?{credentials}"')


if __name__ == '__main__':
    local_data_path = r'D:\data'

    with open(r"D:\data\azure_credentials.txt") as credentials_file:
        azure_blob_url, credentials = (line.strip('\n') for line in credentials_file.readlines())

    sync_with_azure(os.path.join(local_data_path, 'output'), f'{azure_blob_url}/$web/data/output', credentials)
    sync_with_azure(os.path.join(local_data_path, 'reference'), f'{azure_blob_url}/$web/data/reference', credentials)
    sync_with_azure(os.path.join(local_data_path, 'logs'), f'{azure_blob_url}/$logs', credentials)
