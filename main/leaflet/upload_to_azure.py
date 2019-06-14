import os

AZCOPY_PATH = r"C:\Program Files (x86)\Microsoft Azure Storage Explorer\resources\app\node_modules\se-az-copy-exe-win\dist\bin\azcopy_windows_amd64.exe"


def upload_to_azure(local_path: str, remote_path: str, credentials: str, overwrite: bool = False):
    os.environ['AZCOPY_CRED_TYPE'] = 'Anonymous'
    azcopy_dir, azcopy = os.path.split(AZCOPY_PATH)
    os.chdir(azcopy_dir)
    os.system(f'{azcopy} copy "{local_path}" "{remote_path}?{credentials}" --overwrite={str(overwrite).lower()} ' + \
              f'--follow-symlinks --recursive --from-to=LocalBlob --blob-type=BlockBlob --put-md5')


if __name__ == '__main__':
    local_data_path = r'D:\data'
    remote_data_path = 'https://ocscoastalmodelingsa.blob.core.windows.net/$web/data'

    with open(r"D:\www\azure_credentials.txt") as credentials_file:
        credentials = credentials_file.readline()

    upload_to_azure(os.path.join(local_data_path, 'reference'), f'{remote_data_path}/reference', credentials,
                    overwrite=True)
    upload_to_azure(os.path.join(local_data_path, 'output'), f'{remote_data_path}/output', credentials, overwrite=True)
