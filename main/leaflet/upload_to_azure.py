import os

AZCOPY_PATH = r"C:\Program Files (x86)\Microsoft Azure Storage Explorer\resources\app\node_modules\se-az-copy-exe-win\dist\bin\azcopy_windows_amd64.exe"
POWERSHELL_PATH = r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"


def upload_to_azure(local_path: str, remote_path: str, credentials: str, overwrite: bool = False):
    command = f'$env:AZCOPY_CRED_TYPE = "Anonymous"; {AZCOPY_PATH} copy "{local_path}" "{remote_path}?{credentials}" ' + \
              f'--overwrite={str(overwrite).lower()} --follow-symlinks --recursive --from-to=LocalBlob ' + \
              f'--blob-type=BlockBlob --put-md5; $env:AZCOPY_CRED_TYPE = "";'

    os.chdir(os.path.split(POWERSHELL_PATH)[0])
    os.system(f'powershell.exe < {command}')


if __name__ == '__main__':
    local_data_path = r'D:\data'
    remote_data_path = 'https://ocscmmbstore1.blob.core.windows.net/cmmb/data'

    with open(r"D:\www\azure_credentials.txt") as credentials_file:
        credentials = credentials_file.readline()

    upload_to_azure(os.path.join(local_data_path, 'reference'), f'{remote_data_path}/reference', credentials,
                    overwrite=True)
    upload_to_azure(os.path.join(local_data_path, 'output'), f'{remote_data_path}/output', credentials, overwrite=True)
