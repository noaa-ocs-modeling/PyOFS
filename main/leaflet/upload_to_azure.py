import os

AZCOPY_PATH = r"C:\Program Files (x86)\Microsoft Azure Storage Explorer\resources\app\node_modules\se-az-copy-exe-win\dist\bin\azcopy_windows_amd64.exe"


def upload_to_azure(local_path: str, remote_path: str, sas_key: str, overwrite: bool = False):
    command = f'$env:AZCOPY_CRED_TYPE = "Anonymous"; {AZCOPY_PATH} copy "{local_path}" "{remote_path}?se={sas_key}" ' + \
              f'--overwrite={str(overwrite).lower()} --follow-symlinks --recursive --from-to=LocalBlob ' + \
              f'--blob-type=BlockBlob --put-md5; $env:AZCOPY_CRED_TYPE = "";'

    os.system(command)


if __name__ == '__main__':
    local_path = r'D:\data\output'
    remote_path = 'https://ocscmmbstore1.blob.core.windows.net/cmmb/data/output'
    sas_key = ''
    overwrite = True

    upload_to_azure(local_path, remote_path, sas_key, overwrite)
