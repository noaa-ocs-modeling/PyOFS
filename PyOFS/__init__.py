import os

CRS_EPSG = 4326

if 'OFS_DATA' in os.environ:
    DATA_DIR = os.environ['OFS_DATA']
else:
    DATA_DIR = r"C:\data\OFS"

# # for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
