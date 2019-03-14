import os

CRS_EPSG = 4326

DATA_DIR = os.environ['OFS_DATA']

# for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
