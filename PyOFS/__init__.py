import os

import sys

if 'CONDA_GIS' in os.environ:
    CONDA_ENV = os.environ['CONDA_GIS']
else:
    CONDA_ENV = r"D:\Python\conda\GIS"

if 'OFS_DATA' in os.environ:
    DATA_DIR = os.environ['OFS_DATA']
else:
    DATA_DIR = r"D:\Data"

sys.path.append(os.path.join(CONDA_ENV, 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Scripts'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'usr', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'mingw-w64', 'bin'))
os.environ['PATH'] = ';'.join(sys.path)

os.environ['GDAL_DATA'] = os.path.join(CONDA_ENV, 'Library', 'share', 'gdal')

CRS_EPSG = 4326

# for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
