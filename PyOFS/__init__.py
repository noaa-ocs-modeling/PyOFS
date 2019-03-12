import os
import sys

CONDA_BIN = r'C:\Languages\Python\Miniconda3\condabin'
CONDA_ENV = r'C:\Environments\conda\GIS'

sys.path.append(CONDA_BIN)
sys.path.append(os.path.join(CONDA_ENV, 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Scripts'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'usr', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'mingw-w64', 'bin'))
os.environ['PATH'] = ';'.join(sys.path)

os.environ['GDAL_DATA'] = os.path.join(CONDA_ENV, 'Library', 'share', 'gdal')

DATA_DIR = os.environ['OFS_DATA']

CRS_EPSG = 4326

# for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
