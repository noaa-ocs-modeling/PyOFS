import sys
import os

from config import CONDA_ENV, DATA_DIR

sys.path.append(os.path.join(CONDA_ENV, 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Scripts'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'usr', 'bin'))
sys.path.append(os.path.join(CONDA_ENV, 'Library', 'mingw-w64', 'bin'))
os.environ['PATH'] = ';'.join(sys.path)

os.environ['GDAL_DATA'] = os.path.join(CONDA_ENV, 'Library', 'share', 'gdal')

CRS_EPSG = 4326
