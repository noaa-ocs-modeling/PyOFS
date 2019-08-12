import os

CRS_EPSG = 4326

if 'OFS_DATA' in os.environ:
    DATA_DIR = os.environ['OFS_DATA']
else:
    DATA_DIR = r"C:\data\OFS"

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999.0

# # for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
