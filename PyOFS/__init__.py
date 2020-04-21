import os

CRS_EPSG = 4326

if 'OFS_DATA' in os.environ:
    DATA_DIRECTORY = os.environ['OFS_DATA']
else:
    DATA_DIRECTORY = r"C:\data\OFS"

# default nodata value used by leaflet-geotiff renderer
LEAFLET_NODATA_VALUE = -9999.0

# # for development branch
# DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'develop')

TIFF_CREATION_OPTIONS = {
    'TILED': 'YES',
    'COMPRESS': 'DEFLATE',
    'PREDICTOR': 3,
    'NUM_THREADS': 'ALL_CPUS',
    'BIGTIFF': 'IF_SAFER'
}
