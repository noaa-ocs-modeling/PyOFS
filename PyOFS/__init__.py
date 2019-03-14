import os

# PATH=D:\Python\conda\GIS;D:\Python\conda\GIS\lib\site-packages;D:\Python\conda\GIS\bin;D:\Python\conda\GIS\Scripts;D:\Python\conda\GIS\Library\bin;D:\Python\conda\GIS\Library\usr\bin;D:\Python\conda\GIS\Library\mingw-w64\bin
# GDAL_DATA=D:\Python\conda\GIS\Library\share\gdal

CRS_EPSG = 4326

DATA_DIR = os.environ['OFS_DATA']

# for development branch
DATA_DIR = os.path.join(DATA_DIR, 'develop')
