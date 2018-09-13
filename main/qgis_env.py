import os
import subprocess
import sys


def setup_env(osgeo4w_root: str):
    """
    Set up environment for QGIS.

    :param str osgeo4w_root: Directory of OSGeo installation (usually QGIS).
    """

    os.environ['OSGEO4W_ROOT'] = osgeo4w_root

    subprocess.check_output(fr'C:\Windows\System32\cmd.exe /c "{osgeo4w_root}\bin\o4w_env.bat"')
    subprocess.check_output(fr'C:\Windows\System32\cmd.exe /c "{osgeo4w_root}\bin\qt5_env.bat"')
    subprocess.check_output(fr'C:\Windows\System32\cmd.exe /c "{osgeo4w_root}\bin\py3_env.bat"')

    os.environ['PATH'] = r'%OSGEO4W_ROOT%/apps/qgis/bin;%OSGEO4W_ROOT%/apps/Python36/Scripts;%PATH%'
    os.environ['QGIS_PREFIX_PATH'] = r'%OSGEO4W_ROOT%/apps/qgis'

    os.environ['GDAL_FILENAME_IS_UTF8'] = 'YES'
    os.environ['GDAL_HTTP_UNSAFESSL'] = 'YES'

    os.environ['VSI_CACHE'] = 'TRUE'
    os.environ['VSI_CACHE_SIZE'] = '1000000'

    sys.path.append(r'%QGIS_PREFIX_PATH%/python')

    os.environ['QT_PLUGIN_PATH'] = r'%QT_PLUGIN_PATH%;%QGIS_PREFIX_PATH%/qtplugins'
