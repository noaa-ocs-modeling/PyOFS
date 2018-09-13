@echo off

rem set paths to OSGEO4W installation and to desired editor
set OSGEO4W_ROOT="C:\Program Files\QGIS 3.0"
set EDITOR_PATH="C:\Working\PyCharmPortable2018\PyCharmPortable.exe"

rem run environment setup batch files
call %OSGEO4W_ROOT%\bin\o4w_env.bat
call %OSGEO4W_ROOT%\bin\qt5_env.bat
call %OSGEO4W_ROOT%\bin\py3_env.bat

rem Python environment setup from python-quis.bat and set paths
path %OSGEO4W_ROOT%/apps/qgis/bin;%OSGEO4W_ROOT%/apps/Python36/Scripts;%PATH%
set QGIS_PREFIX_PATH=%OSGEO4W_ROOT%/apps/qgis

rem set GDAL variables
set GDAL_FILENAME_IS_UTF8=YES
set GDAL_HTTP_UNSAFESSL=YES

rem set VSI cache to be used as buffer, see #6448
set VSI_CACHE=TRUE
set VSI_CACHE_SIZE=1000000

rem set other paths
set PYTHONPATH=%PYTHONPATH%;%QGIS_PREFIX_PATH%/python;C:\ProgramData\Anaconda3\Lib

set QT_PLUGIN_PATH=%QT_PLUGIN_PATH%;%QGIS_PREFIX_PATH%/qtplugins

rem start editor
start "editor" %EDITOR_PATH% 
rem -vm  "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\javaw.exe"
