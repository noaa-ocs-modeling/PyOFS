"""
Range comparisons of PyOFS source data.

Created on Jun 25, 2018

@author: zachary.burnett
"""

import os
import sys

from qgis.core import QgsApplication

from dataset import hfr, viirs, wcofs

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

DATA_DIR = os.environ['OFS_DATA']

STUDY_AREA_POLYGON_FILENAME = os.path.join(DATA_DIR, r"reference\wcofs.gpkg:study_area")
DAILY_AVERAGE_DIR = os.path.join(DATA_DIR, r'output\daily_averages')

if __name__ == '__main__':
    qgis_application = QgsApplication([], True, None)
    qgis_application.setPrefixPath(os.environ['QGIS_PREFIX_PATH'], True)
    qgis_application.initQgis()

    import argparse
    import dateutil.parser

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', help='Starting datetime in format YYYYmmddHHMMSS')
    parser.add_argument('-e', '--end', help='Ending datetime in format YYYYmmddHHMMSS')
    parser.add_argument('-o', '--outdir', help='Directory in which to store rasters.')
    parser.add_argument('-a', '--area', help='Path to vector file of study area polygon.')

    args = parser.parse_args()
    start_datetime = dateutil.parser.parse(args.start)
    end_datetime = dateutil.parser.parse(args.end)
    output_dir = args.outdir
    study_area_polygon_filename = args.area

    # ensure output directory exists
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print('Processing HFR')
    hfr_range = hfr.HFR_Range(start_datetime, end_datetime)
    hfr_range.write_rasters(output_dir)
    # hfr_range.write_vector(os.path.join(output_dir, 'hfr.gpkg'))

    print('Processing VIIRS')
    viirs_range = viirs.VIIRS_Range(start_datetime, end_datetime)
    viirs_range.write_raster(output_dir)

    print('Processing PyOFS')
    wcofs_range = wcofs.WCOFS_Range(start_datetime, end_datetime, source='avg')
    wcofs_range.write_rasters(output_dir, ['temp', 'u', 'v'])
    # wcofs_range.write_vector(os.path.join(output_dir, 'wcofs.gpkg'))

    print('done')
