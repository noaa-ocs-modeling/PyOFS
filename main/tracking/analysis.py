# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import os
from typing import List

import fiona.crs
import numpy
import shapely.geometry


def diffusion(polygons: List[shapely.geometry.Polygon]):
    for polygon in polygons:
        centroid = polygon.centroid

        max_radius = max(centroid.distance(vertex) for vertex in
                         (shapely.geometry.Point(point) for point in zip(*polygon.exterior.xy)))

        radius_interval = 500

        for radius in range(radius_interval, max_radius, step=radius_interval):
            analysis_area = shapely.geometry.Polygon(centroid.buffer(radius + radius_interval), centroid.buffer(radius))
            polygon.intersection(analysis_area)


if __name__ == '__main__':
    import sys
    from matplotlib import pyplot

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

    from PyOFS import DATA_DIR

    velocity_products = ['hourly_modeled', 'hourly_geostrophic', 'daily_modeled']
    contour_names = [f'{letter}{number}' for letter in ['A', 'B', 'C'] for number in range(1, 5)]

    print(f'[{datetime.datetime.now()}]: Reading...')

    values = {'area': {contour_name: {} for contour_name in contour_names},
              'area_change': {contour_name: {} for contour_name in contour_names}}

    for velocity_product in velocity_products:
        time_delta = datetime.timedelta(hours=1) if 'hourly' in velocity_product else datetime.timedelta(days=1)

        if 'modeled' in velocity_product:
            source = 'wcofs_qck'
        elif 'geostrophic' in velocity_product:
            source = 'wcofs_qck_geostrophic'

        start_time = datetime.datetime(2016, 9, 25, 1)
        period = datetime.timedelta(days=4)

        time_deltas = [time_delta for index in range(int(period / time_delta))]

        output_path = os.path.join(DATA_DIR, 'output', 'test', 'contours.gpkg')
        layer_name = f'{source}_{start_time.strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{(start_time + period).strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{int(time_delta.total_seconds() / 3600)}h'

        with fiona.open(output_path, layer=layer_name) as contours_file:
            for contour_name in sorted(numpy.unique([feature['properties']['name'] for feature in contours_file])):
                contour_steps = {record['properties']['datetime']: record for record in
                                 filter(lambda record: record['properties']['name'] == contour_name, contours_file)}

                values['area'][contour_name][velocity_product] = {}
                values['area_change'][contour_name][velocity_product] = {}

                previous_area = None

                for contour_datetime in sorted(contour_steps):
                    contour_step = contour_steps[contour_datetime]
                    current_area = shapely.geometry.Polygon(contour_step["geometry"]["coordinates"][0]).area

                    area_change = current_area - previous_area if previous_area is not None else 0

                    values['area'][contour_name][velocity_product][contour_datetime] = current_area
                    values['area_change'][contour_name][velocity_product][contour_datetime] = area_change

                    previous_area = current_area

    print(f'[{datetime.datetime.now()}]: Plotting...')

    for contour_name, contour_steps in values['area_change'].items():
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

        for velocity_product, values in contour_steps.items():
            times = numpy.array(list(values.keys())).astype(numpy.datetime64)
            axis.plot(times, values.values(), label=velocity_product)

        axis.legend()

    pyplot.show()

    print('done')
