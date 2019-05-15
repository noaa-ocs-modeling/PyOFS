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

    plot_dir = r"C:\Users\Zach\Documents\school\graduate\GEOG797\plots"
    contour_starting_radius = 10000
    contour_starting_area = numpy.pi * contour_starting_radius ** 2

    velocity_products = ['hourly_modeled', 'hourly_geostrophic', 'daily_modeled']
    contour_names = [f'{letter}{number}' for number in range(1, 5) for letter in ['A', 'B', 'C']]

    print(f'[{datetime.datetime.now()}]: Reading...')

    values = {'area': {contour_name: {} for contour_name in contour_names},
              'area_change': {contour_name: {} for contour_name in contour_names}}

    for velocity_product in velocity_products:
        time_delta = datetime.timedelta(hours=1) if 'hourly' in velocity_product else datetime.timedelta(days=1)
        source = 'wcofs_qck' if 'modeled' in velocity_product else 'wcofs_qck_geostrophic'

        start_time = datetime.datetime(2016, 9, 25, 1)
        period = datetime.timedelta(days=4)

        time_deltas = [time_delta for index in range(int(period / time_delta))]

        output_path = os.path.join(DATA_DIR, 'output', 'test', 'contours.gpkg')
        layer_name = f'{source}_{start_time.strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{(start_time + period).strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{int(time_delta.total_seconds() / 3600)}h'

        with fiona.open(output_path, layer=layer_name) as contours_file:
            for contour_name in sorted(numpy.unique([feature['properties']['name'] for feature in contours_file])):
                contour_steps = {
                    datetime.datetime.strptime(record['properties']['datetime'], '%Y-%m-%dT%H:%M:%S'): record for record
                    in filter(lambda record: record['properties']['name'] == contour_name, contours_file)}

                values['area'][contour_name][velocity_product] = {}
                values['area_change'][contour_name][velocity_product] = {}

                previous_datetime = None
                previous_area = None

                for contour_datetime in sorted(contour_steps):
                    contour_step = contour_steps[contour_datetime]
                    current_area = shapely.geometry.Polygon(contour_step["geometry"]["coordinates"][0]).area

                    area_change = (current_area - previous_area) / (
                            contour_datetime - previous_datetime).total_seconds() if previous_area is not None and previous_datetime is not None else 0

                    values['area'][contour_name][velocity_product][contour_datetime] = current_area
                    values['area_change'][contour_name][velocity_product][contour_datetime] = area_change

                    previous_datetime = contour_datetime
                    previous_area = current_area

    value_type = 'area'

    print(f'[{datetime.datetime.now()}]: Plotting...')
    for contour_name, contour_steps in values[value_type].items():
        print(f'[{datetime.datetime.now()}]: Plotting {contour_name}...')

        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        axis.set_title(f'starting point {contour_name}')

        for velocity_product, contour_values in contour_steps.items():
            times = numpy.array(list(contour_values.keys())).astype(numpy.datetime64)
            axis.plot(times, contour_values.values(), marker='o', markersize=3, label=velocity_product)

        axis.axhline(y=contour_starting_area, linestyle=':', color='k', zorder=0)
        axis.legend()
        pyplot.xticks(rotation=-45, ha='left', rotation_mode='anchor')
        pyplot.tight_layout()

        figure.savefig(os.path.join(plot_dir, value_type, f'{value_type}_{contour_name}.pdf'), orientation='landscape',
                       papertype='A4')
        # pyplot.show()

    value_type = 'area_change'

    boxplot_figure = pyplot.figure()
    boxplot_axis = boxplot_figure.add_subplot(1, 1, 1)
    boxplot_axis.set_ylabel(f'{value_type} (m^2/s)')

    contour_values = {}
    for contour_name in contour_names:
        contour_steps = values[value_type][contour_name]
        contour_values[contour_name] = list(contour_steps['hourly_modeled'].values())

    boxplot_axis.boxplot(contour_values.values(), labels=contour_values.keys())
    # axis.axhline(y=contour_starting_area)
    boxplot_axis.axhline(y=0, color='k', linestyle='--')

    boxplot_figure.savefig(os.path.join(plot_dir, f'{value_type}_boxplot.pdf'), orientation='landscape', papertype='A4')
    # pyplot.show()

    print('done')
