# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import functools
import os
from typing import List

import fiona
import numpy
import pandas
from matplotlib import pyplot
from shapely import geometry


def diffusion(polygons: List[geometry.Polygon]):
    for polygon in polygons:
        centroid = polygon.centroid

        max_radius = max(centroid.distance(vertex) for vertex in
                         (geometry.Point(point) for point in zip(*polygon.exterior.xy)))

        radius_interval = 500

        for radius in range(radius_interval, max_radius, step=radius_interval):
            analysis_area = geometry.Polygon(centroid.buffer(radius + radius_interval), centroid.buffer(radius))
            polygon.intersection(analysis_area)


if __name__ == '__main__':
    import sys

    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

    from PyOFS import DATA_DIR

    plot_dir = r"R:\documents\plots"
    contour_starting_radius = 10000

    velocity_products = ['hourly_modeled', 'hourly_geostrophic', 'daily_modeled']
    contour_names = [f'{letter}{number}' for number in range(1, 5) for letter in ['A', 'B', 'C']]

    start_time = datetime.datetime(2016, 9, 25, 1)
    period = datetime.timedelta(days=4)

    values = {contour_name: {} for contour_name in contour_names}

    for velocity_product in velocity_products:
        time_delta = datetime.timedelta(hours=1) if 'hourly' in velocity_product else datetime.timedelta(days=1)
        source = 'wcofs_qck' if 'modeled' in velocity_product else 'wcofs_qck_geostrophic'

        input_path = os.path.join(DATA_DIR, 'output', 'test', 'contours.gpkg')
        layer_name = f'{source}_{start_time.strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{(start_time + period).strftime("%Y%m%dT%H%M%S")}_' + \
                     f'{int(time_delta.total_seconds() / 3600)}h'

        print(f'[{datetime.datetime.now()}]: Reading {input_path}...')
        with fiona.open(input_path, layer=layer_name) as contours_file:
            contour_names = sorted(numpy.unique([feature['properties']['contour'] for feature in contours_file]))

            for contour_name in contour_names:
                contour_datetimes = []
                contour_areas = []
                contour_perimeters = []

                contour_records = [record for record in
                                   filter(lambda record: record['properties']['contour'] == contour_name,
                                          contours_file)]

                for record in contour_records:
                    contour_datetime = datetime.datetime.strptime(record['properties']['datetime'], '%Y-%m-%dT%H:%M:%S')
                    contour_polygon = geometry.Polygon(record['geometry']['coordinates'][0])

                    contour_datetimes.append(contour_datetime)
                    contour_areas.append(contour_polygon.area)
                    contour_perimeters.append(contour_polygon.length)

                values[contour_name][velocity_product] = pandas.DataFrame(
                    {'datetime': contour_datetimes, 'area': contour_areas, 'perimeter': contour_perimeters})

    plotting_values = {'area': 'm^2', 'area_change': 'm^2/s', 'perimeter': 'm'}

    print(f'[{datetime.datetime.now()}]: Plotting...')

    # for contour_name, contour_velocity_products in values.items():
    #     print(f'[{datetime.datetime.now()}]: Plotting {contour_name}...')
    #
    #     for plotting_value, plotting_unit in plotting_values.items():
    #         figure = pyplot.figure()
    #         axis = figure.add_subplot(1, 1, 1)
    #         axis.set_title(f'starting point {contour_name} {start_time.strftime("%Y%m%dT%H%M%S")}')
    #         axis.set_ylabel(f'{plotting_value} ({plotting_unit})')
    #
    #         for velocity_product, contour_values in contour_velocity_products.items():
    #             if plotting_value == 'area_change':
    #                 plotting_data = numpy.concatenate(([0], numpy.diff(contour_values['area'])))
    #             else:
    #                 plotting_data = contour_values[plotting_value]
    #
    #             axis.plot(contour_values['datetime'], plotting_data, marker='o', markersize=3, label=velocity_product)
    #
    #         if plotting_value == 'area':
    #             starting_value = numpy.pi * contour_starting_radius ** 2
    #         elif plotting_value == 'perimeter':
    #             starting_value = numpy.pi * contour_starting_radius * 2
    #         else:
    #             starting_value = 0
    #
    #         axis.axhline(y=starting_value, linestyle=':', color='k', zorder=0)
    #
    #         axis.legend()
    #         pyplot.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    #         pyplot.tight_layout()
    #
    #         figure.savefig(os.path.join(plot_dir, plotting_value,
    #                                     f'{plotting_value}_{contour_name}.pdf'),
    #                        orientation='landscape', papertype='A4')
    #
    #         # pyplot.show()
    #         pyplot.clf()

    for velocity_product in velocity_products:
        contour_values = functools.reduce(lambda left, right: pandas.merge(left, right, on='datetime'),
                                          [contour_velocity_products[velocity_product] for contour_velocity_products
                                           in values.values()])

        for plotting_value, plotting_unit in plotting_values.items():
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
            axis.set_ylabel(f'{plotting_value} ({plotting_unit})')

            axis.boxplot(contour_values[plotting_value], labels=contour_names)

            if plotting_value == 'area':
                starting_value = numpy.pi * contour_starting_radius ** 2
            elif plotting_value == 'perimeter':
                starting_value = numpy.pi * contour_starting_radius * 2
            else:
                starting_value = 0

            axis.axhline(y=starting_value, linestyle=':', color='k', zorder=0)

            figure.savefig(os.path.join(plot_dir, f'{velocity_product}_{plotting_value}_boxplot.pdf'),
                           orientation='landscape', papertype='A4')
            # pyplot.show()

    print('done')
