# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

from datetime import datetime, timedelta
import os
import sys

import fiona
from matplotlib import pyplot
import numpy
import pandas
from shapely import geometry

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from PyOFS import get_logger, DATA_DIRECTORY

LOGGER = get_logger('PyOFS.track')


def diffusion(polygons: [geometry.Polygon]):
    for polygon in polygons:
        centroid = polygon.centroid

        max_radius = max(centroid.distance(vertex) for vertex in (geometry.Point(point) for point in zip(*polygon.exterior.xy)))

        radius_interval = 500

        for radius in range(radius_interval, max_radius, step=radius_interval):
            analysis_area = geometry.Polygon(centroid.buffer(radius + radius_interval), centroid.buffer(radius))
            polygon.intersection(analysis_area)


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    plot_dir = r"R:\documents\plots"
    contour_starting_radius = 50000

    # velocity_products = ['hourly_modeled', 'hourly_geostrophic', 'daily_modeled']
    velocity_products = ['daily_modeled']
    contour_names = [f'{letter}{number}' for number in range(1, 5) for letter in ['A', 'B', 'C']]
    # contour_names = (str(item) for item in range(1, 8))

    start_time = datetime(2016, 9, 25, 1)
    period = timedelta(days=4)

    # whether to plot percentages of the starting value (instead of actual values)
    plot_percentages = True

    values = {contour_name: {} for contour_name in contour_names}

    for velocity_product in velocity_products:
        time_delta = timedelta(hours=1) if 'hourly' in velocity_product else timedelta(days=1)
        source = 'wcofs_qck' if 'modeled' in velocity_product else 'wcofs_qck_geostrophic'

        input_path = os.path.join(DATA_DIRECTORY, 'output', 'test', 'contours.gpkg')
        layer_name = f'{source}_{start_time:%Y%m%dT%H%M%S}_{(start_time + period):"%Y%m%dT%H%M%S"}_{int(time_delta.total_seconds() / 3600)}h'

        LOGGER.info(f'[{datetime.now()}]: Reading {input_path}...')
        with fiona.open(input_path, layer=layer_name) as contours_file:
            contour_names = sorted(numpy.unique([feature['properties']['contour'] for feature in contours_file]))

            for contour_name in contour_names:
                contour_datetimes = []
                contour_areas = []
                contour_perimeters = []

                contour_records = [record for record in filter(lambda record: record['properties']['contour'] == contour_name, contours_file)]

                for record in contour_records:
                    contour_datetime = datetime.strptime(record['properties']['datetime'], '%Y-%m-%dT%H:%M:%S')
                    contour_polygon = geometry.Polygon(record['geometry']['coordinates'][0])

                    contour_datetimes.append(contour_datetime)
                    contour_areas.append(contour_polygon.area)
                    contour_perimeters.append(contour_polygon.length)

                values[contour_name][velocity_product] = pandas.DataFrame({'datetime': contour_datetimes, 'area': contour_areas, 'perimeter': contour_perimeters})

    plotting_values = {'area': '%' if plot_percentages else 'm^2', 'perimeter': '%' if plot_percentages else 'm'}

    LOGGER.info(f'[{datetime.now()}]: Plotting...')

    # for contour_name, contour_velocity_products in values.items():
    #     LOGGER.info(f'[{datetime.now()}]: Plotting {contour_name}...')
    #
    #     for plotting_value, plotting_unit in plotting_values.items():
    #         figure = pyplot.figure()
    #         axis = figure.add_subplot(1, 1, 1)
    #         axis.set_title(f'starting point {contour_name} {start_time:%Y%m%dT%H%M%S}')
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
    #         figure.savefig(os.path.join(plot_dir, plotting_value,#                                     f'{plotting_value}_{contour_name}.pdf'),#                        orientation='landscape', papertype='A4')
    #
    #         # pyplot.show()
    #         pyplot.clf()

    for velocity_product in velocity_products:
        contours_values = []

        for contour_name, contour_velocity_products in values.items():
            contour_velocity_products[velocity_product].insert(0, 'contour', contour_name)
            contours_values.append(contour_velocity_products[velocity_product])

        contours_values = pandas.concat(contours_values, ignore_index=True)

        for plotting_value, plotting_unit in plotting_values.items():
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
            axis.set_xlabel('time')
            axis.set_ylabel(f'{plotting_value} ({plotting_unit})')

            colors = dict(zip((contour_name[1] for contour_name in contour_names), pyplot.cm.viridis(numpy.linspace(0, 1, 4))))
            # colors = dict(zip(contour_names, pyplot.cm.viridis(numpy.linspace(0, 1, 4))))

            if plotting_value == 'area':
                starting_value = numpy.pi * contour_starting_radius ** 2
            elif plotting_value == 'perimeter':
                starting_value = numpy.pi * contour_starting_radius * 2
            else:
                starting_value = 0

            for color_index, (contour_name, contour_values) in enumerate(contours_values.groupby('contour')):
                color = colors[contour_name[1]]
                y_values = contour_values[plotting_value] / starting_value if plot_percentages else 1
                line, = axis.plot(contour_values['datetime'], y_values, '-o', label=contour_name, color=color)
                axis.annotate(contour_name, xy=(1, line.get_ydata()[-1]), xytext=(6, 0), color=line.get_color(), xycoords=axis.get_yaxis_transform(), textcoords="offset points", size=8, va="center")

            axis.axhline(y=1 if plot_percentages else starting_value, linestyle=':', color='k', zorder=0)

            figure.savefig(os.path.join(plot_dir, f'{velocity_product}_{plotting_value}.pdf'), orientation='landscape', papertype='A4')

    pyplot.show()
    print('done')
