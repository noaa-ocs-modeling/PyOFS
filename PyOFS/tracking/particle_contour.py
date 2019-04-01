# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import os
from typing import List, Tuple

import cartopy.feature
import haversine
import math
import numpy
import pyproj
import shapely.geometry
import xarray
from matplotlib import pyplot

import _utilities

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WebMercator = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs')


class VectorField:
    """
    Vector field with time component.
    """

    def __init__(self, time_deltas: numpy.array):
        self.time_deltas = [numpy.timedelta64(time_delta) for time_delta in time_deltas]
        time_delta = numpy.nanmean(self.time_deltas).item()
        if type(time_delta) is int:
            time_delta *= 1e-9
        elif type(time_delta) is datetime.timedelta:
            time_delta = int(time_delta.total_seconds())

        self.delta_t = datetime.timedelta(seconds=time_delta)

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        u velocity in m/s at coordinates

        :param point: coordinates in linear units
        :param time: time
        :return: u value at coordinate in m/s
        """

        pass

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        v velocity in m/s at coordinates

        :param point: coordinates in linear units
        :param time: time
        :return: v value at coordinate in m/s
        """

        pass

    def velocity(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        absolute velocity in m/s at coordinate

        :param point: coordinates in linear units
        :param time: time
        :return: magnitude of uv vector in m/s
        """

        return math.sqrt(self.u(point, time) ** 2 + self.v(point, time) ** 2)

    def direction(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        angle of uv vector

        :param point: coordinates in linear units
        :param time: time
        :return: radians from east of uv vector
        """

        return math.atan2(self.u(point, time), self.v(point, time))

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs):
        """
        Plot vector field at the given time.

        :param time: time at which to plot
        :param axis: pyplot axis on which to plot
        """

        pass

    def __getitem__(self, position: Tuple[numpy.array, datetime.datetime]) -> numpy.array:
        """
        velocity vector (u, v) in m/s at coordinates

        :param point: coordinates in linear units
        :param time: time
        :return: (u, v) vector
        """

        point, time = position
        if numpy.isnan(point).any() or time is None:
            vector = numpy.array([0.0, 0.0])
        else:
            vector = numpy.array([self.u(point, time), self.v(point, time)])

            if numpy.isnan(vector).any():
                vector = numpy.array([0.0, 0.0])

        return vector

    def __repr__(self):
        return f'{self.__class__.__name__}({self.delta_t})'


class RankineEddy(VectorField):
    def __init__(self, center: Tuple[float, float], radius: float, angular_velocity: float, time_deltas: numpy.array):
        """
        Construct a 2-dimensional Rankine eddy (a solid rotating disk surrounded by inverse falloff of tangential velocity).

        :param center: (lon, lat) point
        :param radius: radius of central solid rotation
        :param angular_velocity: velocity of central solid rotation (rad/s)
        :param time_deltas: time differences
        """

        self.center = numpy.array(center)
        self.radius = radius
        self.angular_velocity = angular_velocity

        super().__init__(time_deltas)

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        return -self.velocity(point, time) * math.cos(math.atan2(*(point - self.center)))

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.velocity(point, time) * math.sin(math.atan2(*(point - self.center)))

    def velocity(self, point: numpy.array, time: datetime.datetime) -> float:
        distance = haversine.haversine(self.center, point, unit='m')

        if distance <= self.radius:
            return self.angular_velocity * distance
        else:
            return self.angular_velocity * self.radius ** 2 / distance

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs):
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        points = []

        radii = numpy.linspace(1, self.radius, 20)

        for radius in radii:
            center_x, center_y = pyproj.transform(WGS84, WebMercator, self.center[0], self.center[1])
            num_points = 50
            points.extend(
                [pyproj.transform(WebMercator, WGS84, math.cos(2 * math.pi / num_points * x) * radius + center_x,
                                  math.sin(2 * math.pi / num_points * x) * radius + center_y) for x in
                 range(0, num_points + 1)])

        u = [self.u(point, time) for point in points]
        v = [self.v(point, time) for point in points]

        quiver_plot = axis.quiver([point[0] for point in points], [point[1] for point in points], u, v, units='width',
                                  **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')


class VectorDataset(VectorField):
    """
    Velocity field of u and v vectors in m/s.
    """

    def __init__(self, dataset: xarray.Dataset, u_name: str = 'u', v_name: str = 'v', x_name: str = 'lon',
                 y_name: str = 'lat', t_name: str = 'time'):
        """
        Create new velocity field from given dataset.

        :param dataset: xarray dataset containing velocity data (u, v)
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_name: name of x coordinate
        :param y_name: name of y coordinate
        :param t_name: name of time coordinate
        """

        self.dataset = dataset.rename({u_name: 'u', v_name: 'v', x_name: 'x', y_name: 'y', t_name: 'time'})

        x_dims = self.dataset['x'].dims
        y_dims = self.dataset['y'].dims

        lon = self.dataset['x'].values
        lat = self.dataset['y'].values

        if len(x_dims) == 1 or len(y_dims) == 1:
            lon, lat = numpy.meshgrid(lon, lat)

        x, y = pyproj.transform(WGS84, WebMercator, lon, lat)

        if len(x_dims) == 1 or len(y_dims) == 1:
            x = x[0, :]
            y = y[:, 0]
            self.has_multidim_coords = False
        else:
            x = (x_dims, x)
            y = (y_dims, y)
            self.has_multidim_coords = True

        self.dataset['x'], self.dataset['y'] = x, y

        super().__init__(numpy.diff(self.dataset['time'].values))

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        num_x_dims = len(self.dataset['x'].dims)
        num_y_dims = len(self.dataset['y'].dims)

        x_query = point[0] if num_x_dims == 1 else point
        y_query = point[1] if num_y_dims == 1 else point

        return self.dataset['u'].sel(time=time, x=x_query, y=y_query, method='nearest').values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        num_x_dims = len(self.dataset['x'].dims)
        num_y_dims = len(self.dataset['y'].dims)

        x_query = point[0] if num_x_dims == 1 else point
        y_query = point[1] if num_y_dims == 1 else point

        return self.dataset['v'].sel(time=time, x=x_query, y=y_query, method='nearest').values.item()

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs):
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84,
                                    *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        quiver_plot = axis.quiver(lon, lat, self.dataset['u'].sel(time=time, method='nearest'),
                                  self.dataset['v'].sel(time=time, method='nearest'), units='width', **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')


class Particle:
    """
    Particle simulation.
    """

    def __init__(self, point: Tuple[float, float], time: datetime.datetime, field: VectorField):
        """
        Create new particle within in the given velocity field.

        :param point: (lon, lat) point
        :param time: starting time
        :param field: velocity field
        """

        self.point = numpy.array(pyproj.transform(WGS84, WebMercator, point[0], point[1]))
        self.time = time
        self.field = field

        self.vector = self.field[self.point, self.time]

    def step(self, delta_t: datetime.timedelta = None, order: int = 1):
        """
        Step particle by given time delta.

        :param delta_t: time delta
        :param order: order method to use (Euler / Runge-Kutta)
        """

        if delta_t is None:
            delta_t = self.field.delta_t

        delta_seconds = delta_t.total_seconds()
        delta_vector = numpy.array([0, 0])

        if not any(numpy.isnan(self.vector)):
            k1 = delta_seconds * velocity_field[self.point, self.time]

            if order == 1:
                delta_vector = k1
            elif order > 1:
                k2 = delta_seconds * velocity_field[self.point + k1 / 2, self.time + (delta_t / 2)]

                if order == 2:
                    delta_vector = k2
                elif order > 2:
                    k3 = delta_seconds * velocity_field[self.point + k2 / 2, self.time + (delta_t / 2)]
                    k4 = delta_seconds * velocity_field[self.point + k3, self.time + delta_t]

                    if order == 4:
                        delta_vector = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        self.point += delta_vector
        self.time += delta_t

        self.vector = self.field[self.point, self.time]

    def coordinates(self) -> tuple:
        """
        current coordinates
        :return tuple of GCS coordinates
        """

        return pyproj.transform(WebMercator, WGS84, *self.point)

    def plot(self, axis: pyplot.Axes = None, **kwargs):
        """
        Plot particle as point.

        :param axis: pyplot axis on which to plot
        """

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84, *self.point)

        axis.plot(lon, lat, **kwargs)

    def __str__(self):
        return f'{self.time} {self.coordinates()} -> {self.vector}'


class ParticleContour:
    """
    Contour of points within a velocity field.
    """

    def __init__(self, points: List[Tuple[float, float]], time: datetime.datetime, field: VectorField):
        """
        Create contour given list of points.

        :param points: list of (lon, lat) points
        :param time: starting time
        :param field: velocity field
        """

        self.time = time
        self.field = field

        self.particles = []

        for point in points:
            self.particles.append(Particle(point, time, field))

    def step(self, delta_t: datetime.timedelta = None, order: int = 1):
        """
        Step particle by given time delta.

        :param delta_t: time delta
        :param order: order method to use (Euler / Runge-Kutta)
        """

        if delta_t is None:
            delta_t = self.field.delta_t

        self.time += delta_t

        for particle in self.particles:
            particle.step(delta_t, order)

    def plot(self, axis: pyplot.Axes = None, **kwargs):
        """
        Plot the current state of the contour.

        :param axis: pyplot axis on which to plot
        """

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = zip(*[pyproj.transform(WebMercator, WGS84, *particle.point) for particle in self.particles])

        axis.plot(lon, lat, **kwargs)

    def geometry(self):
        return shapely.geometry.Polygon([particle.point for particle in self.particles])

    def area(self):
        return self.geometry().area

    def bounds(self):
        return self.geometry().bounds

    def __str__(self):
        return f'contour at time {self.time} with bounds {self.bounds()} and area {self.area()} m^2'


class CircleContour(ParticleContour):
    def __init__(self, center: tuple, radius: float, time: datetime.datetime, field: VectorField,
                 interval: float = 500):
        """
        Create circle contour with given interval between points.

        :param center: central point (lon, lat)
        :param radius: radius in m
        :param time: starting time
        :param field: velocity field
        :param interval: interval between points in m
        """

        center_x, center_y = pyproj.transform(WGS84, WebMercator, center[0], center[1])
        circumference = 2 * math.pi * radius
        num_points = round(circumference / interval)

        points = [pyproj.transform(WebMercator, WGS84, math.cos(2 * math.pi / num_points * x) * radius + center_x,
                                   math.sin(2 * math.pi / num_points * x) * radius + center_y) for x in
                  range(0, num_points + 1)]

        super().__init__(points, time, field)


class RectangleContour(ParticleContour):
    def __init__(self, west_lon: float, east_lon: float, south_lat: float, north_lat: float, time: datetime.datetime,
                 field: VectorField, interval: float = 500):
        """
        Create orthogonal square contour with given bounds.

        :param west_lon: minimum longitude
        :param east_lon: maximum longitude
        :param south_lat: minimum latitude
        :param north_lat: maximum latitude
        :param time: starting time
        :param field: velocity field
        :param interval: interval between points in meters
        """

        corners = {'sw': pyproj.transform(WGS84, WebMercator, west_lon, south_lat),
                   'nw': pyproj.transform(WGS84, WebMercator, west_lon, north_lat),
                   'ne': pyproj.transform(WGS84, WebMercator, east_lon, north_lat),
                   'se': pyproj.transform(WGS84, WebMercator, east_lon, south_lat)}
        points = []

        for corner_name, corner in corners.items():
            points.append(pyproj.transform(WebMercator, WGS84, *corner))

            if corner_name is 'sw':
                edge_length = corners['nw'][1] - corners['sw'][1]
            elif corner_name is 'nw':
                edge_length = corners['ne'][0] - corners['nw'][0]
            elif corner_name is 'ne':
                edge_length = corners['ne'][1] - corners['se'][1]
            elif corner_name is 'se':
                edge_length = corners['se'][0] - corners['sw'][0]
            else:
                edge_length = 0

            for stride in range(int(edge_length / interval)):
                x, y = corner

                if corner_name is 'sw':
                    y += stride
                elif corner_name is 'nw':
                    x += stride
                elif corner_name is 'ne':
                    y -= stride
                elif corner_name is 'se':
                    x -= stride

                points.append(pyproj.transform(WebMercator, WGS84, x, y))

        super().__init__(points, time, field)


class PointContour(ParticleContour):
    def __init__(self, point: numpy.array, time: datetime.datetime, field: VectorField):
        super().__init__([point], time, field)

    def geometry(self):
        return shapely.geometry.Point(self.particles[0].point)

    def plot(self, axis: pyplot.Axes = None, **kwargs):
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84, *self.particles[0].point)

        axis.plot(lon, lat, **kwargs)


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    source = 'rankine'
    order = 2

    contour_center = (-123.79820, 37.31710)
    contour_radius = 6000
    data_time = datetime.datetime(2019, 3, 30)

    contour_shape = 'point'

    if source == 'rankine':
        print('Creating velocity field...')
        eddy_center = contour_center
        time_deltas = [datetime.timedelta(hours=1) for hour in range(24)]
        velocity_field = RankineEddy(eddy_center, contour_radius * 2, 0.5, time_deltas)

        print('Creating starting contour...')
    else:
        data_path = os.path.join(r"C:\Data\OFS\develop\output\test",
                                 f'{source.lower()}_{data_time.strftime("%Y%m%d")}.nc')

        print('Collecting data...')
        if not os.path.exists(data_path):
            if source.upper() == 'HFR':
                from PyOFS.dataset import hfr

                vector_dataset = hfr.HFRRange(data_time, data_time + datetime.timedelta(days=1))
                vector_dataset.to_netcdf(data_path, variables=('ssu', 'ssv'), mean=False)
            elif source.upper() == 'RTOFS':
                from PyOFS.dataset import rtofs

                vector_dataset = rtofs.RTOFSDataset(data_time)
                vector_dataset.to_netcdf(data_path, variables=('ssu', 'ssv'), mean=False)
            elif source.upper() == 'WCOFS':
                from PyOFS.dataset import wcofs

                vector_dataset = wcofs.WCOFSDataset(data_time)
                vector_dataset.to_netcdf(data_path, variables=('ssu', 'ssv'))

        vector_dataset = xarray.open_dataset(data_path)

        print('Creating velocity field...')
        velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')
        data_time = _utilities.datetime64_to_time(velocity_field.dataset['time'][0])

    print('Creating starting contour...')

    if contour_shape == 'circle':
        contour = CircleContour(contour_center, contour_radius, data_time, velocity_field)
    elif contour_shape == 'rectangle':
        point = pyproj.transform(WGS84, WebMercator, *contour_center)
        southwest_corner = pyproj.transform(WebMercator, WGS84, *(point[0] - contour_radius, point[1] - contour_radius))
        northeast_corner = pyproj.transform(WebMercator, WGS84, *(point[0] + contour_radius, point[1] + contour_radius))
        contour = RectangleContour(southwest_corner[0], northeast_corner[0], southwest_corner[1], northeast_corner[1],
                                   data_time, velocity_field)
    else:
        point = numpy.array(pyproj.transform(WGS84, WebMercator, *contour_center))
        point = pyproj.transform(WebMercator, WGS84, *(point + contour_radius))
        contour = PointContour(point, data_time, velocity_field)

    print(f'Contour created: {contour}')

    figure = pyplot.figure()
    ordinal = lambda n: f'{n}{"tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4]}'
    figure.suptitle(
        f'{ordinal(order)} order {source.upper()} contour with {contour_radius / 1000} km radius from {contour_center}')

    area_axis = figure.add_subplot(1, 2, 1)
    area_axis.set_xlabel('time')
    area_axis.set_ylabel('area (m^2)')

    map_axis = figure.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())
    map_axis.set_prop_cycle(
        color=[pyplot.cm.cool(color_index) for color_index in
               numpy.linspace(0, 1, len(velocity_field.time_deltas) + 1)])
    map_axis.add_feature(cartopy.feature.LAND)

    areas = {}

    velocity_field.plot(data_time, map_axis)
    contour.plot(map_axis, markersize=14)
    pyplot.show()

    for time_delta in velocity_field.time_deltas:
        if type(time_delta) is numpy.timedelta64:
            time_delta = time_delta.item()

        if type(time_delta) is int:
            time_delta = datetime.timedelta(seconds=time_delta * 1e-9)

        contour.plot(map_axis)

        previous_area = contour.area()
        contour.step(time_delta, order)
        current_area = contour.area()
        print(f'step {time_delta} to {contour.time}: change in area was {current_area - previous_area} m^2')

        areas[contour.time] = current_area

    area_axis.plot(areas.keys(), areas.values())

    pyplot.show()

    print('done')
