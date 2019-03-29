# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import math
import os
from typing import List, Tuple

import cartopy.feature
import haversine
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
        self.delta_t = datetime.timedelta(seconds=float(numpy.nanmean(self.time_deltas).item().total_seconds()) * 1e-9)

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

    def plot(self, time: datetime.datetime, color: str = None, axis: pyplot.Axes = None):
        """
        Plot vector field at the given time.

        :param time: time at which to plot
        :param color: color of vector arrows
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
        return numpy.array([self.u(point, time), self.v(point, time)])

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
        return -self.velocity(point, time) * math.sin(math.atan2(*(point - self.center)))

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.velocity(point, time) * math.cos(math.atan2(*(point - self.center)))

    def velocity(self, point: numpy.array, time: datetime.datetime) -> float:
        return rankine_velocity(haversine.haversine(self.center, point, unit='m'), self.radius, self.angular_velocity)


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
        return self.dataset['u'].sel(time=time, x=point[0], y=point[1], method='nearest').values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.dataset['v'].sel(time=time, x=point[0], y=point[1], method='nearest').values.item()

    def plot(self, time: datetime.datetime, color: str = None, axis: pyplot.Axes = None):
        if axis is None:
            axis = pyplot.gca()

        quiver_plot = axis.quiver(X=self.dataset['x'], Y=self.dataset['y'],
                                  U=self.dataset['u'].sel(time=time, method='nearest'),
                                  V=self.dataset['v'].sel(time=time, method='nearest'), C=color, units='width')
        axis.quiverkey(quiver_plot, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')


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

    def plot(self, color: str = None, axis: pyplot.Axes = None):
        """
        Plot particle as point.

        :param color: color
        :param axis: pyplot axis on which to plot
        """

        if axis is None:
            axis = pyplot.gca()

        lon, lat = pyproj.transform(WebMercator, WGS84, *self.point)

        if color is None:
            axis.plot(lon, lat)
        else:
            axis.plot(lon, lat, color=color)

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

        self.polygon = shapely.geometry.Polygon([particle.point for particle in self.particles])

    def step(self, t_delta: datetime.timedelta = None, order: int = 1):
        """
        Step particle by given time delta.

        :param t_delta: time delta
        :param order: order method to use (Euler / Runge-Kutta)
        """

        if t_delta is None:
            t_delta = self.field.delta_t

        self.time += t_delta

        for particle in self.particles:
            particle.step(t_delta, order)

        self.polygon = shapely.geometry.Polygon([particle.point for particle in self.particles])

    def plot(self, color: str = None, axis: pyplot.Axes = None):
        """
        Plot the current state of the contour.

        :param color: color
        :param axis: pyplot axis on which to plot
        """

        if axis is None:
            axis = pyplot.gca()

        lon, lat = zip(*[pyproj.transform(WebMercator, WGS84, *particle.point) for particle in self.particles])

        if color is None:
            axis.plot(lon, lat)
        else:
            axis.plot(lon, lat, color=color)

    def area(self):
        return self.polygon.area

    def bounds(self):
        return self.polygon.bounds

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
                 field: VectorDataset, interval: float = 500):
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


def rankine_velocity(radius, eddy_radius, angular_velocity):
    if radius <= eddy_radius:
        return angular_velocity * radius
    else:
        return angular_velocity * eddy_radius ** 2 / radius


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    source = 'rankine'
    order = 1

    contour_center = (-123.79820, 37.31710)
    contour_radius = 1000
    data_time = datetime.datetime(2019, 3, 28)

    if source == 'rankine':
        print('Creating velocity field...')
        eddy_center = contour_center
        time_deltas = [forward_time - data_time for forward_time in
                       _utilities.range_hourly(data_time, data_time + datetime.timedelta(days=1))]
        velocity_field = RankineEddy(eddy_center, contour_radius * 2, 0.5, time_deltas)

        print('Creating starting contour...')
        contour = CircleContour(contour_center, contour_radius, data_time, velocity_field)
    else:
        data_path = os.path.join(r"C:\Data\develop\output\test", f'{source.lower()}_{data_time.strftime("%Y%m%d")}.nc')

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

                vector_dataset = wcofs.WCOFSRange(data_time, data_time + datetime.timedelta(days=1))
                vector_dataset.to_netcdf(data_path, variables=('ssu', 'ssv'), mean=False)

        vector_dataset = xarray.open_dataset(data_path)

        print('Creating velocity field...')
        velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')

        print('Creating starting contour...')
        contour = CircleContour(contour_center, contour_radius,
                                _utilities.datetime64_to_time(velocity_field.dataset['time'][0]), velocity_field)

    print(f'Contour created: {contour}')

    figure = pyplot.figure()
    figure.suptitle(
        f'{source.upper()} SSUV contour at {contour_center} with {contour_radius} m radius with {order} order')

    area_axis = figure.add_subplot(121)
    area_axis.set_xlabel('time')
    area_axis.set_ylabel('area (m^2)')

    map_axis = figure.add_subplot(122, projection=cartopy.crs.PlateCarree())
    map_axis.set_prop_cycle(
        color=[pyplot.cm.cool(color_index) for color_index in
               numpy.linspace(0, 1, len(velocity_field.time_deltas) + 1)])
    map_axis.add_feature(cartopy.feature.LAND)

    areas = {}

    for time_delta in velocity_field.time_deltas:
        timedelta = datetime.timedelta(seconds=int(time_delta.item().total_seconds()) * 1e-9)

        previous_area = contour.area()
        contour.step(timedelta, 1)
        current_area = contour.area()
        print(f'step {timedelta} to {contour.time}: change in area was {current_area - previous_area} m^2')

        contour.plot(None, map_axis)
        areas[contour.time] = current_area

    area_axis.plot(areas.keys(), areas.values())

    pyplot.show()

    print('done')
