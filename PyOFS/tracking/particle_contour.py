# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import os
from typing import List, Tuple, Union

import cartopy.feature
import math
import numpy
import pyproj
import shapely.geometry
import xarray
from matplotlib import pyplot, quiver

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

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        """
        Plot vector field at the given time.

        :param time: time at which to plot
        :param axis: pyplot axis on which to plot
        :return: quiver plot
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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.delta_t})'


class RankineVortex(VectorField):
    def __init__(self, center: Tuple[float, float], radius: float, period: datetime.timedelta,
                 time_deltas: numpy.array):
        """
        Construct a 2-dimensional solid rotating disk surrounded by inverse falloff of tangential velocity.

        :param center: tuple of geographic coordinates
        :param radius: radius of central solid rotation in meters
        :param period: rotational period
        :param time_deltas: time differences
        """

        self.center = numpy.array(pyproj.transform(WGS84, WebMercator, *center))
        self.radius = radius
        self.angular_velocity = 2 * math.pi / period.total_seconds()

        super().__init__(time_deltas)

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        return -self.velocity(point, time) * math.cos(math.atan2(*(point - self.center)))

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.velocity(point, time) * math.sin(math.atan2(*(point - self.center)))

    def velocity(self, point: numpy.array, time: datetime.datetime) -> float:
        radial_distance = numpy.sqrt(numpy.sum((point - self.center) ** 2))

        if radial_distance <= self.radius:
            return self.angular_velocity * radial_distance
        else:
            return self.angular_velocity * self.radius ** 2 / radial_distance

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        points = []
        radii = numpy.linspace(1, self.radius * 2, 20)

        for radius in radii:
            num_points = 50
            points.extend([(math.cos(2 * math.pi / num_points * point_index) * radius + self.center[0],
                            math.sin(2 * math.pi / num_points * point_index) * radius + self.center[1]) for
                           point_index in range(0, num_points + 1)])

        vectors = [self[point, time] for point in points]
        points = list(zip(*pyproj.transform(WebMercator, WGS84, *zip(*points))))

        quiver_plot = axis.quiver(*zip(*points), *zip(*vectors), units='width', **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')

        return quiver_plot


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

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84,
                                    *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        quiver_plot = axis.quiver(lon, lat, self.dataset['u'].sel(time=time, method='nearest'),
                                  self.dataset['v'].sel(time=time, method='nearest'), units='width', **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')

        return quiver_plot


class Particle:
    """
    Particle simulation.
    """

    def __init__(self, location: Tuple[float, float], time: datetime.datetime, field: VectorField):
        """
        Create new particle within in the given velocity field.

        :param location: (lon, lat) point
        :param time: starting time
        :param field: velocity field
        """

        self.locations = [numpy.array(pyproj.transform(WGS84, WebMercator, location[0], location[1]))]
        self.time = time
        self.field = field

        self.vector = self.field[self.coordinates(), self.time]

    def step(self, delta_t: datetime.timedelta = None, order: int = 1):
        """
        Step particle by given time delta.

        :param delta_t: time delta
        :param order: order method to use (Euler / Runge-Kutta)
        """

        if delta_t is None:
            delta_t = self.field.delta_t

        delta_seconds = delta_t.total_seconds()

        if order > 0 and not any(numpy.isnan(self.vector)):
            k1 = delta_seconds * self.field[self.coordinates(), self.time]

            if order > 1:
                k2 = delta_seconds * self.field[self.coordinates() + k1 / 2, self.time + (delta_t / 2)]

                if order > 2:
                    k3 = delta_seconds * self.field[self.coordinates() + k2 / 2, self.time + (delta_t / 2)]

                    if order > 3:
                        k4 = delta_seconds * self.field[self.coordinates() + k3, self.time + delta_t]

                        if order > 4:
                            raise ValueError('Methods above 4th order are not implemented.')
                        else:
                            delta_vector = 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4
                    else:
                        delta_vector = 1 / 6 * k1 + 2 / 3 * k2 + 1 / 6 * k3
                else:
                    delta_vector = k2
            else:
                delta_vector = k1
        else:
            delta_vector = numpy.array([0, 0])

        self.locations.append(self.coordinates() + delta_vector)
        self.time += delta_t

        self.vector = self.field[self.coordinates(), self.time]

    def coordinates(self) -> numpy.array:
        """
        Get current linear coordinates.

        :return tuple of projected coordinates
        """

        return self.locations[-1]

    def geometry(self) -> shapely.geometry.Point:
        """
        Get current location as point.

        :return point geometry
        """

        return shapely.geometry.Point(*self.coordinates())

    def plot(self, locations: Union[int, slice] = -1, axis: pyplot.Axes = None, **kwargs) -> pyplot.Line2D:
        """
        Plot particle as point.

        :param locations: indices of locations to plot
        :param axis: pyplot axis on which to plot
        :return: plot
        """

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        return axis.plot(*pyproj.transform(WebMercator, WGS84, *zip(*self.locations[locations])), linestyle='--',
                         marker='o', **kwargs)

    def __str__(self) -> str:
        return f'{self.time} {self.coordinates()} -> {self.vector}'

    def __repr__(self):
        return str(self)


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

    def plot(self, axis: pyplot.Axes = None, **kwargs) -> pyplot.Line2D:
        """
        Plot the current state of the contour.

        :param axis: pyplot axis on which to plot
        :return: plot
        """

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        return axis.plot(*zip(*[pyproj.transform(WebMercator, WGS84, *particle.coordinates()) for particle in
                                self.particles + [self.particles[0]]]), **kwargs)

    def geometry(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon([particle.coordinates() for particle in self.particles])

    def area(self) -> float:
        return self.geometry().area

    def bounds(self) -> Tuple[float, float, float, float]:
        return self.geometry().bounds

    def __str__(self) -> str:
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

    def geometry(self) -> shapely.geometry.Point:
        return self.particles[0].geometry()

    def coordinates(self):
        return self.particles[0].coordinates()

    def plot(self, axis: pyplot.Axes = None, **kwargs) -> pyplot.Line2D:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84, *self.particles[0].coordinates())

        return axis.plot(lon, lat, **kwargs)


def translate_geographic_coordinates(point: numpy.array, offset: numpy.array) -> numpy.array:
    """
    Add meter offset to geographic coordinates using WebMercator.

    :param point: geographic coordinates
    :param offset: translation vector in meters
    :return: new geographic coordinates
    """

    if type(point) is not numpy.array:
        point = numpy.array(point)

    if type(offset) is not numpy.array:
        offset = numpy.array(offset)

    return numpy.array(pyproj.transform(WebMercator, WGS84, *(pyproj.transform(WGS84, WebMercator, *point)) + offset))


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    source = 'hfr'
    contour_shape = 'circle'
    order = 4

    contour_center = (-123.79820, 37.31710)
    contour_radius = 3000
    data_time = datetime.datetime(2019, 3, 31)

    print('Creating velocity field...')
    if source == 'rankine':
        hours = 36
        vortex_radius = contour_radius * 10
        vortex_center = translate_geographic_coordinates(contour_center, contour_radius * -5)
        vortex_period = datetime.timedelta(hours=12)
        velocity_field = RankineVortex(vortex_center, vortex_radius, vortex_period,
                                       [datetime.timedelta(hours=1) for hour in range(hours)])
    else:
        from PyOFS import DATA_DIR

        data_path = os.path.join(DATA_DIR, 'output', 'test', f'{source.lower()}_{data_time.strftime("%Y%m%d")}.nc')

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

        velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')
        data_time = _utilities.datetime64_to_time(velocity_field.dataset['time'][0])

    print('Creating starting contour...')
    if contour_shape == 'circle':
        contour = CircleContour(contour_center, contour_radius, data_time, velocity_field)
    elif contour_shape == 'square':
        southwest_corner = translate_geographic_coordinates(contour_center, -contour_radius)
        northeast_corner = translate_geographic_coordinates(contour_center, contour_radius)
        contour = RectangleContour(southwest_corner[0], northeast_corner[0], southwest_corner[1], northeast_corner[1],
                                   data_time, velocity_field)
    else:
        contour = PointContour(contour_center, data_time, velocity_field)

    print(f'Contour created: {contour}')

    figure = pyplot.figure()
    ordinal = lambda n: f'{n}{"tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4]}'
    figure.suptitle(f'{ordinal(order)} order {source.upper()} {contour_shape} contour with {contour_radius / 1000} km' +
                    f' radius from {contour_center}')

    if contour_shape == 'point':
        radial_distances_axis = figure.add_subplot(1, 2, 1)
        radial_distances_axis.set_xlabel('time')
        radial_distances_axis.set_ylabel('radial distance (m)')
    else:
        area_axis = figure.add_subplot(1, 2, 1)
        area_axis.set_xlabel('time')
        area_axis.set_ylabel('area (m^2)')

    map_axis = figure.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())
    map_axis.set_prop_cycle(
        color=[pyplot.cm.cool(color_index) for color_index in
               numpy.linspace(0, 1, len(velocity_field.time_deltas) + 1)])
    map_axis.add_feature(cartopy.feature.LAND)

    if contour_shape == 'point':
        radial_distances = {}
    else:
        areas = {}

    velocity_field.plot(data_time, map_axis)

    for time_delta in velocity_field.time_deltas:
        if type(time_delta) is numpy.timedelta64:
            time_delta = time_delta.item()

        if type(time_delta) is int:
            time_delta = datetime.timedelta(seconds=time_delta * 1e-9)

        if contour_shape == 'point':
            previous_radial_distance = numpy.sqrt(numpy.sum(
                (contour.coordinates() - numpy.array(pyproj.transform(WGS84, WebMercator, *contour_center))) ** 2))
        else:
            contour.plot(map_axis)
            previous_area = contour.area()

        if contour_shape == 'point':
            radial_distances[contour.time] = previous_radial_distance
        else:
            areas[contour.time] = previous_area

        contour.step(time_delta, order)

        if contour_shape == 'point':
            current_radial_distance = numpy.sqrt(numpy.sum(
                (contour.coordinates() - numpy.array(pyproj.transform(WGS84, WebMercator, *contour_center))) ** 2))
        else:
            current_area = contour.area()

        if contour_shape == 'point':
            print(
                f'step {time_delta} to {contour.time}: change in radius was {current_radial_distance - previous_radial_distance} m')
        else:
            print(f'step {time_delta} to {contour.time}: change in area was {current_area - previous_area} m^2')

    if contour_shape == 'point':
        radial_distances_axis.plot(radial_distances.keys(), radial_distances.values())
        for particle in contour.particles:
            particle.plot(slice(0, -1), map_axis)
    else:
        area_axis.plot(areas.keys(), areas.values())

    pyplot.show()

    print('done')
