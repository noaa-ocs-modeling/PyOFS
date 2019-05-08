# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import os
from concurrent import futures
from typing import List, Tuple, Union, Dict

import cartopy.feature
import fiona.crs
import math
import numpy
import pyproj
import shapely.geometry
import xarray
from matplotlib import pyplot, quiver

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WebMercator = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs')
fiona_WebMercator = fiona.crs.from_epsg(3857)


class VectorField:
    """
    Vector field of (u, v) values.
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
        if numpy.any(numpy.isnan(point)) or time is None:
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
    Vector field with time component using xarray observation.
    """

    def __init__(self, dataset: xarray.Dataset, u_name: str = 'u', v_name: str = 'v', x_name: str = 'lon',
                 y_name: str = 'lat', t_name: str = 'time', coordinate_system: pyproj.Proj = None):
        """
        Create new velocity field from given observation.

        :param dataset: xarray observation containing velocity data (u, v)
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_name: name of x coordinate
        :param y_name: name of y coordinate
        :param t_name: name of time coordinate
        :param coordinate_system: coordinate system of observation
        """

        self.coordinate_system = coordinate_system if coordinate_system is not None else WGS84

        variables_to_rename = {u_name: 'u', v_name: 'v', x_name: 'x', y_name: 'y', t_name: 'time'}
        self.dataset = dataset.rename(variables_to_rename)

        x, y = pyproj.transform(self.coordinate_system, WebMercator,
                                *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        self.dataset['x'] = x[0, :]
        self.dataset['y'] = y[:, 0]

        super().__init__(numpy.diff(self.dataset['time'].values))

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        time_bounds = [self.dataset['time'].sel(time=time, method='bfill').values,
                       self.dataset['time'].sel(time=time, method='ffill').values]

        if time_bounds[0] == time_bounds[1]:
            time = time_bounds[0]
            return self.dataset['u'].sel(time=time).interp(x=point[0], y=point[1]).values.item()
        else:
            return self.dataset['u'].sel(time=time_bounds).interp(time=time, x=point[0], y=point[1]).values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        time_bounds = [self.dataset['time'].sel(time=time, method='bfill').values,
                       self.dataset['time'].sel(time=time, method='ffill').values]

        if time_bounds[0] == time_bounds[1]:
            time = time_bounds[0]
            return self.dataset['v'].sel(time=time).interp(x=point[0], y=point[1]).values.item()
        else:
            return self.dataset['v'].sel(time=time_bounds).interp(time=time, x=point[0], y=point[1]).values.item()

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84,
                                    *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        quiver_plot = axis.quiver(lon, lat, self.dataset['u'].sel(time=time, method='nearest'),
                                  self.dataset['v'].sel(time=time, method='nearest'), units='width', **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')

        return quiver_plot


class ROMSGridVectorDataset(VectorField):
    def __init__(self, dataset: xarray.Dataset, u_name: str = 'u', v_name: str = 'v',
                 x_names: Tuple[str, str] = ('u_lon', 'v_lon'), y_names: Tuple[str, str] = ('u_lat', 'v_lat'),
                 t_name: str = 'time', coordinate_system: pyproj.Proj = None):
        """
        Create new velocity field from given observation.

        :param dataset: xarray observation containing velocity data (u, v)
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_names: names of x coordinates
        :param y_names: names of y coordinates
        :param t_name: name of time coordinate
        :param coordinate_system: coordinate system of observation
        """

        self.coordinate_system = coordinate_system if coordinate_system is not None else WGS84

        variables_to_rename = {u_name: 'u', v_name: 'v', t_name: 'time'}
        variables_to_rename.update(dict(zip(x_names, ('u_x', 'v_x'))))
        variables_to_rename.update(dict(zip(y_names, ('u_y', 'v_y'))))
        self.dataset = dataset.rename(variables_to_rename)

        self.native_u_x, self.native_u_y = pyproj.transform(WGS84, self.coordinate_system, self.dataset['u_x'].values,
                                                            self.dataset['u_y'].values)
        self.native_v_x, self.native_v_y = pyproj.transform(WGS84, self.coordinate_system, self.dataset['v_x'].values,
                                                            self.dataset['v_y'].values)

        super().__init__(numpy.diff(self.dataset['time'].values))

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        transformed_point = pyproj.transform(WebMercator, self.coordinate_system, *point)

        eta_index = numpy.nanmax(self.dataset['u_eta']) * ((transformed_point[0] - numpy.nanmin(self.native_u_x)) / (
                numpy.nanmax(self.native_u_x) - numpy.nanmin(self.native_u_x)))
        xi_index = numpy.nanmax(self.dataset['u_xi']) * ((transformed_point[1] - numpy.nanmin(self.native_u_y)) / (
                numpy.nanmax(self.native_u_y) - numpy.nanmin(self.native_u_y)))

        eta_index_bounds = [math.floor(eta_index), math.ceil(eta_index)]
        xi_index_bounds = [math.floor(xi_index), math.ceil(xi_index)]
        time_bounds = [self.dataset['time'].sel(time=time, method='bfill').values,
                       self.dataset['time'].sel(time=time, method='ffill').values]

        if time_bounds[0] == time_bounds[1]:
            time = time_bounds[0]
            return self.dataset['u'].sel(u_eta=eta_index_bounds, u_xi=xi_index_bounds, time=time).interp(
                u_eta=eta_index - eta_index_bounds[0], u_xi=xi_index - xi_index_bounds[0]).values.item()
        else:
            return self.dataset['u'].sel(u_eta=eta_index_bounds, u_xi=xi_index_bounds, time=time_bounds).interp(
                time=time, u_eta=eta_index - eta_index_bounds[0], u_xi=xi_index - xi_index_bounds[0]).values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        transformed_point = pyproj.transform(WebMercator, self.coordinate_system, *point)

        eta_index = numpy.nanmax(self.dataset['v_eta']) * ((transformed_point[0] - numpy.nanmin(self.native_v_x)) / (
                numpy.nanmax(self.native_v_x) - numpy.nanmin(self.native_v_x)))
        xi_index = numpy.nanmax(self.dataset['v_xi']) * ((transformed_point[1] - numpy.nanmin(self.native_v_y)) / (
                numpy.nanmax(self.native_v_y) - numpy.nanmin(self.native_v_y)))

        eta_index_bounds = [math.floor(eta_index), math.ceil(eta_index)]
        xi_index_bounds = [math.floor(xi_index), math.ceil(xi_index)]
        time_bounds = [self.dataset['time'].sel(time=time, method='bfill').values,
                       self.dataset['time'].sel(time=time, method='ffill').values]

        if time_bounds[0] == time_bounds[1]:
            time = time_bounds[0]
            return self.dataset['v'].sel(v_eta=eta_index_bounds, v_xi=xi_index_bounds, time=time).interp(
                v_eta=eta_index - eta_index_bounds[0], v_xi=xi_index - xi_index_bounds[0]).values.item()
        else:
            return self.dataset['v'].sel(v_eta=eta_index_bounds, v_xi=xi_index_bounds, time=time_bounds).interp(
                time=time, v_eta=eta_index - eta_index_bounds[0], v_xi=xi_index - xi_index_bounds[0]).values.item()

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(self.coordinate_system, WGS84, *numpy.meshgrid(self.dataset['u_x'].values,
                                                                                   self.dataset['u_y'].values))

        u_shape = self.dataset['u'].shape
        v_shape = self.dataset['v'].shape
        new_shape = (min(u_shape[-2], v_shape[-2]), min(u_shape[-1], v_shape[-1]))

        u = numpy.resize(self.dataset['u'].sel(time=time, method='nearest'), new_shape)
        v = numpy.resize(self.dataset['v'].sel(time=time, method='nearest'), new_shape)

        quiver_plot = axis.quiver(lon, lat, u, v, units='width', **kwargs)
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
            k_1 = delta_seconds * self.field[self.coordinates(), self.time]

            if order > 1:
                k_2 = delta_seconds * self.field[self.coordinates() + 0.5 * k_1, self.time + 0.5 * delta_t]

                if order > 2:
                    k_3 = delta_seconds * self.field[self.coordinates() + 0.5 * k_2, self.time + 0.5 * delta_t]

                    if order > 3:
                        k_4 = delta_seconds * self.field[self.coordinates() + k_3, self.time + delta_t]

                        if order > 4:
                            raise ValueError('Methods above 4th order are not implemented.')
                        else:
                            delta_vector = 1 / 6 * (k_1 + 2 * (k_2 + k_3) + k_4)
                    else:
                        delta_vector = 1 / 6 * (k_1 + 4 * k_2 + k_3)
                else:
                    delta_vector = k_2
            else:
                delta_vector = k_1
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

        if locations is None:
            locations = slice(0, -1)

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

        with futures.ThreadPoolExecutor() as concurrency_pool:
            for particle in self.particles:
                concurrency_pool.submit(particle.step, delta_t, order)

        # for particle in self.particles:
        #     particle.step(delta_t, order)

        self.time += delta_t

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

        # circumference = 2 * math.pi * radius
        # num_points = round(circumference / interval)
        # points = [pyproj.transform(WebMercator, WGS84, math.cos(2 * math.pi / num_points * x) * radius + center_x,
        #                            math.sin(2 * math.pi / num_points * x) * radius + center_y) for x in
        #           range(0, num_points + 1)]

        points = list(zip(*pyproj.transform(WebMercator, WGS84, *shapely.geometry.Point(center_x, center_y).buffer(
            radius).exterior.coords.xy)))

        super().__init__(points, time, field)

    def __str__(self) -> str:
        return f'circular {super().__str__()}'


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

    def __str__(self) -> str:
        return f'rectangular {super().__str__()}'


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

    def __str__(self) -> str:
        return f'point contour at time {self.time} at {self.coordinates()}'


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


def create_contour(contour_center: tuple, contour_radius: float, start_time: datetime.datetime,
                   velocity_field: VectorField, contour_shape: str) -> ParticleContour:
    if contour_shape == 'circle':
        return CircleContour(contour_center, contour_radius, start_time, velocity_field)
    elif contour_shape == 'square':
        southwest_corner = translate_geographic_coordinates(contour_center, -contour_radius)
        northeast_corner = translate_geographic_coordinates(contour_center, contour_radius)
        return RectangleContour(southwest_corner[0], northeast_corner[0], southwest_corner[1],
                                northeast_corner[1], start_time, velocity_field)
    else:
        return PointContour(contour_center, start_time, velocity_field)


def track_contour(contour: ParticleContour, time_deltas: List[datetime.timedelta]) -> Dict[datetime.datetime, dict]:
    print(f'[{datetime.datetime.now()}]: Advecting contour with {len(contour.particles)} particles...')

    polygons = {}
    polygons[contour.time] = contour.geometry()

    for time_delta in time_deltas:
        if type(time_delta) is numpy.timedelta64:
            time_delta = time_delta.item()

        if type(time_delta) is int:
            time_delta = datetime.timedelta(seconds=time_delta * 1e-9)

        contour.step(time_delta, order)

        polygons[contour.time] = contour.geometry()

        print(f'[{datetime.datetime.now()}]: Tracked {contour}')

    return polygons


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

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

    from PyOFS import DATA_DIR
    from PyOFS.model import rtofs, wcofs
    from PyOFS.observation import hf_radar

    source = 'wcofs_qck_geostrophic'
    contour_shape = 'circle'
    order = 4

    contour_radius = 10000

    contour_centers = {}
    start_time = datetime.datetime(2016, 9, 25, 1)

    period = datetime.timedelta(hours=2)
    time_delta = datetime.timedelta(hours=1)

    time_deltas = [time_delta for index in range(int(period / time_delta))]

    output_path = os.path.join(DATA_DIR, 'output', 'test', 'contours.gpkg')
    layer_name = f'{source}_{start_time.strftime("%Y%m%dT%H%M%S")}_{(start_time + period).strftime("%Y%m%dT%H%M%S")}_' + \
                 f'{int(time_delta.total_seconds() / 3600)}h'

    print(f'[{datetime.datetime.now()}]: Started processing...')

    with fiona.open(os.path.join(DATA_DIR, 'reference', 'study_points.gpkg'),
                    layer='study_points') as contour_centers_file:
        for point in contour_centers_file:
            contour_id = int(point['id'])

            if contour_id == 1:
                contour_centers[contour_id] = point['geometry']['coordinates']

    print(f'[{datetime.datetime.now()}]: Creating velocity field...')
    if source == 'rankine':
        vortex_radius = contour_radius * 5
        vortex_center = translate_geographic_coordinates(contour_centers.values()[0],
                                                         numpy.array([contour_radius * -2, contour_radius * -2]))
        vortex_period = datetime.timedelta(days=5)
        velocity_field = RankineVortex(vortex_center, vortex_radius, vortex_period, time_deltas)

        radii = range(1, vortex_radius * 2, 50)
        points = [numpy.array(pyproj.transform(WGS84, WebMercator, *vortex_center)) + numpy.array([radius, 0]) for
                  radius in radii]
        velocities = [velocity_field.velocity(point, start_time) for point in points]
    else:
        data_path = os.path.join(DATA_DIR, 'output', 'test', f'{source.lower()}_{start_time.strftime("%Y%m%d")}.nc')

        print(f'[{datetime.datetime.now()}]: Collecting data...')

        if not os.path.exists(data_path):
            if source.upper() == 'HFR':
                vector_dataset = hf_radar.HFRadarRange(start_time, start_time + datetime.timedelta(days=1)).to_xarray(
                    variables=('ssu', 'ssv'), mean=False)
                vector_dataset.to_netcdf(data_path)
            elif source.upper() == 'RTOFS':
                vector_dataset = rtofs.RTOFSDataset(start_time).to_xarray(variables=('ssu', 'ssv'), mean=False)
                vector_dataset.to_netcdf(data_path)
            elif source.upper() == 'WCOFS':
                source = 'avg' if time_delta.total_seconds() / 3600 / 24 >= 1 else '2ds'

                vector_dataset = wcofs.WCOFSDataset(start_time, source).to_xarray(variables=('ssu', 'ssv'))
                vector_dataset.to_netcdf(data_path)
            if 'WCOFS_QCK' in source.upper():
                qck_path = os.path.join(DATA_DIR, 'input', 'wcofs', 'qck')
                input_filenames = os.listdir(qck_path)

                times = []
                ssu = []
                ssv = []

                u_lon = None
                u_lat = None
                v_lon = None
                v_lat = None

                for input_filename in sorted(input_filenames):
                    input_dataset = xarray.open_dataset(os.path.join(qck_path, input_filename))

                    current_times = input_dataset['ocean_time'].values
                    times.extend(current_times)

                    if 'GEOSTROPHIC' in source.upper():
                        gravitational_acceleration = 9.80665
                        sidereal_rotation_period = datetime.timedelta(hours=23, minutes=53, seconds=4.1)
                        coriolis_frequencies = 4 * math.pi / sidereal_rotation_period.total_seconds() * numpy.sin(
                            input_dataset['lat_rho'].values * math.pi / 180)

                        sea_level = input_dataset['zeta']
                        geostrophic_ssu = -(gravitational_acceleration / coriolis_frequencies * sea_level.differentiate(
                            'eta_rho')).values
                        geostrophic_ssv = (gravitational_acceleration / coriolis_frequencies * sea_level.differentiate(
                            'xi_rho')).values

                        geostrophic_ssu[numpy.isnan(geostrophic_ssu)] = 0
                        geostrophic_ssv[numpy.isnan(geostrophic_ssv)] = 0

                        ssu.append(geostrophic_ssu)
                        ssv.append(geostrophic_ssv)

                        u_lon = input_dataset['lon_rho'].values
                        u_lat = input_dataset['lat_rho'].values
                        v_lon = input_dataset['lon_rho'].values
                        v_lat = input_dataset['lat_rho'].values
                    else:
                        if u_lon is None or u_lat is None or v_lon is None or v_lat is None:
                            u_lon = input_dataset['lon_u'].values
                            u_lat = input_dataset['lat_u'].values
                            v_lon = input_dataset['lon_v'].values
                            v_lat = input_dataset['lat_v'].values

                            u_lon = numpy.column_stack((u_lon, numpy.array(list(v_lon[:, -1]) + [numpy.nan])))
                            u_lat = numpy.column_stack((u_lat, numpy.array(list(v_lat[:, -1]) + [numpy.nan])))
                            v_lon = numpy.row_stack((v_lon, u_lon[-1, :]))
                            v_lat = numpy.row_stack((v_lat, u_lat[-1, :]))

                        extra_column = numpy.empty((current_times.shape[0], u_lon.shape[0], 1), dtype=u_lon.dtype)
                        extra_column[:] = 0

                        extra_row = numpy.empty((current_times.shape[0], 1, v_lon.shape[1]), dtype=v_lon.dtype)
                        extra_row[:] = 0

                        # correct for angles
                        theta = numpy.stack([input_dataset['angle'].values] * current_times.shape[0], axis=0)

                        raw_u = numpy.concatenate((input_dataset['u_sur'].values, extra_column), axis=2)
                        raw_v = numpy.concatenate((input_dataset['v_sur'].values, extra_row), axis=1)

                        geostrophic_ssu = raw_u * numpy.cos(theta) - raw_v * numpy.sin(theta)
                        geostrophic_ssv = raw_u * numpy.sin(theta) + raw_v * numpy.cos(theta)

                        ssu.append(geostrophic_ssu)
                        ssv.append(geostrophic_ssv)

                ssu = numpy.concatenate(ssu, axis=0)
                ssv = numpy.concatenate(ssv, axis=0)

                ssu = xarray.DataArray(ssu, coords={'time': times, 'u_lon': (('u_eta', 'u_xi'), u_lon),
                                                    'u_lat': (('u_eta', 'u_xi'), u_lat)},
                                       dims=('time', 'u_eta', 'u_xi'))
                ssv = xarray.DataArray(ssv, coords={'time': times, 'v_lon': (('v_eta', 'v_xi'), v_lon),
                                                    'v_lat': (('v_eta', 'v_xi'), v_lat)},
                                       dims=('time', 'v_eta', 'v_xi'))

                vector_dataset = xarray.Dataset({'ssu': ssu, 'ssv': ssv})
                vector_dataset.to_netcdf(data_path)
            else:
                raise ValueError(f'Source not recognized: "{source}"')
        else:
            vector_dataset = xarray.open_dataset(data_path)

        if 'WCOFS' in source.upper():
            velocity_field = ROMSGridVectorDataset(vector_dataset, u_name='ssu', v_name='ssv',
                                                   coordinate_system=wcofs.WCOFS_GCS)
        else:
            velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')

    contours = {}

    print(f'[{datetime.datetime.now()}]: Creating {len(contour_centers)} initial contours...')

    with futures.ThreadPoolExecutor() as concurrency_pool:
        running_futures = {
            concurrency_pool.submit(create_contour, contour_center, contour_radius, start_time, velocity_field,
                                    contour_shape): contour_id for contour_id, contour_center in
            contour_centers.items()}

        for completed_future in futures.as_completed(running_futures):
            contour_id = running_futures[completed_future]
            contour = completed_future.result()
            contours[contour_id] = contour
            print(f'[{datetime.datetime.now()}]: Contour {contour_id} created: {contour}')

    print(f'[{datetime.datetime.now()}]: Contours created.')

    schema = {'geometry': 'Polygon', 'properties': {'contour': 'int', 'datetime': 'datetime'}}

    with fiona.open(output_path, 'w', 'GPKG', schema, crs=fiona_WebMercator, layer=layer_name) as output_file:
        for contour_id, contour in contours.items():
            polygons = track_contour(contour, time_deltas)
            # diffusion_coefficient = diffusion(polygons.values())

            records = [{'geometry': shapely.geometry.mapping(polygon),
                        'properties': {'contour': contour_id, 'datetime': contour_time}} for contour_time, polygon in
                       polygons.items()]

            output_file.writerecords(records)

        # with futures.ThreadPoolExecutor() as concurrency_pool:
        #     running_futures = [concurrency_pool.submit(track_contour, contour, time_deltas) for contour in contours]
        #
        #     for completed_future in futures.as_completed(running_futures):
        #         records = completed_future.result()
        #
        #         if records is not None:
        #             output_file.writerecords(records)
        #             print(f'[{datetime.datetime.now()}]: Finished tracking contour.')
        #
        #     del running_futures

    print('done')
