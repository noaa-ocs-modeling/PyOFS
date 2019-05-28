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

from PyOFS import utilities

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WebMercator = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs')


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

        :param point: lon / lat coordinates
        :param time: time
        :return: v value at coordinate in m/s
        """

        pass

    def velocity(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        absolute velocity in m/s at coordinate

        :param point: lon / lat coordinates
        :param time: time
        :return: magnitude of uv vector in m/s
        """

        return math.sqrt(self.u(point, time) ** 2 + self.v(point, time) ** 2)

    def direction(self, point: numpy.array, time: datetime.datetime) -> float:
        """
        angle of uv vector

        :param point: lon / lat coordinates
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
        if numpy.any(numpy.isnan(point)):
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

    def plot(self, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        points = []
        radii = numpy.linspace(1, self.radius * 2, 20)

        for radius in radii:
            num_points = 50
            points.extend([(math.cos(2 * math.pi / num_points * point_index) * radius + self.center[0],
                            math.sin(2 * math.pi / num_points * point_index) * radius + self.center[1]) for
                           point_index in range(0, num_points + 1)])

        vectors = [self[point, datetime.datetime.now()] for point in points]
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
        if time is None:
            time = self.dataset['time'].values[0]

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = pyproj.transform(WebMercator, WGS84,
                                    *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        quiver_plot = axis.quiver(lon, lat, self.dataset['u'].sel(time=time, method='nearest'),
                                  self.dataset['v'].sel(time=time, method='nearest'), units='width', **kwargs)
        axis.quiverkey(quiver_plot, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')

        return quiver_plot


class ROMSGridVectorDataset(VectorField):
    def __init__(self, dataset: xarray.Dataset, u: str = 'u', v: str = 'v', u_x: str = 'u_x', u_y: str = 'u_y',
                 v_x: str = 'v_x', v_y: str = 'v_y', t: str = 'time'):
        """
        Create new velocity field from given observation.

        :param dataset: xarray observation containing velocity data (u, v)
        :param u: u variable
        :param v: v variable
        :param u_lon: u longitude coordinate
        :param u_lat: u latitude coordinate
        :param v_lon: v longitude coordinate
        :param v_lat: v latitude coordinate
        :param t: time coordinate
        """

        self.rotated_pole = utilities.RotatedPoleCoordinateSystem(wcofs.ROTATED_POLE)

        self.dataset = xarray.Dataset(
            {'u': xarray.DataArray(dataset[u].values,
                                   coords={'time': dataset[t], 'u_x': dataset[u_x], 'u_y': dataset[u_y]},
                                   dims=('time', 'u_x', 'u_y')),
             'v': xarray.DataArray(dataset[v].values,
                                   coords={'time': dataset[t], 'v_x': dataset[v_x], 'v_y': dataset[v_y]},
                                   dims=('time', 'v_x', 'v_y'))})

        super().__init__(numpy.diff(self.dataset['time'].values))

    def interp(self, variable: str, point: numpy.array, time: datetime.datetime) -> xarray.DataArray:
        rotated_point = self.rotated_pole.rotate_coordinates(point)

        x_name = f'{variable}_x'
        y_name = f'{variable}_y'

        x_range = [self.dataset[x_name].sel({x_name: rotated_point[0]}, method='bfill').values.item(),
                   self.dataset[x_name].sel({x_name: rotated_point[0]}, method='ffill').values.item()]
        y_range = [self.dataset[y_name].sel({y_name: rotated_point[1]}, method='bfill').values.item(),
                   self.dataset[y_name].sel({y_name: rotated_point[1]}, method='ffill').values.item()]
        time_range = [self.dataset['time'].sel(time=time, method='bfill').values,
                      self.dataset['time'].sel(time=time, method='ffill').values]

        if time_range[0] == time_range[1]:
            time_range = time_range[0]

        cell = self.dataset[variable].sel({'time': time_range, x_name: x_range, y_name: y_range})

        if 'time' in cell.dims:
            return cell.interp({'time': time, x_name: rotated_point[0], y_name: rotated_point[1]})
        else:
            return cell.interp({x_name: rotated_point[0], y_name: rotated_point[1]})

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.interp('u', point, time).values.item() / (
                geodetic_radius(point[1]) * (numpy.cos(point[1] * numpy.pi / 180) * 180 / numpy.pi))

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.interp('v', point, time).values.item() / (geodetic_radius(point[1]))

    def plot(self, time: datetime.datetime, axis: pyplot.Axes = None, **kwargs) -> quiver.Quiver:
        if time is None:
            time = self.dataset['time'].values[0]

        if axis is None:
            axis = pyplot.axes(projection=cartopy.crs.PlateCarree())

        lon, lat = self.rotated_pole.unrotate_coordinates(
            numpy.meshgrid(self.dataset['u_x'].values, self.dataset['u_y'].values))

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

        self.locations = [numpy.array(location)]
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

            if order >= 2:
                k_2 = delta_seconds * self.field[self.coordinates() + 0.5 * k_1, self.time + 0.5 * delta_t]

                if order >= 3:
                    k_3 = delta_seconds * self.field[self.coordinates() + 0.5 * k_2, self.time + 0.5 * delta_t]

                    if order >= 4:
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

        new_coordinates = self.coordinates() + delta_vector

        self.locations.append(new_coordinates)
        self.time += delta_t

        self.vector = self.field[new_coordinates, self.time]

    def coordinates(self, projection: pyproj.Proj = None) -> numpy.array:
        """
        Get current coordinates.

        :param projection: output coordinate projection
        :return tuple of coordinates
        """

        if projection is None or projection == WGS84:
            return self.locations[-1]
        else:
            return pyproj.transform(WGS84, projection, *self.locations[-1])

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

        # with futures.ThreadPoolExecutor() as concurrency_pool:
        #     for particle in self.particles:
        #         concurrency_pool.submit(particle.step, delta_t, order)

        for particle in self.particles:
            particle.step(delta_t, order)

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

        num_points = round(2 * math.pi * radius / interval)
        points = []

        for point_angle in [point_index * 2 * math.pi / num_points for point_index in range(num_points + 1)]:
            point_x = math.cos(point_angle) * radius + center_x
            point_y = math.sin(point_angle) * radius + center_y
            point = pyproj.transform(WebMercator, WGS84, point_x, point_y)
            # point = (point_x, point_y)
            points.append(point)

        # points = list(zip(*pyproj.transform(WebMercator, WGS84, *shapely.geometry.Point(center_x, center_y).buffer(
        #     radius).exterior.coords.xy)))

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


def geodetic_radius(latitude: float) -> float:
    semimajor_radius = 6378137
    semiminor_radius = 6356752.314245

    sine_latitude = numpy.sin(latitude * numpy.pi / 180) * 180 / numpy.pi
    cosine_latitude = numpy.cos(latitude * numpy.pi / 180) * 180 / numpy.pi

    return numpy.sqrt(
        ((semimajor_radius ** 2 * cosine_latitude) ** 2 + (semiminor_radius ** 2 + sine_latitude) ** 2) / (
                (semimajor_radius * cosine_latitude) ** 2 + (semiminor_radius + sine_latitude) ** 2))


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


def track_contour(contour: ParticleContour, time_deltas: List[datetime.timedelta]) -> Dict[
    datetime.datetime, shapely.geometry.Polygon]:
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


if __name__ == '__main__':
    import sys

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

    from PyOFS import DATA_DIR
    from PyOFS.model import rtofs, wcofs
    from PyOFS.observation import hf_radar

    source = 'wcofs_qck'
    contour_shape = 'circle'
    order = 4

    contour_radius = 10000

    contour_centers = {}
    start_time = datetime.datetime(2016, 9, 25, 1)

    period = datetime.timedelta(hours=5)
    time_delta = datetime.timedelta(hours=1)

    time_deltas = [time_delta for index in range(int(period / time_delta))]

    output_path = os.path.join(DATA_DIR, 'output', 'test', 'contours.gpkg')
    layer_name = f'{source}_{start_time.strftime("%Y%m%dT%H%M%S")}_{(start_time + period).strftime("%Y%m%dT%H%M%S")}_' + \
                 f'{int(time_delta.total_seconds() / 3600)}h'

    print(f'[{datetime.datetime.now()}]: Started processing...')

    with fiona.open(os.path.join(DATA_DIR, 'reference', 'study_points.gpkg'),
                    layer='study_points') as contour_centers_file:
        for point in contour_centers_file:
            contour_id = point['properties']['name']

            if contour_id == 'A4':
                contour_centers[contour_id] = point['geometry']['coordinates']

    print(f'[{datetime.datetime.now()}]: Creating velocity field...')
    if source == 'rankine':
        vortex_radius = contour_radius * 5
        vortex_center = translate_geographic_coordinates(next(iter(contour_centers.values())),
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
                rotated_pole = utilities.RotatedPoleCoordinateSystem(wcofs.ROTATED_POLE)

                qck_path = os.path.join(DATA_DIR, 'input', 'wcofs', 'qck')
                input_filenames = os.listdir(qck_path)

                combined_time = []
                combined_ssu = []
                combined_ssv = []

                theta = None
                u_lon = None
                u_lat = None
                v_lon = None
                v_lat = None

                for input_filename in sorted(input_filenames):
                    input_dataset = xarray.open_dataset(os.path.join(qck_path, input_filename))

                    time = input_dataset['ocean_time'].values
                    combined_time.extend(time)

                    if 'GEOSTROPHIC' in source.upper():
                        if u_lon is None or u_lat is None or v_lon is None or v_lat is None:
                            u_lon = v_lon = input_dataset['lon_rho'].values
                            u_lat = v_lat = input_dataset['lat_rho'].values

                        gravitational_acceleration = 9.80665
                        # sidereal_rotation_period = datetime.timedelta(hours=23, minutes=53, seconds=4.1)
                        # coriolis_frequencies = 4 * math.pi / sidereal_rotation_period.total_seconds() * numpy.sin(
                        #     input_dataset['lat_rho'].values * math.pi / 180)

                        coriolis_frequencies = input_dataset['f']
                        delta_x_reciprocal = input_dataset['pn']
                        delta_y_reciprocal = input_dataset['pm']
                        sea_level = input_dataset['zeta']

                        first_term = gravitational_acceleration / coriolis_frequencies

                        raw_ssu = numpy.repeat(numpy.expand_dims(-first_term * input_dataset['pm'], axis=0), len(time),
                                               axis=0) * numpy.concatenate(
                            (numpy.empty((len(time), 1, len(sea_level['xi_rho']))), sea_level.diff('eta_rho')), axis=1)
                        raw_ssv = numpy.repeat(numpy.expand_dims(first_term * input_dataset['pn'], axis=0), len(time),
                                               axis=0) * numpy.concatenate(
                            (numpy.empty((len(time), len(sea_level['eta_rho']), 1)), sea_level.diff('xi_rho')), axis=2)

                        numpy.empty((len(time), sea_level.shape[1], 1))

                        raw_ssu[numpy.isnan(raw_ssu)] = 0
                        raw_ssv[numpy.isnan(raw_ssv)] = 0
                    else:
                        if theta is None:
                            theta = input_dataset['angle'].values

                        if u_lon is None or u_lat is None or v_lon is None or v_lat is None:
                            u_lon = input_dataset['lon_u'].values
                            u_lat = input_dataset['lat_u'].values
                            v_lon = input_dataset['lon_v'].values
                            v_lat = input_dataset['lat_v'].values

                            u_lon = numpy.column_stack((u_lon, numpy.array(list(v_lon[:, -1]) + [numpy.nan])))
                            u_lat = numpy.column_stack((u_lat, numpy.array(list(v_lat[:, -1]) + [numpy.nan])))
                            v_lon = numpy.row_stack((v_lon, u_lon[-1, :]))
                            v_lat = numpy.row_stack((v_lat, u_lat[-1, :]))

                        extra_column = numpy.empty((time.shape[0], u_lon.shape[0], 1), dtype=u_lon.dtype)
                        extra_column[:] = 0

                        extra_row = numpy.empty((time.shape[0], 1, v_lon.shape[1]), dtype=v_lon.dtype)
                        extra_row[:] = 0

                    raw_ssu = numpy.concatenate((input_dataset['u_sur'].values, extra_column), axis=2)
                    raw_ssv = numpy.concatenate((input_dataset['v_sur'].values, extra_row), axis=1)

                    # # correct for angles
                    # sin_theta = numpy.repeat(numpy.expand_dims(numpy.sin(input_dataset['angle']), axis=0), len(time),
                    #                          axis=0)
                    # cos_theta = numpy.repeat(numpy.expand_dims(numpy.cos(input_dataset['angle']), axis=0), len(time),
                    #                          axis=0)
                    # rotated_ssu = raw_ssu * cos_theta - raw_ssv * sin_theta
                    # rotated_ssv = raw_ssu * sin_theta + raw_ssv * cos_theta

                    combined_ssu.append(raw_ssu)
                    combined_ssv.append(raw_ssv)

                combined_ssu = numpy.concatenate(combined_ssu, axis=0)
                combined_ssv = numpy.concatenate(combined_ssv, axis=0)

                u_x, u_y = rotated_pole.rotate_coordinates((u_lon, u_lat))
                v_x, v_y = rotated_pole.rotate_coordinates((v_lon, v_lat))

                ssu_dataarray = xarray.DataArray(combined_ssu,
                                                 coords={'time': combined_time, 'u_x': u_x[:, 0], 'u_y': u_y[0, :]},
                                                 dims=('time', 'u_x', 'u_y'))
                ssv_dataarray = xarray.DataArray(combined_ssv,
                                                 coords={'time': combined_time, 'v_x': v_x[:, 0], 'v_y': v_y[0, :]},
                                                 dims=('time', 'v_x', 'v_y'))
                theta_dataarray = xarray.DataArray(theta, coords={'v_x': v_x[:, 0], 'v_y': v_y[0, :]},
                                                   dims=('v_x', 'v_y'))

                # ssu_dataarray = xarray.DataArray(combined_ssu,
                #                                  coords={'time': combined_time, 'u_lon': (('u_eta', 'u_xi'), u_lon),
                #                                          'u_lat': (('u_eta', 'u_xi'), u_lat)},
                #                                  dims=('time', 'u_eta', 'u_xi'))
                # ssv_dataarray = xarray.DataArray(combined_ssv,
                #                                  coords={'time': combined_time, 'v_lon': (('v_eta', 'v_xi'), v_lon),
                #                                          'v_lat': (('v_eta', 'v_xi'), v_lat)},
                #                                  dims=('time', 'v_eta', 'v_xi'))

                vector_dataset = xarray.Dataset({'ssu': ssu_dataarray, 'ssv': ssv_dataarray, 'angle': theta_dataarray})
                vector_dataset.to_netcdf(data_path)
            else:
                raise ValueError(f'Source not recognized: "{source}"')
        else:
            vector_dataset = xarray.open_dataset(data_path)

        # schema = {'geometry': 'Point', 'properties': {'u': 'float', 'v': 'float'}}
        #
        # for layer_time in vector_dataset['time']:
        #     points = zip(*rotated_pole.unrotate_coordinates((vector_dataset['u_x'], vector_dataset['u_y'])))
        #
        #     records = [dict(zip(('geometry', 'properties'), (points, dict(zip(('u', 'v'), (vector_dataset['u'], vector_dataset['v']))))))]
        #
        #     with fiona.open(os.path.join(DATA_DIR, 'output', 'test', 'ssuv.gpkg'), 'w', 'GPKG', schema,
        #                     crs=fiona.crs.from_epsg(3857),
        #                     layer=f'vectors_{layer_time.strftime("%Y%m%d%H%M%S")}') as output_layer:
        #         output_layer.writerecords(records)

        if time_delta != datetime.timedelta(seconds=(numpy.diff(vector_dataset['time'][0:2]) * 1e-9).item()):
            if time_delta == datetime.timedelta(days=1):
                vector_dataset = vector_dataset.resample(time='D').mean()

        if 'WCOFS' in source.upper():
            velocity_field = ROMSGridVectorDataset(vector_dataset, u='ssu', v='ssv')
        else:
            velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')

    # u = velocity_field.dataset['u'].sel(time=start_time)
    #
    # transformed_point = utilities.rotate_coordinates((-135.08643988445394, 45.821736984330855), wcofs.ROTATED_POLE)
    #
    # u_eta_index = u['u_eta'].max() * ((transformed_point[0] - u['u_x'].min()) / (u['u_x'].max() - u['u_x'].min()))
    # u_xi_index = u['u_xi'].max() * ((transformed_point[1] - u['u_y'].min()) / (u['u_y'].max() - u['u_y'].min()))
    #
    # pyplot.quiver(numpy.flip(numpy.ma.array(u, mask=u == 0)))
    # pyplot.plot(eta_index, xi_index, 'o')
    # pyplot.show()

    # study_point = velocity_field.rotated_pole.rotate_coordinates(contour_centers['A4'])
    #
    # x_slice = slice(study_point[0] - 1, study_point[0] + 1)
    # y_slice = slice(study_point[1] - 1, study_point[1] + 1)
    #
    # pyplot.quiver(velocity_field.dataset['u'].sel(time=start_time, u_x=x_slice, u_y=y_slice),
    #               velocity_field.dataset['v'].sel(time=start_time, v_x=x_slice, v_y=y_slice))
    # pyplot.show()

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

    # define schema
    schema = {'geometry': 'Polygon', 'properties': {'contour': 'str', 'datetime': 'datetime'}}

    # add value fields to schema
    schema['properties'].update({'area': 'float', 'perimeter': 'float'})

    with fiona.open(output_path, 'w', 'GPKG', schema, crs=fiona.crs.from_epsg(4326), layer=layer_name) as output_layer:
        for contour_id, contour in contours.items():
            polygons = track_contour(contour, time_deltas)

            records = [{'geometry': shapely.geometry.mapping(polygon),
                        'properties': {'contour': contour_id, 'datetime': contour_time, 'area': polygon.area,
                                       'perimeter': polygon.length}} for
                       contour_time, polygon in polygons.items()]

            output_layer.writerecords(records)

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
