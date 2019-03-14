# coding=utf-8
"""
Data utility functions.

Created on Feb 27, 2019

@author: zachary.burnett
"""

import datetime
import math
import os
from typing import Tuple

import cartopy.feature
import numpy
import pyproj
import shapely.geometry
import xarray
from matplotlib import pyplot

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WebMercator = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs')


class VelocityField:
    """
    Velocity field of u and v vectors in m/s.
    """

    def __init__(self, field: xarray.Dataset, u_name: str = 'u', v_name: str = 'v', x_name: str = 'lon',
                 y_name: str = 'lat', t_name: str = 'time'):
        """
        Create new velocity field from given dataset.

        :param field: dataset
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_name: name of x coordinate
        :param y_name: name of y coordinate
        :param t_name: name of time coordinate
        """

        self.field = field.rename({u_name: 'u', v_name: 'v', x_name: 'x', y_name: 'y', t_name: 'time'})

        x_dims = self.field['x'].dims
        y_dims = self.field['y'].dims

        lon = self.field['x'].values
        lat = self.field['y'].values

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

        self.field['x'], self.field['y'] = x, y

        self.t_delta = datetime.timedelta(seconds=float(numpy.nanmean(numpy.diff(self.field['time']))) * 1e-9)
        self.x_delta = float(numpy.nanmean(numpy.diff(self.field['x'])))
        self.y_delta = float(numpy.nanmean(numpy.diff(self.field['y'])))

    def u(self, lon: float, lat: float, time: datetime.datetime) -> float:
        """
        u velocity in m/s at coordinates

        :param lon: longitude
        :param lat: latitude
        :param time: time
        :return: u value at coordinate in m/s
        """

        x, y = pyproj.transform(WGS84, WebMercator, lon, lat)
        return self.field['u'].sel(time=time, x=x, y=y, method='nearest').values.item()

    def v(self, lon: float, lat: float, time: datetime.datetime) -> float:
        """
        v velocity in m/s at coordinates

        :param lon: longitude
        :param lat: latitude
        :param time: time
        :return: v value at coordinate in m/s
        """

        x, y = pyproj.transform(WGS84, WebMercator, lon, lat)
        return self.field['v'].sel(time=time, x=x, y=y, method='nearest').values.item()

    def velocity(self, lon: float, lat: float, time: datetime.datetime) -> float:
        """
        absolute velocity in m/s at coordinate

        :param lon: longitude
        :param lat: latitude
        :param time: time
        :return: magnitude of uv vector in m/s
        """

        return math.sqrt(self.u(lon, lat, time) ** 2 + self.v(lon, lat, time) ** 2)

    def direction(self, lon: float, lat: float, time: datetime.datetime) -> float:
        """
        angle of uv vector

        :param lon: longitude
        :param lat: latitude
        :param time: time
        :return: degree from north of uv vector
        """

        return (math.atan2(self.u(lon, lat, time), self.v(lon, lat, time)) + math.pi) * (180 / math.pi)

    def plot(self, time: datetime.datetime, color: str = None):
        quiver_plot = pyplot.quiver(X=self.field['x'], Y=self.field['y'],
                                    U=self.field['u'].sel(time=time, method='nearest'),
                                    V=self.field['v'].sel(time=time, method='nearest'), C=color, units='width')
        pyplot.quiverkey(quiver_plot, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')

    def __getitem__(self, lon: float, lat: float, time: datetime.datetime) -> Tuple[float, float]:
        """
        velocity vector (u, v) in m/s at coordinates

        :param lon: longitude
        :param lat: latitude
        :param time: time
        :return: tuple of (u, v)
        """

        return self.u(lon, lat, time), self.v(lon, lat, time)

    def __repr__(self):
        return str(self.field)


class Particle:
    """
    Particle simulation.
    """

    def __init__(self, field: VelocityField, lon: float, lat: float, time: datetime.datetime):
        """
        Create new particle within in the given velocity field.

        :param field: velocity field
        :param lon: starting longitude
        :param lat: starting latitude
        :param time: starting time
        """

        self.field = field
        self.time = time
        self.x, self.y = pyproj.transform(WGS84, WebMercator, lon, lat)
        self.u = self.field.u(lon, lat, time)
        self.v = self.field.v(lon, lat, time)

    def step(self, t_delta: datetime.timedelta = None):
        """
        Step particle by given time delta.

        :param t_delta: time delta
        """

        if t_delta is None:
            t_delta = self.field.t_delta

        if not numpy.isnan(self.u) and not numpy.isnan(self.v):
            self.x += t_delta.total_seconds() * self.u
            self.y += t_delta.total_seconds() * self.v

        self.time += t_delta

    def coordinates(self) -> tuple:
        """
        current coordinates
        :return tuple of GCS coordinates
        """

        return pyproj.transform(WebMercator, WGS84, self.x, self.y)

    def velocity(self):
        """
        current velocity within velocity field
        :return: tuple of velocity vector (u, v)
        """

        return self.u, self.v

    def plot(self, color: str = None):
        lon, lat = pyproj.transform(WebMercator, WGS84, self.x, self.y)

        if color is None:
            pyplot.plot(lon, lat)
        else:
            pyplot.plot(lon, lat, color=color)

    def __str__(self):
        return f'{self.time} {self.coordinates()} -> {self.velocity()}'


class ParticleContour:
    """
    Contour of points within a velocity field.
    """

    def __init__(self, field: VelocityField, time: datetime.datetime, points: list):
        """
        Create contour given list of points.

        :param field: velocity field
        :param time: starting time
        :param points: list of coordinate tuples (lon, lat)
        """

        self.field = field
        self.time = time
        self.particles = []

        for point in points:
            self.particles.append(Particle(field, point[0], point[1], time))

        self.polygon = shapely.geometry.Polygon([(particle.x, particle.y) for particle in self.particles])

    def step(self, t_delta: datetime.timedelta = None):
        """
        Step particle by given time delta.

        :param t_delta: time delta
        """

        if t_delta is None:
            t_delta = self.field.t_delta

        self.time += t_delta

        for particle in self.particles:
            particle.step(t_delta)

        self.polygon = shapely.geometry.Polygon([(particle.x, particle.y) for particle in self.particles])

    def plot(self, color: str = None):
        lon, lat = zip(*[pyproj.transform(WebMercator, WGS84, particle.x, particle.y) for particle in self.particles])

        if color is None:
            pyplot.plot(lon, lat)
        else:
            pyplot.plot(lon, lat, color=color)

    def area(self):
        return self.polygon.area

    def bounds(self):
        return self.polygon.bounds

    def __str__(self):
        return f'contour at time {self.time} with bounds {self.bounds()} and area {self.area()} m^2'


class CircleContour(ParticleContour):
    def __init__(self, field: VelocityField, time: datetime.datetime, center: tuple, radius: float,
                 interval: float = 500):
        """
        Create circle contour with given interval between points.

        :param field: velocity field
        :param time: starting time
        :param center: central point (lon, lat)
        :param radius: radius in m
        :param interval: interval between points in m
        """

        center_x, center_y = pyproj.transform(WGS84, WebMercator, center[0], center[1])
        circumference = 2 * math.pi * radius
        num_points = round(circumference / interval)

        points = [pyproj.transform(WebMercator, WGS84, math.cos(2 * math.pi / num_points * x) * radius + center_x,
                                   math.sin(2 * math.pi / num_points * x) * radius + center_y) for x in
                  range(0, num_points + 1)]

        super().__init__(field, time, points)


class RectangleContour(ParticleContour):
    def __init__(self, field: VelocityField, time: datetime.datetime, west_lon: float, east_lon: float,
                 south_lat: float, north_lat: float, interval: float = 500):
        """
        Create orthogonal square contour with given bounds.

        :param field: velocity field
        :param time: starting time
        :param west_lon: minimum longitude
        :param east_lon: maximum longitude
        :param south_lat: minimum latitude
        :param north_lat: maximum latitude
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

        super().__init__(field, time, points)


if __name__ == '__main__':
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    data_path = r"C:\Data\develop\output\test\rtofs_20190313.nc"
    data_time = datetime.datetime(2019, 3, 13)
    contour_center = (-123.79820, 37.31710)
    contour_radius = 100000

    print('Collecting data...')
    if os.path.exists(data_path):
        data = xarray.open_dataset(data_path)
    else:
        from PyOFS.dataset import rtofs

        data = rtofs.RTOFSDataset(data_time).to_xarray(variables=['ssu', 'ssv'])
        data.to_netcdf(data_path)

    print('Creating velocity field...')
    velocity_field = VelocityField(data, u_name='ssu', v_name='ssv')

    print('Creating starting contour...')
    contour = CircleContour(velocity_field,
                            datetime.datetime.utcfromtimestamp(velocity_field.field['time'][0].item() * 1e-9),
                            center=contour_center, radius=contour_radius)
    print(f'Contour created: {contour}')

    time_deltas = numpy.diff(velocity_field.field['time'])

    ax = pyplot.axes(projection=cartopy.crs.PlateCarree())
    ax.set_prop_cycle(color=[pyplot.cm.cool(color_index) for color_index in numpy.linspace(0, 1, len(time_deltas))])
    ax.add_feature(cartopy.feature.LAND)

    for time_delta in time_deltas:
        timedelta = datetime.timedelta(seconds=int(time_delta) * 1e-9)

        previous_area = contour.area()
        contour.step(timedelta)
        area_change = contour.area() - previous_area
        print(f'step {timedelta} to {contour.time}: change in area was {area_change} m^2')

        contour.plot()

    pyplot.show()

    print('done')
