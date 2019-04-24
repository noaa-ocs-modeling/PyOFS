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
import fiona
import haversine
import math
import numpy
import pyproj
import shapely.geometry
import xarray
from matplotlib import pyplot, quiver

WGS84 = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WebMercator = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs')


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
    Vector field with time component using xarray dataset.
    """

    def __init__(self, dataset: xarray.Dataset, u_name: str = 'u', v_name: str = 'v', x_name: str = 'lon',
                 y_name: str = 'lat', t_name: str = 'time', coordinate_system: pyproj.Proj = None):
        """
        Create new velocity field from given dataset.

        :param dataset: xarray dataset containing velocity data (u, v)
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_name: name of x coordinate
        :param y_name: name of y coordinate
        :param t_name: name of time coordinate
        :param coordinate_system: coordinate system of dataset
        """

        self.coordinate_system = coordinate_system if coordinate_system is not None else WGS84

        variables_to_rename = {u_name: 'u', v_name: 'v', x_name: 'x', y_name: 'y', t_name: 'time'}
        self.dataset = dataset.rename(variables_to_rename)
        del dataset

        x, y = pyproj.transform(self.coordinate_system, WebMercator,
                                *numpy.meshgrid(self.dataset['x'].values, self.dataset['y'].values))

        self.dataset['x'] = x[0, :]
        self.dataset['y'] = y[:, 0]

        super().__init__(numpy.diff(self.dataset['time'].values))

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.dataset['u'].interp(time=time, x=point[0], y=point[1], method='linear').values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        return self.dataset['v'].interp(time=time, x=point[0], y=point[1], method='linear').values.item()

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
        Create new velocity field from given dataset.

        :param dataset: xarray dataset containing velocity data (u, v)
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_names: names of x coordinates
        :param y_names: names of y coordinates
        :param t_name: name of time coordinate
        :param coordinate_system: coordinate system of dataset
        """

        self.coordinate_system = coordinate_system if coordinate_system is not None else WGS84

        variables_to_rename = {u_name: 'u', v_name: 'v', t_name: 'time'}
        variables_to_rename.update(dict(zip(x_names, ('u_x', 'v_x'))))
        variables_to_rename.update(dict(zip(y_names, ('u_y', 'v_y'))))
        self.dataset = dataset.rename(variables_to_rename)
        del dataset

        self.native_u_x, self.native_u_y = pyproj.transform(WGS84, self.coordinate_system, self.dataset['u_x'].values,
                                                            self.dataset['u_y'].values)
        self.native_v_x, self.native_v_y = pyproj.transform(WGS84, self.coordinate_system, self.dataset['v_x'].values,
                                                            self.dataset['v_y'].values)

        super().__init__(numpy.diff(self.dataset['time'].values))

    def u(self, point: numpy.array, time: datetime.datetime) -> float:
        transformed_point = pyproj.transform(WebMercator, self.coordinate_system, *point)

        eta_index = self.dataset['u_eta'].max() * (
                (transformed_point[0] - self.native_u_x.min()) / (self.native_u_x.max() - self.native_u_x.min()))
        xi_index = self.dataset['u_xi'].max() * (
                (transformed_point[1] - self.native_u_y.min()) / (self.native_u_y.max() - self.native_u_y.min()))

        return self.dataset['u'].interp(time=time, u_eta=eta_index, u_xi=xi_index, method='linear').values.item()

    def v(self, point: numpy.array, time: datetime.datetime) -> float:
        transformed_point = pyproj.transform(WebMercator, self.coordinate_system, *point)

        eta_index = self.dataset['v_eta'].max() * (
                (transformed_point[0] - self.native_u_x.min()) / (self.native_u_x.max() - self.native_u_x.min()))
        xi_index = self.dataset['v_xi'].max() * (
                (transformed_point[1] - self.native_v_y.min()) / (self.native_v_y.max() - self.native_v_y.min()))

        return self.dataset['v'].interp(time=time, v_eta=eta_index, v_xi=xi_index, method='linear').values.item()

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
                k_2 = delta_seconds * self.field[self.coordinates() + k_1 / 2, self.time + (delta_t / 2)]

                if order > 2:
                    k_3 = delta_seconds * self.field[self.coordinates() + k_2 / 2, self.time + (delta_t / 2)]

                    if order > 3:
                        k_4 = delta_seconds * self.field[self.coordinates() + k_3, self.time + delta_t]

                        if order > 4:
                            raise ValueError('Methods above 4th order are not implemented.')
                        else:
                            delta_vector = 1 / 6 * k_1 + 1 / 3 * k_2 + 1 / 3 * k_3 + 1 / 6 * k_4
                    else:
                        delta_vector = 1 / 6 * k_1 + 2 / 3 * k_2 + 1 / 6 * k_3
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

    def write(self, filename: str):
        schema = {
            'geometry': 'Polygon',
            'properties': {'date': 'date'}
        }

        with fiona.open(filename, 'a', 'GPKG', schema) as output_file:
            output_file.write({
                'geometry': shapely.geometry.mapping(self.geometry()),
                'properties': {'date': self.time}
            })

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

    source = 'wcofs'
    contour_shape = 'circle'
    order = 4

    contour_radius = 3000
    contour_centers = [(-123.79820, 37.31710)]
    start_time = datetime.datetime(2019, 4, 22)

    period = datetime.timedelta(days=3)
    time_delta = datetime.timedelta(days=1)

    time_deltas = [time_delta for index in range(int(period / time_delta))]

    output_path = rf"C:\Data\OFS\develop\output\test\contours_{start_time.strftime('%Y%m%d')}.shp"

    print('Creating velocity field...')
    if source == 'rankine':
        vortex_radius = contour_radius * 5
        vortex_center = translate_geographic_coordinates(contour_centers[0],
                                                         numpy.array([contour_radius * -2, contour_radius * -2]))
        vortex_period = datetime.timedelta(days=5)
        velocity_field = RankineVortex(vortex_center, vortex_radius, vortex_period, time_deltas)

        radii = range(1, vortex_radius * 2, 50)
        points = [numpy.array(pyproj.transform(WGS84, WebMercator, *vortex_center)) + numpy.array([radius, 0]) for
                  radius in radii]
        velocities = [velocity_field.velocity(point, start_time) for point in points]

        # main_figure = pyplot.figure()
        # main_figure.suptitle(
        #     f'Rankine vortex with {vortex_period.total_seconds() / 3600} hour period ({velocity_field.angular_velocity:.6f} rad/s) and radius of {vortex_radius} m')
        # plot_axis = main_figure.add_subplot(1, 2, 1)
        # plot_axis.set_xlabel('distance from center (m)')
        # plot_axis.set_ylabel('tangential velocity (m/s)')
        #
        # map_axis = main_figure.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())
        #
        # plot_axis.plot(radii, velocities)
        # velocity_field.plot(start_time, map_axis)
        # pyplot.show()
    else:
        from PyOFS import DATA_DIR

        data_path = os.path.join(DATA_DIR, 'output', 'test', f'{source.lower()}_{start_time.strftime("%Y%m%d")}.nc')

        print('Collecting data...')
        if not os.path.exists(data_path):
            if source.upper() == 'HFR':
                from PyOFS.dataset import hfr

                vector_dataset = hfr.HFRRange(start_time, start_time + datetime.timedelta(days=1)).to_xarray(
                    variables=('ssu', 'ssv'), mean=False)
                vector_dataset.to_netcdf(data_path)
            elif source.upper() == 'RTOFS':
                from PyOFS.dataset import rtofs

                vector_dataset = rtofs.RTOFSDataset(start_time).to_xarray(variables=('ssu', 'ssv'), mean=False)
                vector_dataset.to_netcdf(data_path)
            elif source.upper() == 'WCOFS':
                from PyOFS.dataset import wcofs

                vector_dataset = wcofs.WCOFSDataset(start_time).to_xarray(variables=('ssu', 'ssv'))
                vector_dataset.to_netcdf(data_path)
            else:
                raise ValueError(f'Source not recognized: "{source}"')
        else:
            vector_dataset = xarray.open_dataset(data_path)

        if source.upper() == 'WCOFS':
            from PyOFS.dataset import wcofs

            coordinate_system = wcofs.WCOFS_ROTATED_POLE

            velocity_field = ROMSGridVectorDataset(vector_dataset, u_name='ssu', v_name='ssv',
                                                   coordinate_system=wcofs.WCOFS_ROTATED_POLE)
        else:
            velocity_field = VectorDataset(vector_dataset, u_name='ssu', v_name='ssv')

    contours = {}

    for contour_center in contour_centers:
        print('Creating starting contour...')
        if contour_shape == 'circle':
            contour = CircleContour(contour_center, contour_radius, start_time, velocity_field)
        elif contour_shape == 'square':
            southwest_corner = translate_geographic_coordinates(contour_center, -contour_radius)
            northeast_corner = translate_geographic_coordinates(contour_center, contour_radius)
            contour = RectangleContour(southwest_corner[0], northeast_corner[0], southwest_corner[1],
                                       northeast_corner[1], start_time, velocity_field)
        else:
            contour = PointContour(contour_center, start_time, velocity_field)

        print(f'Contour created: {contour}')

        contours[tuple(contour_center)] = contour

    plot_colors = pyplot.get_cmap("tab10").colors
    contour_colors = [pyplot.cm.cool(color_index) for color_index in
                      numpy.linspace(0, 1, len(velocity_field.time_deltas) + 1)]

    main_figure = pyplot.figure()
    ordinal_string = lambda n: f'{n}{"tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4]}'
    main_figure.suptitle(
        f'{ordinal_string(order)} order {source.upper()} {contour_shape} contours with {contour_radius / 1000} km' +
        f' radius, tracked every {time_delta.total_seconds() / 3600} hours over {period.total_seconds() / 3600} hours total')

    plot_axis = main_figure.add_subplot(1, 2, 1)
    plot_axis.set_xlabel('time')

    if contour_shape == 'point':
        plot_axis.set_ylabel('% of starting radial distance')
    else:
        plot_axis.set_ylabel('% of starting area')
        plot_axis.set_ylim([80, 180])

    map_axis = main_figure.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())
    map_axis.set_prop_cycle(color=contour_colors)
    map_axis.add_feature(cartopy.feature.LAND)

    # velocity_field.plot(start_time, map_axis)

    for contour_center, contour in contours.items():
        values = {}

        if contour_shape == 'point':
            initial_value = haversine.haversine(pyproj.transform(WebMercator, WGS84, *contour.coordinates()),
                                                contour_center, unit='m')
            values[contour.time] = 100

            for time_delta in time_deltas:
                if type(time_delta) is numpy.timedelta64:
                    time_delta = time_delta.item()

                if type(time_delta) is int:
                    time_delta = datetime.timedelta(seconds=time_delta * 1e-9)

                previous_radial_distance = numpy.sqrt(numpy.sum(
                    (contour.coordinates() - numpy.array(pyproj.transform(WGS84, WebMercator, *contour_center))) ** 2))
                # previous_radial_distance = haversine.haversine(
                #     pyproj.transform(WebMercator, WGS84, *contour.coordinates()),
                #     contour_center, unit='m')

                contour.step(time_delta, order)

                current_radial_distance = numpy.sqrt(numpy.sum(
                    (contour.coordinates() - numpy.array(pyproj.transform(WGS84, WebMercator, *contour_center))) ** 2))
                # current_radial_distance = haversine.haversine(
                #     pyproj.transform(WebMercator, WGS84, *contour.coordinates()),
                #     contour_center, unit='m')
                values[contour.time] = (1 + (current_radial_distance - initial_value) / initial_value) * 100

                print(f'step {time_delta} to {contour.time}: change in radius was ' +
                      f'{(current_radial_distance - previous_radial_distance) / previous_radial_distance * 100:.2f}%')

            for particle in contour.particles:
                particle.plot(slice(0, -1), map_axis)
        else:
            initial_value = contour.area()
            values[contour.time] = 100
            contour.write(output_path)

            for time_delta in time_deltas:
                if type(time_delta) is numpy.timedelta64:
                    time_delta = time_delta.item()

                if type(time_delta) is int:
                    time_delta = datetime.timedelta(seconds=time_delta * 1e-9)

                contour.plot(map_axis)
                previous_area = contour.area()

                contour.step(time_delta, order)
                contour.write(output_path)

                current_area = contour.area()
                values[contour.time] = (1 + (current_area - initial_value) / initial_value) * 100

                print(f'step {time_delta} to {contour.time}: change in area was ' +
                      f'{(current_area - previous_area) / previous_area * 100:.2f}%')

        plot_axis.plot(values.keys(), values.values(), '-o')

    pyplot.show()

    print('done')
