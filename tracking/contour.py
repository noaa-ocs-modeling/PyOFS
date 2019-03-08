import datetime
import math

from matplotlib import pyplot
import numpy
import pyproj
import shapely.geometry
import xarray

from dataset import _utilities

GCS = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
PCS = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs')


class VelocityField:
    """
    Velocity field of u and v vectors in m/s.
    """

    def __init__(self, field: xarray.Dataset, u_name: str = 'u', v_name: str = 'v', x_name: str = 'lon',
                 y_name: str = 'lat'):
        """
        Create new velocity field from given dataset.

        :param field: dataset
        :param u_name: name of u variable
        :param v_name: name of v variable
        :param x_name: name of x coordinate
        :param y_name: name of y coordinate
        """

        self.field = field.rename({x_name: 'x', y_name: 'y'})

        x_dims = self.field['x'].dims
        y_dims = self.field['y'].dims

        lon = self.field['x'].values
        lat = self.field['y'].values

        if len(x_dims) == 1 or len(y_dims) == 1:
            lon, lat = numpy.meshgrid(lon, lat)

        x, y = PCS(lon, lat)

        if len(x_dims) == 1 or len(y_dims) == 1:
            x = x[0, :]
            y = y[:, 0]
            self.has_multidim_coords = False
        else:
            x = (x_dims, x)
            y = (y_dims, y)
            self.has_multidim_coords = True

        self.field['x'], self.field['y'] = x, y

        self.u_name = u_name
        self.v_name = v_name

        self.t_delta = datetime.timedelta(seconds=float(numpy.nanmean(numpy.diff(self.field['time']))) * 1e-9)
        self.x_delta = float(numpy.nanmean(numpy.diff(self.field['x'])))
        self.y_delta = float(numpy.nanmean(numpy.diff(self.field['y'])))

    def u(self, time: datetime.datetime, lon: float, lat: float) -> float:
        """
        u velocity in m/s at coordinates

        :param time: time
        :param lon: longitude
        :param lat: latitude
        :return: u value at coordinate in m/s
        """

        x, y = PCS(lon, lat)
        return self.field[self.u_name].sel(time=time, x=x, y=y, method='nearest').values.item()

    def v(self, time: datetime.datetime, lon: float, lat: float) -> float:
        """
        v velocity in m/s at coordinates

        :param time: time
        :param lon: longitude
        :param lat: latitude
        :return: v value at coordinate in m/s
        """

        x, y = PCS(lon, lat)
        return self.field[self.v_name].sel(time=time, x=x, y=y, method='nearest').values.item()

    def velocity(self, time: datetime.datetime, lon: float, lat: float) -> float:
        """
        absolute velocity in m/s at coordinate

        :param time: time
        :param lon: longitude
        :param lat: latitude
        :return: magnitude of uv vector in m/s
        """

        return math.sqrt(self.u(time, lon, lat) ** 2 + self.v(time, lon, lat) ** 2)

    def direction(self, time: datetime.datetime, lon: float, lat: float) -> float:
        """
        angle of uv vector

        :param time: time
        :param lon: longitude
        :param lat: latitude
        :return: degree from north of uv vector
        """

        return (math.atan2(self.u(time, lon, lat), self.v(time, lon, lat)) + math.pi) * (180 / math.pi)

    def plot(self, time: datetime.datetime):
        quiver_plot = pyplot.quiver(self.field['x'], self.field['y'],
                                    self.field[self.u_name].sel(time=time, method='nearest'),
                                    self.field[self.v_name].sel(time=time, method='nearest'), units='width')
        pyplot.quiverkey(quiver_plot, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                         coordinates='figure')

    def __getitem__(self, time: datetime.datetime, lon: float, lat: float):
        return self.u(time, lon, lat), self.v(time, lon, lat)

    def __repr__(self):
        return str(self.field)


class Particle:
    """
    Particle simulation.
    """

    def __init__(self, field: VelocityField, time: datetime.datetime, lon: float, lat: float):
        """
        Create new particle within in the given velocity field.

        :param field: velocity field
        :param time: starting time
        :param lon: starting longitude
        :param lat: starting latitude
        """

        self.field = field
        self.set(time, lon, lat)

    def step(self, t_delta: datetime.timedelta = None):
        """
        Step particle by given time delta.

        :param t_delta: time delta
        """

        if t_delta is None:
            t_delta = self.field.t_delta

        if not numpy.isnan(self.u):
            self.x += self.u * t_delta.total_seconds()

        if not numpy.isnan(self.v):
            self.y += self.v * t_delta.total_seconds()

        self.time += t_delta

        self.set(self.time, *GCS(self.x, self.y))

    def set(self, time: datetime.datetime, lon: float, lat: float):
        """
        Set particle to given time and coordinates within velocity field.

        :param time: time
        :param lon: longitude
        :param lat: latitude
        """

        x, y = PCS(lon, lat)

        if self.field.has_multidim_coords:
            x_delta = self.field.x_delta
            y_delta = self.field.y_delta

            nearest_time = self.field.field.sel(time=numpy.datetime64(time), method='nearest')
            cell = nearest_time.interp(x=x, y=y)
        else:
            cell = self.field.field.sel(time=numpy.datetime64(time), x=x, y=y, method='nearest')

        self.time = _utilities.datetime64_to_datetime(cell['time'])

        self.x = cell['x'].item()
        self.y = cell['y'].item()
        self.u = cell['ssu'].item()
        self.v = cell['ssv'].item()

    def coordinates(self) -> tuple:
        """
        current coordinates
        :return tuple of GCS coordinates
        """

        return GCS(self.x, self.y)

    def velocity(self):
        """
        current velocity within velocity field
        :return: tuple of velocity vector (u, v)
        """

        return self.u, self.v

    def __str__(self):
        return f'{self.time} {self.coordinates()} -> {self.velocity()}'


class Contour:
    """
    Contour of points within a velocity field.
    """

    def __init__(self, field: VelocityField, time: datetime.datetime, points: list):
        """
        Create contour given list of points.

        :param field: velocity field
        :param time: starting time
        :param points: list of coordinate tuples (x, y)
        """

        self.particles = []

        for point in points:
            self.particles.append(Particle(field, time, *point))

    def step(self, t_delta: datetime.datetime = None):
        """
        Step particle by given time delta.

        :param t_delta: time delta
        """

        for particle in self.particles:
            particle.step(t_delta)

    def polygon(self):
        return shapely.geometry.Polygon([(particle.x, particle.y) for particle in self.particles])

    def area(self):
        return self.polygon().area

    def bounds(self):
        return self.polygon().bounds

    def __str__(self):
        return f'contour with bounds {self.bounds()} and area of {self.area()} m^2'


class CircleContour(Contour):
    def __init__(self, field: VelocityField, time: datetime.datetime, center: tuple, radius: float,
                 interval: float = 500):
        """
        Create circle contour with given interval between points.

        :param field: velocity field
        :param time: starting time
        :param center: central point (x, y)
        :param radius: radius in m
        :param interval: interval between points in m
        """

        points = []
        super().__init__(field, time, points)


class RectangleContour(Contour):
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

        corners = {'sw': (west_lon, south_lat), 'nw': (west_lon, north_lat), 'ne': (east_lon, north_lat),
                   'se': (east_lon, south_lat)}
        points = []

        for corner_name, corner in corners.items():
            points.append(corner)

            if corner_name is 'sw':
                edge_length = PCS(*corners['nw'])[1] - PCS(*corners['sw'])[1]
            elif corner_name is 'nw':
                edge_length = PCS(*corners['ne'])[0] - PCS(*corners['nw'])[0]
            elif corner_name is 'ne':
                edge_length = PCS(*corners['ne'])[1] - PCS(*corners['se'])[1]
            elif corner_name is 'se':
                edge_length = PCS(*corners['se'])[0] - PCS(*corners['sw'])[0]

            for stride in range(int(edge_length / interval)):
                x, y = PCS(*corner)

                if corner_name is 'sw':
                    y += stride
                elif corner_name is 'nw':
                    x += stride
                elif corner_name is 'ne':
                    y -= stride
                elif corner_name is 'se':
                    x -= stride

                points.append(GCS(x, y))

        super().__init__(field, time, points)


if __name__ == '__main__':
    from dataset import hfr

    model_datetime = datetime.datetime(2019, 2, 25)
    # model_datetime.replace(hour=3, minute=0, second=0, microsecond=0)

    print('Collecting data...')
    # data = wcofs.WCOFSDataset(model_datetime, source='avg').to_xarray(variables=('ssu', 'ssv'))
    data = hfr.HFRRange(model_datetime, model_datetime + datetime.timedelta(days=1), source='UCSD').to_xarray(
        variables=['ssu', 'ssv'], mean=False)

    print('Creating velocity field...')
    velocity_field = VelocityField(data, u_name='ssu', v_name='ssv')
    print(f'Velocity field created: {velocity_field}')

    for time in velocity_field.field['time']:
        print(f'Plotting {time}')
        velocity_field.plot(time)
        break

    pyplot.show()

    # print('Creating starting contour...')
    # contour = RectangleContour(velocity_field,
    #                            datetime.datetime.utcfromtimestamp(velocity_field.field['time'][0].item() * 1e-9),
    #                            west_lon=-122.99451, east_lon=-122.73859, south_lat=36.82880, north_lat=36.93911)
    # print(f'Contour created: {contour}')
    #
    # for t_delta in numpy.diff(velocity_field.field['time']):
    #     timedelta = datetime.timedelta(seconds=int(t_delta) * 1e-9)
    #
    #     print(f'Time step: {timedelta}')
    #     contour.step(timedelta)
    #     print(f'New state: {contour}')

    print('done')
