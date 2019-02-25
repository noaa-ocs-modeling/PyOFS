import datetime
import math
import pyproj
import xarray

WGS84_LONLAT = pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs')
WEB_MERCATOR = pyproj.Proj(
    '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs')


class VelocityField:
    def __init__(self, field: xarray.Dataset, u_name: str = 'u', v_name: str = 'v'):
        self.field = field.rename({'lon': 'x', 'lat': 'y'})

        x_dims = field['lon'].dims
        y_dims = field['lat'].dims

        x = self.field['x']
        y = self.field['y']

        if len(x_dims) == 1 or len(y_dims) == 1:
            x, y = numpy.meshgrid(self.field['x'], self.field['y'])

        x, y = WEB_MERCATOR(x, y)

        if len(x_dims) == 1 or len(y_dims) == 1:
            x = x[0, :]
            y = y[:, 0]
        else:
            x = (x_dims, x)
            y = (y_dims, y)

        self.field['x'], self.field['y'] = x, y

        self.u_name = u_name
        self.v_name = v_name

        self.t_delta = datetime.timedelta(seconds=float(numpy.nanmean(numpy.diff(self.field['time']))) * 1e-9)
        self.x_delta = float(numpy.nanmean(numpy.diff(self.field['x'])))
        self.y_delta = float(numpy.nanmean(numpy.diff(self.field['y'])))

    def u(self, time: datetime.datetime, lon: float, lat: float):
        return self.field[self.u_name].sel(time=time, lon=lon, lat=lat, method='nearest')

    def v(self, time: datetime.datetime, lon: float, lat: float):
        return self.field[self.v_name].sel(time=time, lon=lon, lat=lat, method='nearest')

    def velocity(self, time: datetime.datetime, lon: float, lat: float):
        return math.sqrt(self.u(time, lon, lat) ** 2 + self.v(time, lon, lat) ** 2)

    def direction(self, time: datetime.datetime, lon: float, lat: float):
        return (math.atan2(self.u(time, lon, lat), self.v(time, lon, lat)) + math.pi) * (180 / math.pi)

    def __getitem__(self, time: datetime.datetime, lon: float, lat: float):
        return (self.u(time, lon, lat), self.v(time, lon, lat))

    def __repr__(self):
        return str(self.field)


class Particle:
    def __init__(self, field: VelocityField, time: datetime.datetime, lon: float, lat: float):
        self.field = field
        x, y = WEB_MERCATOR(lon, lat)
        self.set(time, x, y)

    def step(self, t_delta: datetime.timedelta = None):
        if t_delta is None:
            t_delta = self.field.t_delta

        if not numpy.isnan(self.u):
            self.x += self.u * t_delta.total_seconds()

        if not numpy.isnan(self.v):
            self.y += self.v * t_delta.total_seconds()

        self.time += t_delta

        self.set(self.time, self.x, self.y)

    def set(self, time: datetime.datetime, x: float, y: float) -> bool:
        nearest_cell = self.field.field.sel(time=numpy.datetime64(time), x=x, y=y, method='nearest')
        self.time = datetime.datetime.utcfromtimestamp(nearest_cell['time'].values.astype(datetime.datetime) * 1e-9)

        self.x = nearest_cell['x'].item()
        self.y = nearest_cell['y'].item()
        self.u = nearest_cell['ssu'].item()
        self.v = nearest_cell['ssv'].item()

        return nearest_cell['time'] == self.field.field['time'][-1].item()

    def coords(self):
        return WGS84_LONLAT(self.x, self.y)

    def __str__(self):
        return f'{self.time} ({self.x}, {self.y}) -> ({self.u}, {self.v})'


class Contour:
    pass


class CircleContour(Contour):
    pass


if __name__ == '__main__':
    from dataset import hfr
    import numpy

    model_datetime = datetime.datetime.now()
    model_datetime.replace(hour=3, minute=0, second=0, microsecond=0)

    # data = wcofs.WCOFSDataset(model_datetime, source='avg')
    data = hfr.HFRRange(model_datetime, model_datetime + datetime.timedelta(days=1))

    velocity_field = VelocityField(data.to_xarray(variables=('ssu', 'ssv'), mean=False))

    particle = Particle(velocity_field,
                        datetime.datetime.utcfromtimestamp(velocity_field.field['time'][0].item() * 1e-9),
                        lon=float(numpy.mean(data.netcdf_dataset['lon'])),
                        lat=float(numpy.mean(data.netcdf_dataset['lat'])))

    for t_delta in numpy.diff(velocity_field.field['time']):
        timedelta = datetime.timedelta(seconds=int(t_delta) * 1e-9)

        print(f'Time step: {timedelta}')
        particle.step(timedelta)
        print(f'New state: {particle}')

    print('done')
