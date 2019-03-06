# coding=utf-8

CRS_EPSG = 4326

try:
    from logbook import Logger
except ImportError:
    class Logger(object):
        def __init__(self, name, level=0):
            self.name = name
            self.level = level

        debug = info = warn = warning = notice = error = exception = \
            critical = log = lambda *a, **kw: None
