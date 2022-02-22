import logging

SUPPRESS_LOGGER = False


def getLogger(name, suppress=False):
    if SUPPRESS_LOGGER is not None:
        suppress = SUPPRESS_LOGGER

    return logging.getLogger(name=name) if not suppress else Logger(name)


def getPersistentLogger(name):
    return logging.getLogger(name=name)


class Logger(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass
