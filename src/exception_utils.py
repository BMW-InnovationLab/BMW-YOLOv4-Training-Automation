import sys


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ConfigError(Error):
    def __init__(self, expression, message) -> None:
        self.expression: str = expression
        self.message: str = message


def excepthook(kind, message, traceback) -> None:
    print("{0}: {1}".format(kind.__name__, message))


# To disable traceback
sys.excepthook = excepthook
