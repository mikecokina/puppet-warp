from __future__ import annotations

import logging
from typing import Any

SUPPRESS_LOGGER: bool = False


# noinspection PyPep8Naming
def getLogger(name: str, *, suppress: bool = False) -> logging.Logger | Logger:  # noqa: N802
    """Return a logger instance, optionally suppressing output.

    Return a standard logging.Logger unless logging is suppressed, in which case
    return a dummy Logger instance that ignores all log calls.

    :param name: Name of the logger.
    :param suppress: Whether to suppress logging output.
    :return: Logger-like object.
    """
    if SUPPRESS_LOGGER:
        suppress = True

    if suppress:
        return Logger(name)

    return logging.getLogger(name=name)


# noinspection PyPep8Naming
def getPersistentLogger(name: str) -> logging.Logger:  # noqa: N802
    """Return a persistent logging.Logger instance.

    Always return a real logger regardless of suppression settings.

    :param name: Name of the logger.
    :return: logging.Logger instance.
    """
    return logging.getLogger(name=name)


class Logger:
    """Provide a dummy logger that suppresses all log output.

    Implement the logging.Logger interface partially, so it can be used as a
    drop-in replacement when logging is disabled.
    """

    def __init__(self, name: str, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Initialize the dummy logger.

        :param name: Name of the logger.
        """
        self.name = name

    def info(self, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Ignore info-level log messages."""
        return

    def error(self, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Ignore error-level log messages."""
        return

    def debug(self, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Ignore debug-level log messages."""
        return

    def warning(self, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Ignore warning-level log messages."""
        return

    def warn(self, *_: Any, **__: Any) -> None:  # noqa: ANN401
        """Ignore deprecated warn-level log messages.

        Provided for backward compatibility.
        """
        return
