from datetime import datetime as dt
from logging import Logger, getLogger
from typing import Any, Callable

_logger: Logger = getLogger(__name__)


def timeit(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        start: dt = dt.now()
        _logger.debug(f"{func.__name__} started")
        result = func(*args, **kwargs)
        _logger.debug(f"{func.__name__} took: {str(dt.now() - start)}")
        return result

    return wrapper
