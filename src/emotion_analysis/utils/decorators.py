from datetime import datetime as dt
from functools import wraps
from logging import Logger, getLogger
from typing import Any, Callable

_logger: Logger = getLogger(__name__)


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start, label_preffix = dt.now(), func.__name__
        _logger.debug(label_preffix + " started")
        result = func(*args, **kwargs)
        _logger.debug(label_preffix + f" took: {str(dt.now() - start)}")
        return result

    return wrapper
