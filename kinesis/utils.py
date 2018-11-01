from functools import wraps

import pandas as pd

__all__ = ['cache_to']


def cache_to(path):
    """
    Decorator to cache pandas DataFrame to csv
    """
    def decorator_cache(func):
        @wraps(func)
        def wrapper_cache():
            try:
                r = pd.read_csv(path)
                print("Data loaded from {:s}".format(path))
                return r
            except FileNotFoundError:
                r = func()
                r.to_csv(path)
                print("Data written to {:s}".format(path))
                return r
        return wrapper_cache
    return decorator_cache
