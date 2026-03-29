from typing import Callable, Optional, Union
from warnings import warn


def deprecated(replacement_func: Optional[Union[str, Callable]] = None, message: Optional[str] = None) -> Callable:
    """Mark a function as deprecated"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            a = []
            if replacement_func:
                if isinstance(replacement_func, str):
                    replacement = replacement_func
                else:
                    replacement = replacement_func.__name__
                a.append(f"The function '{func.__name__}' is deprecated, use '{replacement}' instead.")
            else:
                a.append(f"The function '{func.__name__}' is deprecated.")
            if message:
                a.append(message)
            warn(" ".join(a), DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["deprecated"]
