"""SmartTimer decorators."""


__all__ = ('smarttime', 'decorator_timer')


from functools import wraps
from .smarttimer import SmartTimer


# Global instance for all decorators
decorator_timer = SmartTimer("Function decorator")


def smarttime(func, *, timer=None):
    """Measure runtime for functions/methods.

    Args:
        timer (SmartTimer, optional): Instance for measuring time. If None,
            then global instance, *decorator_timer*, is used instead.
    """
    if not timer: timer = decorator_timer

    @wraps(func)
    def wrapper(*args, **kwargs):
        timer.tic(func.__qualname__)
        ret = func(*args, **kwargs)
        timer.toc()
        return ret
    return wrapper
