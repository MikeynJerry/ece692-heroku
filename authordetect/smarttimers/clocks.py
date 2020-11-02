# -*- coding: utf-8 -*-
"""Utilities to manage system and custom clocks/counters.


.. _`namedtuple`:
    https://docs.python.org/3.3/library/collections.html#collections.namedtuple
.. _`OrderedDict`:
    https://docs.python.org/3/library/collections.html#collections.OrderedDict
.. _`time`: https://docs.python.org/3/library/time.html
.. _`time.perf_counter()`:
    https://docs.python.org/3/library/time.html#time.perf_counter
.. _`time.monotonic()`:
    https://docs.python.org/3/library/time.html#time.monotonic
.. _`time.process_time()`:
    https://docs.python.org/3/library/time.html#time.process_time
.. _`time.thread_time()`:
    https://docs.python.org/3/library/time.html#time.thread_time
.. _`time.time()`:
    https://docs.python.org/3/library/time.html#time.time
.. _`time.clock()`:
    https://docs.python.org/3/library/time.html#time.clock


Global variables:
    * CLOCKS (`OrderedDict`_ of {str: callable})
    * ClockInfo (`namedtuple`_)


Global functions:
    * :func:`print_clock`
    * :func:`print_clocks`
    * :func:`get_clock_info`
    * :func:`register_clock`
    * :func:`unregister_clock`
    * :func:`are_clocks_compatible`
    * :func:`is_clock_function_valid`


:mod:`clocks` uses a simple and extensible API which allows registering new
timing functions. A timing function is compliant if it returns a time
measured using numeric values. The function can contain arbitrary
positional and/or keyword arguments or no arguments.


Available time measurement functions in :attr:`CLOCKS`:
    * 'perf_counter' : `time.perf_counter()`_ (system-wide)
    * 'monotonic'    : `time.monotonic()`_ (system-wide)
    * 'process_time' : `time.process_time()`_ (process-wide)
    * 'thread_time'  : `time.thread_time()`_ (thread-wide)
    * 'time'         : `time.time()`_ (system-wide, deprecated)
    * 'clock'        : `time.clock()`_ (implementation-specific, deprecated)


.. literalinclude:: ../examples/clocks.py
    :language: python
    :linenos:
    :name: clocks_API
    :caption: :mod:`clocks` API examples.


Note:
    * The available timing functions are built-ins from the standard
      `time`_ library.
    * Custom timing functions need to have a compliant interface. If
      a custom timing function is non-compliant, then place it
      inside a compliant wrapper function.
"""


__all__ = (
    'CLOCKS',
    'ClockInfo',
    'print_clock',
    'print_clocks',
    'get_clock_info',
    'register_clock',
    'unregister_clock',
    'are_clocks_compatible',
    'is_clock_function_valid',
    )


import os
import sys
import time as std_time
from numbers import Number
from typing import Any, Callable, Optional
from collections import namedtuple, OrderedDict


# Predefined clocks
if sys.version_info >= (3, 7):
    CLOCKS = OrderedDict((
        ("perf_counter", std_time.perf_counter),
        ("monotonic", std_time.monotonic),
        ("process_time", std_time.process_time),
        ("thread_time", std_time.thread_time),
        ("time", std_time.time),
        ))
else:
    CLOCKS = OrderedDict((
        ("perf_counter", std_time.perf_counter),
        ("monotonic", std_time.monotonic),
        ("process_time", std_time.process_time),
        ("time", std_time.time),
        ("clock", std_time.clock),
        ))


# Structure used to represent clock attributes.
_CLOCK_INFO_FIELDS = ('function', *vars(std_time.get_clock_info('time')))
_CLOCK_INFO_DEFAULTS = len(_CLOCK_INFO_FIELDS) * (None,)
if sys.version_info >= (3, 7):
    ClockInfo = namedtuple('ClockInfo', _CLOCK_INFO_FIELDS,
                           defaults=_CLOCK_INFO_DEFAULTS)
else:
    ClockInfo = namedtuple('ClockInfo', _CLOCK_INFO_FIELDS)
    ClockInfo.__new__.__defaults__ = _CLOCK_INFO_DEFAULTS

# Map of clock name to clock attributes, dict of dicts.
_CLOCKS_INFO_ATTR = {}


def get_clock_info(clock_name: str) -> 'ClockInfo':
    # Check local structure in case even a standard clock name has been
    # overwritten which is not recommended, but...
    if clock_name in _CLOCKS_INFO_ATTR:
        return ClockInfo(function=CLOCKS[clock_name],
                         **_CLOCKS_INFO_ATTR[clock_name])
    return ClockInfo(function=CLOCKS[clock_name],
                     **vars(std_time.get_clock_info(clock_name)))


def are_clocks_compatible(clock_name1: str, clock_name2: str) -> bool:
    # [1:] to skip 'function' attribute.
    return get_clock_info(clock_name1)[1:] == get_clock_info(clock_name2)[1:]


def is_clock_function_valid(clock_function: Callable) -> bool:
    # Check that clock function returns a number.
    # If function requires arguments to work correctly, then ignore
    # validation and assume function is valid.
    clock_value = 0.
    try:
        clock_value = clock_function()
    except Exception as ex:
        pass
    return isinstance(clock_value, Number)


def register_clock(clock_name: str,
                   clock_function: Callable,
                   **kwargs: Any) -> None:
    if not is_clock_function_valid(clock_function):
        raise ValueError("clock function to register, '{}', does not returns "
                         "a numeric value".format(clock_function.__qualname__))
    CLOCKS[clock_name] = clock_function
    kwargs['implementation'] = kwargs.get('implementation', clock_function)
    _CLOCKS_INFO_ATTR[clock_name] = kwargs


def unregister_clock(clock_name: str) -> None:
    CLOCKS.pop(clock_name)
    if clock_name in _CLOCKS_INFO_ATTR:
        _CLOCKS_INFO_ATTR.pop(clock_name)


def print_clock(clock_name: str) -> None:
    info = get_clock_info(clock_name)
    field_maxlen = max(map(len, info._fields))
    text = ["'{}'".format(clock_name)]
    for item in info._asdict().items():
        text += ["    {:<{width}} : {}".format(width=field_maxlen, *item)]
    print(*text, sep=os.linesep)


def print_clocks() -> None:
    for i, clock_name in enumerate(CLOCKS):
        if i > 0:
            print()
        print_clock(clock_name)
