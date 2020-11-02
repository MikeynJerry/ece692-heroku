# -*- coding: utf-8 -*-
"""Timer module used as building block for timing libraries."""


__all__ = ('Timer')


import time as std_time
from typing import Any, Union, Optional
from .clocks import CLOCKS, print_clock, get_clock_info, are_clocks_compatible


class Timer:
    """Container to manage time measured from a clock/counter.

    Args:
        label (str, None, optional): Timer identifier. Default is None.

        seconds (float, optional): Time measured in fractional seconds.
            Default is 0.0.

        clock_name (str, optional): Name to select a time measurement
            function from :attr:`clocks.CLOCKS` map. Default is
            :attr:`DEFAULT_CLOCK_NAME`.

        timer (Timer, optional): Instance to copy properties from.
            Default is None.


    A :class:`Timer` allows recording the current time measured by a
    registered timing function. Time is recorded in fractional seconds
    and fractional minutes. :class:`Timer` supports addition,
    difference, and logical operators.


    Note:

        * Only Timers with compatible clocks support arithmetic and
          logical operators. Compatible Timers use the same
          implementation function in the backend, see
          :func:`clocks.get_clock_info`.


    .. literalinclude:: ../examples/timer.py
        :language: python
        :linenos:
        :name: Timer_API
        :caption: Timer API examples.


    .. _`namedtuple`:
        https://docs.python.org/3.3/library/collections.html#collections.namedtuple

    Attributes:
        DEFAULT_CLOCK_NAME (str): Default clock name.
        CLOCKS (dict): Map between clock names and timing functions.
        label (str): Identifier.
        clock_name (str, None): Name to select a timing function from
            :attr:`CLOCKS` map.
        seconds (float): Time measured in fractional seconds.
        minutes (float): Time measured in minutes.
        info (ClockInfo): Clock attributes (`namedtuple`_).
    """
    DEFAULT_CLOCK_NAME = 'perf_counter'

    def __init__(self,
                 label: Optional[Union[None, str]] = None,
                 **kwargs: Optional[Any]) -> None:
        self._seconds = None
        self._minutes = None
        self._clock_name = None
        self._clock = None

        timer = kwargs.get('timer')
        if timer:
            self.label = timer.label
            self.seconds = timer.seconds
            self.clock_name = timer.clock_name
        else:
            self.label = label
            self.seconds = kwargs.get('seconds', 0.)
            self.clock_name = kwargs.get('clock_name',
                                         type(self).DEFAULT_CLOCK_NAME)

    def __repr__(self) -> str:
        return "{cls}(label={label},"\
               " seconds={seconds},"\
               " clock_name={clock_name},"\
               " function='{info.function}',"\
               " adjustable={info.adjustable},"\
               " implementation='{info.implementation}',"\
               " monotonic={info.monotonic},"\
               " resolution={info.resolution})"\
               .format(cls=type(self).__qualname__,
                       label=repr(self.label),
                       seconds=self.seconds,
                       clock_name=repr(self.clock_name),
                       info=self.info)

    def __str__(self) -> str:
        ffmt = "{lw}.{pr}f".format(lw=12, pr=6)
        fmt = "{}" + 2 * (" {:" + ffmt + "}")
        return fmt.format(self.label, self.seconds, self.minutes)

    def __add__(self, other: 'Timer') -> 'Timer':
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        labels = filter(None, [self.label, other.label])
        return type(self)(label='+'.join(labels),
                          seconds=self.seconds + other.seconds,
                          clock_name=self.clock_name)

    def __sub__(self, other: 'Timer') -> 'Timer':
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        labels = filter(None, [self.label, other.label])
        return type(self)(label='-'.join(labels),
                          seconds=self.seconds - other.seconds,
                          clock_name=self.clock_name)

    def __eq__(self, other: 'Timer') -> bool:
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        return self.seconds == other.seconds

    __hash__ = None

    def __lt__(self, other: 'Timer') -> bool:
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        return self.seconds < other.seconds

    def __le__(self, other: 'Timer') -> bool:
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        return self.seconds <= other.seconds

    def __gt__(self, other: 'Timer') -> bool:
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        return self.seconds > other.seconds

    def __ge__(self, other: 'Timer') -> bool:
        type(self)._validate_compatibility(self.clock_name, other.clock_name)
        return self.seconds >= other.seconds

    @property
    def seconds(self) -> float:
        return self._seconds

    @seconds.setter
    def seconds(self, seconds: float):
        self._seconds = float(seconds)
        self._minutes = seconds / 60.

    @property
    def minutes(self) -> float:
        return self._minutes

    @property
    def clock_name(self) -> str:
        return self._clock_name

    @clock_name.setter
    def clock_name(self, clock_name: str):
        if self._clock_name \
                and not are_clocks_compatible(self.clock_name, clock_name):
            self.clear()
        # Set function first to catch dict key error before setting clock name
        self._clock = CLOCKS[clock_name]
        self._clock_name = clock_name

    @property
    def info(self) -> 'ClockInfo':
        return get_clock_info(self.clock_name)

    def print_info(self):
        print_clock(self.clock_name)

    def time(self, *args: Optional[Any], **kwargs: Optional[Any]) -> float:
        self.seconds = self._clock(*args, **kwargs)
        return self.seconds

    def clear(self):
        self.seconds = 0.

    def reset(self):
        self.label = None
        self.clock_name = type(self).DEFAULT_CLOCK_NAME
        self.clear()

    sleep = std_time.sleep

    @staticmethod
    def _validate_compatibility(clock_name1: str, clock_name2: str) -> None:
        if not are_clocks_compatible(clock_name1, clock_name2):
            raise Exception("Timers are not compatible")
