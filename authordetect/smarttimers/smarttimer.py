"""SmartTimer

Classes:
    :class:`SmartTimer`

Note:
    Internal operations may affect the time measurements by a factor of
    milliseconds. In a future release, this noise will be corrected.
"""


__all__ = ('TimerStat', 'SmartTimer')


import os
import re
import numpy
import cProfile
import time as std_time
from .timer import Timer
from functools import wraps, partial
from .clocks import are_clocks_compatible
from collections import namedtuple, defaultdict


_TimerStat_fields = ('min', 'max', 'total', 'avg',)
TimerStat = namedtuple('TimerStat', _TimerStat_fields)


class SmartTimer:
    """`Timer`_ container to perform time measurements in code blocks.

    Args:
        name (str, optional): Name of container. Default is *smarttimer*.

        kwargs (dict, optional): Map of options to configure the internal
            `Timer`_. Default is `Timer`_ defaults.

    A :class:`SmartTimer` allows recording elapsed time in an arbitrary
    number of code blocks. Specified points in the code are marked as either
    the beginning of a block to measure, :meth:`tic`, or as the end of a
    measured block, :meth:`toc`. Times are managed internally and ordered
    based on :meth:`tic` calls. Times can be queried, operated on, and
    written to file.

    The following schemes are supported for timing code blocks
        * Consecutive: ``tic('A')``, ``toc()``, ..., ``tic('B')``, ``toc()``
        * Cascade: ``tic('A')``, ``toc()``, ``toc()``, ...
        * Nested: ``tic('A')``, ``tic('B')``, ..., ``toc()``, ``toc()``
        * Label-paired: ``tic('A')``, ``tic('B')``, ..., ``toc('A')``,
          ``toc('B')``
        * Mixed: arbitrary combinations of schemes

    .. _`namedtuple`:
        https://docs.python.org/3.3/library/collections.html#collections.namedtuple

    Attributes:
        name (str): Name of container. May be used for filename in
            :meth:`write_to_file`.

        labels (list, str): Label identifiers of completed timed code blocks.

        active_labels (list, str): Label identifiers of active code blocks.

        seconds (list, float): Elapsed time for completed code blocks.

        minutes (list, float): Elapsed time for completed code blocks.

        times (dict): Map of times elapsed for completed blocks. Keys are the
            labels used when invoking :meth:`tic`.

        walltime (float): Elapsed time between first and last timings.
    """
    DEFAULT_CLOCK_NAME = 'process_time'

    def __init__(self, name=None, **kwargs):
        self.name = name
        self._timer = Timer(label=None, **kwargs)  # internal Timer
        self._first_tic = None  # pointer used to calculate walltime
        self._last_tic = self._timer  # pointer used to support cascade scheme
        self._timers = []  # completed time blocks
        self._timer_stack = []  # stack of active time blocks
        self._prof = None  # profiling object

    @property
    def labels(self):
        return tuple(t.label for t in self._filter_timers())

    @property
    def active_labels(self):
        return tuple(t.label for t in self._timer_stack)

    @property
    def seconds(self):
        return tuple(t.seconds for t in self._filter_timers())

    @property
    def minutes(self):
        return tuple(t.minutes for t in self._filter_timers())

    @property
    def relative_percent(self):
        return tuple(t.relative_percent for t in self._filter_timers())

    @property
    def cumulative_seconds(self):
        return tuple(t.cumulative_seconds for t in self._filter_timers())

    @property
    def cumulative_minutes(self):
        return tuple(t.cumulative_minutes for t in self._filter_timers())

    @property
    def cumulative_percent(self):
        return tuple(t.cumulative_percent for t in self._filter_timers())

    @property
    def times(self):
        times_map = defaultdict(list)
        for t in self._filter_timers():
            times_map[t.label].append(t.seconds)
        return times_map

    @property
    def clock_name(self):
        return self._timer.clock_name

    @clock_name.setter
    def clock_name(self, clock_name):
        if not are_clocks_compatible(self._timer.clock_name, clock_name):
            self._timers = list(self._filter_timers())
            self._timer_stack = []
            self._first_tic = None
            self._last_tic = self._timer
        self._timer.clock_name = clock_name

    @property
    def info(self):
        return self._timer.info

    @property
    def walltime(self):
        if not any(self._timers):
            return 0.
        return self._timer.seconds - self._first_tic.seconds

    def _filter_timers(self):
        return filter(None, self._timers)

    def __repr__(self):
        return "{cls}(name={name},"\
               " timer={timer})"\
               .format(cls=type(self).__qualname__,
                       name=repr(self.name),
                       timer=repr(self._timer))

    def __str__(self):
        if not self.labels:
            return ""
        lw = max(len('label'), max(map(len, self.labels)))
        fmt_head = "{:>" + str(lw) + "}" + 6 * " {:>12}" + os.linesep
        fmt_data = "{:>" + str(lw) + "}" + 6 * " {:12.4f}" + os.linesep
        data = fmt_head.format('label', 'seconds', 'minutes', 'rel_percent',
                               'cum_sec', 'cum_min', 'cum_percent')
        for t in self._filter_timers():
            data += fmt_data.format(t.label, t.seconds, t.minutes,
                                    t.relative_percent, t.cumulative_seconds,
                                    t.cumulative_minutes, t.cumulative_percent)
        return data

    def __enter__(self):
        self.tic()
        return self

    def __eq__(self, other):
        return NotImplemented

    __hash__ = None

    def __exit__(self, *args):
        self.toc()

    def __getitem__(self, key):
        value = self.times[key]
        return value[0] if len(value) == 1 else value

    def _update_cumulative_and_percent(self):
        total_seconds = sum(self.seconds)
        for i, t in enumerate(self._filter_timers()):
            # Skip timers already processed, only update percentages
            if t.cumulative_seconds < 0. or t.cumulative_minutes < 0.:
                t.cumulative_seconds = t.seconds
                t.cumulative_minutes = t.minutes
                if i > 0:
                    t_prev = self._timers[i - 1]
                    t.cumulative_seconds += t_prev.cumulative_seconds
                    t.cumulative_minutes += t_prev.cumulative_minutes
            t.relative_percent = t.seconds / total_seconds
            t.cumulative_percent = t.cumulative_seconds / total_seconds

    def tic(self, label=None):
        """Start measuring time.

        Measure time at the latest moment possible to minimize noise from
        internal operations.

        Args:
            label (str): Label identifier for current code block.
        """
        # _last_tic -> timer of most recent tic
        self._last_tic = Timer(label=label, clock_name=self._timer.clock_name)

        # _first_tic -> timer of first tic
        if self._first_tic is None:
            self._first_tic = self._last_tic

        # Insert Timer into stack, then record time to minimize noise
        self._timer_stack.append(self._last_tic)

        # Use 'None' as an indicator of active code blocks
        self._timers.append(None)

        # Measure time
        self._last_tic.time()

    def toc(self, label=None):
        """Stop measuring time at end of code block.

        Args:
            label (str): Label identifier for current code block.

        Returns:
            float: Measured time in seconds.

        Raises:
            Exception, KeyError: If there is not a matching :meth:`tic`.
        """
        # Error if no tic pair (e.g., toc() after instance creation)
        # _last_tic -> _timer
        if self._last_tic is self._timer:
            raise Exception("'toc()' has no matching 'tic()'")

        # Measure time at the soonest moment possible to minimize noise from
        # internal operations.
        self._timer.time()

        # Stack is not empty so there is a matching tic
        if self._timer_stack:

            # Last item or item specified by label
            stack_idx = -1

            # Label-paired timer.
            # Label can be "", so explicitly check against None.
            if label is not None:
                # Find index of last timer in stack with matching label
                for i, t in enumerate(self._timer_stack[::-1]):
                    if label == t.label:
                        stack_idx = len(self._timer_stack) - i - 1
                        break
                else:
                    raise KeyError("'{}' has no matching label".format(label))

            # Calculate time elapsed
            t_first = self._timer_stack.pop(stack_idx)
            t_diff = self._timer - t_first

            # Add extra attributes, use a negative sentinel value
            t_diff.relative_percent = -1.
            t_diff.cumulative_seconds = -1.
            t_diff.cumulative_minutes = -1.
            t_diff.cumulative_percent = -1.

            # Place time in corresponding position
            idx = [i for i, v in enumerate(self._timers)
                   if v is None][stack_idx]
            self._timers[idx] = t_diff

        # Empty stack, use _last_tic -> timer from most recent tic
        else:
            t_diff = self._timer - self._last_tic

            # Add extra attributes, use a negative sentinel value
            t_diff.relative_percent = -1.
            t_diff.cumulative_seconds = -1.
            t_diff.cumulative_minutes = -1.
            t_diff.cumulative_percent = -1.

            # Use label.
            # Label can be "", so explicitly check against None.
            if label is not None:
                t_diff.label = label

            self._timers.append(t_diff)

        # Update cumulative and percent times when all timers have completed
        if all(self._timers):
            self._update_cumulative_and_percent()

        return t_diff.seconds

    def print_info(self):
        self._timer.print_info()

    def remove(self, *keys):
        """Remove time(s) of completed code blocks.

        Args:
            keys (str): Keys to select times for removal based on the label
                used in :meth:`tic`.
        """
        for key in keys:
            for t in filter(None, self._timers[:]):
                if key == t.label:
                    self._timers.remove(t)

    def clear(self):
        self._timers = []
        self._timer_stack = []
        self._timer.clear()
        self._first_tic = None
        self._last_tic = self._timer
        if self._prof:
            self._prof.clear()
        self._prof = None

    def reset(self):
        self.name = None
        self._timer.reset()
        self._timer.clock_name = type(self).DEFAULT_CLOCK_NAME
        self.clear()

    def dump_times(self, filename=None, mode='w'):
        """Write timing results to a file.

        If *filename* is provided, then it will be used as the filename.
        Otherwise :attr:`name` is used if non-empty, else the default filename
        is used. The extension *.times* is appended only if filename does not
        already has an extension. Using *mode* the file can be overwritten or
        appended with timing data.

        .. _`open`: https://docs.python.org/3/library/functions.html#open

        Args:
            filename (str, optional): Name of file.

            mode (str, optional): Mode flag passed to `open`_. Default is *w*.
        """
        if not filename:
            if not self.name:
                raise ValueError("either provide an explicit filename or set"
                                 " 'name' attribute")
            filename = self.name
        if not os.path.splitext(filename)[1]:
            filename += '.times'

        with open(filename, mode) as fd:
            # Remove excess whitespace used by __str__
            fd.write(re.sub(r"\s*,\s*", ",",
                     re.sub(r"^\s*|\s*$", "", str(self), flags=re.MULTILINE)))

    def stats(self, label=None):
        """Compute total, min, max, and average stats for timings.

        Note:
            * *label* is compared as a word-bounded expression.

        Args:
            label (str, iterable, None, optional): String used to match timer
                labels to select. To use as a regular expression, *label*
                has to be a raw string. If None, then all completed timings are
                used.

        Returns:
            TimerStat, None: Stats in seconds and minutes (`namedtuple`_).
        """
        timers = list(self._filter_timers())

        # Label can be "", so explicitly check against None
        if label is None:
            seconds = self.seconds
            minutes = self.minutes
            selected = timers
        else:
            # Make strings iterate as strings, not characters
            if isinstance(label, str):
                label = [label]

            seconds = []
            minutes = []
            selected = []
            for ll in label:
                for t in timers:
                    if (ll.isalnum() \
                       and re.search(r"\b{}\b".format(ll), t.label)) \
                       or ll == t.label and t not in selected:
                        seconds.append(t.seconds)
                        minutes.append(t.minutes)
                        selected.append(t)

        if not selected:
            return None

        total_seconds = sum(seconds)
        total_minutes = sum(minutes)
        return TimerStat(
            min=(min(seconds), min(minutes)),
            max=(max(seconds), max(minutes)),
            total=(total_seconds, total_minutes),
            avg=(total_seconds / len(seconds), total_minutes / len(minutes)))

    def asarray(self):
        """Return timing data as a list or numpy array (no labels).

        Data is arranged as a transposed view of :meth:`__str__` and
        :meth:`to_file` formats.

        .. _`numpy.ndarray`: https://www.numpy.org/devdocs/index.html

        Returns:
            `numpy.ndarray`_, list: Timing data.
        """
        return numpy.array([self.seconds, self.minutes, self.relative_percent,
                self.cumulative_seconds, self.cumulative_minutes,
                self.cumulative_percent])

    def pic(self, subcalls=True, builtins=True):
        """Start profiling.

        .. _`profile`: https://docs.python.org/3.3/library/profile.html

        See `profile`_
        """
        self._prof = cProfile.Profile(timer=self._timer.clock,
                                      subcalls=subcalls,
                                      builtins=builtins)
        self._prof.enable()

    def poc(self):
        """Stop profiling."""
        self._prof.disable()
        self._prof.create_stats()
        self._prof.clear()

    def print_profile(self, sort='time'):
        self._prof.print_stats(sort)

    def get_profile(self):
        return self._prof.getstats()

    def dump_profile(self, filename=None, mode='w'):
        """Write profiling results to a file.

        If *filename* is provided, then it will be used as the filename.
        Otherwise :attr:`name` is used if non-empty, else the default filename
        is used. The extension *.prof* is appended only if filename does not
        already has an extension. Using *mode* the file can be overwritten or
        appended with timing data.

        .. _`open`: https://docs.python.org/3/library/functions.html#open

        Args:
            filename (str, optional): Name of file.

            mode (str, optional): Mode flag passed to `open`_. Default is *w*.
        """
        if not filename:
            if not self.name:
                raise ValueError("either provide an explicit filename or set"
                                 " 'name' attribute")
            filename = self.name
        if not os.path.splitext(filename)[1]:
            filename += '.prof'
        self._prof.dump_stats(filename)

    sleep = std_time.sleep
