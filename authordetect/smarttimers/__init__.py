"""SmartTimers package.

Todo:
    Improve handling of module imports vs special attributes if using
    non-standard libraries. Currently applicable when running tox environments
    that do not include the install_requirements.txt:
        * Use try-except in __init__.py (bad hack)
        * Use explicit values in setup.py and doc/conf.py
        * Include install_requirements.txt in tox environment (e.g., doc)
"""


__title__ = "SmartTimers"
# __name__ = "SmartTimers"
__version__ = "1.4.0"
__description__ = "Timing library with a simple and flexible API" \
                  " supporting a wide variety of timing paradigms."
__keywords__ = (
    "timer",
    "runtime",
    "profiling",
    "measurement",
    "performance",
)
__url__ = "https://github.com/edponce/smarttimers"
__author__ = "Eduardo Ponce, The University of Tennessee, Knoxville, TN"
__author_email__ = "edponce2010@gmail.com"
__license__ = "MIT"
__copyright__ = "2018 Eduardo Ponce"


__all__ = (
    'Timer',
    'CLOCKS',
    'smarttime',
    'SmartTimer',
    'print_clock',
    'print_clocks',
    'get_clock_info',
    'register_clock',
    'decorator_timer',
    'unregister_clock',
)


from .timer import Timer
from .smarttimer import SmartTimer
from .decorators import smarttime, decorator_timer
from .clocks import (CLOCKS, print_clock, print_clocks, get_clock_info,
                     register_clock, unregister_clock,
                     are_clocks_compatible, is_clock_function_valid)
