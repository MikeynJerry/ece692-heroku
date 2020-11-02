#! /usr/bin/python3

import copy
from typing import Any, Union, Iterable


__all__ = ['tlist']


class tlist(list):
    """A list container with typing support for items, copy control, and
       metadata.

    Args:
        data (iterable): Sequence of items to initialize list with.

        force_deepcopy (bool): Deepcopy items before adding to list.

        metadata (any): Custom metadata.

        dtype (type): Allowed types of items.
    """
    def __init__(
        self,
        data=None,
        *,
        force_deepcopy: bool = False,
        metadata: Any = None,
        dtype: Union[type, Iterable[type]] = None,
    ):
        self.__dict__['force_deepcopy'] = force_deepcopy
        self.__dict__['metadata'] = metadata
        self.__dict__['dtype'] = dtype
        if dtype is not None:
            # If not a type, assume it is an iterable of types
            dtype = {dtype} if isinstance(dtype, type) else set(dtype)
        if data is None:
            data = []
        else:
            self._type_checks(data)
            if self.force_deepcopy:
                data = copy.deepcopy(data)
        super().__init__(data)

    def __setattr__(self, name, value):
        if name in ('dtype',):
            raise AttributeError("can't set attribtue")
        super().__setattr__(name, value)

    def __setitem__(self, index, item):
        # If not int, assume it is a slice
        if isinstance(index, int):
            self._type_check(item)
        else:
            self._type_checks(item)
        if self.force_deepcopy:
            item = copy.deepcopy(item)
        super().__setitem__(index, item)

    def __reduce__(self):
        # This allows pickle to work correctly for list subclass.
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        return type(self), (list(self),), self.__dict__

    def append(self, item):
        self._type_check(item)
        if self.force_deepcopy:
            item = copy.deepcopy(item)
        super().append(item)

    def insert(self, index, item):
        self._type_check(item)
        if self.force_deepcopy:
            item = copy.deepcopy(item)
        super().insert(index, item)

    def extend(self, items):
        self._type_checks(items)
        if self.force_deepcopy:
            items = copy.deepcopy(items)
        super().extend(items)

    def copy(self):
        return type(self)(self, **self.__dict__)

    def _type_check(self, item: Any):
        """Checks type of single item."""
        if self.dtype is not None:
            for dtype in self.dtype:
                if isinstance(item, dtype):
                    break
            else:
                raise TypeError(
                    f'invalid item {type(item)}, requires a {self.dtype}'
                )

    def _type_checks(self, items: Iterable[Any]):
        """Checks type of multiple items."""
        if self.dtype is not None:
            for item in items:
                self._type_check(item)
