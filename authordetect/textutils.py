#! /usr/bin/python3

import os
import json
import pickle
import functools
from smart_open import open
from typing import Any, Tuple, Union, Iterable, Callable


__all__ = [
    'load_text',
    'save_text',
    'load_json',
    'save_json',
    'load_pickle',
    'save_pickle',
    'merge_texts',
    'is_iter_not_str',
    'iter2str',
    'get_text_from_span',
    'item_count',
    'find_max_depth',
]


def load_text(src: str, *, phony: bool = False) -> str:
    """Get text content from a given source.

    Args:
        phony (bool): If set, 'src' is considered as the text content.
    """
    if not phony:
        try:
            with open(src) as fd:
                return fd.read()
        except (FileNotFoundError, OSError):
            pass
    return src


def save_text(text: str, fn: str, *, flag: str = 'w') -> int:
    """Save a string to a file."""
    # Create output directory (if available)
    outdir = os.path.dirname(fn)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(fn, flag) as fd:
        return fd.write(text)


def load_json(fn: str) -> list:
    with open(fn) as fd:
        return json.load(fd)


def save_json(data: object, fn: str):
    """Write a data structure into a JSON file."""
    # Create output directory (if available)
    outdir = os.path.dirname(fn)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(fn, 'w') as fd:
        json.dump(data, fd)


def load_pickle(fn: str):
    with open(fn, 'rb') as fd:
        return pickle.load(fd)


def save_pickle(data: object, fn: str):
    # Create output directory (if available)
    outdir = os.path.dirname(fn)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(fn, 'wb') as fd:
        pickle.dump(data, fd)


def merge_texts(srcs: Iterable[str], delim: str = '\n') -> str:
    """Get text content from multiple sources and join into a
    delimited string."""
    return delim.join(map(load_text, srcs))


def is_iter_not_str(obj: Any) -> bool:
    return hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes))


def iter2str(
    data: Union[str, Iterable[Any]],
    delim: Union[str, Iterable[str]] = None,
) -> str:
    """Convert arbitrary iterable into a delimited string.

    Args:
        delim (str): Iterable of delimiters used to join strings at
            specific depths. Delimiter order is from top to lower levels.
            If empty delimiter, no delimiter is used throughout all levels.
            If only 1 delimiter, it is only used in top level.
            If multiple delimiters, it is used from top to lower levels.
    """
    if not is_iter_not_str(data):
        return data if isinstance(data, (str, bytes)) else str(data)
    if delim is None or len(delim) == 0:
        delim = ''
    return delim[0:1].join(map(lambda x: iter2str(x, delim[1:]), data))


def get_text_from_span(
    text: str,
    span: Union[Iterable[Tuple[int, int]], Tuple[int, int]],
) -> str:
    """Extract text corresponding to span(s)."""
    if is_iter_not_str(span[0]):
        return list(map(lambda x: text[slice(*x)], span))
    else:
        return text[slice(*span)]


def item_count(
    data: Iterable[Any],
    *,
    attr: str = None,
    func: Callable = None,
    **kwargs,
) -> int:
    """Count the number of elementary items in an arbitrary nested iterable.

    Args:
        attr (str): Object's attribute to use as the data for counting.
            This is used in nested structures that share a common attribute.

        func (callable): Transformation applied to data.

    Kwargs:
        Passed directly to attribute member (if callable) and/or to 'func'.
    """
    if attr is not None and hasattr(data, attr):
        _attr = getattr(data, attr)
        data = _attr(**kwargs) if callable(_attr) else _attr
    if func is not None:
        data = func(data, **kwargs)
    _item_count = functools.partial(item_count, attr=attr, func=func, **kwargs)
    return sum(map(_item_count, data)) if is_iter_not_str(data) else 1


def find_max_depth(
    data: Iterable[Any],
    *,
    dtype: Union[type, Tuple[type]] = (str, bytes),
) -> int:
    """Find maximum nested depth in iterable structures.

    Args:
        dtype (type): Object type to use as base case and stop recursion.
    """
    if isinstance(data, dtype) or len(data) == 0:
        return 0
    _depth = functools.partial(find_max_depth, dtype=dtype)
    return 1 + max(map(_depth, data))
