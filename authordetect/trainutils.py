#! /usr/bin/python3

import sklearn
from typing import Any, List, Tuple, Iterable, Iterator


__all__ = [
    'shuffle_split',
    'split_data_into_train_test',
]


def shuffle_split(
    data: Iterable[Any],
    *,
    n_splits: int = 1,
    test_size: float = 0.1,
    **kwargs,
) -> Iterator[Iterable[Iterable[Any]]]:
    """Random permutation of iterable data.

    Returns: iter(train_idxs, test_idxs)
    """
    return sklearn.model_selection.ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        **kwargs,
    ).split(data)


def split_data_into_train_test(
    data: Iterable[Any],
    *,
    sort: bool = False,
    **kwargs,
) -> Tuple[List[Any], List[Any]]:
    """Permute documents into training and test datasets."""
    train_idxs, test_idxs = next(shuffle_split(data, **kwargs))

    # Sort indexes
    if sort:
        train_idxs.sort()
        test_idxs.sort()

    # Extract documents
    train = [data[i] for i in train_idxs]
    test = [data[i] for i in test_idxs]
    return train, test
