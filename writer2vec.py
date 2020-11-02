#! /usr/bin/python3

import numpy
import random
import itertools
from authordetect import Author
from typing import Any, Union, Iterable


def writer2vec(
    data: Union[str, Iterable[str]],
    labels: Iterable[Any],
    outfiles: Union[str, Iterable[str]] = None,  # filenames for embedding matrices
    *,
    embedding: 'EmbeddingModel' = None,
    **kwargs,
):
    if isinstance(data, str):
        data = [data]
        labels = [labels]

    authors = []
    for i, (corpus, label) in enumerate(zip(data, labels), start=1):
        author = Author(corpus, label)
        author.writer2vec(embedding=embedding, **kwargs)
        authors.append(author)

    vectors = []
    labels = []
    for author in authors:
        vectors.append(author.docs_vectors)
        labels.append([author.label] * len(author.docs_vectors))

    if outfiles:
        if isinstance(outfiles, str):
            outfiles = [outfiles]
        for author, outfile in zip(authors, outfiles):
            if outfile:
                author.embedding.save(outfile)

    return vectors, labels


def split_combine_data(data, labels, pos_frac=0.5, neg_frac=None, *, seed=None):
    """Split a nested dataset of the form [[pos], [neg], [neg], ...],
    using the fractions provided.

    The positive data is first and is used completely with the pos_frac
    stating the fraction it represents. Partial negative datasets are
    generated via random sampling. Data is returned in the same format.
    """
    random.seed(seed)

    total_size = len(data[0]) / pos_frac

    # Calculate negative fraction, uniformly for each negative dataset
    _neg_frac = (1 - pos_frac) / (len(data) - 1)
    if neg_frac is not None:
        _neg_frac = min(_neg_frac, neg_frac)
    neg_frac = _neg_frac

    # Randomly select negative data
    combined_data = [data[0]]
    combined_labels = [labels[0]]
    neg_size = int(total_size * neg_frac)
    for neg_data, neg_labels in zip(data[1:], labels[1:]):
        _neg_size = min(neg_size, len(neg_data))
        idxs = random.sample(range(len(neg_data)), _neg_size)
        _neg_data = [neg_data[i] for i in idxs]
        _neg_labels = [neg_labels[i] for i in idxs]
        combined_data.append(_neg_data)
        combined_labels.append(_neg_labels)

    return combined_data, combined_labels


def flatten(data):
    return numpy.array(list(itertools.chain.from_iterable(data)))
