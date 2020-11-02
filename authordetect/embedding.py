#! /usr/bin/python3

import os
# import multiprocessing
import psutil
from gensim.models import Word2Vec
from typing import Iterable


__all__ = ['EmbeddingModel']


class EmbeddingModel:
    def __init__(self, **kwargs):
        if 'sentences' in kwargs or 'corpus_file' in kwargs:
            raise Exception('training is not allowed during initialization, use train()')

        # Handle seeding consistently
        seed = kwargs.pop('seed', None)
        if seed is None:
            seed = 1  # Gensim's default value
            workers = kwargs.pop('workers', psutil.cpu_count(False))
            # workers=kwargs.pop('workers', multiprocessing.cpu_count() / 2)
        else:
            workers = kwargs.pop('workers', 1)

        self._model = Word2Vec(
            size=kwargs.pop('size', 50),
            window=kwargs.pop('window', 5),
            min_count=kwargs.pop('min_count', 1),
            workers=workers,
            sg=kwargs.pop('sg', 0),
            hs=kwargs.pop('hs', 0),
            negative=kwargs.pop('negative', 20),
            alpha=kwargs.pop('alpha', 0.03),
            min_alpha=kwargs.pop('min_alpha', 0.0007),
            seed=seed,
            max_vocab_size=kwargs.pop('max_vocab_size', None),
            max_final_vocab=kwargs.pop('max_final_vocab', None),
            sample=kwargs.pop('sample', 6e-5),
            iter=kwargs.pop('iter', 10),
            sorted_vocab=kwargs.pop('sorted_vocab', 1),
            **kwargs,
        )

    def __getitem__(self, key):
        """Returns the vector associated with a given word."""
        return self._model.wv.word_vec(key)

    @property
    def model(self):
        return self._model

    @property
    def vocabulary(self):
        return self._model.wv.vocab

    @property
    def vectors(self):
        return self._model.wv.vectors

    @property
    def vectors_norm(self):
        return self._model.wv.vectors_norm

    def train(self, sentences: Iterable[Iterable[str]] = None, **kwargs):
        self._model.build_vocab(sentences, progress_per=10000)
        self._model.train(
            sentences,
            total_examples=kwargs.pop('total_examples', self._model.corpus_count),
            epochs=kwargs.pop('epochs', self._model.epochs),
            report_delay=kwargs.pop('report_delay', 1),
            **kwargs,
        )
        self._model.wv.init_sims()

    def save(self, outfile):
        outdir = os.path.dirname(outfile)
        if outdir and not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        if outfile:
            self._model.save(outfile)

    def load(self, infile: str):
        self._model = Word2Vec.load(infile)
        self._model.wv.init_sims()
