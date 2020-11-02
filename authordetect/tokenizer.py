#! /usr/bin/python3

import re
import functools
from .tokenizers import NLTKTokenizer, WhitespaceTokenizer
from typing import Iterable


__all__ = ['Tokenizer']


class Tokenizer:
    """Tokenizer.

        use_stopwords (bool): If set, use stopwords, else otherwise.

    Kwargs: (common options across tokenizers.*Tokenizer)
        converters (str, iterable[callable]): Available options are 'lower',
            'upper', 'unidecode'
    """
    STOPWORDS = NLTKTokenizer._STOPWORDS

    def __init__(
        self,
        *,
        use_stopwords=False,
        min_token_length=1,
        lemmatizer='wordnet',
        **kwargs,
    ):

        # Create tokenizers for sentences, tokens, and lemmatize
        # Do not use stopwords
        self._sentencizer = NLTKTokenizer(
            use_stopwords=use_stopwords,
            lemmatizer=lemmatizer,
            **kwargs,
        )
        self._lemmatizer = (
            None
            if lemmatizer is None
            else self._sentencizer._lemmatizer
        )
        self._tokenizer = WhitespaceTokenizer(
            use_stopwords=use_stopwords,
            min_token_length=min_token_length,
            **kwargs,
        )

        # Regexes to transform input text of tokenization process
        # NOTE: Transformations need to maintain same alignment so that
        # spans are consistent with input corpus.
        self._tokenizer_transforms = [
            # Remove non-alphanumerics (apostrophe for contractions)
            functools.partial(re.sub, r"[^\w']", ' ', flags=re.ASCII),
            # Remove enclosing quotes
            functools.partial(re.sub, r"'(\w+)'", r' \1 ', flags=re.ASCII),
        ]

        # Regexes to select valid tokens
        self._tokenizer_filters = [
            # Check that tokens include at least one alpha character
            functools.partial(re.search, r'[A-Za-z]', flags=re.ASCII),
        ]

    def sentencize(self, text: str):
        for b, e, t in self._sentencizer.sentencize(text):
            yield (b, e + 1, t)

    def tokenize(self, text: str):
        # Apply transformations at the sentence level so that output
        # from tokenizer are individual tokens.
        for token_transform in self._tokenizer_transforms:
            text = token_transform(text)

        for b, e, t in self._tokenizer.tokenize(text):
            for token_filter in self._tokenizer_filters:
                if token_filter(t):
                    break
            else:
                continue
            yield (b, e + 1, t)

    def lemmatize(self, text: str, pos: Iterable[str] = 'vn') -> str:
        """Lemmatize selected parts-of-speech."""
        if self._lemmatizer is not None:
            # NOTE: Do not lemmatize words not matching selected POS.
            for p in pos:
                text = self._lemmatizer.lemmatize(text, p)
        return text
