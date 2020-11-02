from .textspan import TextSpan
from .author import Author, np_avg, np_sum
from .tokenizer import Tokenizer
from .embedding import EmbeddingModel
from .classifier import Classifier
from .textutils import (
    load_text, save_text,
    load_pickle, save_pickle,
    load_json, save_json,
    get_text_from_span,
)
from . import textutils
from . import trainutils
from . import tokenizers
from .smarttimers import SmartTimer
