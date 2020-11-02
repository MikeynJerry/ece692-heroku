import nltk
import numpy as np
from authordetect import Tokenizer, EmbeddingModel
from writer2vec import writer2vec, flatten
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
from sklearn.neural_network._base import ACTIVATIONS


# load model
data_splits = [350, 1400, 3500]
vector_widths = [50, 300]
models = {
      width: {
        split: pickle.load(open(f'./models/mlp_{width}dim_{split}part.pkl', 'rb'))
        for split in data_splits
      }
    for width in vector_widths
}

def load_embedding(fn):
    embedding = EmbeddingModel()
    embedding.load(fn)
    return embedding

embeddings = {
    width: {
        split: load_embedding(f'./embeddings/doyle_{width}dim_{split}part.bin')
        for split in data_splits
    }
    for width in vector_widths
}

tokenizer = Tokenizer(min_token_length=1, use_stopwords=False)
stopwords = Tokenizer.STOPWORDS

w2v_params = {
    'tokenizer': tokenizer,
    'part_size': None,
    'window': 5,
    'min_count': 1,
    'workers': 4,
    'sg': 0,
    'hs': 0,
    'negative': 20,
    'alpha': 0.03,
    'min_alpha': 0.0007,
    'seed': 0,
    'sample': 6e-5,
    'iter': 10,
    'stopwords': stopwords,
    'use_norm': True,
}

test_data = 'Doyle_10.txt'


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'


@app.route('/wakeup', methods=['GET'])
@cross_origin()
def wake_up():
  return {
    'awake': True,
    'weights': [ coefs.tolist() for coefs in models[50][350].coefs_ ],
    'layout': [50, 100, 1]
  }


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
  test_data = request.get_json()
  data = {
    'activations': {}
  }
  for width in vector_widths:
      data[str(width)] = {}
      for split in data_splits:
          data[str(width)][str(split)] = {}
          vectors, _ = writer2vec(test_data, None, embedding=embeddings[width][split], size=width, **w2v_params)
          vectors = flatten(vectors)
          data[str(width)][str(split)]['prediction'] = models[width][split].predict(vectors).flatten().tolist()
          data[str(width)][str(split)]['probabilities'] = models[width][split].predict_proba(vectors).flatten().tolist()

  embedding = embeddings[50][350]
  model = models[50][350]
  vectors, _ = writer2vec(test_data, None, embedding=embedding, size=width, **w2v_params)
  vectors = flatten(vectors)
  data['activations']['input'] = vectors[0].astype(float).tolist()
  hidden = np.matmul(vectors[0], model.coefs_[0]) + model.intercepts_[0]
  data['activations']['hidden'] = ACTIVATIONS['relu'](hidden).astype(float).tolist()
  out = np.matmul(hidden, model.coefs_[1]) + model.intercepts_[1]
  data['activations']['output'] = ACTIVATIONS['logistic'](out).astype(float).tolist()

  return jsonify(data)


if __name__ == '__main__':
    app.run(threaded=True, port=5000, debug=True)