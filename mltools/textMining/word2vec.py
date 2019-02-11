import gensim
import numpy as np


def load_model(path):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, data, field):
    embeddings = data[field].apply(lambda x: get_average_word2vec(x, vectors))
    return list(embeddings)
