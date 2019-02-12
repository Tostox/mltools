import numpy as np
from nltk import FreqDist
from ..plottingTool.mltools_plot import text_mining_plot
import concurrent.futures

def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[model.classes_[class_index]] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


def plot_important_words(importance, classes, figsize=(800, 600)):

    text_mining_plot._plot_important_words(importance, classes, figsize)


def plot_word_freq(df, target, col, plot_n=50, figsize=(800, 600)):
    frequencies_dist = {}

    for label in np.unique(df[target]):
        words_list = []

        for line in df[df[target] == label][col]:
            for word in line:
                words_list.append(word)

        frequencies_dist[label] = FreqDist(words_list)

    text_mining_plot._plot_word_freq(frequencies_dist, plot_n, figsize)
