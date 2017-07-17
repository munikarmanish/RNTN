"""
Utility functions
"""

import pickle

import numpy as np


def save_to_file(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / sum(e)
