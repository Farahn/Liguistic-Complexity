from __future__ import print_function
import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def zero_pad_test(X, seq_len_div):
    diff = seq_len_div - (len(X)%seq_len_div)
    return np.concatenate((np.array([x for x in X]),np.zeros((diff,len(X[0])))), axis = 0)

def batch_generator(X, y, batch_size, seq_len = 1):
    """Primitive batch generator 
    """
    size = X.shape[0]//seq_len
    X_copy = X.copy()
    y_copy = y.copy()

    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i * seq_len:(i + batch_size)* seq_len], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            continue

def test_batch_generator(X, batch_size, seq_len = 1):
    """Primitive batch generator 
    """
    size = X.shape[0]//seq_len
    X_copy = X.copy()

    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i * seq_len:(i + batch_size)* seq_len]
            i += batch_size
        else:
            i = 0
            continue

