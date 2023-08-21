"""
import ChEMBL dataset and convert it to filtered np.array
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale


def dataset(filename, shuffle_state=None, y_col_in_the_last=False):
    orig_data = import_data(filename, shuffle_state)

    if y_col_in_the_last:
        orig_data[:, [0, -1]] = orig_data[:, [-1, 0]]

    filter1 = uniform_features(orig_data)
    filter2 = duplicated_features(filter1)
    filter2[:, 1:] = minmax_scale(filter2[:, 1:])
    return filter2


def filter_data(train_test, shuffle_state=None):
    if shuffle_state is not None:
        train_test = shuffle(train_test, random_state=shuffle_state)

    filter1 = uniform_features(train_test)
    filter2 = duplicated_features(filter1)
    filter2[:, 1:] = minmax_scale(filter2[:, 1:])
    return filter2

def import_data(filename, shuffle_state):
    """
    to import raw data from csv to numpy array
    :param filename: str - string of (directory of the dataset + filename)
    :param shuffle_state: int - random seed
    :return: np.array - [y, x1, x2, ..., xn]
    """
    df = pd.read_csv(filename)
    try:
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)
    except:
        del df['molecule_id']
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)

    if shuffle_state is not None:
        data = shuffle(data, random_state=shuffle_state)
    return data


def uniform_features(data):
    """
    Remove the all-zeros and all-ones features from the dataset
    :param data: np.array - [y, x1, x2, ..., xn]
    :return: np.array - [y, x1, x2, ..., xn']
    """
    n_samples, n_columns = np.shape(data)
    redundant_features = []
    for feature in range(1, n_columns):
        if np.all(data[:, feature] == data[0, feature]):
            redundant_features.append(feature)
    filtered_data = np.delete(data, redundant_features, axis=1)
    return filtered_data


def duplicated_features(data):
    """
    Remove the duplicated features from the dataset
    **Note: this function can change the sequences of features

    :param data: np.array - [y, x1, x2, ..., xn]
    :return: np.array - [y, x1, x2, ..., xn']
    """
    y = data[:, 0:1]
    a = data[:, 1:].T
    order = np.lexsort(a.T)
    a = a[order]

    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    new_train = a[ui].T
    new_train_test = np.concatenate((y, new_train), axis=1)
    return new_train_test
