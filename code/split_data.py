import numpy as np
from sklearn.model_selection import KFold
from itertools import permutations, product


def data_check(train_test):
    """
    Check if a dataset has too many repeated activities values. For 5 fold CV, a repeat rate of 15% is used.
    For time-saving experiments, small datasets are used. The range of size of datasets is specified here.
    :param train_test: np.ndarray of filtered dataset - [y, x1, x2, ..., xn]
    :return: bool (True means the dataset is OK for experiment, and vice versa)
    """
    sample_size = np.shape(train_test)[0]
    if sample_size > 100 or sample_size < 90: return False

    my_dict = {i: list(train_test[:, 0]).count(i) for i in list(train_test[:, 0])}
    max_repetition = max(my_dict.values())
    if max_repetition > 0.15 * sample_size: return False

    return True

def get_repetition_rate(train_test):
    sample_size = np.shape(train_test)[0]
    my_dict = {i: list(train_test[:, 0]).count(i) for i in list(train_test[:, 0])}
    max_repetition = max(my_dict.values())
    return max_repetition / sample_size

def pair_test_with_train(train_ids, test_ids):
    """
    Generate C2-type pairs (test samples pairing with train samples)
    :param train_ids: list of int for training sample IDs
    :param test_ids: list of int for test sample IDs
    :return: list of tuples of sample IDs
    """
    c2test_combs = []
    for comb in product(test_ids, train_ids):
        c2test_combs.append(comb)
        c2test_combs.append(comb[::-1])
    return c2test_combs


def train_test_split(pairs, train_ids, test_ids):
    """
    Generate different types of pairs depending on the given sample IDs for training samples and test samples
    Types of pairing:
        C1-type: train ID - train ID
        C2-type: train ID - test ID
        C3-type: test ID - test ID

    :param pairs: dict of all possible pairs for a filtered dataset
    :param train_ids: list of int for training sample IDs
    :param test_ids: list of int for test sample IDs
    :return: a dict of different types of pairs. It contains the info for samples IDs for pairs, and feature and target
             values for them.
    """
    train_pairs = dict(pairs)
    c2_test_pairs = []
    c3_test_pairs = []
    c2_keys_del = pair_test_with_train(train_ids, test_ids)
    c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]

    for key in c2_keys_del:
        c2_test_pairs.append(train_pairs.pop(key))
    for key in c3_keys_del:
        c3_test_pairs.append(train_pairs.pop(key))
    c2_test_pairs = np.array(c2_test_pairs)
    c3_test_pairs = np.array(c3_test_pairs)

    c1_keys_del, trainp = [], []
    for a, b in train_pairs.items():
        c1_keys_del.append(a)
        trainp.append(b)
    train_pairs = np.array(trainp)

    return {'train_pairs': train_pairs,
            'train_pair_ids': c1_keys_del,
            'c2_test_pairs': c2_test_pairs,
            'c2_test_pair_ids': c2_keys_del,
            'c3_test_pairs': c3_test_pairs,
            'c3_test_pair_ids': c3_keys_del}


def generate_train_test_sets_ids(train_test, fold):
    """
    Generate training sets and test sets for standard approach(regression on FP and activities) and for pairwise approach
     (regression on pairwise features and differences in activities) for each fold of cross validation
    :param train_test: np.array of filtered dataset - [y, x1, x2, ..., xn]
    :return: a dict, keys =  fold number, values = the corresponding pre-processed training and test data and
             sample information
    """
    y_true = np.array(train_test[:, 0])

    train_test_data = {}
    kf = KFold(n_splits=fold)
    n_fold = 0
    for train_ids, test_ids in kf.split(train_test):
        c1_keys_del = list(permutations(train_ids, 2)) + [(a, a) for a in train_ids]
        c2_keys_del = pair_test_with_train(train_ids, test_ids)
        c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]
        train_test_data_per_fold = {'train_test': train_test,
                                    'train_ids': train_ids, 'test_ids': test_ids,
                                    'train_set': train_test[train_ids], 'test_set': train_test[test_ids],
                                    'y_true': y_true, "train_pair_ids": c1_keys_del,
                                    "c2_test_pair_ids": c2_keys_del, "c3_test_pair_ids": c3_keys_del}

        # a dict of different types of pairs and their samples IDs
        # pairs_data = train_test_split(pairs, train_ids, test_ids)
        train_test_data[n_fold] = train_test_data_per_fold

        n_fold += 1

    return train_test_data
