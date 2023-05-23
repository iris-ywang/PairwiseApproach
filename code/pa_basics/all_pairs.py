"""
Create all pairs
"""

import numpy as np
import multiprocessing
from itertools import permutations
from time import time

from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from scipy.spatial.distance import dice, yule, kulsinski, sokalmichener


def pair_2samples(n_columns, sample_a, sample_b, feature_variation):
    """
    Transform the information from two single samples to a pair
    Note the Rules of pairwise features:
          x_A = 1 & x_B = 1 -> X_AB = 2
          x_A = 1 & x_B = 0 -> X_AB = 1
          x_A = 0 & x_B = 1 -> X_AB = -1
          x_A = 1 & x_B = 0 -> X_AB = 0
    :param n_columns: int
    :param sample_a: np.array - Sample A in the shape of (1, n_columns), [y, x1, x2, ...]
    :param sample_b: np.array - Sample B in the shape of (1, n_columns), [y, x1, x2, ...]
    :param feature_variation: list of bool - if any of them is true, the pairwise features vary according to the request
    :return:
    """
    with_similarity, with_fp, only_fp, multiple_tanimoto = feature_variation
    delta_y = sample_a[0, 0] - sample_b[0, 0]
    new_sample = [delta_y]
    if only_fp:
        new_sample += list(sample_a[0, 1:]) + list(sample_b[0, 1:])
        return new_sample

    a = sample_a[0, 1:]
    b = sample_b[0, 1:]
    new_sample += list(a - b + (a + b - 0.5).astype(int) * 2)

    if with_similarity:
        new_sample += similarity_metrics(np.array([sample_a[0, 1:]]), np.array([sample_b[0, 1:]]))

    if with_fp:
        new_sample += list(sample_a[0, 1:]) + list(sample_b[0, 1:])

    if multiple_tanimoto:
        tanimoto = jaccard_score(sample_a[0, 1:], sample_b[0, 1:])
        new_sample = [i * tanimoto for i in new_sample]

    return new_sample


def similarity_metrics(fp1, fp2):
    # TOCHECK: manhattan_distances(fp1, fp2)[0, 0] ?
    return [
        jaccard_score(fp1[0], fp2[0]),
        cosine_similarity(fp1, fp2)[0, 0],
        manhattan_distances(fp1, fp2),
        euclidean_distances(fp1, fp2)[0, 0],
        dice(fp1, fp2),
        kulsinski(fp1, fp2),
        yule(fp1, fp2),
        sokalmichener(fp1, fp2)
    ]


def paired_data_by_pair_id(data, pair_ids, sign_only=False):
    """
    Generate all possible pairs from a QSAR dataset
    :param only_fp: bool - if true, the pairwise features only contains original samples' FP
    :param data: np.array of all samples (train_test) - [y, x1, x2, ..., xn]
    :param with_similarity: bool - if true, the pairwise features include pairwise similarity measures
    :param with_fp: bool - if true, the pairwise features include original samples' FP
    :return: a dict - keys = (ID_a, ID_b); values = [Y_ab, X1, X2, ...Xn]
    """

    pairing_tool = PairingDatasetByPairID(data,
                                          pair_ids,
                                          sign_only)

    with multiprocessing.Pool(processes=None) as executor:
        results = executor.map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    return np.array([values for _, values in dict(results).items()])


class PairingDatasetByPairID:
    def __init__(self, data, pair_ids, sign_only=True):
        self.data = data
        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = pair_ids
        self.n_combinations = len(self.permutation_pairs)
        self.sign_only = sign_only

    def parallelised_pairing_process(self, combination_id):
        sample_id_a, sample_id_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sample_id_a, :]
        sample_b = self.data[sample_id_b, :]

        pair_ab = sample_a - sample_b

        if self.sign_only:
            return (sample_id_a, sample_id_b), list(np.sign(pair_ab))
        return (sample_id_a, sample_id_b), list(pair_ab) + list(sample_a[1:]) + list(sample_b[1:])


class PairingDataByFeature():
    def __init__(self, data, pair_ids, mapping):
        self.data = data
        self.mapping = mapping

        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = pair_ids
        self.n_combinations = len(self.permutation_pairs)

    def parallelised_pairing_process(self, combination_id):

        sample_id_a, sample_id_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sample_id_a: sample_id_a + 1, :]
        sample_b = self.data[sample_id_b: sample_id_b + 1, :]
        pair_ab = pair_2samples_discretise(sample_a, sample_b, self.mapping)

        return (sample_id_a, sample_id_b), pair_ab


def pair_2samples_discretise(sample_a, sample_b, mapping):
    a = sample_a[0, :]
    b = sample_b[0, :]
    new_sample = [a[0] - b[0]]

    for feature_id in range(1, len(a)):
        feature_combi = (a[feature_id], b[feature_id])
        new_sample.append(mapping[feature_combi])
    return new_sample


def pair_by_pair_id_per_feature(data, pair_ids):
    t1 = time()
    n_bins_max = 10
    data = np.array(data)
    n_samples, n_columns = data.shape
    for feature in range(1, n_columns):
        n_unique = len(np.unique(data[:, feature]))
        if n_unique <= n_bins_max:
            data[:, feature: feature+1] = np.array([LabelEncoder().fit_transform(data[:, feature])]).T
        else:
            data[:, feature: feature+1] = transform_into_ordinal_features(data[:, feature: feature+1],
                                                  n_bins=n_bins_max)
    mapping = make_mapping(n_bins_max)

    pairing_tool = PairingDataByFeature(data, pair_ids, mapping)
    with multiprocessing.Pool(processes=None) as executor:
        results = executor.map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    print("Time used to pair = ", time() - t1)
    return np.array([values for _, values in dict(results).items()])



# utils
def make_mapping(n_bins: int) -> dict:
    mapping = dict(enumerate(
        list(permutations(range(n_bins), 2)) + [(a, a) for a in range(n_bins)]
    ))
    mapping = {v: k for k, v in mapping.items()}
    return mapping


def transform_into_ordinal_features(x: np.array, n_bins=2) -> np.array:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    est.fit(x)
    x_transformed = est.transform(x)
    return x_transformed