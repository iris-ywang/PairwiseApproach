"""
Create all pairs
"""

import numpy as np
from itertools import permutations
import multiprocessing
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from scipy.spatial.distance import dice, yule, kulsinski, sokalmichener


def paired_data(data, with_similarity=False, with_fp=False, only_fp=False, multiple_tanimoto=False):
    """
    Generate all possible pairs from a QSAR dataset

    :param only_fp: bool - if true, the pairwise features only contains original samples' FP
    :param data: np.array of all samples (train_test) - [y, x1, x2, ..., xn]
    :param with_similarity: bool - if true, the pairwise features include pairwise similarity measures
    :param with_fp: bool - if true, the pairwise features include original samples' FP
    :return: a dict - keys = (ID_a, ID_b); values = [Y_ab, X1, X2, ...Xn]
    """
    pairing_tool = PairingDataset(data, with_similarity, with_fp, only_fp, multiple_tanimoto)

    with multiprocessing.Pool() as executor:
        results = executor.map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    return dict(results)


class PairingDataset:
    """
    This class is built to save/initialise some reference info that is needed repeatly for the pairing process,
    so that in line 37, we only need to pass iteratively different combinations of sample IDs as argument to
    generate all the pairs.
    """
    def __init__(self, data, with_similarity, with_fp, only_fp, multiple_tanimoto):
        self.data = data
        self.feature_variation = [with_similarity, with_fp, only_fp, multiple_tanimoto]

        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = list(permutations(range(self.n_samples), 2)) + [(a, a) for a in range(self.n_samples)]
        self.n_combinations = len(self.permutation_pairs)

    def parallelised_pairing_process(self, combination_id):
        sample_id_a, sample_id_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sample_id_a: sample_id_a + 1, :]
        sample_b = self.data[sample_id_b: sample_id_b + 1, :]

        pair_ab = pair_2samples(self.n_columns, sample_a, sample_b, self.feature_variation)
        return (sample_id_a, sample_id_b), pair_ab


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
    # a = np.array([1, 0, 0, 1])
    # b = np.array([0, 0, 1, 1])
    # new_sample = array([-1,  2, -2,  1])
    new_sample += get_pairwise_features(a, b)

    # new_sample += list(a - b + (a + b - 0.5).astype(int) * 2)

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


def paired_data_by_pair_id(data, pair_ids, with_similarity=False, with_fp=False, only_fp=False, multiple_tanimoto=False):
    """
    Generate all possible pairs from a QSAR dataset
    :param only_fp: bool - if true, the pairwise features only contains original samples' FP
    :param data: np.array of all samples (train_test) - [y, x1, x2, ..., xn]
    :param with_similarity: bool - if true, the pairwise features include pairwise similarity measures
    :param with_fp: bool - if true, the pairwise features include original samples' FP
    :return: a dict - keys = (ID_a, ID_b); values = [Y_ab, X1, X2, ...Xn]
    """
    pairing_tool = PairingDatasetByPairID(data, pair_ids, with_similarity, with_fp, only_fp, multiple_tanimoto)

    with multiprocessing.Pool(processes=None) as executor:
        results = executor.map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    return np.array([values for _, values in dict(results).items()])


class PairingDatasetByPairID:
    def __init__(self, data, pair_ids, with_similarity, with_fp, only_fp, multiple_tanimoto):
        self.data = data
        self.feature_variation = [with_similarity, with_fp, only_fp, multiple_tanimoto]

        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = pair_ids
        self.n_combinations = len(self.permutation_pairs)

    def parallelised_pairing_process(self, combination_id):
        sample_id_a, sample_id_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sample_id_a: sample_id_a + 1, :]
        sample_b = self.data[sample_id_b: sample_id_b + 1, :]

        pair_ab = pair_2samples(self.n_columns, sample_a, sample_b, self.feature_variation)
        return (sample_id_a, sample_id_b), pair_ab

def get_pairwise_features(x1, x2):
    # Convert input lists to NumPy arrays
    x1 = np.array(x1)
    x2 = np.array(x2)
    # Initialize the result array with zeros
    y = np.zeros_like(x1, dtype=int)
    # Conditions for different cases
    condition1 = np.logical_and(x1 == 1, x2 == 1)
    condition2 = np.logical_and(x1 == 0, x2 == 0)
    condition3 = np.logical_and(x1 == 1, x2 == 0)
    condition4 = np.logical_and(x1 == 0, x2 == 1)
    # Apply the conditions to set values in the result array
    y[condition1] = 2
    y[condition2] = 1
    y[condition3] = -1
    y[condition4] = -2
    return y.tolist()