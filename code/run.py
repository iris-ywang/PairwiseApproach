import os
import numpy as np
import pandas as pd
import time
import warnings

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
from build_model import run_model


def load_datasets():
    """
    Create a list of directories for all the qsar datasets to be evaluated
    :return:
    list of strings
    """
    filename_lst = []
    # TODO: may need to change the way of getting parent directory if this does not work on windows
    directory = os.getcwd() + '/input/qsar_data_unsorted'

    for root, dirs, files in os.walk(directory):
        for each_file in files:
            if each_file.endswith(".csv"):
                f = open(os.path.join(root, each_file), 'r')
                filename_lst.append(os.path.join(root, each_file))
                f.close()
    return filename_lst


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    filename = 'concrete_data.csv'

    connection = "/input/"
    train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)[:200, :]
    print("Generating datasets...")
    start = time.time()
    data = generate_train_test_sets_ids(train_test, fold=10)
    print(":::Time used: ", time.time() - start)

    print("Running models...")
    start = time.time()
    metrics = run_model(data, percentage_of_top_samples=0.1)
    print(":::Time used: ", time.time() - start, "\n")
    np.save("concrete_data_results.npy", metrics[0])


