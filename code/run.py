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

    chembl_info = pd.read_csv("input//chembl_datasets_info.csv").sort_values(by=["N(sample)"])
    all_metrics = []

    try:
        existing_results = np.load("extrapolation_kfold_cv_all_data_rf_ranking.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("extrapolation_temporary_dataset_count_rf_ranking.npy")
    except:
        np.save("extrapolation_temporary_dataset_count_rf_ranking.npy", [0])

    count = 0
    for file in range(len(chembl_info)):
        # if len(all_metrics) > 100: break
        # if chembl_info["Repetition Rate"][file] > 0.15: continue
        # if chembl_info["N(sample)"][file] > 300 or chembl_info["N(sample)"][file] < 90: continue
        # If dataset passes the above criteria, then it gives a dict of fold number and their corresponding
        # pre-processed data

        count += 1
        if count <= existing_count:
            continue
        # TODO: may need to change the way of getting parent directory if this does not work on windows
        filename = chembl_info.iloc[file]["File name"]
        print("On Dataset No.", count, ", ", filename)

        with open('dataset_running_order_rf_ranking.txt', 'a') as f:
            f.write(filename + "\n")

        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)

        if len(np.unique(train_test[:, 0])) == 1:
            with open('dataset_running_order_rf_ranking.txt', 'a') as f:
                f.write("WARNING: Cannot build model with only one target value for Dataset " + filename + "\n")
            print("WARNING: Cannot build model with only one target value for Dataset " + filename)
            continue

        data = generate_train_test_sets_ids(train_test, fold=10)

        print("Running models...")
        start = time.time()
        metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
        print(":::Time used: ", time.time() - start, "\n")

        all_metrics.append(metrics)
        np.save("extrapolation_kfold_cv_all_data_rf_ranking.npy", np.array(all_metrics))

