import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
from build_model_Yc2_accuracy import run_model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    chembl_info = pd.read_csv("input//chembl_datasets_info.csv").sort_values(by=["N(sample)"])
    log_file_name = 'dataset_running_order_rf_sign_acccuracy.txt'

    # For re-running purpose:
    try:
        existing_results = np.load("10fold_cv_chembl_rf_sign_acccuracy.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("temporary_dataset_count_rf_sign_acccuracy.npy")
    except:
        np.save("temporary_dataset_count_rf_sign_acccuracy.npy", [0])

    count = 0
    for file in range(len(chembl_info)):
        count += 1
        if count <= existing_count:
            continue
        # Start from the dataset following the last run.

        filename = chembl_info.iloc[file]["File name"]
        print(datetime.now(), " -- ", "On Dataset No.", count, ", ", filename)
        with open(log_file_name, 'a') as f:
            f.write(str(datetime.now()) + " -- " + filename + "\n")

        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)

        if len(np.unique(train_test[:, 0])) == 1:
            with open(log_file_name, 'a') as f:
                f.write("WARNING: Cannot build model with only one target value for Dataset " + filename + "\n")
            print("WARNING: Cannot build model with only one target value for Dataset " + filename)
            continue

        data = generate_train_test_sets_ids(train_test, fold=10)
        try:
            metrics = run_model(data, current_dataset_count=count)
        except ValueError:
            with open(log_file_name, 'a') as f:
                f.write("WARNING: Cannot build model for Dataset " + filename + "\n")
            print("WARNING: Cannot build model for Dataset " + filename)
            continue

        all_metrics.append(metrics)
        m = np.nanmean(metrics, axis=(0, 1))
        print("metrics:")
        print(m)
        np.save("10fold_cv_chembl_rf_sign_acccuracy.npy", np.array(all_metrics))

