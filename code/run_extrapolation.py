import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
from build_model_extrapolation import run_model


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    chembl_info = pd.read_csv("input//chembl_datasets_info.csv").sort_values(by=["N(sample)"])
    log_file_name = 'extrapolation_10fold_cv_chembl_rf_log.txt'

    # For re-running purpose:
    try:
        existing_results = np.load("extrapolation_10fold_cv_chembl_rf.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("temporary_dataset_count_rf.npy")
    except:
        np.save("temporary_dataset_count_rf.npy", [0])

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
            metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
        except ValueError:
            with open(log_file_name, 'a') as f:
                f.write("WARNING: Cannot build model for Dataset " + filename + "\n")
            print("WARNING: Cannot build model for Dataset " + filename)
            continue

        all_metrics.append(metrics)
        np.save("extrapolation_10fold_cv_chembl_rf.npy", np.array(all_metrics))

