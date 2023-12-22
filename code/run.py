import logging

import numpy as np
import pandas as pd
import warnings

from itertools import product
from sklearn.preprocessing import LabelEncoder


from pa_basics.import_chembl_data import filter_data
from split_data import generate_train_test_sets_ids
from build_model import run_model

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def transform_categorical_columns(train_test, col_not_value):
    print("Transforming categorical features...")
    label_encoder = LabelEncoder()
    for col_name in col_not_value:
        train_test[col_name] = label_encoder.fit_transform(train_test[col_name])
    return train_test


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    filename = "icsd_formation_energy.csv"
    whole_dataset = pd.read_csv(f"input//{filename}")
    random_states = list(range(0, 10))
    sizes = [50, 100, 200, 300, 400, 500]

    try:
        existing_results = np.load("extrapolation_icsd_formation_energy.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("extrapolation_icsd_formation_energy_temp_dataset_count.npy")
    except:
        np.save("extrapolation_icsd_formation_energy_temp_dataset_count.npy", [0])

    count = 0
    for rs, n_samples in product(random_states, sizes):
        count += 1
        if count <= existing_count:
            continue
        train_test = whole_dataset.sample(n=n_samples, random_state=rs)

        col_non_numerical = list(train_test.dtypes[train_test.dtypes == "category"].index) + \
                            list(train_test.dtypes[train_test.dtypes == "object"].index)
        if col_non_numerical:
            train_test = transform_categorical_columns(train_test, col_non_numerical)

        train_test = train_test.to_numpy().astype(np.float64)
        train_test = filter_data(train_test, shuffle_state=1)

        data = generate_train_test_sets_ids(train_test, fold=10)
        logging.info(f"Running dataset {filename}, size {n_samples}, sampling random state {rs}.")
        logging.info("Running models...")
        metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
        all_metrics.append(metrics[0])
        logging.info("Finished")
        print(np.nanmean(metrics[0], axis=0))
        np.save("extrapolation_icsd_formation_energy.npy", np.array(all_metrics))
