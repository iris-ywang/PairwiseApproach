import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from pa_basics.import_chembl_data import filter_data, dataset
from split_data import generate_train_test_sets_ids, get_repetition_rate
from build_model import run_model

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def get_chembl_info():
    chembl_info = pd.DataFrame(columns=[
        "File name",
        "Chembl ID",
        "N(sample)",
        "N(features)",
        "OpenML ID"])
    connection = "/code/input/qsar_data_meta/24850-27000/"
    file_names = os.listdir(os.getcwd() + connection)
    for filename in file_names:
        if ".csv" not in filename:
            continue
        dataset = pd.read_csv(os.getcwd() + connection + filename, index_col=None)
        n_samples, n_features = dataset.shape
        n_features -= 1
        # if (dataset.iloc[:, 1:].nunique() <= 10).any():
        #     print(filename)
        #     print(dataset.iloc[:, 1:].nunique())
        chembl_id = filename.split("-Nf-")[0].split("CHEMBL")[1]
        if chembl_id.endswith(str(n_features)):
            chembl_id = chembl_id[:-len(str(n_features))]

        chembl_info = chembl_info.append({
            "File name": filename,
            "Chembl ID": chembl_id,
            "N(sample)": n_samples,
            "N(features)": n_features,
            "OpenML ID": "24850-27000"},
            ignore_index=True
        )
    return chembl_info


def check_dataset(train_test):
    y = train_test[:, 0]
    if pd.Series(y).nunique() == 1:
        print("Can't pass dataset check with single y value.")
        return False
    elif get_repetition_rate(train_test) >= 0.85:
        print("Can't pass dataset check with repeated y value.")
        return False
    return True


def transform_categorical_columns(train_test, col_not_value):
    print("Transforming categorical features...")
    label_encoder = LabelEncoder()
    for col_name in col_not_value:
        train_test[col_name] = label_encoder.fit_transform(train_test[col_name])
    return train_test


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    input_data_dir = os.getcwd() + "//input//mentch_regression_clean//"  ##########
    list_of_dataset = os.listdir(input_data_dir)

    try:
        existing_results = np.load("extrapolation_mentch1.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    count = 0
    for file in list_of_dataset:
        count += 1
        if count <= existing_count:
            continue

        logging.info(f"Now running dataset: {file}")
        train_test = dataset(input_data_dir + file, shuffle_state=1, y_col_in_the_last=True)  #############
        if not check_dataset(train_test):
            print(f"Dataset {file} does not pass on the check. Ignore this dataset.")
            continue

        subset = 0
        while True:
            subset += 100
            train_test_shuffled = shuffle(np.array(train_test), random_state=subset)
            train_test_subset = train_test_shuffled[:subset]

            if not check_dataset(train_test_subset):
                for try_resubset in range(1, 6):
                    train_test_shuffled = shuffle(np.array(train_test), random_state=subset + try_resubset)
                    train_test_subset = train_test_shuffled[:subset]
                    if check_dataset(train_test_subset):
                        break
                if try_resubset == 5:
                    print(
                        f"Dataset {file} of subset size {subset} does not pass on the check with 5 random subsets. Ignore this dataset.")
                    break
            logging.info(f"Sub dataset size: {subset}")
            try:
                logging.info(f"Re-subdivide dataset run (1-5): {try_resubset}")
            except:
                pass

            data = generate_train_test_sets_ids(train_test_subset, fold=10)   ########

            logging.info("Running models...")
            metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
            all_metrics.append(metrics[0])

            logging.info("Model Runs Finished. ")
            print(np.nanmean(metrics[0], axis=0))

            np.save("extrapolation_mentch1.npy", np.array(all_metrics))

            if subset >= len(train_test) or subset >= 300:
                logging.info(f"Finished Dataset {file}. ")
                print("\n \n \n")
                break
