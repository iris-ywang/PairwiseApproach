import logging
import os
import numpy as np
import pandas as pd
import warnings
import functools
import multiprocessing
import openml
from sklearn.preprocessing import LabelEncoder


from pa_basics.import_chembl_data import filter_data
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


def transform_categorical_columns(train_test, col_not_value):
    print("Transforming categorical features...")
    label_encoder = LabelEncoder()
    for col_name in col_not_value:
        train_test[col_name] = label_encoder.fit_transform(train_test[col_name])
    return train_test


def run_datasets_in_parallel(file, chembl_info):
    data_id = int(chembl_info.iloc[file]["OpenML ID"])
    chembl_id = int(chembl_info.iloc[file]["ChEMBL ID"])
    data = openml.datasets.get_dataset(data_id)
    X, y, categorical_indicator, attribute_names = data.get_data(target=data.default_target_attribute)


    # Exclude datasets with following traits
    if y.nunique() == 1 or get_repetition_rate(np.array([y]).T) >= 0.85:
        logging.info(
            f"Dataset ID. {data_id}, ChEMBL ID {chembl_id}, only has one value of y, "
            f"or has too many repeated y ( > 85% of y are the same). Abort.")

        invalid_datasets = np.load(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "invalid_datasets.npy")
        invalid_datasets.append(data_id)
        np.save(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "invalid_datasets.npy", invalid_datasets)
        return

    logging.info(f"Running on ChEMBL ID {chembl_id}, OpenML ID {data_id}")
    train_test = pd.concat([y, X], axis=1)
    col_non_numerical = list(train_test.dtypes[train_test.dtypes == "category"].index) + \
                        list(train_test.dtypes[train_test.dtypes == "object"].index)
    if col_non_numerical:
        train_test = transform_categorical_columns(train_test, col_non_numerical)

    train_test = train_test.to_numpy().astype(np.float64)
    train_test = filter_data(train_test, shuffle_state=1)

    data = generate_train_test_sets_ids(train_test, fold=10)

    logging.info("Running models...")
    metrics = run_model(data, current_data_id=data_id, percentage_of_top_samples=0.1)
    logging.info("Finished")

    np.save(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "extrapolation_lr_reg_chembl_cv_"+str(data_id)+".npy", metrics)
    return


def count_finished_datasets(sorted_chembl_info):
    existing_count = 0
    invalid_datasets = np.load(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "invalid_datasets.npy")

    for file in range(len(sorted_chembl_info)):
        data_id = int(sorted_chembl_info.iloc[file]["OpenML ID"])
        if data_id in invalid_datasets:
            existing_count += 1
            continue
        try:
            _ = np.load(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "extrapolation_lr_reg_chembl_cv_"+str(data_id)+".npy",)
            existing_count += 1
        except FileNotFoundError:
            return existing_count



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    try:
        _ = np.load(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "invalid_datasets.npy")
    except:
        np.save(os.getcwd() + "/extrapolation_lr_reg_chembl/" + "invalid_datasets.npy", [])

    chembl_info_all = pd.read_csv("input//chembl_meta_ml_info.csv")
    chembl_info = chembl_info_all[(chembl_info_all["All boolean?"] == False) &
                                 (chembl_info_all["Half Boolean?"] == False) &
                                  (chembl_info_all["N(feature)"] <= 50) &
                                 (chembl_info_all["N(sample)"] >= 30)]

    chembl_info = chembl_info.sort_values(by=["N(sample)"])
    existing_count = count_finished_datasets(chembl_info)
    logging.info(f"{existing_count} datasets has been ")
    run_datasets_in_parallel_partial = functools.partial(run_datasets_in_parallel, chembl_info=chembl_info)

    with multiprocessing.Pool() as executor:
        executor.map(run_datasets_in_parallel_partial, range(existing_count, len(chembl_info)), chunksize=1)

