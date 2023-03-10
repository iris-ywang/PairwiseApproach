import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
from build_model import run_model

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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    chembl_info_all = pd.read_csv("input//chembl_reg_info.csv")
    chembl_info = chembl_info_all[chembl_info_all["OpenML ID"] == "24850-27000"]
    chembl_info = chembl_info[chembl_info_all["N(features)"] > 1000]

    chembl_info = chembl_info.sort_values(by=["N(sample)"])
    chembl_info = chembl_info[chembl_info["N(sample)"] >= 30]

    try:
        existing_results = np.load("extrapolation_kfold_cv_reg_trial4.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("extrapolation_temporary_dataset_count_reg_trial4.npy")
    except:
        np.save("extrapolation_temporary_dataset_count_reg_trial4.npy", [0])

    count = 0
    for file in range(len(chembl_info)):
        count += 1
        if count <= existing_count:
            continue
        connection = "/input/qsar_data_meta/24850-27000/"
        filename = chembl_info.iloc[file]["File name"]
        print(datetime.now(), " -- ", "On Dataset No.", count, ", ", filename)

        train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)
        print(datetime.now(), " -- ", "Generating datasets...")
        data = generate_train_test_sets_ids(train_test, fold=10)

        print(datetime.now(), " -- ", "Running models...")
        metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
        all_metrics.append(metrics[0])
        print(datetime.now(), " -- ")
        print(np.nanmean(metrics[0], axis=0))
        np.save("extrapolation_kfold_cv_reg_trial4", np.array(all_metrics))
