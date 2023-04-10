import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from extrapolation_evaluation import EvaluateAbilityToIdentifyTopTestSamples
from pa_basics.all_pairs import paired_data_by_pair_id
from pa_basics.rating import rating_trueskill


def build_ml_model(model, train_data, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model


def performance_standard_approach(all_data, percentage_of_top_samples):
    '''Run the standard regression, evaluated by extrapolation metrics.'''
    sa_model, y_SA = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_set'], all_data['test_set'])
    y_pred_all = np.array(all_data["y_true"])
    y_pred_all[all_data["test_ids"]] = y_SA

    metrics = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_pred_all, all_data).run_evaluation()
    return metrics, sa_model


def performance_pairwise_approach(all_data, percentage_of_top_samples, batch_size=200000):
    '''Run the pairwise approach, evaluated by extrapolation metrics.'''
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size

    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], all_data['train_pair_ids'])

        train_pairs_for_sign = np.array(train_pairs_batch)
        train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
        rfc = RandomForestClassifier(n_jobs=-1, random_state=1)
        rfc = build_ml_model(rfc, train_pairs_for_sign)

        train_pairs_for_abs = np.absolute(train_pairs_batch)
        rfr = RandomForestRegressor(n_jobs=-1, random_state=1)
        rfr = build_ml_model(rfr, train_pairs_for_abs)

    else:

        for run in range(runs_of_estimators + 1):
            if run < runs_of_estimators:
                train_ids_per_batch = all_data["train_pair_ids"][run*batch_size:(run + 1) * batch_size]

            else:
                train_ids_per_batch = all_data["train_pair_ids"][run*batch_size:]

            train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], train_ids_per_batch)

            train_pairs_for_sign = np.array(train_pairs_batch)
            train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
            rfc = RandomForestRegressor(n_jobs=-1, random_state=1, warm_start=True)
            rfc = build_ml_model(rfc, train_pairs_for_sign)

            train_pairs_for_abs = np.absolute(train_pairs_batch)
            rfr = RandomForestRegressor(n_jobs=-1, random_state=1, warm_start=True)
            rfr = build_ml_model(rfr, train_pairs_for_abs)

            rfc.n_estimators += 100
            rfr.n_estimators += 100

    c2_test_pair_ids = all_data["c2_test_pair_ids"]
    number_test_batches = len(c2_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c2_sign, Y_pa_c2_dist = [], []
    Y_pa_c2_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c2_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c2_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_dist += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c2_true += list(test_pairs_batch[:,0])

    c3_test_pair_ids = all_data["c3_test_pair_ids"]
    number_test_batches = len(c3_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c3_sign = []
    Y_pa_c3_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches:
            test_pair_id_batch = c3_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c3_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c3_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_true += list(test_pairs_batch[:, 0])

    Y_c2_sign_and_abs_predictions = dict(zip(all_data["c2_test_pair_ids"], np.array([Y_pa_c2_dist, Y_pa_c2_sign]).T))
    y_ranking = rating_trueskill(list(train_pairs_for_sign[:, 0]) +Y_pa_c2_sign + Y_pa_c3_sign,
                                 all_data["train_pair_ids"] + all_data["c2_test_pair_ids"] + all_data["c3_test_pair_ids"],
                                 all_data["y_true"])

    metrics = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    return metrics, rfc, rfr


def run_model(data, current_dataset_count, percentage_of_top_samples):
    temporary_file_dataset_count = int(np.load("temporary_dataset_count_rf.npy"))

    # Continue from the last run of 10-fold CV.
    if current_dataset_count == temporary_file_dataset_count:
        existing_iterations = np.load("extrapolation_10fold_cv_chembl_rf_temporary.npy")
        existing_count = len(existing_iterations)
        metrics = list(existing_iterations)
    else:
        metrics = []
        existing_count = 0

    count = 0
    for outer_fold, datum in data.items():
        count += 1
        if count <= existing_count: continue
        metric_sa, rfr_sa = performance_standard_approach(datum, percentage_of_top_samples)
        metric_pa, rfc_pa, rfr_pa = performance_pairwise_approach(datum, percentage_of_top_samples)
        metrics.append([metric_sa, metric_pa])

        np.save("temporary_dataset_count_rf.npy", [current_dataset_count])
        np.save("extrapolation_10fold_cv_chembl_rf_temporary.npy", np.array(metrics))

    return np.array([metrics])
