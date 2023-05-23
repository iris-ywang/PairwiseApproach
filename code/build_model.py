import numpy as np
from time import time

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score, accuracy_score
from scipy.stats import spearmanr, kendalltau
from extrapolation_evaluation import EvaluateAbilityToIdentifyTopTestSamples
from pa_basics.all_pairs import pair_by_pair_id_per_feature, paired_data_by_pair_id
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


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    ndcg = ndcg_score([y_true], [y_predict])
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    tau = kendalltau(y_true, y_predict)[0]
    r2 = r2_score(y_true, y_predict)
    return [mse, mae, r2, rho, ndcg, tau]


def performance_standard_approach(all_data, percentage_of_top_samples):
    sa_model, y_SA = build_ml_model(GradientBoostingRegressor(random_state=1), all_data['train_set'],
                                    all_data['test_set'])
    y_pred_all = np.array(all_data["y_true"])
    y_pred_all[all_data["test_ids"]] = y_SA

    metrics_direct_regression = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_pred_all, all_data).run_evaluation()
    metrics_direct_regression += [
        spearmanr(all_data['y_true'], y_pred_all, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_pred_all])
    ]

    Y_c2_true, Y_c2_pred = pairwise_differences_for_standard_approach(all_data, "c2", y_pred_all)
    Y_pa_c2_dist = np.absolute(Y_c2_pred)
    Y_pa_c2_sign = np.sign(Y_c2_pred)
    acc_c2 = accuracy_score(Y_c2_true, Y_c2_pred)

    Y_c3_true, Y_c3_pred = pairwise_differences_for_standard_approach(all_data, "c3", y_pred_all)
    Y_pa_c3_sign = np.sign(Y_c3_pred)
    acc_c3 = accuracy_score(Y_c3_true, Y_c3_pred)

    Y_c1_true, Y_c1_pred = pairwise_differences_for_standard_approach(all_data, "c1", y_pred_all)
    Y_pa_c1_sign = np.sign(Y_c1_pred)

    metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3 = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_dist
    )
    return [metrics_direct_regression, metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3], [acc_c2, acc_c3]


def pairwise_differences_for_standard_approach(all_data, type: str, y_pred_all):
    Y_true, Y_pred = [], []
    if type == "c2":
        combs = all_data["c2_test_pair_ids"]
    elif type == "c3":
        combs = all_data["c3_test_pair_ids"]
    elif type =="c1":
        combs = all_data["train_pair_ids"]
    for comb in combs:
        a, b = comb
        Y_true.append(np.sign(all_data['y_true'][a] - all_data['y_true'][b]))
        Y_pred.append(np.sign(y_pred_all[a] - y_pred_all[b]))
    return Y_true, Y_pred


def results_of_pairwise_combinations(
        all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_dist
):
    Y_c2_sign_and_abs_predictions = dict(zip(all_data["c2_test_pair_ids"], np.array([Y_pa_c2_dist, Y_pa_c2_sign]).T))

    y_ranking_c2 = rating_trueskill(Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"])
    metrics_sign_Yc2 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c2, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc2 += [
        spearmanr(all_data['y_true'], y_ranking_c2, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c2]),
        ]
    y_ranking_c1c2c3 = rating_trueskill(list(np.concatenate([Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign])),
                                        all_data["train_pair_ids"] + all_data["c2_test_pair_ids"] + all_data["c3_test_pair_ids"],
                                        all_data["y_true"])
    metrics_sign_Yc1c2c3 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c1c2c3, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc1c2c3 += [
        spearmanr(all_data['y_true'], y_ranking_c1c2c3, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c1c2c3])
        ]
    y_ranking_c2c3 = rating_trueskill(list(np.concatenate([Y_pa_c2_sign, Y_pa_c3_sign])),
                                      all_data["c2_test_pair_ids"] + all_data["c3_test_pair_ids"],
                                      all_data["y_true"])
    metrics_sign_Yc2c3 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c2c3, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc2c3 += [
        spearmanr(all_data['y_true'], y_ranking_c2c3, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c2c3])
        ]
    return metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3


def performance_pairwise_approach(all_data, percentage_of_top_samples, batch_size=1000000):
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size
    Y_pa_c1_sign = []

    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
                                                        pair_ids=all_data['train_pair_ids'],
                                                   sign_only=True)

        train_pairs_for_sign = np.array(train_pairs_batch)
        train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
        rfc = GradientBoostingClassifier(random_state=1)
        rfc = build_ml_model(rfc, train_pairs_for_sign)

        train_pairs_for_abs = np.absolute(train_pairs_batch)
        rfr = GradientBoostingRegressor(random_state=1)
        rfr = build_ml_model(rfr, train_pairs_for_abs)
        Y_pa_c1_sign += list(train_pairs_for_sign[:, 0])

    else:

        for run in range(runs_of_estimators + 1):
            if run < runs_of_estimators:
                train_ids_per_batch = all_data["train_pair_ids"][run * batch_size:(run + 1) * batch_size]

            else:
                train_ids_per_batch = all_data["train_pair_ids"][run * batch_size:]

            train_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
                                                            pair_ids=train_ids_per_batch,
                                                       sign_only=True)

            train_pairs_for_sign = np.array(train_pairs_batch)
            train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
            rfc = GradientBoostingClassifier(random_state=1, warm_start=True)
            rfc = build_ml_model(rfc, train_pairs_for_sign)

            train_pairs_for_abs = np.absolute(train_pairs_batch)
            rfr = GradientBoostingRegressor( random_state=1, warm_start=True)
            rfr = build_ml_model(rfr, train_pairs_for_abs)
            Y_pa_c1_sign += list(train_pairs_for_sign[:, 0])

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
        test_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
                                                       pair_ids=test_pair_id_batch,
                                                  sign_only=True)

        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_dist += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c2_true += list(test_pairs_batch[:, 0])
        if (test_batch + 1) * batch_size >= len(c2_test_pair_ids): break


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

        test_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
                                                       pair_ids=test_pair_id_batch,
                                                  sign_only=True)
        Y_pa_c3_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_true += list(test_pairs_batch[:, 0])

    acc_c2 = accuracy_score(np.sign(Y_pa_c2_true), Y_pa_c2_sign)
    acc_c3 = accuracy_score(np.sign(Y_pa_c3_true), Y_pa_c3_sign)

    metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3 = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_dist
    )
    return [metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3], [acc_c2, acc_c3]


def estimate_y_from_final_ranking_and_absolute_Y(test_ids, ranking, y_true, Y_c2_sign_and_abs_predictions):
    final_estimate_of_y_and_delta_y = {test_id: [] for test_id in test_ids}
    for pair_id, values in Y_c2_sign_and_abs_predictions.items():
        test_a, test_b = pair_id
        sign_ab = np.sign(ranking[test_a] - ranking[test_b])

        if test_a in test_ids and test_b not in test_ids:
            final_estimate_of_y_and_delta_y[test_a].append(y_true[test_b] + (values[0] * sign_ab))
        elif test_b in test_ids and test_a not in test_ids:
            final_estimate_of_y_and_delta_y[test_b].append(y_true[test_a] - (values[0] * sign_ab))

    mean_estimates = [np.mean(estimate_list)
                      for test_id, estimate_list in final_estimate_of_y_and_delta_y.items()]
    return mean_estimates


def run_model(data, current_dataset_count, percentage_of_top_samples):
    temporary_file_dataset_count = int(np.load("extrapolation_temporary_dataset_count_reg_trial12.npy"))

    if current_dataset_count == temporary_file_dataset_count:
        existing_iterations = np.load("extrapolation_kfold_cv_reg_trial12_temporary.npy")
        existing_count = len(existing_iterations)
        metrics = list(existing_iterations)
    else:
        metrics = []
        existing_count = 0

    count = 0
    for outer_fold, datum in data.items():
        count += 1
        if count <= existing_count: continue
        metrics_sa, acc_sa = performance_standard_approach(datum, percentage_of_top_samples)
        metrics_pa, acc_pa = performance_pairwise_approach(datum, percentage_of_top_samples)
        acc = acc_sa + acc_pa + [0] * (len(metrics_sa[0]) - 4)
        metrics.append(metrics_sa + metrics_pa + [acc])
        np.save("extrapolation_temporary_dataset_count_reg_trial12.npy", [current_dataset_count])
        np.save("extrapolation_kfold_cv_reg_trial12_temporary.npy", np.array(metrics))

    return np.array([metrics])
