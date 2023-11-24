import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score, accuracy_score
from scipy.stats import spearmanr, kendalltau
from extrapolation_evaluation import EvaluateAbilityToIdentifyTopTestSamples
from pa_basics.all_pairs import paired_data_by_pair_id
from pa_basics.rating import rating_trueskill, rating_sbbr


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
    sa_model, y_SA = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_set'], all_data['test_set'])
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
    acc_c3 = accuracy_score(np.sign(Y_c2_true), Y_pa_c2_sign)

    Y_c3_true, Y_c3_pred = pairwise_differences_for_standard_approach(all_data, "c3", y_pred_all)
    Y_pa_c3_sign = np.sign(Y_c3_pred)
    acc_c2 = accuracy_score(np.sign(Y_c3_true), Y_pa_c3_sign)

    Y_c1_true, Y_c1_pred = pairwise_differences_for_standard_approach(all_data, "c1", y_pred_all)
    Y_pa_c1_sign = np.sign(Y_c1_pred)

    metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3, metrics_est_reranked = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_dist
    )
    metrics_est = metrics_evaluation(all_data['test_set'][:, 0], y_SA)
    return [metrics_direct_regression, metrics_sign_Yc2, metrics_sign_Yc2c3], [acc_c2, acc_c3], [metrics_est, metrics_est_reranked]


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
        Y_true.append(all_data['y_true'][a] - all_data['y_true'][b])
        Y_pred.append(y_pred_all[a] - y_pred_all[b])
    return Y_true, Y_pred


def results_of_pairwise_combinations(
     all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_dist=None, rank_method=rating_trueskill
):
    if Y_pa_c2_dist is not None:
        Y_c2_sign_and_abs_predictions = dict(
            zip(all_data["c2_test_pair_ids"], np.array([Y_pa_c2_dist, Y_pa_c2_sign]).T)
        )
    else:
        # Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign are not signs then
        Y_c2_sign_and_abs_predictions = dict(zip(
            all_data["c2_test_pair_ids"],
            np.array([np.absolute(Y_pa_c2_sign), np.sign(Y_pa_c2_sign)]).T
        ))
        Y_pa_c2 = Y_pa_c2_sign

    y_ranking_c2 = rank_method(Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"])
    metrics_sign_Yc2 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c2, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc2 += [
        spearmanr(all_data['y_true'], y_ranking_c2, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c2]),
        ]
    y_ranking_c1c2c3 = rank_method(list(np.concatenate([Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign])),
                                        all_data["train_pair_ids"] + all_data["c2_test_pair_ids"] + all_data["c3_test_pair_ids"],
                                        all_data["y_true"])
    metrics_sign_Yc1c2c3 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c1c2c3, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc1c2c3 += [
        spearmanr(all_data['y_true'], y_ranking_c1c2c3, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c1c2c3])
        ]
    y_ranking_c2c3 = rank_method(list(np.concatenate([Y_pa_c2_sign, Y_pa_c3_sign])),
                                      all_data["c2_test_pair_ids"] + all_data["c3_test_pair_ids"],
                                      all_data["y_true"])
    metrics_sign_Yc2c3 = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking_c2c3, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    metrics_sign_Yc2c3 += [
        spearmanr(all_data['y_true'], y_ranking_c2c3, nan_policy="omit")[0],
        ndcg_score([all_data['y_true']], [y_ranking_c2c3])
        ]

    if Y_pa_c2_dist is not None:
        Y_pa_c2_sign_pred = np.sign(pairwise_differences_for_standard_approach(all_data, "c2", y_ranking_c2)[1])
        Y_pa_c2 = Y_pa_c2_sign_pred * Y_pa_c2_dist
    y_est = estimate_y_from_averaging(Y_pa_c2, all_data["c2_test_pair_ids"], all_data["test_ids"], all_data["y_true"])

    metrics_est = metrics_evaluation(all_data["test_set"][:, 0], y_est)

    return [metrics_sign_Yc2, metrics_sign_Yc2c3, metrics_sign_Yc1c2c3, metrics_est]


def performance_pairwise_approach(all_data, percentage_of_top_samples, batch_size=200000):
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size
    Y_pa_c1_sign, Y_pa_c1 = [], []

    if runs_of_estimators >= 1:
        raise ValueError("Training size too large. Reduce training set size or increase batch size allowrance")

    train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], all_data['train_pair_ids'])

    # training on signs - rfc
    train_pairs_for_sign = np.array(train_pairs_batch)
    train_pairs_for_sign[:, 0] = 2 * (train_pairs_for_sign[:, 0] >= 0) - 1
    rfc = RandomForestClassifier(n_jobs=-1, random_state=1)
    rfc = build_ml_model(rfc, train_pairs_for_sign)
    Y_pa_c1_sign += list(train_pairs_for_sign[:, 0])

    # training on abs - rfr
    train_pairs_for_abs = np.absolute(train_pairs_batch)
    rfr = RandomForestRegressor(n_jobs=-1, random_state=1)
    rfr = build_ml_model(rfr, train_pairs_for_abs)

    # training on Y - rfrb
    rfrb = RandomForestRegressor(n_jobs=-1, random_state=1)
    rfrb = build_ml_model(rfrb, train_pairs_batch)
    Y_pa_c1 += list(train_pairs_batch[:, 0])

    # testing
    c2_test_pair_ids = all_data["c2_test_pair_ids"]
    number_test_batches = len(c2_test_pair_ids) // batch_size

    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c2_sign, Y_pa_c2_abs, Y_pa_c2, Y_pa_c2_sign_prob = [], [], [], []
    Y_pa_c2_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c2_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c2_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)

        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_abs += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c2 += list(rfrb.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_true += list(test_pairs_batch[:, 0])

        proba = rfc.predict_proba(test_pairs_batch[:, 1:])
        classes = list(rfc.classes_)
        win_proba = proba[:, classes.index(1)]
        Y_pa_c2_sign_prob += list(win_proba)

    c3_test_pair_ids = all_data["c3_test_pair_ids"]
    number_test_batches = len(c3_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c3_sign, Y_pa_c3, Y_pa_c3_abs, Y_pa_c3_sign_prob = [], [], [], []
    Y_pa_c3_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches:
            test_pair_id_batch = c3_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c3_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        # Y_pa_c3_batch_prediction = rfc.predict(test_pairs_batch[:, 1:])

        Y_pa_c3_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_abs += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c3 += list(rfrb.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_true += list(test_pairs_batch[:, 0])

        proba = rfc.predict_proba(test_pairs_batch[:, 1:])
        classes = list(rfc.classes_)
        assert len(classes) == 2
        win_proba = proba[:, classes.index(1)]
        Y_pa_c3_sign_prob += list(win_proba)

    acc_c2 = accuracy_score(np.sign(Y_pa_c2_true), Y_pa_c2_sign)
    acc_c3 = accuracy_score(np.sign(Y_pa_c3_true), Y_pa_c3_sign)

    metrics_trueskill = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples, Y_pa_c1_sign, Y_pa_c2_sign, Y_pa_c3_sign, Y_pa_c2_abs
    )

    metrics_sd_trueskill_regY = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples,
        Y_pa_c1, Y_pa_c2, Y_pa_c3,
        rank_method=rating_sbbr
    )

    Y_pa_c2_proba_balanced = list(np.array(Y_pa_c2_sign) * np.array(Y_pa_c2_abs) * np.array(Y_pa_c2_sign_prob))
    Y_pa_c3_proba_balanced = list(np.array(Y_pa_c3_sign) * np.array(Y_pa_c3_abs) * np.array(Y_pa_c3_sign_prob))

    metrics_sd_trueskill_proba_sign_abs = results_of_pairwise_combinations(
        all_data, percentage_of_top_samples,
        Y_pa_c1, Y_pa_c2_proba_balanced, Y_pa_c3_proba_balanced,
        rank_method=rating_sbbr
    )

    # list of 9 rows of size 17. per 3 of them is from on rating scheme.
    metrics_pair_res = metrics_trueskill[:3] + metrics_sd_trueskill_regY[:3] + metrics_sd_trueskill_proba_sign_abs[:3]
    # list of 3 rows of size 6. each of them is from on rating scheme.
    metrics_est = [metrics_trueskill[3]] + [metrics_sd_trueskill_regY[3]] + [metrics_sd_trueskill_proba_sign_abs[3]]
    return metrics_pair_res, [acc_c2, acc_c3], metrics_est


def run_model(data, current_dataset_count, percentage_of_top_samples):
    temporary_file_dataset_count = int(np.load("boolean_chembl_trueskill_sb_variations_rf_run1_temp_dataset_count.npy"))

    if current_dataset_count == temporary_file_dataset_count:
        existing_iterations = np.load("boolean_chembl_trueskill_sb_variations_rf_run1_temp.npy")
        existing_count = len(existing_iterations)
        metrics = list(existing_iterations)
    else:
        metrics = []
        existing_count = 0

    count = 0
    for outer_fold, datum in data.items():
        count += 1
        if count <= existing_count: continue
        metrics_sa, acc_sa, metrics_est_sa = performance_standard_approach(datum, percentage_of_top_samples)
        # [[x1, x2, ..., x17] * 4], [acc1, acc2], [[x1, ..., x6] * 2]
        metrics_pa, acc_pa, metrics_est_pa = performance_pairwise_approach(datum, percentage_of_top_samples)
        # [[x1, x2, ..., x17] * 3], [acc1, acc2], [[x1, ..., x6]*3]

        acc = acc_sa + acc_pa + [0] * (len(metrics_sa[0]) - 4)

        metrics_est = [ls + [0] * (len(metrics_sa[0]) - 6) for ls in (metrics_est_sa + metrics_est_pa)]
        metrics.append(metrics_sa + metrics_pa + [acc] + metrics_est)

        np.save("boolean_chembl_trueskill_sb_variations_rf_run1_temp_dataset_count.npy", [current_dataset_count])
        np.save("boolean_chembl_trueskill_sb_variations_rf_run1_temp.npy", np.array(metrics))

    return np.array([metrics])
