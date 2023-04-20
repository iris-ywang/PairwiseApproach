import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, ndcg_score


class EvaluateAbilityToIdentifyTopTestSamples:
    def __init__(self, percentage_of_top_samples, y_train_with_true_test, y_train_with_predicted_test,
                 all_data):
        self.y_pred_all = y_train_with_predicted_test
        self.y_true_all = y_train_with_true_test
        self.all_data = all_data
        self.test_ids = all_data["test_ids"]
        self.train_ids = all_data['train_ids']
        self.pc = percentage_of_top_samples

    def find_x_extreme_pairs(self, y_pred_all, Y_sign_and_abs_predictions=None):
        if Y_sign_and_abs_predictions is not None:
            Y_c2_abs = np.array(list(Y_sign_and_abs_predictions.values()))[:, 0]
        else:
            Y_c2_abs = self.pairwise_differences_for_standard_approach(y_pred_all)
        orders_test_Y_abs = np.argsort(-Y_c2_abs)
        top_x_tests = orders_test_Y_abs[:self.x * 10]
        bottom_x_tests = orders_test_Y_abs[-self.x * 10:]

        return top_x_tests, bottom_x_tests

    def pairwise_differences_for_standard_approach(self, y_pred_all):
        Y_c2_abs_derived = []
        for pair_id in self.all_data["c2_test_pair_ids"]:
            id_a, id_b = pair_id
            Y_c2_abs_derived.append(abs(y_pred_all[id_a] - y_pred_all[id_b]))
        return np.array(Y_c2_abs_derived)

    def find_sum_of_estimates_of_top_x_tests(self, y_all_pred):
        # a list of sample IDs in the descending order of activity values
        orders_overall = np.argsort(-y_all_pred)
        orders_tests = [i for i in orders_overall if i in self.test_ids]
        top_x_tests = orders_tests[:self.x]  # index of top 5 testids
        sum_y_true_of_identified_test = []

        for test_id in top_x_tests:
            sum_y_true_of_identified_test.append(self.y_true_all[test_id])
        return sum(sum_y_true_of_identified_test)

    def find_top_test_ids(self, y_pred_all, Y_sign_and_abs_predictions=None):
        # trains == train samples; tests == test samples.

        overall_orders = np.argsort(-y_pred_all)  # a list of sample IDs in the descending order of activity values
        top_trains_and_tests = overall_orders[0: int(self.pc * len(overall_orders))]
        top_tests = [idx for idx in top_trains_and_tests if idx in self.test_ids]

        # Find the ID of top train sample in the overall_order
        top_train_order_position = 0
        while True:
            if overall_orders[top_train_order_position] in self.train_ids: break
            top_train_order_position += 1
        top_train_id = overall_orders[top_train_order_position]

        tests_better_than_top_train = list(overall_orders[:top_train_order_position])

        if Y_sign_and_abs_predictions is None:
            final_estimate_of_y_and_delta_y = None
        else:
            final_estimate_of_y_and_delta_y = self.estimate_y_from_final_ranking_and_absolute_Y(
                top_tests, tests_better_than_top_train, top_train_id, overall_orders, Y_sign_and_abs_predictions
            )
        return top_tests, tests_better_than_top_train, top_train_id, final_estimate_of_y_and_delta_y

    def estimate_y_from_final_ranking_and_absolute_Y(self, top_tests, tests_better_than_top_train, top_train_id,
                                                     overall_orders, Y_sign_and_abs_predictions):
        final_estimate_of_y_and_delta_y = {}
        for test_id in top_tests:
            y_test_estimate = self.estimate_averaged_y_from_final_ranking(overall_orders, test_id,
                                                                          Y_sign_and_abs_predictions)
            if test_id in tests_better_than_top_train:
                Y_rough = Y_sign_and_abs_predictions[(test_id, top_train_id)][0] * 1
            if test_id in top_tests:
                Y_rough = Y_sign_and_abs_predictions[(test_id, top_train_id)][0] * -1

            Y_ave = y_test_estimate - self.y_true_all[top_train_id]
            final_estimate_of_y_and_delta_y[test_id] = [Y_rough, Y_ave, self.y_true_all[test_id], y_test_estimate]
        return final_estimate_of_y_and_delta_y

    def estimate_averaged_y_from_final_ranking(self, overall_orders, test_id, Y_sign_and_abs_predictions):
        test_order_position = int(np.where(overall_orders == test_id)[0])
        test_estimates = []
        for train_id in self.train_ids:
            train_order_position = int(np.where(overall_orders == train_id)[0])
            if test_order_position < train_order_position:
                # i.e.test is ranked higher than the top train -> higher activity than that of the top train sample
                test_estimates.append(Y_sign_and_abs_predictions[(test_id, train_id)][0] * 1
                                      + self.y_true_all[train_id])
                # No need of the below because:
                #     sv[(test_id, train_id)] = sv[(train_id, test_id)] = abs(y_train_id - y_test_id)
                # test_estimates.append(sv[(train_id, test_id)][0] * -1 + self.y_true[train_id])
            elif test_order_position > train_order_position:
                test_estimates.append(Y_sign_and_abs_predictions[(test_id, train_id)][0] * -1
                                      + self.y_true_all[train_id])
        return np.mean(test_estimates)

    def estimate_precision_recall(self, top_tests_true, top_tests):
        test_samples_boolean_true = [0 for _ in range(len(self.test_ids))]
        for top_test_id_true in top_tests_true:
            position_in_test_ids = int(np.where(self.test_ids == top_test_id_true)[0])
            test_samples_boolean_true[position_in_test_ids] = 1

        test_samples_boolean_pred = [0 for _ in range(len(self.test_ids))]
        for top_test_id in top_tests:
            position_in_test_ids = int(np.where(self.test_ids == top_test_id)[0])
            test_samples_boolean_pred[position_in_test_ids] = 1

        precision = precision_score(test_samples_boolean_true, test_samples_boolean_pred)
        recall = recall_score(test_samples_boolean_true, test_samples_boolean_pred)
        f1 = f1_score(test_samples_boolean_true, test_samples_boolean_pred)

        return precision, recall, f1

    def ndcg_top_pc(self):
        if len(self.test_ids) <= 5:
            return ndcg_score([self.y_true_all[self.test_ids]], [self.y_pred_all[self.test_ids]])
        k = int(self.pc * len(self.test_ids)) if int(self.pc * len(self.test_ids)) > 5 else 5
        return ndcg_score([self.y_true_all[self.test_ids]], [self.y_pred_all[self.test_ids]],
                          k=k)

    def RIE(self, top_tests_true):
        alpha = 1 / self.pc
        overall_orders = np.argsort(-self.y_pred_all[self.test_ids]) # a list of test sample indice in the descending order of activity values
        ordered_ids = np.array(self.test_ids)[overall_orders] # a list of test IDs in the descending order of activity values
        sorter = np.argsort(ordered_ids)
        ranks_of_actives = sorter[np.searchsorted(ordered_ids, top_tests_true, sorter=sorter)] + 1
        relative_rank = ranks_of_actives / (max(overall_orders) + 1)
        numerator = sum( np.exp(-1 * alpha * relative_rank))
        return numerator


    def run_evaluation(self, Y_sign_and_abs_predictions=None):
        top_tests_true, tests_better_than_top_train_true, _, _ = self.find_top_test_ids(self.y_true_all)
        top_tests, tests_better_than_top_train, top_train_id, final_estimate_of_y_and_Y = \
            self.find_top_test_ids(self.y_pred_all, Y_sign_and_abs_predictions)
        # print("Number of tops: " + str(len(top_tests_true)) +", " + str(len(tests_better_than_top_train_true)))
        # print('top_tests_true:', top_tests_true)
        # print('top_tests:', top_tests)
        # print('tests_better_than_top_train_true:', tests_better_than_top_train_true)
        # print('tests_better_than_top_train', tests_better_than_top_train)

        if len(top_tests_true) > 0:
            # Correct Ratio:
            correct_ratio_exceeding_train = self.calculate_correct_ratio(tests_better_than_top_train_true,
                                                                         tests_better_than_top_train)
            correct_ratio_top_of_dataset = self.calculate_correct_ratio(top_tests_true, top_tests)

            # precision & recall:
            precision_top, recall_top, f1_top = self.estimate_precision_recall(top_tests_true, top_tests)
            precision_better, recall_better, f1_better = self.estimate_precision_recall(tests_better_than_top_train_true,
                                                                                        tests_better_than_top_train)
            # Summation Ratio:
            self.x = 5 if len(self.test_ids) >= 5 else len(self.test_ids)
            sum_y_true_of_pred_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_pred_all)
            sum_y_true_of_true_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_true_all)
            summation_ratio_at_5 = sum_y_true_of_pred_top_test / sum_y_true_of_true_top_test

            self.x = 10 if len(self.test_ids) >= 10 else len(self.test_ids)
            sum_y_true_of_pred_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_pred_all)
            sum_y_true_of_true_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_true_all)
            summation_ratio_at_10 = sum_y_true_of_pred_top_test / sum_y_true_of_true_top_test

            self.x = int(0.2*len(self.all_data["test_ids"])) if len(top_tests_true) >= 5 else 1
            sum_y_true_of_pred_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_pred_all)
            sum_y_true_of_true_top_test = self.find_sum_of_estimates_of_top_x_tests(self.y_true_all)
            summation_ratio_at_20pc = sum_y_true_of_pred_top_test / sum_y_true_of_true_top_test

            # MSE(corectly identified tests in top 20%)
            mse_of_tests_top_pc = self.calculate_mse_top_tests_identified(top_tests_true, top_tests,
                                                                          final_estimate_of_y_and_Y)
            mse_of_tests_better = self.calculate_mse_top_tests_identified(tests_better_than_top_train_true,
                                                                          tests_better_than_top_train,
                                                                          final_estimate_of_y_and_Y)

            # Correct Ratio for Y extremes
            top_x_pairs_true, bottom_x_pairs_true = self.find_x_extreme_pairs(self.y_true_all)
            top_x_pairs_pred, bottom_x_pairs_pred = self.find_x_extreme_pairs(self.y_pred_all,
                                                                              Y_sign_and_abs_predictions)
            # correct_ratio_top_pairs = self.calculate_correct_ratio(top_x_pairs_true, top_x_pairs_pred)
            # correct_ratio_bottom_pairs = self.calculate_correct_ratio(bottom_x_pairs_true, bottom_x_pairs_pred)
            rie = self.RIE(top_tests_true)
            ndcg_top_pc = self.ndcg_top_pc()

            return [correct_ratio_exceeding_train, correct_ratio_top_of_dataset,
                    summation_ratio_at_5, summation_ratio_at_10,
                    summation_ratio_at_20pc,
                    mse_of_tests_top_pc, mse_of_tests_better,
                    rie, ndcg_top_pc,
                    precision_top, recall_top, f1_top,
                    precision_better, recall_better, f1_better]
        else:
            return [np.nan for _ in range(15)]

    @staticmethod
    def calculate_correct_ratio(top_samples_true, top_samples_pred):
        correctly_identified_top_samples = list(set(top_samples_pred) & set(top_samples_true))
        return len(correctly_identified_top_samples) / len(top_samples_true) if len(top_samples_true) > 0 else np.nan

    def calculate_mse_top_tests_identified(self, top_samples_true, top_samples_pred, pairwise_estimates=None):
        correctly_identified_top_samples = list(set(top_samples_pred) & set(top_samples_true))

        if len(correctly_identified_top_samples) > 0:
            if pairwise_estimates is None:
                return mean_squared_error(self.y_true_all[correctly_identified_top_samples],
                                          self.y_pred_all[correctly_identified_top_samples])
            if pairwise_estimates is not None:
                y_true_tests = [pairwise_estimates[test_id][2] for test_id in correctly_identified_top_samples]
                y_pred_tests = [pairwise_estimates[test_id][3] for test_id in correctly_identified_top_samples]
                return mean_squared_error(y_true_tests, y_pred_tests)
        else:
            return np.nan
