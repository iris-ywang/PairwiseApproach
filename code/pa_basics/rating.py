#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:00:02 2022

@author: dangoo
"""

from trueskill import Rating, rate_1vs1 
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from ScoreBasedTrueSkill.score_based_bayesian_rating import ScoreBasedBayesianRating as sd_rate_1vs1
from ScoreBasedTrueSkill.rating import Rating as SDRating



def rating_trueskill(comparison_results_lst, test_combs_lst, y_true):
    nsamples = len(y_true)
    ncomparisons = len(comparison_results_lst)
    ranking = [Rating() for _ in range(nsamples)]

    for comp_id in range(ncomparisons):
        
        id_a, id_b = test_combs_lst[comp_id]
        comp_result = comparison_results_lst[comp_id]
        
        # trueskill:
            
        if comp_result > 0:
            # i.e. id_a wins
            ranking[id_a], ranking[id_b] = rate_1vs1(ranking[id_a], ranking[id_b]) 
        
        elif comp_result < 0:
            ranking[id_b], ranking[id_a] = rate_1vs1(ranking[id_b], ranking[id_a]) 
        elif comp_result == 0:
            ranking[id_b], ranking[id_a] = rate_1vs1(ranking[id_b], ranking[id_a], drawn=True) 

    final_ranking = np.array([i.mu for i in ranking])
    
    return final_ranking


def rating_sbbr(comparison_results_lst, test_combs_lst, y_true):
    nsamples = len(y_true)
    ncomparisons = len(comparison_results_lst)
    # mean_train = np.mean(y_true[train_ids])
    # dev_train = mean_train / 3
    # beta = mean_train / 6
    ranking = [[SDRating()] for _ in range(nsamples)]

    for comp_id in range(ncomparisons):
        ida, idb = test_combs_lst[comp_id]
        comp_result = comparison_results_lst[comp_id]
        sd_rate_1vs1([ranking[ida], ranking[idb]], [comp_result, 0]).update_skills()

    return np.array([i[0].mean for i in ranking])


def evaluation(dy, combs_lst, train_ids, test_ids, y_true, func = rating_trueskill, params = None):
    y0 = func(dy, combs_lst, y_true, train_ids)
    # print(y_true[test_ids], y0[test_ids])
    rho = spearmanr(y_true[test_ids], y0[test_ids], nan_policy = "omit")[0]
    ndcg = ndcg_score([y_true[test_ids]], [y0[test_ids]])
    mse = mean_squared_error(y_true[test_ids], y0[test_ids])
    tau = kendalltau(y_true[test_ids], y0[test_ids])[0]
    # print(rho, ndcg, tau, mse)

    return y0, (rho, ndcg, tau, mse)
