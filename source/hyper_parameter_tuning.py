import itertools
import warnings
from statistics import mean

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.models import models
from packages.notification import notificate
from packages.processing import postprocess, preprocess, split

warnings.filterwarnings('ignore')


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS
compute_login = config.COMPUTE_LOGIN

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


use = ['knn', 'svm', 'isolationF', 'lsanomaly'][0]
all_params_comb = []
name = 'knn'

if use == 'knn':
    model = 'knn'

    all_knn_l = list(range(1, 6))
    all_knn_n_neighbors = list(range(1, 6))
    iterables = [all_knn_l, all_knn_n_neighbors]
    for p, n in itertools.product(*iterables):
        all_params_comb.append({'n': n, 'p': p})


elif use == 'svm':
    model = 'svm'
    all_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    iterables = [all_kernels]
    for comb in itertools.product(*iterables):
        all_params_comb.append({"kernel": comb[0]})


elif use == 'isolationF':
    model = 'ísolationForest'
    all_n_estimators = list(range(0, 800, 100))
    all_n_estimators[0] += 50
    iterables = [all_n_estimators]
    for comb in itertools.product(*iterables):
        all_params_comb.append({"n_estimators": comb[0]})

elif use == 'lsanomaly':
    model = 'lsanomaly'
    all_sigma = [0.5, 1, 2, 3, 4, 5, 6, 7, 10]
    all_rho = [0.001, 0.01, 0.1, 1, 10]
    iterables = [all_sigma, all_rho]
    for sigma, rho in itertools.product(*iterables):
        all_params_comb.append({'sigma': sigma, 'rho': rho})

all_features_subset = list(selected_features_dict.keys())
all_features_subset += [all_features_subset.pop(0)]
all_predict_based_on_whole_pattern = [True]
kind_of_patterns = [0, 1, 2]

iterables = [all_features_subset,
             all_predict_based_on_whole_pattern, kind_of_patterns, all_params_comb]

users_to_cv = postprocess.get_combinations_for_cv(
    df_raw_train[y_column].unique(), 1, compute_login)

comb_number = len(list(itertools.product(*iterables)))
checkpoit_number = comb_number/len(all_features_subset)
print(name + str(comb_number))
rows = []
df_tuning = pd.DataFrame(rows, columns=["features_subset", "predict_based_on_whole_pattern", "kind_of_patten", "model", "params", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/cont_tuning_result_" + name + ".csv", encoding='utf-8', index=False)

multiplier = 0
for features_subset, predict_based_on_whole_pattern, kind_of_patten, params in itertools.product(*iterables):

    train_eer, val_eer, test_eer = evaluation.cross_validate(
        selected_features_dict[features_subset], y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, model, params, predict_based_on_whole_pattern, kind_of_patten)

    rows.append([features_subset,
                 predict_based_on_whole_pattern, kind_of_patten, model, params, train_eer, val_eer, test_eer])

    print(name + str(len(rows) + (checkpoit_number * multiplier)))
    if len(rows) == checkpoit_number:
        multiplier += 1
        df_tuning = pd.DataFrame(rows, columns=[
                                "features_subset", "predict_based_on_whole_pattern", "kind_of_patten", "model", "params", "train_eer", "val_eer", "test_eer"])

        df_tuning.to_csv("../results/cont_tuning_result_" + name + ".csv",
                        encoding='utf-8', mode='a', index=False, header=False)
        rows = []

try:
    notificate.sendnotificate(message='DONE!')
except:
    print("Notificate not sent!")
finally:
    print("Job done!")
