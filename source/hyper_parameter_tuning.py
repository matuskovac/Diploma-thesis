import itertools
import warnings
from statistics import mean

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.models import models
from packages.notification import notificate
from packages.processing import postprocess, split

warnings.filterwarnings('ignore')


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


use = ['knn', 'svm'][1]
all_params_comb = []

if use == 'knn':
    model = 'knn'

    all_knn_l = list(range(1, 3))
    all_knn_n_neighbors = list(range(1, 3))
    iterables = [all_knn_l, all_knn_n_neighbors]
    for n, p in itertools.product(*iterables):
        all_params_comb.append({'n': n, 'p': p})


elif use == 'svm':
    model = 'svm'
    all_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    iterables = [all_kernels]
    for comb in itertools.product(*iterables):
        print(comb[0])
        all_params_comb.append({"kernel": comb[0]})


all_features_subset = selected_features_dict.keys()
all_predict_based_on_whole_pattern = [True, False]
kind_of_patterns = [0, 1, 2]

iterables = [all_features_subset,
             all_predict_based_on_whole_pattern, kind_of_patterns, all_params_comb]


rows = []
for features_subset, predict_based_on_whole_pattern, kind_of_patten, params in itertools.product(*iterables):
    users_to_cv = df_raw_train[y_column].unique()

    train_eer, val_eer, test_eer = evaluation.cross_validate(
        selected_features_dict[features_subset], y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, model, params, predict_based_on_whole_pattern)

    rows.append([features_subset,
                 predict_based_on_whole_pattern, kind_of_patten, model, params, train_eer, val_eer, test_eer])
    print(len(rows))

df_tuning = pd.DataFrame(rows, columns=[
                         "features_subset", "predict_based_on_whole_pattern", "kind_of_patten", "model", "params", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/tuning_result4.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendemail(subject='Script', message='DONE!')
except:
    print("Mail not sent!")
finally:
    print("Job done!")
