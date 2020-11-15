import warnings
warnings.filterwarnings('ignore')
import itertools
from statistics import mean

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.models import models
from packages.notification import notificate
from packages.processing import postprocess, split


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


all_features_subset = selected_features_dict.keys()
all_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
all_predict_based_on_whole_pattern = [True, False]
kind_of_patterns = [0, 1, 2]
iterables = [all_features_subset, all_kernels,
             all_predict_based_on_whole_pattern, kind_of_patterns]

rows = []
for features_subset, svm_kernel, predict_based_on_whole_pattern, kind_of_patten in itertools.product(*iterables):

    test_eer_array = []
    val_eer_array = []
    train_eer_array = []

    predict_based_on_whole_pattern = True
    users_to_cv = df_raw_train[y_column].unique()
    for selected_owners in users_to_cv:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, 0)

        svm = models.get_svm(svm_kernel)
        svm.fit(df_train[selected_features_dict[features_subset]])

        predicted_train = svm.score_samples(
            df_train[selected_features_dict[features_subset]])
        predicted_val = svm.score_samples(
            df_val[selected_features_dict[features_subset]])
        predicted_test = svm.score_samples(
            df_test[selected_features_dict[features_subset]])

        ground_truth_train, predicted_train = postprocess.adapt_columns_for_evaluation(
            df_train[[y_column, 'id']], predicted_train, y_column, predict_based_on_whole_pattern)
        train_eer_array.append(evaluation.get_eer(
            ground_truth_train, predicted_train, selected_owners))

        ground_truth_val, predicted_val = postprocess.adapt_columns_for_evaluation(
            df_val[[y_column, 'id']], predicted_val, y_column, predict_based_on_whole_pattern)
        val_eer_array.append(evaluation.get_eer(
            ground_truth_val, predicted_val, selected_owners))

        ground_truth_test, predicted_test = postprocess.adapt_columns_for_evaluation(
            df_test[[y_column, 'id']], predicted_test, y_column, predict_based_on_whole_pattern)
        test_eer_array.append(evaluation.get_eer(
            ground_truth_test, predicted_test, selected_owners))

    train_eer = mean(train_eer_array)
    val_eer = mean(val_eer_array)
    test_eer = mean(test_eer_array)

    rows.append([features_subset, svm_kernel,
                 predict_based_on_whole_pattern, kind_of_patten, train_eer, val_eer, test_eer])
    print(len(rows))

df_tuning = pd.DataFrame(rows, columns=[
                         "features_subset", "svm_kernel", "predict_based_on_whole_pattern", "kind_of_patten", "train_eer", "val_eer", "test_eer"])
df_tuning.to_csv("../results/tuning_result_svm.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendemail(subject='Script', message='DONE!')
except:
    print("Mail not sent!")
finally:
    print("Job done!")
