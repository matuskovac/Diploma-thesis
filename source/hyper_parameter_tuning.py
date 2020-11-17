import warnings
warnings.filterwarnings('ignore')

import itertools
from statistics import mean

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.models import models
from packages.processing import postprocess, split
from packages.notification import notificate


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


all_features_subset = selected_features_dict.keys()
all_knn_l = list(range(1, 6))
all_knn_n_neighbors = list(range(1, 6))
all_predict_based_on_whole_pattern = [True, False]
kind_of_patterns = [0, 1, 2]
iterables = [all_features_subset, all_knn_l, all_knn_n_neighbors,
             all_predict_based_on_whole_pattern, kind_of_patterns]


rows = []
for features_subset, knn_l, knn_n_neighbors, predict_based_on_whole_pattern, kind_of_patten in itertools.product(*iterables):

    test_eer_array = []
    val_eer_array = []

    users_to_cv = df_raw_train[y_column].unique()
    for selected_owners in users_to_cv:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, kind_of_patten)

        knn = models.get_knn(knn_n_neighbors, knn_l)
        knn.fit(df_train[selected_features_dict[features_subset]])

        predicted_val = [np.mean(i) for i in knn.kneighbors(
            df_val[selected_features_dict[features_subset]])[0]]
        predicted_test = [np.mean(i) for i in knn.kneighbors(
            df_test[selected_features_dict[features_subset]])[0]]

        ground_truth_val, predicted_val = postprocess.adapt_columns_for_evaluation(
            df_val[[y_column, 'id']], predicted_val, y_column, predict_based_on_whole_pattern)
        val_eer_array.append(evaluation.get_eer(
            ground_truth_val, predicted_val, selected_owners))

        ground_truth_test, predicted_test = postprocess.adapt_columns_for_evaluation(
            df_test[[y_column, 'id']], predicted_test, y_column, predict_based_on_whole_pattern)
        test_eer_array.append(evaluation.get_eer(
            ground_truth_test, predicted_test, selected_owners))

    val_eer = mean(val_eer_array)
    test_eer = mean(test_eer_array)

    rows.append([features_subset, knn_l, knn_n_neighbors,
                 predict_based_on_whole_pattern, kind_of_patten, val_eer, test_eer])
    print(len(rows))

df_tuning = pd.DataFrame(rows, columns=[
                         "features_subset", "knn_l", "knn_n_neighbors", "predict_based_on_whole_pattern", "kind_of_patten", "val_eer", "test_eer"])
df_tuning.to_csv("../results/tuning_result_knn.csv", encoding='utf-8', index=False)


try:
    notificate.sendemail(subject='Script', message='DONE!')
except:
    print("Mail not sent!")
finally:
    print("Job done!")
