import time
from statistics import mean

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.models import models
from packages.processing import postprocess, split

path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS
x_columns = config.X_COLUMNS


df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')

test_eer_array = []
val_eer_array = []
train_eer_array = []

predict_based_on_whole_pattern = True
start = time.time()
users_to_cv = df_raw_train[y_column].unique()
for selected_owners in users_to_cv:
    df_train, df_val, df_test = split.adapt_dfs_to_users(
        df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, 2)

    knn = models.get_knn()
    knn.fit(df_train[x_columns])

    # predicted_train = [np.mean(i)
    #                    for i in knn.kneighbors(df_train[x_columns])[0]]
    predicted_val = [np.mean(i) for i in knn.kneighbors(df_val[x_columns])[0]]
    predicted_test = [np.mean(i)
                      for i in knn.kneighbors(df_test[x_columns])[0]]

    # ground_truth_train, predicted_train = postprocess.adapt_columns_for_evaluation(
    #     df_train[[y_column, 'id']], predicted_train, y_column, predict_based_on_whole_pattern)
    # train_eer_array.append(evaluation.get_eer(
    #     ground_truth_train, predicted_train, selected_owners))

    ground_truth_val, predicted_val = postprocess.adapt_columns_for_evaluation(
        df_val[[y_column, 'id']], predicted_val, y_column, predict_based_on_whole_pattern)
    val_eer_array.append(evaluation.get_eer(
        ground_truth_val, predicted_val, selected_owners))

    ground_truth_test, predicted_test = postprocess.adapt_columns_for_evaluation(
        df_test[[y_column, 'id']], predicted_test, y_column, predict_based_on_whole_pattern)
    test_eer_array.append(evaluation.get_eer(
        ground_truth_test, predicted_test, selected_owners))

# train_eer = mean(train_eer_array)
val_eer = mean(val_eer_array)
test_eer = mean(test_eer_array)
end = time.time()
print(end-start)
# print("TRAIN EER: " + str(train_eer))
print("VAL EER: " + str(val_eer))
print("TEST EER: " + str(test_eer))
