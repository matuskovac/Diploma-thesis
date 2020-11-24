
import time

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
models_dict =config.MODELS_DICT

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


predict_based_on_whole_pattern = True

use = ['knn', 'svm', 'isolationF', 'autoencoder'][3]
model=models_dict[use]



selected_owners = ['Stevo']
start = time.time()
df_train, df_val, df_test = split.adapt_dfs_to_users(
    df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, 0)

predicted_train, predicted_test = models.use_model(
    model['name'], [df_train, df_test], selected_features_dict[model['x_columns']], model['params'])


ground_truth_test, predicted_test = postprocess.adapt_columns_for_evaluation(
    df_test[[y_column, 'id']], predicted_test, y_column, predict_based_on_whole_pattern)

evaluation.plot_far_eer(ground_truth_test, predicted_test, selected_owners)
end = time.time()
print(end-start)
