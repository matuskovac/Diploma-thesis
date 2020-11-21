import time
import warnings

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.processing import postprocess, preprocess

warnings.filterwarnings('ignore')


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS
x_columns = config.X_COLUMNS
models_dict = config.MODELS_DICT


df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')

use = ['knn', 'svm', 'isolationF'][0]
model = models_dict[use]

predict_based_on_whole_pattern = True
kind_of_patten = 2

users_to_cv = preprocess.get_combinations_for_cv(
    df_raw_train[y_column].unique(), 1)
print(users_to_cv)


start = time.time()
train_eer, val_eer, test_eer = evaluation.cross_validate(
    selected_features_dict[model['x_columns']], y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, model['name'], model['params'], predict_based_on_whole_pattern, kind_of_patten)

end = time.time()
print(end - start)
print("TRAIN EER: " + str(train_eer))
print("VAL EER: " + str(val_eer))
print("TEST EER: " + str(test_eer))
