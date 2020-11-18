import time
import warnings

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.processing import postprocess

warnings.filterwarnings('ignore')


path_to_featutes = config.PATH_TO_FEATURES
selected_features_dict = config.SELECTED_FEATURES_DICT
y_column = config.Y_COLUMNS
x_columns = config.X_COLUMNS


df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')

use = ['knn', 'svm', 'isolationF'][2]
if use == 'knn':
    model = 'knn'
    params = {'n': 1, 'p': 1}

elif use == 'svm':
    model = 'svm'
    params = {'kernel': 'poly'}

elif use == 'isolationF':
    model = 'ísolationForest'
    params = {'n_estimators': 500}

predict_based_on_whole_pattern = True

users_to_cv = df_raw_train[y_column].unique()

start = time.time()
train_eer, val_eer, test_eer = evaluation.cross_validate(
    x_columns, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, model, params, predict_based_on_whole_pattern)


print("TRAIN EER: " + str(train_eer))
print("VAL EER: " + str(val_eer))
print("TEST EER: " + str(test_eer))
