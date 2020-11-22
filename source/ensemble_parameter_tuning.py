import itertools
import time
import warnings

import numpy as np
import pandas as pd

from packages.config import config
from packages.evaluation import evaluation
from packages.notification import notificate
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


options = pd.Series(['knn', 'svm', 'isolationF'])
selected = [0,2]
models_to_use = list(options[selected])

predict_based_on_whole_pattern = True
kind_of_patten = 2
all_ensemble_based_on_segments = [True, False]
all_fun = ['max', 'min', 'sum', 'prod']
all_scale_functions = ['use_standard_scaler_list', 'use_minmax_scaler_list']

users_to_cv = preprocess.get_combinations_for_cv(
    df_raw_train[y_column].unique(), 1)

iterables = [all_ensemble_based_on_segments, all_fun, all_scale_functions]

rows = []
for ensemble_based_on_segments, fun, scale in itertools.product(*iterables):

    ensemble_function = getattr(np, fun)
    scale_function = getattr(postprocess, scale)

    train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble(
        models_dict, models_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, ensemble_function, scale_function)

    rows.append([scale.split('_')[1], '_'.join(list(options[i] for i in selected)),
                 ensemble_based_on_segments, fun, train_eer, val_eer, test_eer])
    print(len(rows))


df_tuning = pd.DataFrame(rows, columns=[
                         "norm", "model", "apply_on_segments", "function", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/ensemble_1_if_knn.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendemail(subject='Script', message='DONE!')
except:
    print("Mail not sent!")
finally:
    print("Job done!")
