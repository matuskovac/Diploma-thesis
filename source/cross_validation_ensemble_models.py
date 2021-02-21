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
compute_login = config.COMPUTE_LOGIN


df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


options = pd.Series(['knn', 'svm', 'isolationF', 'autoencoder'])
selected = [2]
models_to_use = list(options[selected])

predict_based_on_whole_pattern = True
kind_of_patten = 2
ensemble_based_on_segments = False
fun = 'max'
# ['use_standard_scaler_list', 'use_minmax_scaler_list']
scale_function = 'use_minmax_scaler_list'
scale_function = getattr(postprocess, scale_function)
function = getattr(np, fun)

users_to_cv = postprocess.get_combinations_for_cv(
    df_raw_train[y_column].unique(), 2, compute_login)

start = time.time()
train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble(
    models_dict, models_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, function, scale_function)

end = time.time()
print(end - start)
print("TRAIN EER: " + str(train_eer))
print("VAL EER: " + str(val_eer))
print("TEST EER: " + str(test_eer))
