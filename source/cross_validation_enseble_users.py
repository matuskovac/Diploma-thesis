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


options = ['knn', 'svm', 'isolationF', 'autoencoder']
selected = 2
model_to_use = options[selected]

predict_based_on_whole_pattern = True
kind_of_patten = 2
ensemble_based_on_segments = False

ensemble_based_on_users = True
function_to_ensemble_users = 'max'

function_to_ensemble_users = getattr(np, function_to_ensemble_users)

users_to_cv = postprocess.get_combinations_for_cv(
    df_raw_train[y_column].unique(), 2)

start = time.time()
train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble3(
    models_dict, model_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, ensemble_based_on_users, function_to_ensemble_users)

end = time.time()
print(end - start)
print("TRAIN EER: " + str(train_eer))
print("VAL EER: " + str(val_eer))
print("TEST EER: " + str(test_eer))
