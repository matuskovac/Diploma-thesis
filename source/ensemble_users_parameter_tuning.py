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
compute_login = config.COMPUTE_LOGIN

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')
df_raw_val = pd.read_csv(path_to_featutes + "imputed/" + "val.csv", sep=',')
df_raw_test = pd.read_csv(path_to_featutes + "imputed/" + "test.csv", sep=',')


all_models = ['knn', 'isolationF', 'autoencoder', 'svm', 'lsanomaly']

predict_based_on_whole_pattern = True
all_kind_of_patten = [0, 1, 2]
all_ensemble_based_on_segments = [False] 

all_ensemble_based_on_users = [True, False]
all_functions_to_ensemble_users = ['max', 'min', 'sum', 'prod']


iterables = [all_models, all_ensemble_based_on_users,
             all_ensemble_based_on_segments, all_functions_to_ensemble_users, all_kind_of_patten]

rows = []
for count_of_owners in range(2, 5):
    users_to_cv = postprocess.get_combinations_for_cv(
        df_raw_train[y_column].unique(), count_of_owners, compute_login)

    for model_to_use, ensemble_based_on_users, ensemble_based_on_segments,  function_to_ensemble_users_raw, kind_of_patten in itertools.product(*iterables):
        function_to_ensemble_users = getattr(
            np, function_to_ensemble_users_raw)

        train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble3(
            models_dict, model_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, ensemble_based_on_users, function_to_ensemble_users)

        rows.append([model_to_use, ensemble_based_on_users, count_of_owners, kind_of_patten,
                     ensemble_based_on_segments, function_to_ensemble_users_raw, train_eer, val_eer, test_eer])
        print(len(rows))


df_tuning = pd.DataFrame(rows, columns=[
    "model", "ensemble_based_on_users", "count_of_owners", "kind_of_patten", "ensemble_based_on_segments", "function_to_users", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/ensemble_users.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendnotificate(message='DONE!')
except:
    print("Notificate not sent!")
finally:
    print("Job done!")
