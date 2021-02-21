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


all_models = ['knn', 'isolationF', 'autoencoder']

predict_based_on_whole_pattern = True
kind_of_patten = 2
all_ensemble_based_on_segments = [True, False]

all_ensemble_based_on_users = [True]
all_functions_to_ensemble_users = ['max', 'min', 'sum', 'prod']


iterables = [all_models, all_ensemble_based_on_users,
             all_ensemble_based_on_segments, all_functions_to_ensemble_users]

rows = []
for count_of_owners in range(2, 5):
    users_to_cv = postprocess.get_combinations_for_cv(
        df_raw_train[y_column].unique(), count_of_owners, compute_login)

    for model_to_use, ensemble_based_on_users, ensemble_based_on_segments,  function_to_ensemble_users_raw in itertools.product(*iterables):
        function_to_ensemble_users = getattr(
            np, function_to_ensemble_users_raw)

        train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble3(
            models_dict, model_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, ensemble_based_on_users, function_to_ensemble_users)

        rows.append([model_to_use, ensemble_based_on_users, count_of_owners,
                     ensemble_based_on_segments, function_to_ensemble_users_raw, train_eer, val_eer, test_eer])
        print(len(rows))


df_tuning = pd.DataFrame(rows, columns=[
    "model", "ensemble_based_on_users", "count_of_owners", "ensemble_based_on_segments", "function_to_users", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/ensemble_users.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendemail(subject='Script', message='DONE!')
except:
    print("Mail not sent!")
finally:
    print("Job done!")
