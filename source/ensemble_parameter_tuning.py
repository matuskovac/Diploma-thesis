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


models = ['knn', 'svm', 'lsanomaly']

all_comb_models = []
for i in range(2, 3):
    comb = list(itertools.combinations(models, i))
    all_comb_models += [list(elem) for elem in comb]

all_predict_based_on_whole_pattern = [True, False]
all_kind_of_patten = [0, 1, 2]
all_ensemble_based_on_segments = [True, False]
all_fun = ['max', 'min', 'sum', 'prod']
all_scale_functions = ['use_standard_scaler_list', 'use_minmax_scaler_list']
all_count_of_owners = list(range(1, 5))

iterables = [all_count_of_owners, all_comb_models, all_ensemble_based_on_segments,
             all_fun, all_scale_functions, all_kind_of_patten, all_predict_based_on_whole_pattern]

rows = []

for count_of_owners, models_to_use, ensemble_based_on_segments, fun, scale, kind_of_patten, predict_based_on_whole_pattern in itertools.product(*iterables):
    users_to_cv = postprocess.get_combinations_for_cv(
        df_raw_train[y_column].unique(), count_of_owners, compute_login)

    ensemble_function = getattr(np, fun)
    scale_function = getattr(postprocess, scale)

    train_eer, val_eer, test_eer = evaluation.cross_validate_with_ensemble(
        models_dict, models_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, users_to_cv, predict_based_on_whole_pattern, kind_of_patten, ensemble_based_on_segments, ensemble_function, scale_function)

    rows.append([scale.split('_')[1], '_'.join(models_to_use),
                 ensemble_based_on_segments, count_of_owners, predict_based_on_whole_pattern, fun, kind_of_patten, train_eer, val_eer, test_eer])
    print(len(rows))


df_tuning = pd.DataFrame(rows, columns=[
                         "normalization", "model", "apply_on_segments", "count_of_owners", "predict_based_on_whole_pattern", "function", "kind_of_patten", "train_eer", "val_eer", "test_eer"])

df_tuning.to_csv("../results/cont_ensemble_models.csv",
                 encoding='utf-8', index=False)


try:
    notificate.sendnotificate(message='DONE!')
except:
    print("Notificate not sent!")
finally:
    print("Job done!")
