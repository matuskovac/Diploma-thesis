import pandas as pd

from packages.config import config
from packages.processing import postprocess, split


compute_features_for_segment = config.COMPUTE_FEATURES_FOR_SEGMENT
path_to_featutes = config.PATH_TO_FEATURES
delete_nan_features = config.DELETE_NAN_FEATURES
columns_to_identify_features = config.get_columns_to_identify_features()
y_column = config.USER_NAME_COLUMN

all_features = pd.read_csv(path_to_featutes + "all_feautures.csv")
if compute_features_for_segment:
    all_features['id'] = all_features['id'].str[:-1]

all_features = all_features.dropna(thresh=80)
x_columns = [x for x in list(
    all_features.columns) if x not in columns_to_identify_features]


df_raw_train, df_raw_val, df_raw_test = split.split_to_train_val_test_raw(
    all_features, y_column)

df_raw_train, df_raw_val, df_raw_test = postprocess.use_standard_scaler(
    [df_raw_train, df_raw_val, df_raw_test], x_columns)
print(df_raw_train[:10])
if not delete_nan_features:
    df_raw_train, df_raw_val, df_raw_test = postprocess.use_imputation(
        [df_raw_train, df_raw_val, df_raw_test], x_columns)


df_raw_train.to_csv(path_to_featutes + "imputed/" +
                    "train.csv", encoding='utf-8', index=False)
df_raw_val.to_csv(path_to_featutes + "imputed/" +
                  "val.csv", encoding='utf-8', index=False)
df_raw_test.to_csv(path_to_featutes + "imputed/" +
                   "test.csv", encoding='utf-8', index=False)
