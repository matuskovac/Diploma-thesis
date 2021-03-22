import pandas as pd

from packages.config import config
from packages.processing import postprocess, split
from packages.notification import notificate

compute_features_for_segment = config.COMPUTE_FEATURES_FOR_SEGMENT
path_to_featutes = config.PATH_TO_FEATURES
delete_nan_features = config.DELETE_NAN_FEATURES
columns_to_identify_features = config.get_columns_to_identify_features()
y_column = config.USER_NAME_COLUMN
filter_user_with_many_strokes = config.CROPPED_OVER_1000

all_features = pd.read_csv(path_to_featutes + "all_feautures.csv")
if compute_features_for_segment:
    all_features['id'] = all_features['id'].str[:-1]


all_features = all_features.dropna(thresh=80).reset_index(drop=True)
x_columns = [x for x in list(
    all_features.columns) if x not in columns_to_identify_features]


if filter_user_with_many_strokes:
    sipledf = all_features.loc[all_features['scenario'] == 'scenario_show_simple'].dropna().reset_index(drop=True)
    complexdf = all_features.loc[all_features['scenario'] == 'scenario_show_complex'].dropna().reset_index(drop=True)
    sipledf = sipledf.groupby('username').head(500)
    complexdf = complexdf.groupby('username').head(500)
    only_nan = all_features[all_features.isna().any(axis=1)].reset_index(drop=True)

    merged_without_nans = sipledf.append(complexdf).reset_index(drop=True)
    merged = merged_without_nans.append(only_nan).reset_index(drop=True)
    sipledf = merged.loc[merged['scenario'] == 'scenario_show_simple']
    complexdf = merged.loc[merged['scenario'] == 'scenario_show_complex']
    sipledf = sipledf.groupby('username').head(500)
    complexdf = complexdf.groupby('username').head(500)
    final = sipledf.append(complexdf).reset_index(drop=True)
    all_features = final.sort_values(['username','pattern_id']).reset_index(drop=True)

df_raw_train, df_raw_val, df_raw_test = split.split_to_train_val_test_raw(
    all_features, y_column)

df_raw_train, df_raw_val, df_raw_test = postprocess.use_standard_scaler(
    [df_raw_train, df_raw_val, df_raw_test], x_columns)
print(df_raw_train[:10])

df_raw_train.to_csv(path_to_featutes  +
                    "train.csv", encoding='utf-8', index=False)
df_raw_val.to_csv(path_to_featutes +
                  "val.csv", encoding='utf-8', index=False)
df_raw_test.to_csv(path_to_featutes +
                   "test.csv", encoding='utf-8', index=False)

df_raw_train = pd.read_csv(path_to_featutes + "train.csv")
df_raw_val = pd.read_csv(path_to_featutes + "val.csv")
df_raw_test = pd.read_csv(path_to_featutes + "test.csv")

if not delete_nan_features:
    df_raw_train, df_raw_val, df_raw_test = postprocess.use_imputation(
        [df_raw_train, df_raw_val, df_raw_test], x_columns)


df_raw_train.to_csv(path_to_featutes + "imputed/" +
                    "train.csv", encoding='utf-8', index=False)
df_raw_val.to_csv(path_to_featutes + "imputed/" +
                  "val.csv", encoding='utf-8', index=False)
df_raw_test.to_csv(path_to_featutes + "imputed/" +
                   "test.csv", encoding='utf-8', index=False)

try:
    notificate.sendnotificate(message='DONE!')
except:
    print("Notificate not sent!")
finally:
    print("Job done!")
