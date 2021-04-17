from packages.processing import preprocess
from packages.config import config
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore')


path_to_raw_data = config.PATH_TO_RAW_DATA
compute_features_for_segment = config.COMPUTE_FEATURES_FOR_SEGMENT
user_column = config.USER_COLUMN
path_to_featutes = config.PATH_TO_FEATURES
columns_to_identificate_features = config.get_columns_to_identify_features()
delete_nan_features = config.DELETE_NAN_FEATURES
print(path_to_raw_data)
print(path_to_featutes)

touch_data = pd.read_csv(path_to_raw_data + 'touch.csv', sep=',')
acc_data = pd.read_csv(path_to_raw_data + 'linear_accelerometer.csv', sep=',')
gyro_data = pd.read_csv(path_to_raw_data + 'gyroscope.csv', sep=',')

touch_data["id"] = touch_data["pattern_id"].astype(str) + touch_data["device"].astype(str)
acc_data["id"] = acc_data["pattern_id"].astype(str) + acc_data["device"].astype(str)
gyro_data["id"] = gyro_data["pattern_id"].astype(str) + gyro_data["device"].astype(str)

if compute_features_for_segment:
    touch_data["id"] += touch_data['segment'].astype(str)
    acc_data["id"] += acc_data['segment'].astype(str)
    gyro_data["id"] += gyro_data['segment'].astype(str)
else:
    touch_data.drop('segment', axis=1, inplace=True)
    acc_data.drop('segment', axis=1, inplace=True)
    gyro_data.drop('segment', axis=1, inplace=True)

le = LabelEncoder()
le.fit(pd.concat([touch_data['id'], acc_data['id'], gyro_data['id']]))
touch_data[user_column] = le.transform(touch_data['id'])
acc_data[user_column] = le.transform(acc_data['id'])
gyro_data[user_column] = le.transform(gyro_data['id'])


features = [
    'duration',
    ('length', {
        'columns': ['x', 'y'],
    }),
    ('start', {
        'columns': ['x', 'y'],
    }),
    ('velocity', {
        'columns': ['x', 'y'],
    }),
    ('acceleration', {
        'columns': ['x', 'y'],
    }),
    ('jerk', {
        'columns': ['x', 'y'],
    }),
    ('angular_velocity', {
        'columns': ['x', 'y'],
    }),
    ('angular_acceleration', {
        'columns': ['x', 'y'],
    }),
]
print(time.strftime("%Y-%m-%d %H:%M"))
touch_features = preprocess.compute_features(
    touch_data, features, user_column, delete_nan_features)
touch_features = touch_features.merge(
    touch_data[columns_to_identificate_features], on=[user_column]).drop_duplicates()
features = [
    ('velocity', {
        'columns': ['x', 'y', 'z'],
    }),
]
print(time.strftime("%Y-%m-%d %H:%M"))

acc_features = preprocess.compute_features(
    acc_data, features, user_column, delete_nan_features, "accelerometer_jerk")
acc_statistics_from_raw_data = preprocess.compute_statistics(
    acc_data, ['x', 'y', 'z'], "accelerometer_", user_column)
acc_features = acc_features.merge(
    acc_statistics_from_raw_data, on='user', how='inner').drop_duplicates()
print(time.strftime("%Y-%m-%d %H:%M"))

gyro_features = preprocess.compute_features(
    gyro_data, features, user_column, delete_nan_features, "gyro_jerk")
gyro_statistics_from_raw_data = preprocess.compute_statistics(
    gyro_data, ['x', 'y', 'z'], "gyro_", user_column)
gyro_features = gyro_features.merge(
    gyro_statistics_from_raw_data, on='user', how='inner').drop_duplicates()
print(time.strftime("%Y-%m-%d %H:%M"))

all_features = touch_features.merge(acc_features, on='user', how='inner').merge(
    gyro_features, on='user', how='inner')
preprocess.normalize_columns_names(all_features)

touch_features.to_csv(path_to_featutes + "touch_feautures.csv", encoding='utf-8', index=False)
acc_features.to_csv(path_to_featutes + "acc_feautures.csv", encoding='utf-8', index=False)
gyro_features.to_csv(path_to_featutes + "gyro_feautures.csv", encoding='utf-8', index=False)
all_features.to_csv(path_to_featutes + "all_feautures.csv", encoding='utf-8', index=False)
