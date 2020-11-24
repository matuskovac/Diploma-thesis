import numpy as np
import random
from behalearn.features import FeatureExtractor
from behalearn.preprocessing import columns


def get_columns_combinations(col_names, combinations=None):
    combs = columns._get_column_combinations(col_names, combinations)
    final_combs = []
    for comb in combs:
        if len(comb) > 1:
            final_combs.append(comb)
    return final_combs


def calculate_maginute_to_df(df, columns_name):
    final_combinations = []
    for combination in get_columns_combinations(columns_name):
        sum = [0]*len(df)
        for dimension in combination:
            sum += df[dimension] ** 2
        magnitude = sum ** (1/2)
        new_column = '_'.join(combination)
        df[new_column] = magnitude
        final_combinations.append(new_column)

    return final_combinations


def compute_features(df, features, user_column, delete_nan_features, prefix=None):
    extractor = FeatureExtractor(features, [user_column])

    features_df = extractor.fit_transform(df)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    if delete_nan_features:
        features_df = features_df.dropna()

    if prefix is not None:
        features_df.columns = features_df.columns.str.replace(
            r"velocity", prefix)

    return features_df


def renaming_condition(x, columns_name, prefix):
    if x in columns_name:
        return prefix + x
    return x


def add_prefix_to_columns(df, columns_name, prefix):
    df.columns = [renaming_condition(
        col, columns_name, prefix) for col in df.columns]

    return [prefix + s for s in columns_name]


def compute_statistics(df, columns_to_compute_statistic, prefix, user_column):
    columns_to_compute_statistic += calculate_maginute_to_df(
        df, columns_to_compute_statistic)
    columns_to_compute_statistic = add_prefix_to_columns(
        df, columns_to_compute_statistic, prefix)

    statistics = df.groupby([user_column])[
        columns_to_compute_statistic].describe()
    statistics.columns = statistics.columns.to_flat_index()
    statistics.rename(columns='_'.join, inplace=True)
    statistics = statistics[statistics.columns.drop(
        list(statistics.filter(regex='count')))]

    return statistics


def normalize_columns_names(df):
    df.columns = df.columns.str.replace(r"25%", "_lower_q")
    df.columns = df.columns.str.replace(r"50%", "_median")
    df.columns = df.columns.str.replace(r"75%", "_upper_q")
    df.columns = df.columns.str.replace(r"__", "_")


def get_combinations_for_cv(list, i_comb):
    random.seed(0)
    random.shuffle(list)
    if i_comb == 1:
        return list

    if i_comb == 2:
        end = len(list)
        step = 6
        users_comb = []
        for i in range(end):
            for j in range(i+step, end, step):
                # print(i, j)
                users_comb.append([list[i], list[j]])
        return users_comb

    if i_comb == 3:
        end = len(list)
        step = 3
        users_comb = []
        for i in range(end):
            for j in range(i+step, end, step*2):
                for k in range(j+step, end, step):
                    # print(i, j, k)
                    users_comb.append([list[i], list[j], list[k]])
                    break
        return users_comb

    if i_comb == 4:
        end = len(list)
        step = 2
        users_comb = []
        for i in range(end):
            for j in range(i+step, end, step*3):
                for k in range(j+step, end, step*2):
                    for l in range(k+step, end, step):
                        # print(i, j, k, l)
                        users_comb.append([list[i], list[j], list[k], list[l]])
                        break
                    break
        return users_comb
