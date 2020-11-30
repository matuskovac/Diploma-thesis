import itertools
from collections import Counter

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def use_imputation(df_list, train_x_columns):
    imputer = IterativeImputer(random_state=0, max_iter=30, verbose=2)
    imputer.fit(df_list[0][train_x_columns])

    for i in range(len(df_list)):
        df_list[i][train_x_columns] = imputer.transform(
            df_list[i][train_x_columns])

    return df_list


def use_standard_scaler(df_list, train_x_columns):
    scaler = StandardScaler()
    scaler.fit(df_list[0][train_x_columns])

    for i in range(len(df_list)):
        df_list[i][train_x_columns] = scaler.transform(
            df_list[i][train_x_columns])

    return df_list


def use_standard_scaler_list(list_of_lists):
    list_of_lists = [np.array(a).reshape(-1, 1) for a in list_of_lists]
    for i in range(len(list_of_lists)):
        scaler = StandardScaler()
        scaler.fit(list_of_lists[i])
        list_of_lists[i] = list(
            np.array(scaler.transform(list_of_lists[i])).flat)

    return list_of_lists if len(list_of_lists) > 1 else list_of_lists[0]


def use_minmax_scaler_list(list_of_lists):
    list_of_lists = [np.array(a).reshape(-1, 1) for a in list_of_lists]
    for i in range(len(list_of_lists)):
        scaler = MinMaxScaler()
        scaler.fit(list_of_lists[i])
        list_of_lists[i] = list(
            np.array(scaler.transform(list_of_lists[i])).flat)

    return list_of_lists if len(list_of_lists) > 1 else list_of_lists[0]


def use_max_scaler(list):
    maxa = max(list)
    converted_list = [(a/maxa) for a in list]
    return converted_list


def unify_y_column_format(test_y, predicted, selected_owners, treshold):
    test_converted = [1 if i in selected_owners else 0 for i in test_y]

    predict_converted = [1 if i <= treshold else 0 for i in predicted]

    return test_converted, predict_converted


def adapt_columns_for_evaluation(Y, pred, y_column, predict_based_on_whole_pattern):
    if predict_based_on_whole_pattern:

        df_results = Y[[y_column, 'id']]
        df_results['prediction'] = pred
        nieco = df_results.groupby([y_column, 'id']).median()
        nieco = nieco.reset_index().drop('id', axis=1)

        ground_truth, predicted = nieco[y_column], nieco['prediction']
    else:
        ground_truth, predicted = Y[y_column], pred

    return ground_truth, predicted


def balance_the_combinations(users_comb):
    count = Counter(list(itertools.chain(*users_comb))).most_common()
    most_common = count[0]
    least_common = count[-1]
    last_most_common_name = most_common[0]
    last_most_common_count = most_common[1]
    while True:

        if (most_common[1] - least_common[1]) > 1:
            for i in range(len(users_comb)):
                if most_common[0] in users_comb[i] and least_common[0] not in users_comb[i]:
                    index_of_most_common = users_comb[i].index(most_common[0])
                    users_comb[i][index_of_most_common] = least_common[0]
                    break

            count = Counter(list(itertools.chain(*users_comb))).most_common()
            most_common = count[0]
            least_common = count[-1]
            if(last_most_common_name == most_common[0] and last_most_common_count == most_common[1]):
                break
            last_most_common_name = most_common[0]
            last_most_common_count = most_common[1]

        else:
            break


def get_combinations_for_cv(list, i_comb):
    # random.seed(0)
    # random.shuffle(list)
    if i_comb == 1:
        return list

    users_comb = []
    end = len(list)

    if i_comb == 2:
        step = 6
        for i in range(end):
            for j in range(i+step, end, step):
                # print(i, j)
                users_comb.append([list[i], list[j]])

    elif i_comb == 3:
        step = 3
        for i in range(end):
            for j in range(i+step, end, step*2):
                for k in range(j+step, end, step):
                    # print(i, j, k)
                    users_comb.append([list[i], list[j], list[k]])
                    break

    elif i_comb == 4:
        step = 2
        for i in range(end):
            for j in range(i+step, end, step*3):
                for k in range(j+step, end, step*2):
                    for l in range(k+step, end, step):
                        # print(i, j, k, l)
                        users_comb.append([list[i], list[j], list[k], list[l]])
                        break
                    break
    
    balance_the_combinations(users_comb)
    return users_comb
