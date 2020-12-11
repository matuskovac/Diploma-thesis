from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from behalearn.metrics import eer_score, fmr_score, fnmr_score
from packages.models import models
from packages.processing import postprocess, split
from shapely.geometry import LineString, Point
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


def show_results(test_y, predicted_y):
    print('Accuracy:', accuracy_score(test_y, predicted_y))
    print('F1 score:', f1_score(test_y, predicted_y, pos_label=1))
    print('Recall:', recall_score(test_y, predicted_y, pos_label=1))
    print('Precision:', precision_score(test_y, predicted_y, pos_label=1))
    print('\n confussion matrix:\n', confusion_matrix(test_y, predicted_y))


def plot_far_eer(test_y_raw, tresholds, selected_owners):
    tresholds, test_y_raw = zip(*sorted(zip(tresholds, test_y_raw)))
    fmr_array = []
    fnmr_array = []
    for treshold in tresholds:
        test_y, predicted_y = postprocess.unify_y_column_format(
            test_y_raw, tresholds, selected_owners, treshold)

        fmr_array.append(fmr_score(test_y, predicted_y))
        fnmr_array.append(fnmr_score(test_y, predicted_y))

#     point = eer_score(list(tresholds), fmr_array, fnmr_array)

    line1 = LineString(list(zip(tresholds, fmr_array)))
    line2 = LineString(list(zip(tresholds, fnmr_array)))

    int_pt = line1.intersection(line2)

    plt.plot(tresholds, fmr_array, 'r')  # plotting t, a separately
    plt.plot(tresholds, fnmr_array, 'b')  # plotting t, b separately
    plt.plot(int_pt.x, int_pt.y, marker='o', markersize=5, color="green")
    plt.show()

    print("EER: "+str(int_pt.y))


def get_eer(test_y_raw, tresholds, selected_owners):

    tresholds, test_y_raw = zip(*sorted(zip(tresholds, test_y_raw)))

    fmr_array = []
    fnmr_array = []
    for treshold in tresholds:
        test_y, predicted_y = postprocess.unify_y_column_format(
            test_y_raw, tresholds, selected_owners, treshold)

        fmr_array.append(fmr_score(test_y, predicted_y))
        fnmr_array.append(fnmr_score(test_y, predicted_y))

    line1 = LineString(list(zip(tresholds, fmr_array)))
    line2 = LineString(list(zip(tresholds, fnmr_array)))
    int_pt = line1.intersection(line2)
    
    if hasattr(int_pt, 'y'):
        result = int_pt.y
    else:
        result = 1
    
    return result

def cross_validate(x_columns, y_column, df_raw_train, df_raw_val, df_raw_test, owners, model, params, predict_based_on_whole_pattern, kind_of_patten):

    test_eer_array = []
    val_eer_array = []
    train_eer_array = []

    for selected_owners in owners:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, kind_of_patten)

        predicted_train, predicted_val, predicted_test = models.use_model(
            model, [df_train, df_val, df_test], x_columns, params)

        ground_truth_train, predicted_train = postprocess.adapt_columns_for_evaluation(
            df_train[[y_column, 'id']], predicted_train, y_column, predict_based_on_whole_pattern)

        train_eer_array.append(get_eer(
            ground_truth_train, predicted_train, selected_owners))

        ground_truth_val, predicted_val = postprocess.adapt_columns_for_evaluation(
            df_val[[y_column, 'id']], predicted_val, y_column, predict_based_on_whole_pattern)
        val_eer_array.append(get_eer(
            ground_truth_val, predicted_val, selected_owners))

        ground_truth_test, predicted_test = postprocess.adapt_columns_for_evaluation(
            df_test[[y_column, 'id']], predicted_test, y_column, predict_based_on_whole_pattern)
        test_eer_array.append(get_eer(
            ground_truth_test, predicted_test, selected_owners))

    train_eer = mean(train_eer_array)
    val_eer = mean(val_eer_array)
    test_eer = mean(test_eer_array)
    return train_eer, val_eer, test_eer


def cross_validate_with_ensemble(models_dict, models_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, owners, predict_based_on_whole_pattern, kind_of_patten, apply_to_segments, function, scale_function):

    test_eer_array = []
    val_eer_array = []
    train_eer_array = []

    for selected_owners in owners:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, kind_of_patten)

        predicted_trains = []
        predicted_vals = []
        predicted_tests = []
        for model_to_use in models_to_use:

            predicted_train, predicted_val, predicted_test = models.use_model(
                models_dict[model_to_use]['name'], [df_train, df_val, df_test], selected_features_dict[models_dict[model_to_use]['x_columns']], models_dict[model_to_use]['params'])
            predicted_train, predicted_val, predicted_test = scale_function(
                [predicted_train, predicted_val, predicted_test])
            predicted_trains.append(predicted_train)
            predicted_vals.append(predicted_val)
            predicted_tests.append(predicted_test)

        if apply_to_segments:
            x = np.array(predicted_trains)
            predicted_trains = [list(function(x, axis=0))]

            x = np.array(predicted_vals)
            predicted_vals = [list(function(x, axis=0))]

            x = np.array(predicted_tests)
            predicted_tests = [list(function(x, axis=0))]

        for i in range(len(predicted_trains)):
            ground_truth_train, predicted_trains[i] = postprocess.adapt_columns_for_evaluation(
                df_train[[y_column, 'id']], predicted_trains[i], y_column, predict_based_on_whole_pattern)

            ground_truth_val, predicted_vals[i] = postprocess.adapt_columns_for_evaluation(
                df_val[[y_column, 'id']], predicted_vals[i], y_column, predict_based_on_whole_pattern)

            ground_truth_test, predicted_tests[i] = postprocess.adapt_columns_for_evaluation(
                df_test[[y_column, 'id']], predicted_tests[i], y_column, predict_based_on_whole_pattern)

        x = np.array(predicted_trains)
        predicted_train = function(x, axis=0)

        x = np.array(predicted_vals)
        predicted_val = function(x, axis=0)

        x = np.array(predicted_tests)
        predicted_test = function(x, axis=0)

        train_eer_array.append(get_eer(
            ground_truth_train, predicted_train, selected_owners))
        val_eer_array.append(get_eer(
            ground_truth_val, predicted_val, selected_owners))
        test_eer_array.append(get_eer(
            ground_truth_test, predicted_test, selected_owners))

    train_eer = mean(train_eer_array)
    val_eer = mean(val_eer_array)
    test_eer = mean(test_eer_array)
    return train_eer, val_eer, test_eer


def split_by_owners(df_train, y_column, split):
    if not split:
        return [df_train]

    names_to_split = df_train[y_column].unique()
    df_trains = []
    for name_to_split in names_to_split:
        df_trains.append(df_train.loc[df_train[y_column] == name_to_split])

    return df_trains


def cross_validate_with_ensemble3(models_dict, model_to_use, selected_features_dict, y_column, df_raw_train, df_raw_val, df_raw_test, owners, predict_based_on_whole_pattern, kind_of_patten, apply_to_segments, ensemble_based_on_users=False, function_to_ensemble_users=None):

    test_eer_array = []
    val_eer_array = []
    train_eer_array = []

    for selected_owners in owners:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, kind_of_patten)

        df_trains = split_by_owners(
            df_train, y_column, ensemble_based_on_users)

        predicted_trains_models = []
        predicted_vals_models = []
        predicted_tests_models = []
        for df_train in df_trains:
            predicted_train_models, predicted_val_models, predicted_test_models = models.use_model(
                models_dict[model_to_use]['name'], [df_train, df_val, df_test], selected_features_dict[models_dict[model_to_use]['x_columns']], models_dict[model_to_use]['params'])
            predicted_trains_models.append(predicted_train_models)
            predicted_vals_models.append(predicted_val_models)
            predicted_tests_models.append(predicted_test_models)

        if apply_to_segments:
            # x = np.array(predicted_trains_models)
            # predicted_trains_models = [
            #     list(function_to_ensemble_users(x, axis=0))]

            x = np.array(predicted_vals_models)
            predicted_vals_models = [
                list(function_to_ensemble_users(x, axis=0))]

            x = np.array(predicted_tests_models)
            predicted_tests_models = [
                list(function_to_ensemble_users(x, axis=0))]

        for i in range(len(predicted_vals_models)):
            # ground_truth_train, predicted_trains_models[i] = postprocess.adapt_columns_for_evaluation(
            #     df_train[[y_column, 'id']], predicted_trains_models[i], y_column, predict_based_on_whole_pattern)

            ground_truth_val, predicted_vals_models[i] = postprocess.adapt_columns_for_evaluation(
                df_val[[y_column, 'id']], predicted_vals_models[i], y_column, predict_based_on_whole_pattern)

            ground_truth_test, predicted_tests_models[i] = postprocess.adapt_columns_for_evaluation(
                df_test[[y_column, 'id']], predicted_tests_models[i], y_column, predict_based_on_whole_pattern)

        # x = np.array(predicted_trains_models)
        # predicted_train = function_to_ensemble_users(x, axis=0)

        x = np.array(predicted_vals_models)
        predicted_val = function_to_ensemble_users(x, axis=0)

        x = np.array(predicted_tests_models)
        predicted_test = function_to_ensemble_users(x, axis=0)

        # train_eer_array.append(get_eer(
        #     ground_truth_train, predicted_train, selected_owners))
        val_eer_array.append(get_eer(
            ground_truth_val, predicted_val, selected_owners))
        test_eer_array.append(get_eer(
            ground_truth_test, predicted_test, selected_owners))

    # train_eer = mean(train_eer_array)
    train_eer = None
    val_eer = mean(val_eer_array)
    test_eer = mean(test_eer_array)
    return train_eer, val_eer, test_eer
