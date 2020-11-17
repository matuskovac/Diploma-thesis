from statistics import mean

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

    if(all(x == 0.0 for x in tresholds)):
        return 0

    line1 = LineString(list(zip(tresholds, fmr_array)))
    line2 = LineString(list(zip(tresholds, fnmr_array)))

    int_pt = line1.intersection(line2)

    return int_pt.y


def cross_validate(x_columns, y_column, df_raw_train, df_raw_val, df_raw_test, owners, model, params, predict_based_on_whole_pattern):
    test_eer_array = []
    val_eer_array = []
    train_eer_array = []

    for selected_owners in owners:
        df_train, df_val, df_test = split.adapt_dfs_to_users(
            df_raw_train, df_raw_val, df_raw_test, selected_owners, y_column, 2)

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
