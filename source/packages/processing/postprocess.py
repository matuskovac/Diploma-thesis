from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


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
