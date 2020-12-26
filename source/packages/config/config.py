import pickle


SEGMENT_COLUMN = 'segment'
USER_COLUMN = 'user'
USER_NAME_COLUMN = 'username'
COMPUTE_FEATURES_FOR_SEGMENT = False
DELETE_NAN_FEATURES = False

PATH_TO_RAW_DATA = '../login_datasets/2019-01-08_FIIT_-2-poschodie_po_skuske_KPAIS_correct_patterns_only/'
PATH_TO_FEATURES = "../login_features/" + \
    ("segments" if COMPUTE_FEATURES_FOR_SEGMENT else "paterns") + \
    ("" if DELETE_NAN_FEATURES else "_nan") + "/"


def get_columns_to_identify_features():
    columns_to_identify_features = [
        'id', 'pattern_id', 'device', 'scenario', USER_NAME_COLUMN, USER_COLUMN]
    if COMPUTE_FEATURES_FOR_SEGMENT:
        columns_to_identify_features.append(SEGMENT_COLUMN)
    return columns_to_identify_features


# settings for segment based predictions
MODELS_DICT1 = {
    'knn': {'name': 'knn', 'params': {'n': 2, 'p': 1}, 'x_columns': '10_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'svm': {'name': 'svm', 'params': {'kernel': 'rbf'}, 'x_columns': '20_LogisticRegression()'},
    'isolationF': {'name': 'ísolationForest', 'params': {'n_estimators': 300}, 'x_columns': '10_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'autoencoder': {'name': 'autoencoder', 'params': {'hidden_neurons': [10, 2, 2, 10]}, 'x_columns': '10_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'lsanomaly': {'name': 'lsanomaly', 'params': {'sigma': 6, 'rho': 0.01}, 'x_columns': '50_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'}
}

# settings for whole pattern
MODELS_DICT2 = {
    'knn': {'name': 'knn', 'params': {'n': 2, 'p': 2}, 'x_columns': '20_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'svm': {'name': 'svm', 'params': {'kernel': 'rbf'}, 'x_columns': '20_LogisticRegression()'},
    'isolationF': {'name': 'ísolationForest', 'params': {'n_estimators': 500}, 'x_columns': '10_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'autoencoder': {'name': 'autoencoder', 'params': {'hidden_neurons': [20, 10, 3, 10, 20]}, 'x_columns': '20_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'},
    'lsanomaly': {'name': 'lsanomaly', 'params': {'sigma': 2, 'rho': 0.01}, 'x_columns': '10_RandomForestClassifierWithCoef(min_samples_leaf=5, n_estimators=500, n_jobs=-1)'}
}
MODELS_DICT = (MODELS_DICT1 if COMPUTE_FEATURES_FOR_SEGMENT else MODELS_DICT2)

Y_COLUMNS = USER_NAME_COLUMN

file = open("./packages/config/selected_features.pickle", 'rb')
SELECTED_FEATURES_DICT1 = pickle.load(file)
file.close()

file = open("./packages/config/selected_features2.pickle", 'rb')
SELECTED_FEATURES_DICT2 = pickle.load(file)
file.close()


SELECTED_FEATURES_DICT = (
    SELECTED_FEATURES_DICT1 if COMPUTE_FEATURES_FOR_SEGMENT else SELECTED_FEATURES_DICT2)
X_COLUMNS = SELECTED_FEATURES_DICT['0']
