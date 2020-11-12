import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV

from packages.config import config
from packages.models.models import RandomForestClassifierWithCoef

x_columns = config.X_COLUMNS
y_column = config.Y_COLUMNS
path_to_featutes = config.PATH_TO_FEATURES

df_raw_train = pd.read_csv(
    path_to_featutes + "imputed/" + "train.csv", sep=',')

selected_features_dict = {}
selected_features_dict[0] = x_columns


models = [RandomForestClassifierWithCoef(
    n_estimators=500, min_samples_leaf=5, n_jobs=-1)]


for model in models:
    print(model)
    rfe = RFE(model, n_features_to_select=1)
    fit = rfe.fit(df_raw_train[x_columns], df_raw_train[y_column])

    for number in [x * 10 for x in list(range(1, 15))]:
        indexes_to_delete = []
        for i in range(len(fit.ranking_)):
            if(fit.ranking_[i] > number):
                indexes_to_delete.append(i)
        selected_features = [i for j, i in enumerate(
            x_columns) if j not in indexes_to_delete]
        selected_features_dict[str(number) + "_" +
                               str(model)] = selected_features

    rfe = RFECV(estimator=model, verbose=2)
    fit = rfe.fit(df_raw_train[x_columns], df_raw_train[y_column])

    indexes_to_delete = []
    for i in range(len(fit.ranking_)):
        if(fit.ranking_[i] != 1):
            indexes_to_delete.append(i)
    selected_features = [i for j, i in enumerate(
        x_columns) if j not in indexes_to_delete]
    selected_features_dict["A_" + str(model)] = selected_features

for k in selected_features_dict.keys():
    new_key = ' '.join(str(k).replace("\\n", " ").split())
    selected_features_dict[new_key] = selected_features_dict.pop(k)


with open('./packages/config/selected_features.pickle', 'wb') as f:
    pickle.dump(selected_features_dict, f)
