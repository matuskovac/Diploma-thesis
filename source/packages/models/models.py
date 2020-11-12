from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def use_knn(df_x_train, df_x_test, count_neighbors=1, l=2):
    neigh = NearestNeighbors(n_neighbors=count_neighbors, p=l)
    neigh.fit(df_x_train)
    knn = neigh.kneighbors(df_x_test)
    return [np.mean(i) for i in knn[0]]


def use_svm(df_x_train, df_x_test, c=1):
    clf = OneClassSVM(kernel='sigmoid').fit(df_x_train)
    svm = clf.score_samples(df_x_test)
    return svm


def get_knn(count_neighbors=1, l=2):
    neigh = NearestNeighbors(n_neighbors=count_neighbors, p=l)
    return neigh


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
