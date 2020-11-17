from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import numpy as np



class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

def use_model(model, df_list, x_columns, params):
    predicted=[]
    
    if model == 'knn':
        neigh = NearestNeighbors(n_neighbors=params['n'],p=params['p'])
        neigh.fit(df_list[0][x_columns])

        for i in range(len(df_list)):
            pred = neigh.kneighbors(df_list[i][x_columns])
            pred = [np.mean(i) for i in pred[0]]
            predicted.append(pred)
            
    elif model == 'svm':
        svm = OneClassSVM(kernel=params['kernel'])
        svm.fit(df_list[0][x_columns])

        predicted=[]
        for i in range(len(df_list)):
            pred = svm.score_samples(df_list[i][x_columns])
            predicted.append(pred)
    
    return predicted

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

def get_svm(kernel='sigmoid'):
    svm = OneClassSVM(kernel=kernel)
    return svm

def use_knn2(df_list, x_columns, **kwargs):
    neigh = NearestNeighbors(n_neighbors=kwargs['n'],p=kwargs['p'])
    neigh.fit(df_list[0][x_columns])
    
    predicted=[]
    for i in range(len(df_list)):
        pred = neigh.kneighbors(df_list[i][x_columns])
        pred = [np.mean(i) for i in pred[0]]
        predicted.append(pred)
        
    return predicted

def use_svm2(df_list, x_columns, **kwargs):
    svm = OneClassSVM(kernel=kwargs['kernel'])
    svm.fit(df_list[0][x_columns])
    
    predicted=[]
    for i in range(len(df_list)):
        pred = svm.score_samples(df_list[i][x_columns])
        predicted.append(pred)
        
    return predicted
