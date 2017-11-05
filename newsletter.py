# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
import numpy as np

from sklearn import __version__ as sk_version
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedShuffleSplit
)
from sklearn.preprocessing import Imputer, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from matplotlib import __version__ as plt_version
import matplotlib.pyplot as plt


FILE_NAME = 'data_set.csv'

def check_library_version():
    if pd.__version__ == '0.18.1':
        print('pandas version: OK')
    else:
        print('pandas version: WRONG \n\tcorrect version -> 0.18.1')
    if np.__version__ == '1.13.1':
        print('numpy version: OK')
    else:
        print('numpy version: WRONG \n\tcorrect version ->1.13.1')
    if sk_version == '0.18.2':
        print('sklearn version: OK')
    else:
        print('sklearn version: WRONG \n\tcorrect version -> 0.18.2')
    if plt_version == '1.5.1':
        print('matplotlib version: OK')
    else:
        print('matplotlib version: WRONG \n\tcorrect version -> 1.5.1')


def download_data(file):
    df = pd.read_csv(file, index_col='us_id')
    # create target series
    target = df['target']
    # deleate tearget from data set
    df.drop('target', 1, inplace=True)
    # mapping variables to intiger values
    df.sex = df.sex.map({'M':0, 'F':1})
    # create training and test sets
    return train_test_split(df, target, test_size=0.2, random_state=43)


def multiple_traning_set(x_train, y_train, target, times):
    ''' multiple training set, times difine how many times a data will be append to set '''
    series = y_train[y_train==target]
    dupli = x_train.loc[series.index.tolist(), :]
    for _ in range(times):
        x_train = x_train.append(dupli)
        y_train = y_train.append(series)
    
    return x_train, y_train


def create_pipeline(classifier):
    imp = Imputer(strategy="most_frequent", axis=0)
    var_thr = VarianceThreshold(threshold=1.7)
    #rbf_feature = RBFSampler(gamma=1, random_state=1)
    #pca = PCA(n_components=16)
    #scaler = StandardScaler()

    return Pipeline(steps=[('imp', imp),
                           ('var_thr', var_thr),
                           #('pca', pca),
                           #('scaler', scaler),
                           #('rbf_feature', rbf_feature),
                           ('clf', classifier)
    ])


def plot_scatter(data, target):
    colors_palette = {0: 'yellow', 1: 'black'}
    colors = [colors_palette[x] for x in target]
    data = pd.DataFrame(data)
    pd.tools.plotting.scatter_matrix(data, alpha=0.2, color=colors)


def fit_and_test_model(clf, x_test, y_test, x_train, y_train):
    cross = cross_val_score(clf, x_train, y_train, cv=5)
    print(cross)
    clf.fit(x_train, y_train)
    predict_train = clf.predict(x_train)
    predict_test = clf.predict(x_test)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('\nscore train result: {}'.format(clf.score(x_train, y_train)))
    print(classification_report(y_train, predict_train))
    print(confusion_matrix(y_train, predict_train))
    print('\nscore test result: {}'.format(clf.score(x_test, y_test)))
    print(classification_report(y_test, predict_test))
    print(confusion_matrix(y_test, predict_test))
    
    return clf


def fit_and_grid_model(clf, params, x_train, y_train):
    search_func = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1)
    search_func.fit(x_train, y_train)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('best params: {}'.format(search_func.best_params_))
    print('best score: {}'.format(search_func.best_score_))


def recruitment_elimination(clf, x_train, y_train, x_test, y_test):
    
    imp = Imputer(missing_values=0, strategy="median", axis=0)
    x_train = imp.fit_transform(x_train)
    x_test = imp.transform(x_test)

    selector = RFECV(estimator=clf)
    selector.fit(x_train, y_train)
    print('select features: {}'.format(selector.n_features_))
    x_train_s = selector.transform(x_train)
    x_test_s = selector.transform(x_test)
    fit_and_test_model(clf, x_test_s, y_test, x_train_s, y_train)


def union_estimators(last_estimator, x_train, y_train, x_test, y_test, *estimators):
    df = pd.DataFrame()
    clf_list = []
    for index, est in enumerate(estimators):
        est.fit(x_train, y_train)
        predict = est.predict_proba(x_train)
        df[index*2+1] = predict[:,0]
        df[index*2+2] = predict[:,1]
        clf_list.append(est)
    last_estimator.fit(df, y_train)
    predict_train = last_estimator.predict(df)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('\nscore train result: {}'.format(last_estimator.score(df, y_train)))
    print(classification_report(y_train, predict_train))
    print(confusion_matrix(y_train, predict_train))
    test_union_estimators(last_estimator, x_test, y_test, clf_list)


def test_union_estimators(last_estimator, x_test, y_test, clf_list):
    df = pd.DataFrame()
    for index, est in enumerate(clf_list):
        predict = est.predict_proba(x_test)
        df[index*2+1] = predict[:,0]
        df[index*2+2] = predict[:,1]
    predict_test = last_estimator.predict(df)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('\nscore test result: {}'.format(last_estimator.score(df, y_test)))
    print(classification_report(y_test, predict_test))
    print(confusion_matrix(y_test, predict_test))


def prepare_raport(clf, x_test, y_test):
    result = [(index, predict[index]) for predict, index in zip(clf.predict_proba(x_test), y_test)]
    df_result = pd.DataFrame(data=result, index=y_test.index, columns=['target', 'target_score'])
    df_result.to_csv(path_or_buf='submmision_form.csv', sep=',')

    
if __name__ == '__main__':
    check_library_version()
    x_train, x_test, y_train, y_test = download_data(FILE_NAME)
    x_train, y_train = multiple_traning_set(x_train, y_train, 1, 4)
    
    clf = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, max_features=0.5, max_depth=15, 
        random_state=1
    )

    clf = fit_and_test_model(create_pipeline(clf), x_test, y_test, x_train, y_train)
    prepare_raport(clf, x_test, y_test)