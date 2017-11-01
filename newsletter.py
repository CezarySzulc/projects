# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
import numpy as np

from sklearn import __version__ as sk_version
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import Imputer, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.decomposition import RandomizedPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
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
    x_train, x_test, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=43
    )
    
    # multiple training set
    series = y_train[y_train==1]
    dupli = x_train.loc[series.index.tolist(), :]
    for _ in range(6):
        x_train = x_train.append(dupli)
        y_train = y_train.append(series)
    
    return x_train, x_test, y_train, y_test


def create_pipeline(classifier):
    imp = Imputer(strategy="mean", axis=0)
    var_thr = VarianceThreshold(threshold=1.8)
    pca = RandomizedPCA(n_components=34)
    scaler = RobustScaler()
    
    return Pipeline(steps=[('imp', imp),
                           ('var_thr', var_thr),
                           ('pca', pca),
                           ('scaler', scaler),
                           ('clf', classifier)
    ])


def fit_and_test_model(clf, x_test, y_test, x_train, y_train):
    cross = cross_val_score(clf, x_train, y_train)
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


if __name__ == '__main__':
    check_library_version()
    x_train, x_test, y_train, y_test = download_data(FILE_NAME)
    # score test result: 0.7806977797915723
    #clf = DecisionTreeClassifier()
    # score test result: 0.8309167799426068 recall: 0.05
    #clf = RandomForestClassifier(n_estimators=47, n_jobs=-1, random_state=1)
    # score test result: 0.8080350400241655 recall: 0.12
    
    clf = ExtraTreesClassifier(
        n_estimators=500, max_features=0.5, max_depth=10,
        n_jobs=-1, random_state=1,  class_weight={0:.1, 1:.5}
    )
    
    # score test result: 0.8389971303428485  recall: 0.01!
    #clf = AdaBoostClassifier(n_estimators=5)
    # score test result: 0.8383929919951669 recall: 0.00!
    #clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5)
    
    #clf = OneClassSVM(kernel='sigmoid')
    # 0.8385440265820873
    #clf = LogisticRegression()
    '''
    params = {'clf__n_estimators': np.arange(100, 1000, 10),
              'clf__max_features': np.arange(0.3, 0.6, 0.05), 
              'clf__max_depth': np.arange(5, 15)
    }
    '''
    pipeline = create_pipeline(clf)
    #fit_and_grid_model(pipeline, params, x_train, y_train)
    fit_and_test_model(pipeline, x_test, y_test, x_train, y_train)
    #recruitment_elimination(clf, x_train, y_train, x_test, y_test)
