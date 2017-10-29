# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
import numpy as np

from sklearn import __version__ as sk_version
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    #df.dropna(inplace=True)
    # create target series
    target = df['target']
    # deleate tearget from data set
    df.drop('target', 1, inplace=True)
    # mapping variables to intiger values
    df.sex = df.sex.map({'M':0, 'F':1})
    # create training and test sets
    
    #global corr
    #corr = np.abs((np.corrcoef(df.T)))
    #import pdb; pdb.set_trace()
    
    return train_test_split(df, target, test_size=0.2, random_state=43)
    
    
def create_pipeline(classifier):
    imp = Imputer(missing_values=0, strategy="mean", axis=0)
    var_thr = VarianceThreshold(threshold=1)
    scaler = RobustScaler()
    clf = classifier
    
    return Pipeline(steps=[('imp', imp),
                           ('var_thr', var_thr),
                           ('scaler', scaler),
                           ('clf', clf)])


def fit_and_test_model(clf, x_test, y_test, x_train, y_train):
    clf.fit_transform(x_train, y_train)
    predict = clf.predict(x_test)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('score train result: {}'.format(clf.score(x_train, y_train)))
    print('score test result: {}'.format(clf.score(x_test, y_test)))
    print(classification_report(y_test, predict))
    print(confusion_matrix(y_test, predict))


def fit_and_grid_model(clf, params, x_train, y_train):
    search_func = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1)
    search_func.fit(x_train, y_train)
    print('#'*70 + '\nRAPORT\n' + '#'*70)
    print('best params: {}'.format(search_func.best_params_))
    print('best score: {}'.format(search_func.best_score_))
    

if __name__ == '__main__':
    check_library_version()
    x_train, x_test, y_train, y_test = download_data(FILE_NAME)
    # score test result: 0.7806977797915723
    #clf = DecisionTreeClassifier()
    # score test result: 0.8309167799426068
    clf = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=43)
    params = {'clf__n_estimators': np.arange(5,50)}
    pipeline = create_pipeline(clf)
    fit_and_grid_model(pipeline, params, x_train, y_train)
    #fit_and_test_model(pipeline, x_test, y_test, x_train, y_train)
