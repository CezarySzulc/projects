# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
from sklearn import __version__ as sk_version
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


FILE_NAME = 'data_set.csv'

def check_library_version():
    if pd.__version__ == '0.18.1':
        print('pandas version: OK')
    else:
        print('pandas version: WRONG \n\tcorrect version -> 0.18.1')
    if sk_version == '0.18.2':
        print('sklearn version: OK')
    else:
        print('sklearn version: WRONG \n\tcorrect version -> 0.18.2')


def download_data(file):
    df = pd.read_csv(file, index_col='us_id')
    # DELETE!!!
    df.dropna(inplace=True)
    #
    # create target series
    target = df['target']
    # deleate tearget from data set
    df.drop('target', 1, inplace=True)
    # mapping variables to intiger values
    df.sex = df.sex.map({'M':0, 'F':1})
    # create training and test sets
    return train_test_split(df, target, test_size=0.2)
    
def create_pipeline():
    var_thr = VarianceThreshold(threshold=1)
    clf = DecisionTreeClassifier()
    
    return Pipeline(steps=[('var_thr', var_thr),
                           ('clf', clf)])
                           
                           
def fit_and_test_model(clf, x_test, y_test, x_train, y_train):
    clf.fit_transform(x_train, y_train)
    print('score train result: {}'.format(clf.score(x_train, y_train)))
    print('score test result: {}'.format(clf.score(x_test, y_test)))
    print(classification_report(y_test, clf.predict(x_test)))
                           
        
if __name__ == '__main__':
    check_library_version()
    x_train, x_test, y_train, y_test = download_data(FILE_NAME)
    pipeline = create_pipeline()
    fit_and_test_model(pipeline, x_test, y_test, x_train, y_train)
