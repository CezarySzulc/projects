# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
from sklearn import __version__ as sk_version
from sklearn.model_selection import train_test_split


def check_library_version():
    if pd.__version__ == '0.18.1':
        print('pandas version: OK')
    else:
        print('pandas version: WRONG \n\tcorrect version -> 0.18.1')
    if sk_version == '0.18.2':
        print('sklearn version: OK')
    else:
        print('sklearn version: WRONG \n\tcorrect version -> 0.18.2')


df = pd.read_csv('data_set.csv', index_col='us_id')
# create target series
target = df['target']
# deleate tearget from data set
df.drop('target', 1, inplace=True)
"""
print(df.info())
print(df.describe())
print(df.dtypes)
"""
def prepare_data(df):
    # mapping variables to intiger values
    df.sex = df.sex.map({'M':0, 'F':1})
    # create training and test sets
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
    
if __name__ == '__main__':
    check_library_version()