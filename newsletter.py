# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:28:39 2017

@author: C
"""


import pandas as pd
from sklearn.model_selection import train_test_split
print(pd.__version__)

df = pd.read_csv('data_set.csv', index_col='us_id')
"""
print(df.info())
print(df.describe())
print(df.dtypes)
"""

df.sex = df.sex.map({'M':0, 'F':1})
x_train, x_test, y_train, y_test = train_test_split(df, test_size=0.2)