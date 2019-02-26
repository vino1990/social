# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:02:11 2018

@author: kunals
"""

'''
Sale price (predicted) = (a number) + (another number)(house size) + 
(yet another number)(number of rooms) + 
(a fourth number)*(house condition) + …
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/train.csv')
data_test = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/test.csv')
data_train.info()

from sklearn import linear_model

X = data_train.drop('SalePrice',1)
y = np.log(data_train.SalePrice)

ols = linear_model.LinearRegression()
ols.fit(X, y)
y_test_predicted = ols.predict(data_test)
print(y_test_predicted)























