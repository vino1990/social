# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 08:33:06 2018

@author: kunals
"""

# Section 2: Getting to know the data. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/train.csv')
data_test = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/test.csv')
data_train.info()

def count_missing(data):
    null_cols = data.columns[data.isnull().any(axis=0)]
    X_null = data[null_cols].isnull().sum()
    X_null = X_null.sort_values(ascending=False)
    print(X_null)
 
data_X = pd.concat([data_train.drop('SalePrice',1), data_test])
print(len(data_X))
count_missing(data_X)

# Missing for a reason

# some of the missing values are in fact meaningful. 
# For example, missing values for garage, pool or basement-related features simply imply that the house does not have a garage, pool or basement respectively.
# In this case, it makes sense to fill these missing values with something that captures this information.

# we can replace missing values in such cases with a new value called ‘None’
catfeats_fillnaNone = ['Alley',
    'BsmtCond','BsmtQual','BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature']
 
data_X.loc[:,catfeats_fillnaNone] = data_X[catfeats_fillnaNone].fillna('None')

# for most numerical features of this kind, it makes sense to replace the missing values with zero:

numfeats_fillnazero = ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF',
     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
     'GarageArea', 'GarageCars']
 
data_X.loc[:,numfeats_fillnazero] = data_X[numfeats_fillnazero].fillna(0)

data_X.loc[:,'GarageYrBlt'] = data_X['GarageYrBlt'].fillna(data_X.YearBuilt)

count_missing(data_X)

# delete, impute or leave.

# 1. Delete

'''

In cases where a very large portion of values are missing for a given feature, 
we may simply decide to drop that feature (column) altogether. 
Similarly, if almost all feature values are missing for a given entry (row),
we may decide to delete that row. 

The downside, of course, 
is losing potentially valuable information, which is particularly problematic
if the dataset is small compared to the number of rows or columns 
with missing values. In the house prices dataset, the fraction of missing
entries in any given row or column is not very high 
(at most 17% missing for LotFrontage), and it will likely be better 
to keep all of our data.
'''

# 2. Impute

'''
If we do decide to keep all of our data, we will generally need to fill 
in – or ‘impute’ – the missing entries. The majority of machine learning 
algorithms cannot handle null values, so modelling will only be possible 
after we do so.

The crudest option is to simply replace each missing entry by the mean, 
median or mode of the given feature, which gives us the roughest possible 
estimate for what the missing value might be. We can implement this for 
our house prices dataset as follows (using mode and median for categorical 
and numerical features respectively)


A more sophisticated approach might involve using what we know about the 
feature’s relationship to other features to guess the missing values.
 For example, if we have 5 features (F1, … F5) and F1 has some missing 
 values, we can treat F1 as our target variable and train a model on 
 (F2, … F5) to predict what the missing values might be.

'''


catfeats_fillnamode = ['Electrical', 'MasVnrType', 'MSZoning', 'Functional', 'Utilities',
     'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
 
data_X.loc[:, catfeats_fillnamode] = data_X[catfeats_fillnamode].fillna(data_X[catfeats_fillnamode].mode().iloc[0])
 
numfeats_fillnamedian = ['MasVnrArea', 'LotFrontage']
 
data_X.loc[:, numfeats_fillnamedian] = data_X[numfeats_fillnamedian].fillna(data_X[numfeats_fillnamedian].median())

# 3. Leave

'''
While the majority of machine learning algorithms cannot handle missing 
values natively, exceptions do exist. For example, the powerful and 
widely used xgboost library happily handles data of this kind. 
'''
# NON-NUMERICAL FEATURES

print(data_X.dtypes.value_counts())

print(data_X.select_dtypes(include = [object]).columns)

'''
Since the majority of available machine learning algorithms can only take numbers 
(floats or integers) as inputs, we must encode these features numerically 
if we are to use them in our models.

non-numerical variables tend to come in two flavours: ordinal and categorical. 
Ordinal variables – such as OverallQual or LotShape – have an intrinsic order to them, 
while purely categorical variables – such as Neighborhood or Foundation
'''

# Ordinal features

'''
Since ordinal features are inherently ordered, they lend themselves 
naturally to numerical encoding. For example, the possible values for 
LotShape are Reg (regular), IR1 (slightly irregular), IR2 (moderately irregular)
and IR3 (irregular), to which we could assign the values (0,1,2,3) respectively:

'''

data_X.LotShape = data_X.LotShape.replace({'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3})

'''
This is known as ordinal encoding and is the most straightforward approach for 
encoding non-numerical variables: we simply assign a number to each possible 
value a feature can take. Mapping the levels of an ordinal variable to 
consecutive integers, as we have done above, is good in that it keeps 
the relative relationship between values intact.

'''

# Categorical features

# Ordinal encoding

print(data_test.Neighborhood.head(15))
print(pd.Series(pd.factorize(data_test.Neighborhood.head(15))[0]))

# Dummy encoding (aka one-hot encoding)

print(pd.get_dummies(data_test.Neighborhood.head(15), drop_first=True))

'''
The great advantage of dummy encoding is that it doesn’t impose any ordering 
on our data and ensures that the distance between each pair of values 
(neighbourhoods in this case) is the same.
'''


from sklearn import linear_model

X = data_X
y = np.log(data_train.SalePrice)

ols = linear_model.LinearRegression()
ols.fit(X, y)
y_test_predicted = ols.predict(data_test)
print(y_test_predicted)




















































