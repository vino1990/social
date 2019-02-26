# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 06:35:29 2018

@author: kunals
"""

'''

The general process to get from data to predictive model tends to involve three major components:

Getting to know the data.
Cleaning and preparing the data for modelling.
Fitting models and evaluating their performance.

'''
# Section 1: Getting to know the data. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/train.csv')
data_test = pd.read_csv('E:/ds_practice/case_study/housePricing/dataset/test.csv')
data_train.info()

# ome features are inherently numerical - total living area (GrLivArea)
# Some features are inherently discrete - the number of rooms (TotRmsAbvGrd).
# Other features are categorical - the neighbourhood in which the house is located (Neighborhood)
# the type of foundation the house was built on (Foundation)
# some of ordinal: the overall quality rating (OverallQual) or the irregularity of the lot (LotShape)

# EXPLORE AND VISUALIZE THE DATA


'''
The distribution of the target variable and of individual features (univariate analysis).
The relationship between pairs of variables (bivariate analysis).

'''

#  1. Univariate analysis---------------------------------------------

# for Numerical variables
# Let’s begin by generating one for SalePrice, our target variable.

#plt.hist(data_train.SalePrice, bins = 25)
# To make the distribution more symmetric, we can try taking its logarithm:
#plt.hist(np.log(data_train.SalePrice), bins = 25)

# we can think of log(SalePrice) as our true target variable.


# Categorical variables

# For categorical variables, bar charts and frequency counts are the natural alternative to histograms

data_train.Foundation.value_counts()
sns.countplot(data_train.Foundation)

# Here we can immediately see that only two types of 
# foundation (poured concrete (PConc) and 
#cinderblock (CBlock)) dominate in our dataset.

# This might think  to make a transformation.
# For example, depending on the type of model we decide to use, we may want to merge Stone, Wood and Slab into a single (‘other’) category.

# Alternatively, if we think stone and wood houses are very important, it may alert us to the fact that we have a deficiency in our data, and need to go out and collect more stone and wood examples.
# Related to this last point, it is also important to check whether the distributions in our training and test sets are similar to each other.

#sns.countplot(data_test.Foundation)

# Bivariate analysis-----------------------------------------------

'''
most interesting will be the relationship between the 
target variable (sale price) and the features we will 
use for prediction. However, as we will see, studying 
relationships among features can also be important.

'''

# Numerical variables

# For numerical features, scatter plots are the go-to tool.

plt.plot(data_train.GrLivArea, data_train.SalePrice,
         '.', alpha = 0.3)
 
plt.plot(data_train.GrLivArea, np.log(data_train.SalePrice),
         '.', alpha = 0.3)


plt.plot(data_train.YearBuilt, data_train.GarageYrBlt,
         '.', alpha=0.5, label = 'training set')
 
plt.legend()
plt.plot(data_test.YearBuilt, data_test.GarageYrBlt,
         '.', alpha=0.5, label = 'test set')
 



# we might consider creating a new feature that tells us whether or not a garage was originally constructed with the house or how many years later one was added.


# Categorical variables
# http://seaborn.pydata.org/tutorial/categorical.html
# stripplot, pointplot, boxplot and violinplot
sns.stripplot(x = data_train.Neighborhood.values, y = data_train.SalePrice.values,
              order = np.sort(data_train.Neighborhood.unique()),
              jitter=0.1, alpha=0.5)
 
plt.xticks(rotation=45)


Neighborhood_meanSalePrice = data_train.groupby('Neighborhood')['SalePrice'].mean()
 
Neighborhood_meanSalePrice = Neighborhood_meanSalePrice.sort_values()

sns.pointplot(x = data_train.Neighborhood.values, y = data_train.SalePrice.values,
              order = Neighborhood_meanSalePrice.index)
 
plt.xticks(rotation=45)























































